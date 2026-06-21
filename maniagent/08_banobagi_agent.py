"""
08_banobagi_agent.py
바노바기 성형외과 실장님 에이전트 — "지수"

데이터 소스:
  - data/processed/plastic/homepage_바노바기_성형외과.jsonl  (홈페이지 59청크)
  - data/processed/plastic/manidata_plastic_gangnam_v1.jsonl  (블로그/카페 중 바노바기 필터)

LLM 라우터: DeepSeek → Qwen → Claude

실행:
  # 대화형 CLI
  python maniagent/08_banobagi_agent.py

  # 단발 질문
  python maniagent/08_banobagi_agent.py --query "코성형 가격이 얼마예요?" --locale ko

  # 중국어 모드
  python maniagent/08_banobagi_agent.py --locale zh
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    import numpy as np
except ImportError as e:
    raise SystemExit(f"[ERROR] 패키지 없음: {e}\npip install python-dotenv numpy")

_HERE  = Path(__file__).parent
_ROOT  = _HERE.parent

load_dotenv(_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── config 로드 ───────────────────────────────────────────────────────────
def _load_mod(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_cfg = _load_mod(_ROOT / "manidata" / "06_banobagi_config.py")

AGENT_PROFILE        = _cfg.AGENT_PROFILE
PERSONA_SYSTEM_PROMPTS = _cfg.PERSONA_SYSTEM_PROMPTS
INTRO_MESSAGES       = _cfg.INTRO_MESSAGES
TASK_INSTRUCTIONS    = _cfg.TASK_INSTRUCTIONS
NO_RESULT_MESSAGES   = _cfg.NO_RESULT_MESSAGES


# ════════════════════════════════════════════════════════════════════════════
# 1. JSONL 인메모리 벡터 스토어 (Milvus 없이도 작동)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    chunk_id:     str
    text:         str
    text_zh:      str
    text_ja:      str
    text_en:      str
    source_file:  str
    chunk_type:   str
    section_type: str
    keywords:     list[str]
    clinic_name:  str
    embedding:    list[float] = field(default_factory=list)


class InMemoryVectorStore:
    """
    JSONL을 메모리에 로드하고 코사인 유사도로 검색하는 경량 벡터 스토어.
    Milvus 없이 바로 실행 가능합니다.
    """

    def __init__(self, embed_model) -> None:
        self._model  = embed_model
        self._chunks: list[Chunk]       = []
        self._vecs:   Optional[np.ndarray] = None

    def load_jsonl(self, path: Path, clinic_filter: str = "") -> int:
        """JSONL 파일을 로드합니다. clinic_filter가 있으면 해당 병원만 필터링합니다."""
        loaded = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                name = d.get("clinic_name", "")
                if clinic_filter and clinic_filter not in name:
                    continue
                self._chunks.append(Chunk(
                    chunk_id     = d.get("chunk_id", str(uuid.uuid4())),
                    text         = d.get("text", ""),
                    text_zh      = d.get("text_zh", ""),
                    text_ja      = d.get("text_ja", ""),
                    text_en      = d.get("text_en", ""),
                    source_file  = d.get("source_file", ""),
                    chunk_type   = d.get("chunk_type", ""),
                    section_type = d.get("section_type", d.get("chunk_type", "")),
                    keywords     = d.get("keywords", []),
                    clinic_name  = name,
                    embedding    = d.get("embedding", []),
                ))
                loaded += 1
        logger.info(f"  로드: {path.name} → {loaded}청크")
        return loaded

    def build_index(self) -> None:
        """청크 텍스트를 임베딩하여 인덱스를 구축합니다."""
        logger.info(f"임베딩 인덱스 구축 중 ({len(self._chunks)}청크)...")
        texts = [c.text for c in self._chunks]
        vecs  = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._vecs = np.array(vecs, dtype=np.float32)
        logger.info("인덱스 구축 완료")

    def search(self, query: str, top_k: int = 5,
               score_threshold: float = 0.3) -> list[tuple[Chunk, float]]:
        if self._vecs is None or len(self._chunks) == 0:
            return []
        q_vec = self._model.encode([query], normalize_embeddings=True)[0]
        scores = self._vecs @ q_vec           # 코사인 유사도 (정규화 벡터)
        top_idx = np.argsort(scores)[::-1][:top_k * 3]
        results = []
        for idx in top_idx:
            score = float(scores[idx])
            if score < score_threshold:
                break
            results.append((self._chunks[idx], score))
        return results[:top_k]


# ════════════════════════════════════════════════════════════════════════════
# 2. LLM 라우터 (DeepSeek → Qwen → Claude)
# ════════════════════════════════════════════════════════════════════════════

class LLMRouter:
    def __init__(self) -> None:
        self._providers: list[dict] = []
        self._setup()

    def _setup(self) -> None:
        try:
            import openai as _oi
            dk = os.getenv("DEEPSEEK_API_KEY", "")
            if dk and "here" not in dk:
                self._providers.append({
                    "name":   "deepseek",
                    "client": _oi.OpenAI(api_key=dk, base_url="https://api.deepseek.com"),
                    "model":  "deepseek-chat",
                    "mode":   "openai",
                })
            qk = os.getenv("QWEN_API_KEY", "")
            if qk and "here" not in qk:
                self._providers.append({
                    "name":   "qwen",
                    "client": _oi.OpenAI(api_key=qk,
                                         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
                    "model":  "qwen-max",
                    "mode":   "openai",
                })
        except ImportError:
            pass

        ak = os.getenv("ANTHROPIC_API_KEY", "")
        if ak and "here" not in ak:
            try:
                import anthropic as _ant
                self._providers.append({
                    "name":   "claude",
                    "client": _ant.Anthropic(api_key=ak),
                    "model":  "claude-sonnet-4-6",
                    "mode":   "anthropic",
                })
            except ImportError:
                pass

        if not self._providers:
            raise EnvironmentError("LLM API 키가 없습니다. .env 파일을 확인하세요.")
        logger.info(f"LLM 라우터 등록: {[p['name'] for p in self._providers]}")

    def complete(self, system: str, user: str,
                 temperature: float = 0.5,
                 max_tokens:  int   = 1024) -> tuple[str, str]:
        for p in self._providers:
            try:
                if p["mode"] == "openai":
                    resp = p["client"].chat.completions.create(
                        model=p["model"],
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user",   "content": user},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return resp.choices[0].message.content.strip(), p["name"]
                else:
                    resp = p["client"].messages.create(
                        model=p["model"],
                        max_tokens=max_tokens,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    return resp.content[0].text.strip(), p["name"]
            except Exception as e:
                logger.warning(f"[{p['name']}] 실패, 폴백: {e}")
        raise RuntimeError("모든 LLM 호출 실패")


# ════════════════════════════════════════════════════════════════════════════
# 3. 바노바기 실장님 에이전트
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentResponse:
    query:        str
    answer:       str
    locale:       str
    llm_provider: str
    sources:      list[dict]
    session_id:   str


class BanobagiAgent:
    """
    바노바기 성형외과 상담 실장 "지수" AI 에이전트.

    - 홈페이지 데이터 + 블로그/카페 데이터를 인메모리 벡터 스토어로 관리
    - 다국어 RAG + 페르소나 응답
    - 대화 히스토리 유지 (세션 기반)
    """

    def __init__(self, top_k: int = 5, score_threshold: float = 0.3) -> None:
        self.top_k           = top_k
        self.score_threshold = score_threshold
        self._sessions: dict[str, list[dict]] = {}

        # 임베딩 모델
        logger.info("임베딩 모델 로딩: BAAI/bge-m3")
        try:
            from sentence_transformers import SentenceTransformer
            self._embed = SentenceTransformer("BAAI/bge-m3", device="cpu")
        except ImportError:
            raise SystemExit("pip install sentence-transformers")
        logger.info("임베딩 모델 로드 완료")

        # 벡터 스토어 초기화
        self._store = InMemoryVectorStore(self._embed)
        self._load_data()

        # LLM 라우터
        self._llm = LLMRouter()

    def _load_data(self) -> None:
        """바노바기 관련 데이터를 모두 로드합니다."""
        processed = _ROOT / "data" / "processed" / "plastic"

        # 1) 홈페이지 JSONL (바노바기 전용)
        hp = processed / "homepage_바노바기_성형외과.jsonl"
        if hp.exists():
            self._store.load_jsonl(hp)
        else:
            logger.warning(f"홈페이지 JSONL 없음: {hp}")

        # 2) 통합 블로그/카페 JSONL (바노바기 필터)
        bg = processed / "manidata_plastic_gangnam_v1.jsonl"
        if bg.exists():
            self._store.load_jsonl(bg, clinic_filter="바노바기")
        else:
            logger.warning(f"통합 JSONL 없음: {bg}")

        self._store.build_index()
        logger.info(f"총 로드 청크: {len(self._store._chunks)}개")

    # ── 컨텍스트 빌더 ─────────────────────────────────────────────────────

    def _build_context(self, results: list[tuple[Chunk, float]], locale: str) -> str:
        if not results:
            return ""
        locale_field = {"zh": "text_zh", "ja": "text_ja", "en": "text_en"}.get(locale, "text")
        parts = []
        for i, (chunk, score) in enumerate(results, 1):
            body = getattr(chunk, locale_field, "") or chunk.text
            src  = chunk.chunk_type or chunk.source_file
            parts.append(f"[참고{i}] ({src}, 유사도={score:.2f})\n{body[:600]}")
        return "\n\n---\n\n".join(parts)

    # ── 히스토리 요약 ─────────────────────────────────────────────────────

    def _history_text(self, session_id: str, last_n: int = 4) -> str:
        history = self._sessions.get(session_id, [])[-last_n:]
        if not history:
            return ""
        lines = []
        for h in history:
            role = "고객" if h["role"] == "user" else "지수"
            lines.append(f"{role}: {h['content'][:200]}")
        return "\n".join(lines)

    # ── 메인 generate ─────────────────────────────────────────────────────

    def generate(self, query: str, locale: str = "ko",
                 session_id: str = "") -> AgentResponse:
        if not session_id:
            session_id = str(uuid.uuid4())
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        # 1. 검색
        results = self._store.search(query, self.top_k, self.score_threshold)

        # 2. 컨텍스트
        context = self._build_context(results, locale)
        history = self._history_text(session_id)

        # 3. 프롬프트
        system_prompt = PERSONA_SYSTEM_PROMPTS.get(locale, PERSONA_SYSTEM_PROMPTS["ko"])
        instruction   = TASK_INSTRUCTIONS.get(locale, TASK_INSTRUCTIONS["ko"])

        user_prompt_parts = []
        if history:
            user_prompt_parts.append(f"[이전 대화]\n{history}\n")
        if context:
            user_prompt_parts.append(f"[참고 자료]\n{context}\n")
        user_prompt_parts.append(f"{instruction}\n\n[고객 질문]\n{query}")
        user_prompt = "\n".join(user_prompt_parts)

        # 4. LLM 호출
        if not results:
            answer = NO_RESULT_MESSAGES.get(locale, NO_RESULT_MESSAGES["ko"])
            provider = "none"
        else:
            answer, provider = self._llm.complete(system_prompt, user_prompt)

        # 5. 히스토리 저장
        self._sessions[session_id].append({"role": "user",      "content": query})
        self._sessions[session_id].append({"role": "assistant",  "content": answer})

        sources = [
            {"source": c.source_file, "type": c.chunk_type,
             "keywords": c.keywords[:4], "score": round(s, 3)}
            for c, s in results
        ]

        return AgentResponse(
            query=query, answer=answer, locale=locale,
            llm_provider=provider, sources=sources, session_id=session_id,
        )

    def intro(self, locale: str = "ko") -> str:
        return INTRO_MESSAGES.get(locale, INTRO_MESSAGES["ko"])


# ════════════════════════════════════════════════════════════════════════════
# 4. 대화형 CLI
# ════════════════════════════════════════════════════════════════════════════

def _locale_from_input(text: str) -> Optional[str]:
    cmd = text.strip().lower()
    mapping = {"/ko": "ko", "/zh": "zh", "/ja": "ja", "/en": "en"}
    return mapping.get(cmd)


def run_cli(locale: str = "ko") -> None:
    print("\n" + "=" * 62)
    print(f"  바노바기 성형외과 AI 상담 실장 '지수'")
    print(f"  언어: {locale.upper()}  |  종료: /quit  |  언어변경: /ko /zh /ja /en")
    print("=" * 62)

    agent      = BanobagiAgent()
    session_id = str(uuid.uuid4())

    print(f"\n{agent.intro(locale)}\n")

    while True:
        try:
            prompt = "💬 고객: " if locale == "ko" else "💬 Client: "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n상담을 종료합니다. 감사합니다! 😊")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "/종료"):
            print("상담을 종료합니다. 감사합니다! 😊")
            break

        new_locale = _locale_from_input(user_input)
        if new_locale:
            locale = new_locale
            print(f"[언어 변경: {locale.upper()}]\n{agent.intro(locale)}\n")
            continue

        resp = agent.generate(user_input, locale=locale, session_id=session_id)
        print(f"\n🏥 지수 실장님: {resp.answer}\n")


# ════════════════════════════════════════════════════════════════════════════
# 5. 싱글톤 & 진입점
# ════════════════════════════════════════════════════════════════════════════

_agent_instance: Optional[BanobagiAgent] = None


def get_agent(**kwargs) -> BanobagiAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = BanobagiAgent(**kwargs)
    return _agent_instance


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="바노바기 실장님 AI 에이전트")
    p.add_argument("--query",  "-q", default="",
                   help="단발 질문 (미입력 시 대화 모드)")
    p.add_argument("--locale", "-l", default="ko",
                   choices=["ko", "zh", "ja", "en"])
    p.add_argument("--json",   action="store_true", help="JSON 출력")
    args = p.parse_args()

    if args.query:
        agent = get_agent()
        resp  = agent.generate(args.query, locale=args.locale)
        if args.json:
            print(json.dumps({
                "query": resp.query, "answer": resp.answer,
                "locale": resp.locale, "llm": resp.llm_provider,
                "sources": resp.sources,
            }, ensure_ascii=False, indent=2))
        else:
            print(f"\n🏥 지수 실장님:\n{resp.answer}\n")
            print(f"[LLM: {resp.llm_provider} | 참고: {len(resp.sources)}개]")
    else:
        run_cli(locale=args.locale)
