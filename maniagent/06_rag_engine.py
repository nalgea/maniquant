"""
05_rag_engine.py
ManiAgentRAG — RAG 검색 + LLM 라우터 + locale 다국어 응답

클래스: ManiAgentRAG
LLM 라우터: DeepSeek-V4(주력) → Qwen-max(보조) → Claude(폴백)
도메인:     anti_aging (기본) | plastic (강남 성형외과 뷰티투어)

사용법 (CLI):
    python maniagent/05_rag_engine.py \\
        --query "바노바기 성형외과 가격 알려줘" \\
        --locale zh \\
        --domain plastic

    python maniagent/05_rag_engine.py \\
        --query "콜라겐이 피부에 좋은 이유는?" \\
        --locale ko \\
        --domain anti_aging
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ─── 의존성 ────────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    from pymilvus import connections, Collection, MilvusException
    from dotenv import load_dotenv
except ImportError as e:
    print(f"[ERROR] 필수 패키지 없음: {e}")
    print("       pip install sentence-transformers pymilvus python-dotenv")
    sys.exit(1)

import importlib.util as _ilu

_HERE      = Path(__file__).parent
_MANIDATA  = _HERE.parent / "manidata"


def _load_config(filename: str):
    """번호 접두어가 붙은 config 파일을 이름 충돌 없이 로드합니다."""
    spec = _ilu.spec_from_file_location(filename.replace(".py", ""), _MANIDATA / filename)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_aa_cfg = _load_config("02_anti_aging_config.py")
_pl_cfg = _load_config("03_plastic_config.py")

# 도메인별 설정 레지스트리
_DOMAIN_CONFIGS: dict[str, dict] = {
    "anti_aging": {
        "embedding":      _aa_cfg.EMBEDDING_CONFIG,
        "schema":         _aa_cfg.MILVUS_COLLECTION_SCHEMA,
        "llm_router":     _aa_cfg.LLM_ROUTER_CONFIG,
        "locale_prompts": _aa_cfg.LOCALE_SYSTEM_PROMPTS,
    },
    "plastic": {
        "embedding":      _pl_cfg.EMBEDDING_CONFIG,
        "schema":         _pl_cfg.MILVUS_COLLECTION_SCHEMA,
        "llm_router":     _pl_cfg.LLM_ROUTER_CONFIG,
        "locale_prompts": _pl_cfg.LOCALE_SYSTEM_PROMPTS,
    },
}
_DEFAULT_DOMAIN = "anti_aging"

EMBEDDING_CONFIG         = _aa_cfg.EMBEDDING_CONFIG
MILVUS_COLLECTION_SCHEMA = _aa_cfg.MILVUS_COLLECTION_SCHEMA
LLM_ROUTER_CONFIG        = _aa_cfg.LLM_ROUTER_CONFIG
LOCALE_SYSTEM_PROMPTS    = _aa_cfg.LOCALE_SYSTEM_PROMPTS

load_dotenv(_HERE.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# 데이터 모델
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievedChunk:
    chunk_id:    str
    text:        str
    text_zh:     str
    text_ja:     str
    text_en:     str
    source_file: str
    page_number: int
    score:       float
    domain:      str
    category:    str
    keywords:    list[str]
    year:        int
    author:      str


@dataclass
class ManiAgentResponse:
    query:            str
    locale:           str
    answer:           str
    retrieved_chunks: list[RetrievedChunk]
    sources:          list[dict]
    llm_provider:     str
    model:            str
    token_usage:      dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["retrieved_chunks"] = [asdict(c) for c in self.retrieved_chunks]
        return d


# ════════════════════════════════════════════════════════════════════════════
# LLM 라우터
# ════════════════════════════════════════════════════════════════════════════

class LLMRouter:
    """
    DeepSeek-V4 → Qwen-max → Claude(Anthropic) 순서로 폴백하는 LLM 라우터.

    모든 제공자는 OpenAI 호환 API를 사용하거나 Anthropic SDK로 처리됩니다.
    """

    def __init__(self, llm_router_config: dict | None = None) -> None:
        self._router_cfg = llm_router_config or LLM_ROUTER_CONFIG
        self._providers: list[dict] = []
        self._build_providers()

    def _build_providers(self) -> None:
        """환경 변수에서 사용 가능한 LLM 제공자를 수집합니다."""
        import openai

        # 1순위: DeepSeek
        deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
        if deepseek_key:
            cfg = self._router_cfg["primary"]
            self._providers.append({
                "name":    "deepseek",
                "client":  openai.OpenAI(api_key=deepseek_key, base_url=cfg["api_base"]),
                "model":   cfg["model"],
                "temperature": cfg["temperature"],
                "max_tokens":  cfg["max_tokens"],
                "mode":    "openai",
            })
            logger.info("LLM 라우터: DeepSeek 등록")

        # 2순위: Qwen (OpenAI 호환)
        qwen_key = os.getenv("QWEN_API_KEY", "")
        if qwen_key:
            cfg = self._router_cfg["secondary"]
            self._providers.append({
                "name":    "qwen",
                "client":  openai.OpenAI(api_key=qwen_key, base_url=cfg["api_base"]),
                "model":   cfg["model"],
                "temperature": cfg["temperature"],
                "max_tokens":  cfg["max_tokens"],
                "mode":    "openai",
            })
            logger.info("LLM 라우터: Qwen 등록")

        # 3순위: Claude (Anthropic)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            try:
                import anthropic as _anthropic
                cfg = self._router_cfg["fallback"]
                self._providers.append({
                    "name":    "claude",
                    "client":  _anthropic.Anthropic(api_key=anthropic_key),
                    "model":   cfg["model"],
                    "temperature": cfg["temperature"],
                    "max_tokens":  cfg["max_tokens"],
                    "mode":    "anthropic",
                })
                logger.info("LLM 라우터: Claude 등록")
            except ImportError:
                logger.warning("anthropic 패키지 없음 — Claude 비활성화 (pip install anthropic)")

        if not self._providers:
            raise EnvironmentError(
                "LLM API 키가 없습니다. DEEPSEEK_API_KEY, QWEN_API_KEY, ANTHROPIC_API_KEY 중 하나를 설정하세요."
            )

    def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
    ) -> tuple[str, str, str, dict]:
        """
        등록된 제공자 순서대로 호출하고, 실패 시 다음으로 폴백합니다.

        Returns:
            (answer, provider_name, model_name, token_usage)
        """
        for provider in self._providers:
            try:
                if provider["mode"] == "openai":
                    return self._call_openai(provider, system_prompt, user_prompt)
                elif provider["mode"] == "anthropic":
                    return self._call_anthropic(provider, system_prompt, user_prompt)
            except Exception as e:
                logger.warning(f"[{provider['name']}] 호출 실패, 폴백: {e}")

        raise RuntimeError("모든 LLM 제공자 호출에 실패했습니다.")

    @staticmethod
    def _call_openai(
        provider:      dict,
        system_prompt: str,
        user_prompt:   str,
    ) -> tuple[str, str, str, dict]:
        response = provider["client"].chat.completions.create(
            model=provider["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=provider["temperature"],
            max_tokens= provider["max_tokens"],
        )
        answer = response.choices[0].message.content or ""
        usage  = {
            "prompt_tokens":     response.usage.prompt_tokens     if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens":      response.usage.total_tokens      if response.usage else 0,
        }
        return answer, provider["name"], provider["model"], usage

    @staticmethod
    def _call_anthropic(
        provider:      dict,
        system_prompt: str,
        user_prompt:   str,
    ) -> tuple[str, str, str, dict]:
        response = provider["client"].messages.create(
            model=provider["model"],
            max_tokens=provider["max_tokens"],
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        answer = response.content[0].text if response.content else ""
        usage  = {
            "prompt_tokens":     response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens":      response.usage.input_tokens + response.usage.output_tokens,
        }
        return answer, provider["name"], provider["model"], usage


# ════════════════════════════════════════════════════════════════════════════
# ManiAgentRAG 클래스
# ════════════════════════════════════════════════════════════════════════════

class ManiAgentRAG:
    """
    ManiAgent 핵심 RAG 엔진.

    메서드:
        retrieve(query, top_k, filter_expr)  — Milvus 검색
        generate(query, locale)              — 검색 + LLM 응답 생성
        _build_prompt(query, context, locale) — 프롬프트 템플릿 구성
    """

    def __init__(
        self,
        collection_name:  str   = "",
        domain:           str   = _DEFAULT_DOMAIN,
        host:             str   = "",
        port:             str   = "",
        top_k:            int   = 5,
        score_threshold:  float = 0.5,
    ) -> None:
        if domain not in _DOMAIN_CONFIGS:
            raise ValueError(f"지원하지 않는 도메인: {domain}. 사용 가능: {list(_DOMAIN_CONFIGS)}")

        self.domain           = domain
        self._cfg             = _DOMAIN_CONFIGS[domain]
        self._embed_cfg       = self._cfg["embedding"]
        self._schema          = self._cfg["schema"]
        self._locale_prompts  = self._cfg["locale_prompts"]

        self.collection_name  = collection_name or self._schema["collection_name"]
        self.host             = host or os.getenv("MILVUS_HOST", "localhost")
        self.port             = port or os.getenv("MILVUS_PORT", "19530")
        self.top_k            = top_k
        self.score_threshold  = score_threshold

        # 임베딩 모델
        logger.info(f"임베딩 모델 로딩: {self._embed_cfg['model_name']} [도메인={domain}]")
        self._embed_model = SentenceTransformer(
            self._embed_cfg["model_name"],
            device=self._embed_cfg.get("device", "cpu"),
        )
        logger.info("임베딩 모델 로드 완료")

        # Milvus 연결
        connections.connect(alias="default", host=self.host, port=self.port)
        self._collection = Collection(self.collection_name)
        self._collection.load()
        logger.info(f"Milvus 연결 완료: {self.host}:{self.port} / {self.collection_name}")

        # LLM 라우터
        self._llm = LLMRouter(llm_router_config=self._cfg["llm_router"])

    # ── 검색 ────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:       str,
        top_k:       int  = None,
        filter_expr: str  = "",
    ) -> list[RetrievedChunk]:
        """
        쿼리를 임베딩하여 Milvus에서 유사 청크를 검색합니다.

        Args:
            query:       검색 질문
            top_k:       반환할 최대 청크 수
            filter_expr: Milvus 부울 표현식 (예: 'year > 2020')
        """
        k    = top_k if top_k is not None else self.top_k
        vec  = self._embed_model.encode(
            [query],
            normalize_embeddings=self._embed_cfg.get("normalize", True),
        ).tolist()

        search_kwargs: dict = {
            "data":         vec,
            "anns_field":   "embedding",
            "param": {
                "metric_type": self._schema["metric_type"],
                "params":      self._schema["search_params"],
            },
            "limit":        k,
            "output_fields": [
                "chunk_id", "text", "text_zh", "text_ja", "text_en",
                "source_file", "page_number", "domain", "category",
                "keywords", "year", "author",
            ],
        }
        if filter_expr:
            search_kwargs["expr"] = filter_expr

        try:
            results = self._collection.search(**search_kwargs)
        except MilvusException as e:
            logger.error(f"Milvus 검색 오류: {e}")
            return []

        chunks: list[RetrievedChunk] = []
        for hit in results[0]:
            score  = hit.score
            if score < self.score_threshold:
                continue
            entity = hit.entity

            raw_kw = entity.get("keywords", "[]")
            try:
                keywords = json.loads(raw_kw) if isinstance(raw_kw, str) else raw_kw
            except json.JSONDecodeError:
                keywords = []

            chunks.append(
                RetrievedChunk(
                    chunk_id=    entity.get("chunk_id", ""),
                    text=        entity.get("text", ""),
                    text_zh=     entity.get("text_zh", ""),
                    text_ja=     entity.get("text_ja", ""),
                    text_en=     entity.get("text_en", ""),
                    source_file= entity.get("source_file", ""),
                    page_number= entity.get("page_number", 0),
                    score=       round(score, 4),
                    domain=      entity.get("domain", ""),
                    category=    entity.get("category", ""),
                    keywords=    keywords,
                    year=        entity.get("year", 0),
                    author=      entity.get("author", ""),
                )
            )

        logger.info(f"검색 결과: {len(chunks)}개 (threshold={self.score_threshold})")
        return chunks

    # ── 프롬프트 구성 ────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        query:   str,
        context: str,
        locale:  str = "ko",
    ) -> tuple[str, str]:
        """
        locale에 맞는 시스템 프롬프트와 사용자 프롬프트를 구성합니다.

        Returns:
            (system_prompt, user_prompt)
        """
        system_prompt = self._locale_prompts.get(locale, self._locale_prompts["ko"])

        # locale에 따라 context 텍스트 선택
        locale_field = {
            "zh": "text_zh", "ja": "text_ja", "en": "text_en",
        }.get(locale, "text")

        # 사용자 프롬프트 (locale별 지시)
        task_instructions = {
            "ko": "다음 참고 문서를 바탕으로 질문에 답변하십시오. 출처(파일명, 페이지)를 명시하십시오.",
            "zh": "请根据以下参考文献回答问题。请注明出处（文件名、页码）。",
            "ja": "以下の参考文書に基づいて質問に答えてください。出典（ファイル名、ページ）を明記してください。",
            "en": "Answer the question based on the reference documents below. Cite the source (filename, page number).",
        }
        instruction = task_instructions.get(locale, task_instructions["ko"])

        user_prompt = (
            f"{instruction}\n\n"
            f"=== 참고 문서 ===\n{context}\n\n"
            f"=== 질문 ===\n{query}"
        )
        return system_prompt, user_prompt

    # ── 컨텍스트 빌더 ────────────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk], locale: str) -> str:
        """검색된 청크로 컨텍스트 문자열을 구성합니다."""
        if not chunks:
            return "관련 문서 없음."

        locale_field = {"zh": "text_zh", "ja": "text_ja", "en": "text_en"}.get(locale, "text")
        parts: list[str] = []
        for i, c in enumerate(chunks, 1):
            # locale 번역본이 있으면 사용, 없으면 원본 한국어
            body = getattr(c, locale_field, "") or c.text
            ref  = f"{c.source_file}, p.{c.page_number}"
            if c.year:
                ref += f" ({c.year})"
            parts.append(f"[{i}] ({ref}, score={c.score:.3f})\n{body}")
        return "\n\n---\n\n".join(parts)

    # ── 메인 generate ────────────────────────────────────────────────────────

    def generate(
        self,
        query:       str,
        locale:      str  = "ko",
        filter_expr: str  = "",
    ) -> ManiAgentResponse:
        """
        질문을 받아 RAG 파이프라인을 실행하고 ManiAgentResponse를 반환합니다.

        Args:
            query:       사용자 질문
            locale:      응답 언어 (ko / zh / ja / en)
            filter_expr: Milvus 필터 표현식
        """
        logger.info(f"generate() 호출 — query='{query[:60]}...', locale={locale}")

        # 1. 검색
        chunks = self.retrieve(query, filter_expr=filter_expr)

        if not chunks:
            no_result_msgs = {
                "ko": "관련 정보를 찾지 못했습니다. 다른 키워드로 다시 질문해 주세요.",
                "zh": "未找到相关信息，请使用其他关键词重新提问。",
                "ja": "関連情報が見つかりませんでした。別のキーワードで再度質問してください。",
                "en": "No relevant information found. Please try with different keywords.",
            }
            return ManiAgentResponse(
                query=query, locale=locale,
                answer=no_result_msgs.get(locale, no_result_msgs["ko"]),
                retrieved_chunks=[], sources=[],
                llm_provider="none", model="none",
            )

        # 2. 컨텍스트 구성
        context = self._build_context(chunks, locale)

        # 3. 프롬프트 구성
        system_prompt, user_prompt = self._build_prompt(query, context, locale)

        # 4. LLM 라우터 호출
        answer, provider, model_name, usage = self._llm.complete(system_prompt, user_prompt)

        # 5. 소스 요약
        sources = [
            {
                "source_file": c.source_file,
                "page_number": c.page_number,
                "score":       c.score,
                "domain":      c.domain,
                "category":    c.category,
                "year":        c.year,
                "author":      c.author,
            }
            for c in chunks
        ]

        return ManiAgentResponse(
            query=query,
            locale=locale,
            answer=answer,
            retrieved_chunks=chunks,
            sources=sources,
            llm_provider=provider,
            model=model_name,
            token_usage=usage,
        )


# ════════════════════════════════════════════════════════════════════════════
# 전역 싱글톤 (API 서버와 공유)
# ════════════════════════════════════════════════════════════════════════════

_engine_registry: dict[str, ManiAgentRAG] = {}


def get_engine(domain: str = _DEFAULT_DOMAIN, **kwargs) -> ManiAgentRAG:
    """도메인별 ManiAgentRAG 싱글톤 인스턴스를 반환합니다."""
    key = domain + (kwargs.get("collection_name", "") or "")
    if key not in _engine_registry:
        _engine_registry[key] = ManiAgentRAG(domain=domain, **kwargs)
    return _engine_registry[key]


# ════════════════════════════════════════════════════════════════════════════
# CLI 진입점
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="ManiAgentRAG CLI")
    p.add_argument("--query",      "-q", required=True)
    p.add_argument("--locale",     "-l", default="ko",
                   choices=["ko", "zh", "ja", "en"])
    p.add_argument("--domain",     "-d", default=_DEFAULT_DOMAIN,
                   choices=list(_DOMAIN_CONFIGS), help="도메인 (anti_aging | plastic)")
    p.add_argument("--collection", "-c", default="")
    p.add_argument("--host",       default=os.getenv("MILVUS_HOST", "localhost"))
    p.add_argument("--port",       default=os.getenv("MILVUS_PORT", "19530"))
    p.add_argument("--top-k",      type=int, default=5)
    p.add_argument("--filter",     default="")
    p.add_argument("--json",       action="store_true", help="JSON 출력")
    args = p.parse_args()

    engine   = get_engine(
        domain=args.domain,
        collection_name=args.collection or None,
        host=args.host, port=args.port,
        top_k=args.top_k,
    )
    response = engine.generate(args.query, locale=args.locale, filter_expr=args.filter)

    if args.json:
        print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  질문 ({args.locale.upper()}): {response.query}")
        print(f"{'='*60}")
        print(f"\n{response.answer}\n")
        print(f"── 출처 ({len(response.sources)}건) ──")
        for s in response.sources:
            print(f"  • {s['source_file']} p.{s['page_number']} (score={s['score']:.3f})")
        print(f"\n[LLM: {response.llm_provider}/{response.model}]")
        print(f"[토큰: {response.token_usage}]")


if __name__ == "__main__":
    main()
