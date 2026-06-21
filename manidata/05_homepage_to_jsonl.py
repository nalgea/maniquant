"""
05_homepage_to_jsonl.py
성형외과 홈페이지 → 다국어 ManiQuant JSONL 변환기

단계:
  1. master_table에서 인기 병원 선정 (방문자리뷰 + 온라인언급 복합 점수)
  2. 홈페이지 텍스트 수집 (requests + BeautifulSoup, Playwright 폴백)
  3. 섹션별 청킹 (소개/시술/의료진/가격/Q&A/후기)
  4. Claude API로 zh/ja/en 번역
  5. ManiQuant JSONL 저장

사용법:
    python manidata/05_homepage_to_jsonl.py
    python manidata/05_homepage_to_jsonl.py --clinic "바노바기 성형외과" --url http://www.banobagi.com/
    python manidata/05_homepage_to_jsonl.py --no-translate   # 번역 없이 한국어만
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

# ── 의존성 ────────────────────────────────────────────────────────────────────
try:
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    from dotenv import load_dotenv
except ImportError as e:
    raise SystemExit(f"[ERROR] 필수 패키지 없음: {e}\n"
                     "pip install requests beautifulsoup4 pandas python-dotenv")

import os
import sys

_HERE   = Path(__file__).parent
_ROOT   = _HERE.parent

load_dotenv(_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── plastic_config 로드 ───────────────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location("03_plastic_config", _HERE / "03_plastic_config.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
tag_plastic_chunk = _mod.tag_plastic_chunk

# ── 섹션 키워드 (홈페이지 구조 파악용) ────────────────────────────────────────
SECTION_PATTERNS = {
    "소개":   ["about", "소개", "병원소개", "인사말", "원장", "철학", "비전", "story"],
    "시술":   ["procedure", "surgery", "시술", "수술", "눈", "코", "얼굴", "가슴", "지방", "리프팅"],
    "의료진": ["doctor", "surgeon", "의료진", "원장", "의사", "전문의", "staff"],
    "가격":   ["price", "cost", "비용", "가격", "이벤트", "프로모션", "할인"],
    "후기":   ["review", "before", "after", "후기", "결과", "사례", "gallery"],
    "FAQ":    ["faq", "자주묻는", "질문", "궁금"],
    "예약":   ["contact", "reservation", "예약", "상담", "오시는"],
}

# ── 번역 타겟 ────────────────────────────────────────────────────────────────
TRANSLATE_TARGETS = [
    ("zh", "중국어(简体中文)"),
    ("ja", "일본어"),
    ("en", "영어"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
}


# ════════════════════════════════════════════════════════════════════════════
# 1. 인기 병원 선정
# ════════════════════════════════════════════════════════════════════════════

def pick_top_clinic(master_csv: Path) -> dict:
    df = pd.read_csv(master_csv, encoding="utf-8-sig", dtype=str)

    for col in ["방문자리뷰수", "총_언급수"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 복합 점수: 방문자리뷰(정규화) * 0.7 + 온라인언급(정규화) * 0.3
    df["_score"] = (
        df["방문자리뷰수"] / df["방문자리뷰수"].max() * 0.7
        + df["총_언급수"]  / df["총_언급수"].max()  * 0.3
    )
    row = df.sort_values("_score", ascending=False).iloc[0]

    return {
        "clinic_name": str(row.get("기관명", "")),
        "naver_name":  str(row.get("병원명_네이버", "")),
        "homepage":    str(row.get("홈페이지", "")),
        "visitor_reviews": int(row["방문자리뷰수"]),
        "online_mentions": int(row["총_언급수"]),
        "address":     str(row.get("도로명주소", "")),
        "tel":         str(row.get("전화번호", "")),
        "naver_id":    str(row.get("고유ID", "")),
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. 홈페이지 텍스트 수집
# ════════════════════════════════════════════════════════════════════════════

_NOISE_TAGS = {"script", "style", "noscript", "header", "footer", "nav",
               "iframe", "svg", "img", "video", "audio", "button", "form",
               "input", "select", "textarea", "meta", "link"}

_MIN_TEXT_LEN = 30


def _clean_html(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()
    return soup


def _extract_sections(soup: BeautifulSoup, base_url: str) -> list[dict]:
    """
    페이지에서 섹션별 텍스트 블록을 추출합니다.
    <section>, <article>, <div id/class=...> 등을 탐색합니다.
    """
    sections: list[dict] = []
    seen: set[str] = set()

    candidates = soup.find_all(["section", "article", "div", "main"], recursive=True)

    for el in candidates:
        # 섹션 레이블 추출
        label_src = " ".join([
            el.get("id", ""),
            " ".join(el.get("class", [])),
            el.get("aria-label", ""),
        ]).lower()

        section_type = "일반"
        for stype, patterns in SECTION_PATTERNS.items():
            if any(p in label_src for p in patterns):
                section_type = stype
                break

        text = el.get_text(separator="\n", strip=True)
        # 중복·노이즈 제거
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

        if len(text) < _MIN_TEXT_LEN:
            continue
        sig = text[:120]
        if sig in seen:
            continue
        seen.add(sig)

        sections.append({
            "section_type": section_type,
            "text":         text[:2000],  # 청크 최대 2000자
            "source_tag":   el.name,
            "label":        label_src[:80],
        })

    return sections


def fetch_homepage(url: str, timeout: int = 15) -> list[dict]:
    """홈페이지 + 주요 하위 페이지 텍스트를 수집합니다."""
    logger.info(f"홈페이지 수집 시작: {url}")
    all_sections: list[dict] = []
    visited: set[str] = set()
    queue = [url]

    parsed_base = urlparse(url)
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

    session = requests.Session()
    session.headers.update(HEADERS)

    depth = 0
    while queue and depth < 3:
        current_url = queue.pop(0)
        if current_url in visited:
            continue
        visited.add(current_url)

        try:
            resp = session.get(current_url, timeout=timeout)
            resp.encoding = resp.apparent_encoding or "utf-8"
            if resp.status_code != 200:
                logger.warning(f"  HTTP {resp.status_code}: {current_url}")
                continue

            soup = _clean_html(resp.text)
            sections = _extract_sections(soup, base_domain)
            for s in sections:
                s["page_url"] = current_url
            all_sections.extend(sections)
            logger.info(f"  수집 [{depth}] {current_url} → {len(sections)}섹션")

            # 링크 수집 (같은 도메인, 1depth만)
            if depth == 0:
                for a in soup.find_all("a", href=True):
                    href = urljoin(base_domain, a["href"])
                    if (href.startswith(base_domain)
                            and href not in visited
                            and not href.endswith((".pdf", ".jpg", ".png", ".gif"))
                            and "#" not in href.split("?")[0]):
                        queue.append(href)

            time.sleep(0.8)

        except Exception as e:
            logger.warning(f"  수집 실패 {current_url}: {e}")

        depth += 1

    logger.info(f"총 수집 섹션: {len(all_sections)}개 / 방문 페이지: {len(visited)}개")
    return all_sections


# ════════════════════════════════════════════════════════════════════════════
# 3. Claude API 번역
# ════════════════════════════════════════════════════════════════════════════

def _build_translate_prompt(text: str, target_lang: str, target_label: str,
                             clinic_name: str) -> str:
    return (
        f"다음은 한국 강남 성형외과 '{clinic_name}'의 홈페이지 텍스트입니다. "
        f"{target_label}로 자연스럽게 번역해 주세요. "
        f"병원명, 시술명, 의학 용어는 정확하게 번역하고, "
        f"원문에 없는 내용을 추가하지 마세요. "
        f"번역 결과만 출력하세요.\n\n"
        f"=== 원문 ===\n{text}"
    )


def _get_all_translators() -> list[tuple]:
    """사용 가능한 번역 엔진 전체를 우선순위 순으로 반환합니다."""
    result = []
    try:
        import openai as _openai
        _have_openai = True
    except ImportError:
        _have_openai = False

    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    if deepseek_key and _have_openai and "here" not in deepseek_key:
        result.append((
            _openai.OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com"),
            "deepseek", "deepseek-chat",
        ))

    qwen_key = os.getenv("QWEN_API_KEY", "")
    if qwen_key and _have_openai and "here" not in qwen_key:
        result.append((
            _openai.OpenAI(api_key=qwen_key,
                           base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "qwen", "qwen-max",
        ))

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key and "here" not in anthropic_key:
        try:
            import anthropic as _anthropic
            result.append((_anthropic.Anthropic(api_key=anthropic_key),
                           "claude", "claude-haiku-4-5-20251001"))
        except ImportError:
            pass

    return result


def _get_translator():
    """첫 번째 사용 가능한 번역 엔진을 반환합니다."""
    engines = _get_all_translators()
    if engines:
        client, mode, model = engines[0]
        logger.info(f"번역 엔진: {mode} ({model})")
        return client, mode, model
    return None, None, None


def _translate_one(client, mode: str, model: str,
                   prompt: str) -> str:
    """단일 번역 호출 (모드별 분기)."""
    if mode in ("deepseek", "qwen"):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()
    elif mode == "claude":
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    return ""


def translate_texts(sections: list[dict], clinic_name: str,
                    max_chars: int = 800) -> list[dict]:
    """DeepSeek → Qwen → Claude 순으로 zh/ja/en 번역합니다."""
    client, mode, model = _get_translator()

    if client is None:
        logger.warning("번역 API 키 없음 (DEEPSEEK_API_KEY / QWEN_API_KEY / ANTHROPIC_API_KEY) — 번역 건너뜀")
        for s in sections:
            s.update({"text_zh": "", "text_ja": "", "text_en": ""})
        return sections

    total = len(sections)
    for i, section in enumerate(sections):
        src = section["text"][:max_chars]

        for lang, label in TRANSLATE_TARGETS:
            try:
                prompt = _build_translate_prompt(src, lang, label, clinic_name)
                section[f"text_{lang}"] = _translate_one(client, mode, model, prompt)
                time.sleep(0.2)
            except Exception as e:
                logger.warning(f"  [{mode}] 번역 실패 [{lang}]: {e} — 다음 엔진 폴백")
                # 현재 엔진 실패 시 다음 우선순위 엔진 시도
                fallback_result = ""
                for fb_client, fb_mode, fb_model in _get_all_translators():
                    if fb_mode == mode:
                        continue
                    try:
                        fallback_result = _translate_one(fb_client, fb_mode, fb_model, prompt)
                        logger.info(f"  폴백 성공: {fb_mode}")
                        break
                    except Exception as fe:
                        logger.warning(f"  [{fb_mode}] 폴백도 실패: {fe}")
                section[f"text_{lang}"] = fallback_result

        if (i + 1) % 10 == 0 or i == total - 1:
            logger.info(f"  번역 진행 [{mode}]: {i+1}/{total}")

    return sections


# ════════════════════════════════════════════════════════════════════════════
# 4. JSONL 청크 생성
# ════════════════════════════════════════════════════════════════════════════

def _token_count(text: str) -> int:
    return max(1, len(text) // 3)


def sections_to_chunks(sections: list[dict], clinic_info: dict) -> list[dict]:
    chunks: list[dict] = []
    clinic_name = clinic_info["clinic_name"]

    for i, s in enumerate(sections):
        text = s["text"]
        if len(text) < _MIN_TEXT_LEN:
            continue

        tags = tag_plastic_chunk(text)

        chunk = {
            "chunk_id":      str(uuid.uuid4()),
            "source_file":   "homepage_" + urlparse(clinic_info["homepage"]).netloc.replace(".", "_"),
            "page_url":      s.get("page_url", clinic_info["homepage"]),
            "page_number":   i + 1,
            "chunk_index":   i,
            "chunk_type":    "homepage_" + s["section_type"],
            "text":          text,
            "language":      "ko",
            "domain":        "plastic",
            "category":      tags["category"],
            "domain_tags":   tags["domain_tags"],
            "keywords":      tags["keywords"],
            "year":          2026,
            "author":        clinic_name,
            "token_count":   _token_count(text),
            "text_zh":       s.get("text_zh", ""),
            "text_ja":       s.get("text_ja", ""),
            "text_en":       s.get("text_en", ""),
            "embedding":     [],
            # 병원 메타
            "clinic_name":    clinic_name,
            "naver_place_id": clinic_info.get("naver_id", ""),
            "address":        clinic_info.get("address", ""),
            "homepage":       clinic_info.get("homepage", ""),
            "section_type":   s["section_type"],
        }
        chunks.append(chunk)

    return chunks


# ════════════════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════════════════

def run(clinic_name: str = "", url: str = "", no_translate: bool = False,
        output_path: str = "") -> Path:

    master_csv = _ROOT / "data" / "processed" / "plastic" / "master_table.csv"

    # 병원 선정
    if clinic_name and url:
        clinic_info = {"clinic_name": clinic_name, "homepage": url,
                       "naver_name": clinic_name, "address": "",
                       "tel": "", "naver_id": "", "visitor_reviews": 0, "online_mentions": 0}
    else:
        clinic_info = pick_top_clinic(master_csv)

    logger.info(
        f"\n{'='*60}\n"
        f"선정 병원: {clinic_info['clinic_name']}\n"
        f"홈페이지:  {clinic_info['homepage']}\n"
        f"방문자리뷰: {clinic_info['visitor_reviews']:,}건 / 온라인언급: {clinic_info['online_mentions']:,}건\n"
        f"{'='*60}"
    )

    homepage = clinic_info["homepage"]
    if not homepage or homepage in ("nan", "None", ""):
        raise ValueError(f"홈페이지 URL 없음: {clinic_info['clinic_name']}")

    # 수집
    sections = fetch_homepage(homepage)
    if not sections:
        raise RuntimeError("텍스트 수집 실패 — Playwright 수동 실행 필요")

    # 번역
    if not no_translate:
        sections = translate_texts(sections, clinic_info["clinic_name"])
    else:
        logger.info("번역 건너뜀 (--no-translate)")
        for s in sections:
            s.update({"text_zh": "", "text_ja": "", "text_en": ""})

    # 청크 생성
    chunks = sections_to_chunks(sections, clinic_info)
    logger.info(f"생성된 청크: {len(chunks)}개")

    # 저장
    out_dir = _ROOT / "data" / "processed" / "plastic"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w]", "_", clinic_info["clinic_name"])
    out = Path(output_path) if output_path else out_dir / f"homepage_{safe_name}.jsonl"

    with open(out, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(
        f"\n{'='*60}\n"
        f"JSONL 저장 완료 → {out}\n"
        f"  총 청크:     {len(chunks)}개\n"
        f"  번역 포함:   {'예' if not no_translate else '아니오'} (zh/ja/en)\n"
        f"{'='*60}"
    )

    # 섹션 분포 출력
    from collections import Counter
    dist = Counter(c["section_type"] for c in chunks)
    for stype, cnt in dist.most_common():
        logger.info(f"  {stype:8s}: {cnt}청크")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="홈페이지 → 다국어 JSONL")
    parser.add_argument("--clinic",       default="", help="병원명 (미입력 시 자동 선정)")
    parser.add_argument("--url",          default="", help="홈페이지 URL")
    parser.add_argument("--no-translate", action="store_true", help="번역 건너뜀")
    parser.add_argument("--output",       default="", help="출력 JSONL 경로")
    args = parser.parse_args()

    run(
        clinic_name  = args.clinic,
        url          = args.url,
        no_translate = args.no_translate,
        output_path  = args.output,
    )
