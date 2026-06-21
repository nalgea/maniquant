"""
07_naver_to_jsonl.py
강남 성형외과 네이버 데이터 → ManiQuant JSONL 변환

입력:
  - C:/bocademy/gangnam_scraper/data/processed/master_table.csv   (103개 병원 메타)
  - C:/bocademy/gangnam_scraper/data/processed/blog_matched.csv   (병원별 매핑 포스트)

출력:
  - data/processed/manidata_plastic_gangnam_v1.jsonl

청크 유형:
  A. 병원 프로필 청크 (clinic_profile)  — 병원 1개당 1 청크
  B. 블로그/카페 포스트 청크 (blog_post) — 병원에 매핑된 포스트 1개당 1 청크

사용법:
    python manidata/07_naver_to_jsonl.py
    python manidata/07_naver_to_jsonl.py --output data/processed/custom.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

_HERE    = Path(__file__).parent
_ROOT    = _HERE.parent
_SCRAPER = _ROOT / "data" / "processed" / "plastic"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("03_plastic_config", _HERE / "03_plastic_config.py")
_mod  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
tag_plastic_chunk       = _mod.tag_plastic_chunk
PLASTIC_CHUNKING_CONFIG = _mod.PLASTIC_CHUNKING_CONFIG


# ════════════════════════════════════════════════════════════════════════════
# 유틸
# ════════════════════════════════════════════════════════════════════════════

_PRICE_RE = re.compile(r"(\d{1,4})\s*만\s*원")


def _safe(val: Any, default: Any = "") -> Any:
    if val is None or (isinstance(val, float) and val != val):
        return default
    return val


def _token_count(text: str) -> int:
    return max(1, len(text) // 3)


def _make_chunk(text: str, source_file: str, page_number: int,
                chunk_index: int, chunk_type: str,
                extra_meta: dict | None = None) -> dict:
    """공통 청크 레코드 생성 (JSONL 1행)"""
    tags = tag_plastic_chunk(text)
    rec: dict = {
        "chunk_id":    str(uuid.uuid4()),
        "source_file": source_file,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "chunk_type":  chunk_type,
        "text":        text,
        "language":    "ko",
        "domain":      "plastic",
        "category":    tags["category"],
        "domain_tags": tags["domain_tags"],
        "keywords":    tags["keywords"],
        "year":        2026,
        "author":      "",
        "token_count": _token_count(text),
        "text_zh":     "",
        "text_ja":     "",
        "text_en":     "",
        "embedding":   [],
    }
    if extra_meta:
        rec.update(extra_meta)
    return rec


# ════════════════════════════════════════════════════════════════════════════
# A. 병원 프로필 청크
# ════════════════════════════════════════════════════════════════════════════

def _clinic_profile_text(row: pd.Series) -> str:
    """병원 1개의 메타데이터를 자연어 텍스트로 직렬화"""
    name      = _safe(row.get("기관명"))
    naver_nm  = _safe(row.get("병원명_네이버"))
    addr      = _safe(row.get("도로명주소"))
    tel       = _safe(row.get("전화번호"))
    category  = _safe(row.get("카테고리"))
    v_review  = _safe(row.get("방문자리뷰수"), 0)
    b_review  = _safe(row.get("블로그리뷰수"), 0)
    rating    = _safe(row.get("별점"))
    status    = _safe(row.get("영업상태"))
    homepage  = _safe(row.get("홈페이지"))
    mentions  = _safe(row.get("총_언급수"), 0)
    pos_rate  = _safe(row.get("긍정률(%)"), 0)
    procedure = _safe(row.get("주요시술1"))
    price_med = _safe(row.get("가격_중앙값"))

    price_str = f"시술 가격 중앙값 약 {int(float(price_med)) // 10_000}만원." if price_med and str(price_med) not in ("", "nan") else ""

    text = (
        f"병원명: {name}"
        + (f" (네이버 등록명: {naver_nm})" if naver_nm and naver_nm != name else "")
        + f"\n주소: {addr}"
        + (f"\n전화: {tel}" if tel else "")
        + (f"\n카테고리: {category}" if category else "")
        + (f"\n영업상태: {status}" if status else "")
        + (f"\n홈페이지: {homepage}" if homepage else "")
        + f"\n방문자 리뷰 수: {int(float(v_review)):,}건"
        + f"\n블로그 리뷰 수: {int(float(b_review)):,}건"
        + (f"\n평점: {float(rating):.1f}" if rating and rating == rating else "")
        + f"\n온라인 블로그/카페 언급 수: {int(float(mentions)):,}건"
        + (f"\n긍정 반응 비율: {float(pos_rate):.1f}%" if pos_rate else "")
        + (f"\n주요 시술 분야: {procedure}" if procedure and procedure != "기타" else "")
        + (f"\n{price_str}" if price_str else "")
    )
    return text.strip()


def build_clinic_profile_chunks(df_master: pd.DataFrame) -> list[dict]:
    chunks = []
    for i, (_, row) in enumerate(df_master.iterrows()):
        text = _clinic_profile_text(row)
        if len(text) < 30:
            continue

        clinic_id = _safe(row.get("고유ID"), "")
        chunk = _make_chunk(
            text        = text,
            source_file = "naver_place_master",
            page_number = i + 1,
            chunk_index = i,
            chunk_type  = "clinic_profile",
            extra_meta  = {
                "clinic_name":    _safe(row.get("기관명")),
                "naver_place_id": str(clinic_id),
                "address":        _safe(row.get("도로명주소")),
                "tel":            _safe(row.get("전화번호")),
                "visitor_reviews": int(float(_safe(row.get("방문자리뷰수"), 0) or 0)),
                "blog_reviews":    int(float(_safe(row.get("블로그리뷰수"), 0) or 0)),
                "rating":          float(_safe(row.get("별점"), 0) or 0),
                "online_mentions": int(float(_safe(row.get("총_언급수"), 0) or 0)),
                "positive_rate":   float(_safe(row.get("긍정률(%)"), 0) or 0),
                "price_median_man": int(float(row["가격_중앙값"])) if pd.notna(row.get("가격_중앙값")) else None,
                "main_procedure":  _safe(row.get("주요시술1")),
                "homepage":        _safe(row.get("홈페이지")),
            },
        )
        chunks.append(chunk)

    logger.info(f"병원 프로필 청크: {len(chunks)}개")
    return chunks


# ════════════════════════════════════════════════════════════════════════════
# B. 블로그/카페 포스트 청크
# ════════════════════════════════════════════════════════════════════════════

def _post_text(row: pd.Series) -> str:
    title  = _safe(row.get("제목", ""))
    desc   = _safe(row.get("설명_미리보기", ""))
    clinic = _safe(row.get("매핑_기관명", ""))
    src    = _safe(row.get("소스", "blog"))
    date   = _safe(row.get("발행일", ""))
    kw     = _safe(row.get("검색키워드", ""))

    src_label = "블로그" if src == "blog" else "네이버카페"
    text = (
        f"[{src_label}] {title}\n"
        + (f"병원: {clinic}\n" if clinic else "")
        + (f"검색키워드: {kw}\n" if kw and kw != clinic else "")
        + (f"발행일: {date}\n" if date else "")
        + f"{desc}"
    )
    return text.strip()


def build_blog_post_chunks(df_blog: pd.DataFrame,
                            min_text_len: int = 30) -> list[dict]:
    # 병원 매핑된 포스트만 사용
    df_matched = df_blog[df_blog["매핑_기관명"].notna()].copy()
    logger.info(f"매핑된 포스트: {len(df_matched):,}건")

    chunks = []
    for i, (_, row) in enumerate(df_matched.iterrows()):
        text = _post_text(row)
        if len(text) < min_text_len:
            continue

        # 가격 추출
        prices = [int(m) * 10_000
                  for m in _PRICE_RE.findall(text)
                  if 50 <= int(m) <= 2000]

        src  = _safe(row.get("소스", "blog"))
        date = _safe(row.get("발행일", ""))
        year = int(str(date)[:4]) if str(date)[:4].isdigit() else 2026

        chunk = _make_chunk(
            text        = text,
            source_file = f"naver_{src}",
            page_number = i + 1,
            chunk_index = i,
            chunk_type  = "blog_post",
            extra_meta  = {
                "clinic_name":   _safe(row.get("매핑_기관명")),
                "post_source":   src,
                "post_url":      _safe(row.get("URL")),
                "post_date":     str(date),
                "search_keyword":_safe(row.get("검색키워드")),
                "prices_won":    prices,
                "is_official":   str(_safe(row.get("공식여부", "False"))).lower() == "true",
                "year":          year,
            },
        )
        chunk["year"] = year
        chunks.append(chunk)

    logger.info(f"블로그/카페 청크: {len(chunks)}개")
    return chunks


# ════════════════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════════════════

def convert(output_path: str | None = None) -> Path:
    out = Path(output_path) if output_path else \
          _ROOT / "data" / "processed" / "plastic" / "manidata_plastic_gangnam_v1.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    master_path = _SCRAPER / "master_table.csv"
    blog_path   = _SCRAPER / "blog_matched.csv"

    if not master_path.exists():
        raise FileNotFoundError(f"master_table.csv 없음: {master_path}")
    if not blog_path.exists():
        raise FileNotFoundError(f"blog_matched.csv 없음: {blog_path}")

    df_master = pd.read_csv(master_path, encoding="utf-8-sig", dtype=str)
    df_blog   = pd.read_csv(blog_path,   encoding="utf-8-sig", dtype=str)

    logger.info(f"master_table: {len(df_master)}행 / blog_matched: {len(df_blog):,}행")

    # ── 청크 생성 ────────────────────────────────────────────────────────────
    profile_chunks = build_clinic_profile_chunks(df_master)
    post_chunks    = build_blog_post_chunks(df_blog)
    all_chunks     = profile_chunks + post_chunks

    # ── JSONL 저장 ───────────────────────────────────────────────────────────
    with open(out, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(
        f"JSONL 저장 완료 → {out}\n"
        f"  프로필 청크: {len(profile_chunks)}개\n"
        f"  포스트 청크: {len(post_chunks):,}개\n"
        f"  합계:        {len(all_chunks):,}개"
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="출력 JSONL 경로")
    args = parser.parse_args()
    convert(args.output)
