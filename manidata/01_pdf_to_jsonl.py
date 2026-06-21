"""
01_pdf_to_jsonl.py
ManiData 파이프라인 Step 1
PDF/EPUB → 청킹 + 도메인 태깅 + 통역 페어 생성 → JSONL

사용법:
    python manidata/01_pdf_to_jsonl.py \\
        --pdf data/raw/anti_aging_book.pdf \\
        --output data/processed/manidata_anti_aging_v1.jsonl \\
        --domain anti_aging

    python manidata/01_pdf_to_jsonl.py \\
        --pdf data/raw/ \\
        --output data/processed/ \\
        --domain anti_aging \\
        --locales zh ja en
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Optional

# ─── 의존성 ────────────────────────────────────────────────────────────────
try:
    import fitz                         # PyMuPDF
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langdetect import detect as _lang_detect, LangDetectException
    from tqdm import tqdm
    from dotenv import load_dotenv
except ImportError as e:
    print(f"[ERROR] 필수 패키지 없음: {e}")
    print("       pip install PyMuPDF langchain-text-splitters langdetect tqdm python-dotenv")
    sys.exit(1)

# 도메인 설정 모듈
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from anti_aging_config import (   # noqa: E402
    CHUNKING_CONFIG,
    tag_anti_aging_chunk,
)
from plastic_config import tag_plastic_chunk  # noqa: E402

load_dotenv(_HERE.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 도메인 → 태거 함수 매핑
DOMAIN_TAGGERS = {
    "anti_aging": tag_anti_aging_chunk,
    "plastic":    tag_plastic_chunk,
}

# ════════════════════════════════════════════════════════════════════════════
# PDF 텍스트 추출 (PyMuPDF)
# ════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    PyMuPDF(fitz)로 PDF 각 페이지의 텍스트와 메타데이터를 추출합니다.

    Returns:
        [{"page_number": int, "text": str, "pdf_metadata": dict}, ...]
    """
    pages: list[dict] = []
    doc = fitz.open(str(pdf_path))
    pdf_meta = doc.metadata or {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text("text")   # type: ignore[attr-defined]
        cleaned  = _clean_page_text(raw_text or "")
        if cleaned.strip():
            pages.append({
                "page_number":  page_num + 1,
                "text":         cleaned,
                "pdf_metadata": pdf_meta,
            })

    doc.close()
    logger.info(f"  PyMuPDF 추출: {len(pages)}페이지 ← {pdf_path.name}")
    return pages


def _clean_page_text(text: str) -> str:
    """페이지 텍스트를 기본 정제합니다."""
    text = re.sub(r"-\n([a-z])", r"\1", text)   # 줄바꿈 하이픈 처리
    text = re.sub(r" {2,}", " ", text)            # 연속 공백 → 단일
    text = re.sub(r"\n{3,}", "\n\n", text)        # 3+ 개행 → 2개
    return text.strip()


def _parse_pdf_metadata(pdf_meta: dict, sample_text: str) -> dict:
    """PDF 메타데이터에서 author·year 파싱을 시도합니다."""
    author = pdf_meta.get("author", "") or ""
    year   = 0

    raw_date = pdf_meta.get("creationDate", "") or ""
    m = re.search(r"(\d{4})", raw_date)
    if m:
        year = int(m.group(1))
    else:
        # 텍스트 앞부분에서 연도 추정
        found = re.findall(r"\b(19[89]\d|20[012]\d)\b", sample_text[:1000])
        if found:
            year = int(sorted(found, reverse=True)[0])

    return {
        "author": str(author)[:512],
        "year":   year,
    }


# ════════════════════════════════════════════════════════════════════════════
# 언어 감지
# ════════════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    try:
        return _lang_detect(text[:500])
    except LangDetectException:
        return "unknown"


# ════════════════════════════════════════════════════════════════════════════
# 통역 페어 생성
# ════════════════════════════════════════════════════════════════════════════

class InterpreterClient:
    """LLM API를 통해 통역 페어를 생성합니다 (DeepSeek 우선)."""

    _TRANSLATE_PROMPT = (
        "Translate the following text to {lang}. "
        "Preserve all technical terms, numbers, and proper nouns. "
        "Return only the translated text without any explanation.\n\n"
        "Text:\n{text}"
    )
    _LANG_NAMES = {"zh": "Chinese (Simplified)", "ja": "Japanese", "en": "English"}

    def __init__(self) -> None:
        import openai  # openai 패키지를 DeepSeek 호환 모드로 사용

        deepseek_key  = os.getenv("DEEPSEEK_API_KEY", "")
        qwen_key      = os.getenv("QWEN_API_KEY", "")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

        if deepseek_key:
            self.client   = openai.OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
            self.model    = "deepseek-chat"
            self._provider = "deepseek"
        elif qwen_key:
            self.client   = openai.OpenAI(
                api_key=qwen_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            self.model    = "qwen-max"
            self._provider = "qwen"
        else:
            raise EnvironmentError(
                "DEEPSEEK_API_KEY 또는 QWEN_API_KEY 중 하나가 필요합니다."
            )

    def translate(self, text: str, target_lang: str) -> str:
        """텍스트를 target_lang으로 번역합니다."""
        lang_name = self._LANG_NAMES.get(target_lang, target_lang)
        prompt    = self._TRANSLATE_PROMPT.format(lang=lang_name, text=text[:1500])
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning(f"번역 실패 ({target_lang}): {e}")
            return ""

    def generate_interpreter_pairs(
        self,
        chunk_text: str,
        locale_list: list[str],
    ) -> dict[str, str]:
        """
        청크 텍스트를 여러 언어로 번역한 페어 딕셔너리를 반환합니다.

        Returns:
            {"zh": "...", "ja": "...", "en": "..."}
        """
        pairs: dict[str, str] = {}
        for locale in locale_list:
            if locale == "ko":
                continue  # 원본이 한국어이면 번역 불필요
            pairs[locale] = self.translate(chunk_text, locale)
        return pairs


# ════════════════════════════════════════════════════════════════════════════
# PDFToJSONL 클래스
# ════════════════════════════════════════════════════════════════════════════

class PDFToJSONL:
    """
    PDF → JSONL 변환 파이프라인 클래스.

    주요 메서드:
        extract_text_from_pdf(pdf_path)
        chunk_text(text)
        tag_metadata(chunk, domain)
        generate_interpreter_pairs(chunk, locale_list)
        save_to_jsonl(records, output_path)
        process(pdf_path, output_path, domain, locale_list)
    """

    def __init__(
        self,
        domain:       str        = "anti_aging",
        locale_list:  list[str]  = None,
        translate:    bool       = False,
    ) -> None:
        self.domain       = domain
        self.locale_list  = locale_list or ["ko"]
        self.translate    = translate
        self._tagger      = DOMAIN_TAGGERS.get(domain, tag_anti_aging_chunk)

        # 청킹 설정
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=    CHUNKING_CONFIG["chunk_size"],
            chunk_overlap= CHUNKING_CONFIG["chunk_overlap"],
            separators=    CHUNKING_CONFIG["separators"],
            length_function=len,
        )

        # 통역 클라이언트 (필요 시)
        self._interpreter: Optional[InterpreterClient] = None
        if translate and len(self.locale_list) > 1:
            try:
                self._interpreter = InterpreterClient()
                logger.info(f"통역 클라이언트 초기화 (provider: {self._interpreter._provider})")
            except EnvironmentError as e:
                logger.warning(f"통역 비활성화: {e}")

    # ── 퍼블릭 메서드 ────────────────────────────────────────────────────────

    def extract_text_from_pdf(self, pdf_path: Path) -> list[dict]:
        """PyMuPDF로 PDF 페이지별 텍스트를 추출합니다."""
        return extract_text_from_pdf(pdf_path)

    def chunk_text(self, text: str) -> list[str]:
        """LangChain RecursiveCharacterTextSplitter로 텍스트를 청킹합니다."""
        return self._splitter.split_text(text)

    def tag_metadata(self, chunk: str, domain: str) -> dict:
        """도메인 설정 파일을 참조하여 청크에 메타데이터 태그를 부여합니다."""
        tagger = DOMAIN_TAGGERS.get(domain, tag_anti_aging_chunk)
        return tagger(chunk)

    def generate_interpreter_pairs(
        self,
        chunk: str,
        locale_list: list[str],
    ) -> dict[str, str]:
        """LLM API로 통역 페어를 생성합니다."""
        if self._interpreter is None:
            return {}
        return self._interpreter.generate_interpreter_pairs(chunk, locale_list)

    def save_to_jsonl(self, records: list[dict], output_path: Path) -> None:
        """레코드 리스트를 JSONL 파일로 저장합니다."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"  JSONL 저장: {len(records)} 청크 → {output_path}")

    def process(
        self,
        pdf_path:    Path,
        output_path: Path,
    ) -> int:
        """
        단일 PDF 파일을 처리하여 JSONL로 저장하고 청크 수를 반환합니다.

        흐름:
            PDF → 페이지 추출 → 청킹 → 태깅 → (통역) → JSONL
        """
        pages = self.extract_text_from_pdf(pdf_path)
        if not pages:
            logger.warning(f"텍스트 추출 실패: {pdf_path.name}")
            return 0

        # PDF 메타데이터 파싱
        sample_text = " ".join(p["text"] for p in pages[:3])
        pdf_meta    = _parse_pdf_metadata(pages[0]["pdf_metadata"], sample_text)

        records: list[dict] = []
        chunk_index = 0

        for page in pages:
            page_text = page["text"]
            page_num  = page["page_number"]
            lang      = detect_language(page_text)

            chunks = self.chunk_text(page_text)
            for chunk_text in chunks:
                if not chunk_text.strip():
                    continue

                # 도메인 메타 태깅
                meta = self.tag_metadata(chunk_text, self.domain)

                # 통역 페어 생성
                translations = self.generate_interpreter_pairs(
                    chunk_text, self.locale_list
                ) if self._interpreter else {}

                record: dict = {
                    "chunk_id":    str(uuid.uuid4()),
                    "source_file": pdf_path.name,
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                    "text":        chunk_text,
                    "language":    lang,
                    "domain":      meta.get("domain", self.domain),
                    "category":    meta.get("category", ""),
                    "keywords":    meta.get("keywords", []),
                    "year":        pdf_meta["year"],
                    "author":      pdf_meta["author"],
                    "token_count": len(chunk_text.split()),   # 단어 수 근사
                    # 통역 페어
                    "text_zh":     translations.get("zh", ""),
                    "text_ja":     translations.get("ja", ""),
                    "text_en":     translations.get("en", ""),
                    # 임베딩은 04_upload_to_milvus.py에서 생성
                    "embedding":   [],
                }
                records.append(record)
                chunk_index += 1

        self.save_to_jsonl(records, output_path)
        return len(records)


# ════════════════════════════════════════════════════════════════════════════
# CLI 진입점
# ════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PDF → JSONL 변환 파이프라인 (ManiData Step 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pdf",    "-p", required=True,  type=Path, help="입력 PDF 파일 또는 디렉토리")
    p.add_argument("--output", "-o", required=True,  type=Path, help="출력 JSONL 파일 또는 디렉토리")
    p.add_argument(
        "--domain", "-d",
        default="anti_aging",
        choices=list(DOMAIN_TAGGERS.keys()),
        help="도메인 설정 (기본: anti_aging)",
    )
    p.add_argument(
        "--locales", "-l",
        nargs="+",
        default=["ko"],
        help="통역할 로케일 목록 (예: --locales zh ja en)",
    )
    p.add_argument(
        "--translate", "-t",
        action="store_true",
        help="LLM API로 통역 페어 생성 (API 키 필요)",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline = PDFToJSONL(
        domain=      args.domain,
        locale_list= args.locales,
        translate=   args.translate,
    )

    input_path  = args.pdf.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        logger.error(f"입력 경로 없음: {input_path}")
        sys.exit(1)

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        # 단일 파일
        out_file = output_path if output_path.suffix == ".jsonl" else output_path.with_suffix(".jsonl")
        count = pipeline.process(input_path, out_file)
        logger.info(f"완료: {count} 청크 → {out_file}")

    elif input_path.is_dir():
        # 디렉토리 일괄 처리
        pdf_files = sorted(input_path.glob("**/*.pdf"))
        if not pdf_files:
            logger.warning(f"PDF 없음: {input_path}")
            sys.exit(0)

        total = 0
        for pdf_file in tqdm(pdf_files, desc="PDF 처리"):
            rel      = pdf_file.relative_to(input_path)
            out_file = (output_path / rel).with_suffix(".jsonl")
            try:
                total += pipeline.process(pdf_file, out_file)
            except Exception as e:
                logger.error(f"[SKIP] {pdf_file.name}: {e}")

        logger.info(f"\n전체 완료: {total} 청크")

    else:
        logger.error("--pdf 는 PDF 파일 또는 디렉토리여야 합니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
