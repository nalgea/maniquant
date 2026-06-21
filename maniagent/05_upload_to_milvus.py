"""
04_upload_to_milvus.py
ManiAgent Step 1 — JSONL → 임베딩 생성 → Milvus 적재

임베딩 모델: BAAI/bge-m3 (768차원, 다국어)
인덱스: IVF_FLAT / IP (내적 유사도)
클래스: MilvusUploader

사용법:
    python maniagent/04_upload_to_milvus.py \\
        --input data/processed/manidata_anti_aging_v1.jsonl \\
        --collection manidata_anti_aging

    python maniagent/04_upload_to_milvus.py \\
        --input data/processed/ \\
        --collection manidata_anti_aging \\
        --recreate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# ─── 의존성 ────────────────────────────────────────────────────────────────
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from pymilvus import (
        connections,
        utility,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        MilvusException,
    )
    from tqdm import tqdm
    from dotenv import load_dotenv
except ImportError as e:
    print(f"[ERROR] 필수 패키지 없음: {e}")
    print("       pip install sentence-transformers pymilvus numpy tqdm python-dotenv")
    sys.exit(1)

_HERE = Path(__file__).parent

import importlib.util as _ilu

def _load_config(name: str):
    spec = _ilu.spec_from_file_location(name, _HERE.parent / "manidata" / f"{name}.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_aa = _load_config("02_anti_aging_config")
EMBEDDING_CONFIG = _aa.EMBEDDING_CONFIG
MILVUS_COLLECTION_SCHEMA = _aa.MILVUS_COLLECTION_SCHEMA

load_dotenv(_HERE.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# MilvusUploader 클래스
# ════════════════════════════════════════════════════════════════════════════

class MilvusUploader:
    """
    JSONL → pandas-like 레코드 → BAAI/bge-m3 임베딩 → Milvus insert

    사용 예:
        uploader = MilvusUploader(collection_name="manidata_anti_aging")
        uploader.upload("data/processed/paper.jsonl")
    """

    def __init__(
        self,
        collection_name: str  = "",
        host:            str  = "",
        port:            str  = "",
        recreate:        bool = False,
    ) -> None:
        self.collection_name = collection_name or MILVUS_COLLECTION_SCHEMA["collection_name"]
        self.host            = host or os.getenv("MILVUS_HOST", "localhost")
        self.port            = port or os.getenv("MILVUS_PORT", "19530")
        self.recreate        = recreate

        # bge-m3 모델 로드
        logger.info(f"임베딩 모델 로딩: {EMBEDDING_CONFIG['model_name']}")
        self._model = SentenceTransformer(
            EMBEDDING_CONFIG["model_name"],
            device=EMBEDDING_CONFIG.get("device", "cpu"),
        )
        logger.info("임베딩 모델 로드 완료")

        # Milvus 연결
        self._connect()
        self._collection: Collection = self._ensure_collection()

    # ── Milvus 연결 ──────────────────────────────────────────────────────────

    def _connect(self) -> None:
        token = os.getenv("MILVUS_TOKEN", "")
        host  = self.host
        if host.startswith("http"):
            connections.connect(alias="default", uri=host, token=token)
            logger.info(f"Milvus(Zilliz) 연결: {host}")
        else:
            kwargs: dict[str, Any] = {"alias": "default", "host": host, "port": self.port}
            if token:
                kwargs["token"] = token
            connections.connect(**kwargs)
            logger.info(f"Milvus 연결: {host}:{self.port}")

    # ── 컬렉션 스키마 구성 ───────────────────────────────────────────────────

    def _build_schema(self) -> CollectionSchema:
        dtype_map = {
            "VARCHAR": DataType.VARCHAR,
            "INT32":   DataType.INT32,
            "INT64":   DataType.INT64,
            "BOOL":    DataType.BOOL,
            "FLOAT":   DataType.FLOAT,
        }

        fields: list[FieldSchema] = []
        for f in MILVUS_COLLECTION_SCHEMA["scalar_fields"]:
            kwargs: dict[str, Any] = {
                "name":       f["name"],
                "dtype":      dtype_map[f["dtype"]],
                "is_primary": f.get("is_primary", False),
            }
            if f["dtype"] == "VARCHAR":
                kwargs["max_length"] = f.get("max_length", 256)
            fields.append(FieldSchema(**kwargs))

        # 벡터 필드 (bge-m3, 768차원)
        fields.append(
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=MILVUS_COLLECTION_SCHEMA["vector_dim"],
            )
        )

        return CollectionSchema(
            fields=fields,
            description=f"ManiData 컬렉션: {self.collection_name}",
            enable_dynamic_field=True,
        )

    def _ensure_collection(self) -> Collection:
        """컬렉션이 없으면 생성, recreate=True 이면 삭제 후 재생성."""
        exists = utility.has_collection(self.collection_name)

        if exists and self.recreate:
            utility.drop_collection(self.collection_name)
            logger.warning(f"기존 컬렉션 삭제: {self.collection_name}")
            exists = False

        if not exists:
            schema     = self._build_schema()
            collection = Collection(name=self.collection_name, schema=schema)

            # IVF_FLAT / IP 인덱스 생성
            collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": MILVUS_COLLECTION_SCHEMA["index_type"],   # IVF_FLAT
                    "metric_type": MILVUS_COLLECTION_SCHEMA["metric_type"], # IP
                    "params": MILVUS_COLLECTION_SCHEMA["index_params"],     # nlist=1024
                },
            )
            logger.info(f"컬렉션 생성: {self.collection_name} (IVF_FLAT/IP, 768d)")
        else:
            collection = Collection(self.collection_name)
            logger.info(f"기존 컬렉션 사용: {self.collection_name}")

        collection.load()
        return collection

    # ── 임베딩 생성 ──────────────────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        BAAI/bge-m3로 텍스트 배치를 임베딩합니다.
        normalize_embeddings=True → IP 메트릭에서 코사인 유사도와 동일 효과.
        """
        batch_size = EMBEDDING_CONFIG.get("batch_size", 32)
        all_vecs: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs  = self._model.encode(
                batch,
                batch_size=batch_size,
                normalize_embeddings=EMBEDDING_CONFIG.get("normalize", True),
                show_progress_bar=False,
            )
            all_vecs.extend(vecs.tolist())

        return all_vecs

    # ── JSONL 로드 ───────────────────────────────────────────────────────────

    @staticmethod
    def load_jsonl(path: Path) -> list[dict]:
        records: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 파싱 오류 line {line_no}: {e}")
        return records

    # ── 레코드 전처리 ─────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_row(record: dict) -> dict:
        """Milvus 삽입용으로 타입/길이를 정규화합니다."""
        row = dict(record)
        # list → JSON 문자열
        if isinstance(row.get("keywords"), list):
            row["keywords"] = json.dumps(row["keywords"], ensure_ascii=False)[:1024]
        # 문자열 길이 제한
        str_limits = {
            "chunk_id": 64, "source_file": 256, "text": 4096,
            "language": 16, "domain": 64, "category": 64,
            "author": 512, "text_zh": 4096, "text_ja": 4096, "text_en": 4096,
        }
        for field, limit in str_limits.items():
            row[field] = str(row.get(field, ""))[:limit]
        # 정수 타입 보장
        for int_field in ("page_number", "chunk_index", "year", "token_count"):
            try:
                row[int_field] = int(row.get(int_field, 0))
            except (ValueError, TypeError):
                row[int_field] = 0
        # embedding 필드 제거 (별도 채움)
        row.pop("embedding", None)
        return row

    # ── 업로드 메인 ──────────────────────────────────────────────────────────

    def upload(self, jsonl_path: Path, insert_batch: int = 100) -> int:
        """
        JSONL 파일을 읽어 임베딩 생성 후 Milvus에 삽입합니다.

        Args:
            jsonl_path:   입력 JSONL 경로
            insert_batch: Milvus 삽입 배치 크기

        Returns:
            삽입된 레코드 수
        """
        records = self.load_jsonl(jsonl_path)
        if not records:
            logger.warning(f"빈 파일: {jsonl_path.name}")
            return 0

        logger.info(f"임베딩 생성: {len(records)} 청크 ({jsonl_path.name})")
        texts      = [r.get("text", "") for r in records]
        embeddings = self.embed_texts(texts)

        inserted_total = 0
        for i in tqdm(
            range(0, len(records), insert_batch),
            desc=f"Milvus 삽입 [{jsonl_path.name}]",
            unit="batch",
        ):
            batch_records = records[i : i + insert_batch]
            batch_vecs    = embeddings[i : i + insert_batch]

            rows = [self._prepare_row(r) for r in batch_records]

            # 스키마 필드 순서대로 리스트 구성 (Milvus column-based insert)
            schema_fields = [f["name"] for f in MILVUS_COLLECTION_SCHEMA["scalar_fields"]]
            data: list[list] = []
            for key in schema_fields:
                col = [r.get(key, "") for r in rows]
                # 정수 필드 타입 보장
                if key in ("page_number", "chunk_index", "year", "token_count"):
                    col = [int(v) if v else 0 for v in col]
                data.append(col)
            data.append(batch_vecs)  # embedding 마지막

            try:
                result = self._collection.insert(data)
                inserted_total += result.insert_count
            except MilvusException as e:
                logger.error(f"삽입 오류 (batch {i}): {e}")

        self._collection.flush()
        logger.info(f"  → {inserted_total} 레코드 업로드 완료: {self.collection_name}")
        return inserted_total

    def upload_directory(self, input_dir: Path) -> int:
        """디렉토리 내 모든 JSONL 파일을 업로드합니다."""
        jsonl_files = sorted(input_dir.glob("**/*.jsonl"))
        if not jsonl_files:
            logger.warning(f"JSONL 없음: {input_dir}")
            return 0

        total = 0
        for path in jsonl_files:
            try:
                total += self.upload(path)
            except Exception as e:
                logger.error(f"[SKIP] {path.name}: {e}")

        logger.info(f"\n전체 업로드: {total} 레코드")
        return total


# ════════════════════════════════════════════════════════════════════════════
# CLI 진입점
# ════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="JSONL → Milvus 적재 (ManiAgent Step 1)")
    p.add_argument("--input",      "-i", required=True, type=Path)
    p.add_argument("--collection", "-c", default="manidata_anti_aging")
    p.add_argument("--host",       default=os.getenv("MILVUS_HOST", "localhost"))
    p.add_argument("--port",       default=os.getenv("MILVUS_PORT", "19530"))
    p.add_argument("--recreate",   action="store_true", help="컬렉션 삭제 후 재생성")
    p.add_argument("--batch-size", type=int, default=100)
    return p


def main() -> None:
    args     = build_arg_parser().parse_args()
    uploader = MilvusUploader(
        collection_name=args.collection,
        host=args.host,
        port=args.port,
        recreate=args.recreate,
    )

    input_path = args.input.resolve()
    if not input_path.exists():
        logger.error(f"입력 경로 없음: {input_path}")
        sys.exit(1)

    if input_path.is_file():
        uploader.upload(input_path, insert_batch=args.batch_size)
    elif input_path.is_dir():
        uploader.upload_directory(input_path)
    else:
        logger.error("--input 은 JSONL 파일 또는 디렉토리여야 합니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
