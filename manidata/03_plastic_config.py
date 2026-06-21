"""
03_plastic_config.py
성형외과(Plastic Surgery) 도메인 설정

키워드 사전, Milvus 스키마, 임베딩/LLM 설정, 로케일 프롬프트,
tag_plastic_chunk() 함수를 정의합니다.
"""

from __future__ import annotations
from typing import Any

# ════════════════════════════════════════════════════════════════════════════
# 1. 성형외과 키워드 사전
# ════════════════════════════════════════════════════════════════════════════

PLASTIC_KEYWORDS: dict[str, dict[str, list[str]]] = {

    # ── 시술 카테고리 ─────────────────────────────────────────────────────
    "procedure_category": {
        "surgical": [
            "rhinoplasty", "코 성형", "鼻整形",
            "blepharoplasty", "눈꺼풀 수술", "双眼皮手术",
            "rhytidectomy", "facelift", "안면거상", "拉皮手术",
            "liposuction", "지방흡입", "抽脂",
            "breast augmentation", "가슴 성형", "隆胸",
            "abdominoplasty", "복부 성형", "腹部整形",
            "orthognathic surgery", "턱 교정", "正颌手术",
        ],
        "non_surgical": [
            "botulinum toxin", "보톡스", "肉毒素",
            "dermal filler", "필러", "玻尿酸",
            "thread lift", "실리프팅", "线雕",
            "HIFU", "울쎄라", "超声刀",
            "radiofrequency", "고주파", "射频",
            "laser treatment", "레이저", "激光",
            "chemical peel", "필링", "化学换肤",
        ],
    },

    # ── 시술 부위 ──────────────────────────────────────────────────────────
    "body_area": {
        "face": [
            "forehead", "이마", "额头",
            "eye", "눈", "眼睛",
            "nose", "코", "鼻子",
            "cheek", "볼", "脸颊",
            "jaw", "턱", "下颌",
            "lip", "입술", "嘴唇",
            "neck", "목", "颈部",
        ],
        "body": [
            "breast", "가슴", "胸部",
            "abdomen", "복부", "腹部",
            "thigh", "허벅지", "大腿",
            "arm", "팔", "手臂",
            "buttock", "엉덩이", "臀部",
        ],
    },

    # ── 회복 & 경과 ────────────────────────────────────────────────────────
    "recovery": {
        "post_op_care": [
            "recovery time", "회복 기간", "恢复期",
            "downtime", "다운타임",
            "swelling", "붓기", "肿胀",
            "bruising", "멍", "瘀伤",
            "scar", "흉터", "疤痕",
            "wound healing", "상처 회복", "伤口愈合",
            "compression garment", "압박복", "压力服",
        ],
    },

    # ── 부작용 & 위험 ──────────────────────────────────────────────────────
    "risks": {
        "complications": [
            "complication", "합병증", "并发症",
            "infection", "감염", "感染",
            "asymmetry", "비대칭", "不对称",
            "revision surgery", "재수술", "修复手术",
            "keloid", "켈로이드", "瘢痕疙瘩",
            "nerve damage", "신경 손상", "神经损伤",
            "capsular contracture", "구형구축", "包膜挛缩",
        ],
    },

    # ── 결과 & 평가 ────────────────────────────────────────────────────────
    "outcomes": {
        "satisfaction": [
            "patient satisfaction", "환자 만족도", "患者满意度",
            "natural result", "자연스러운 결과", "自然效果",
            "before after", "전후 비교", "术前术后",
            "long-term result", "장기 결과", "长期效果",
        ],
    },

    # ── 병원 & 의료 환경 ───────────────────────────────────────────────────
    "medical_context": {
        "hospital": [
            "plastic surgery clinic", "성형외과", "整形外科",
            "board certified", "전문의", "专科医生",
            "Gangnam", "강남", "江南",
            "medical tourism", "의료관광", "医疗旅游",
            "consultation", "상담", "咨询",
            "anesthesia", "마취", "麻醉",
        ],
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 2. 청킹 파라미터 (anti_aging_config와 동일 구조)
# ════════════════════════════════════════════════════════════════════════════

PLASTIC_CHUNKING_CONFIG: dict[str, Any] = {
    "chunk_size":    500,
    "chunk_overlap": 50,
    "separators":    ["\n\n", "\n", ". ", "。", " ", ""],
}

# ════════════════════════════════════════════════════════════════════════════
# 3. Milvus 컬렉션 스키마
# ════════════════════════════════════════════════════════════════════════════

MILVUS_COLLECTION_SCHEMA: dict[str, Any] = {
    "collection_name": "manidata_plastic_gangnam",
    "vector_dim":       1024,
    "index_type":       "IVF_FLAT",
    "metric_type":      "IP",
    "index_params":     {"nlist": 1024},
    "search_params":    {"nprobe": 16},
    "scalar_fields": [
        {"name": "chunk_id",        "dtype": "VARCHAR", "max_length": 64,   "is_primary": True},
        {"name": "source_file",     "dtype": "VARCHAR", "max_length": 256},
        {"name": "page_number",     "dtype": "INT32"},
        {"name": "chunk_index",     "dtype": "INT32"},
        {"name": "chunk_type",      "dtype": "VARCHAR", "max_length": 32},
        {"name": "text",            "dtype": "VARCHAR", "max_length": 4096},
        {"name": "language",        "dtype": "VARCHAR", "max_length": 16},
        {"name": "domain",          "dtype": "VARCHAR", "max_length": 64},
        {"name": "category",        "dtype": "VARCHAR", "max_length": 64},
        {"name": "keywords",        "dtype": "VARCHAR", "max_length": 1024},
        {"name": "year",            "dtype": "INT32"},
        {"name": "author",          "dtype": "VARCHAR", "max_length": 512},
        {"name": "token_count",     "dtype": "INT32"},
        {"name": "text_zh",         "dtype": "VARCHAR", "max_length": 4096},
        {"name": "text_ja",         "dtype": "VARCHAR", "max_length": 4096},
        {"name": "text_en",         "dtype": "VARCHAR", "max_length": 4096},
        # 병원 프로필 전용 필드
        {"name": "clinic_name",     "dtype": "VARCHAR", "max_length": 128},
        {"name": "naver_place_id",  "dtype": "VARCHAR", "max_length": 64},
        {"name": "address",         "dtype": "VARCHAR", "max_length": 256},
        {"name": "visitor_reviews", "dtype": "INT32"},
        {"name": "online_mentions", "dtype": "INT32"},
        {"name": "positive_rate",   "dtype": "FLOAT"},
        {"name": "rating",          "dtype": "FLOAT"},
    ],
}

# ════════════════════════════════════════════════════════════════════════════
# 4. 임베딩 & LLM 설정
# ════════════════════════════════════════════════════════════════════════════

EMBEDDING_CONFIG: dict[str, Any] = {
    "model_name": "BAAI/bge-m3",
    "dimensions": 1024,
    "batch_size": 32,
    "normalize":  True,
    "device":     "cpu",
}

LLM_ROUTER_CONFIG: dict[str, Any] = {
    "primary": {
        "provider":    "deepseek",
        "model":       "deepseek-chat",
        "api_base":    "https://api.deepseek.com",
        "temperature": 0.3,
        "max_tokens":  2048,
    },
    "secondary": {
        "provider":    "qwen",
        "model":       "qwen-max",
        "api_base":    "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "temperature": 0.3,
        "max_tokens":  2048,
    },
    "fallback": {
        "provider":    "anthropic",
        "model":       "claude-sonnet-4-6",
        "temperature": 0.3,
        "max_tokens":  2048,
    },
}

LOCALE_SYSTEM_PROMPTS: dict[str, str] = {
    "ko": (
        "당신은 강남 성형외과 전문 AI 뷰티투어 어드바이저입니다. "
        "네이버 플레이스 데이터와 블로그/카페 리뷰를 기반으로 "
        "방문자에게 병원 추천, 시술 정보, 가격 안내를 제공하십시오. "
        "의학적 판단이 필요한 경우 반드시 전문의 상담을 권유하십시오."
    ),
    "zh": (
        "您是江南整形外科专业AI美容旅游顾问。"
        "基于Naver地图数据和博客/咖啡评论，"
        "为访客提供医院推荐、手术信息和价格指导。"
        "如需医学判断，请务必建议咨询专业医生。"
    ),
    "ja": (
        "あなたは江南整形外科専門AIビューティーツアーアドバイザーです。"
        "Naver Placeデータとブログ/カフェレビューをもとに、"
        "訪問者に病院の推薦、施術情報、料金案内を提供してください。"
        "医学的判断が必要な場合は、必ず専門医への相談を勧めてください。"
    ),
    "en": (
        "You are a professional AI beauty tour advisor specializing in Gangnam plastic surgery clinics. "
        "Based on Naver Place data and blog/cafe reviews, "
        "provide hospital recommendations, procedure information, and pricing guidance to visitors. "
        "If medical judgment is required, always recommend consulting a specialist."
    ),
}

# ════════════════════════════════════════════════════════════════════════════
# 5. 헬퍼 함수
# ════════════════════════════════════════════════════════════════════════════

def _flatten_plastic_keywords() -> list[str]:
    result: list[str] = []
    for cat in PLASTIC_KEYWORDS.values():
        for kws in cat.values():
            result.extend(kws)
    return list(set(result))


_ALL_PLASTIC_KW: list[str] = _flatten_plastic_keywords()


def get_plastic_domain_tags(text: str) -> list[str]:
    """텍스트에서 성형외과 서브 카테고리 태그를 추출합니다."""
    text_lower = text.lower()
    matched: list[str] = []
    for cat_name, subcats in PLASTIC_KEYWORDS.items():
        for sub_name, kws in subcats.items():
            if any(kw.lower() in text_lower for kw in kws):
                matched.append(f"{cat_name}/{sub_name}")
    return list(set(matched))


def get_plastic_keywords(text: str, top_n: int = 15) -> list[str]:
    """텍스트에서 성형외과 관련 키워드를 추출합니다."""
    text_lower = text.lower()
    matched = [kw for kw in _ALL_PLASTIC_KW if kw.lower() in text_lower]
    return sorted(set(matched), key=len, reverse=True)[:top_n]


def tag_plastic_chunk(text: str) -> dict:
    """
    성형외과 청크에 메타데이터 태그를 부여합니다.

    Returns:
        {
            "domain":      "plastic",
            "category":    "procedure_category/surgical",
            "domain_tags": [...],
            "keywords":    [...],
        }
    """
    tags     = get_plastic_domain_tags(text)
    keywords = get_plastic_keywords(text)
    primary  = tags[0] if tags else "plastic/general"

    return {
        "domain":      "plastic",
        "category":    primary,
        "domain_tags": tags,
        "keywords":    keywords,
    }
