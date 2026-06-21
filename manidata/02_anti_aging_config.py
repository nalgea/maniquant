"""
02_anti_aging_config.py
항노화(Anti-Aging) 도메인 설정
- 키워드 사전 (카테고리·피부고민·치료유형별)
- 메타데이터 스키마 (Milvus 컬렉션 기준)
- 청킹 파라미터
- tag_anti_aging_chunk() 함수
"""

from __future__ import annotations
from typing import Any

# ════════════════════════════════════════════════════════════════════════════
# 1. 항노화 키워드 사전 (카테고리별)
# ════════════════════════════════════════════════════════════════════════════

ANTI_AGING_KEYWORDS: dict[str, dict[str, list[str]]] = {

    # ── 세포·분자 메커니즘 ─────────────────────────────────────────────────
    "cellular_mechanisms": {
        "senescence": [
            "cellular senescence", "senescent cells", "SASP",
            "senescence-associated secretory phenotype",
            "p16INK4a", "p21", "p53", "Rb pathway",
            "replicative senescence", "stress-induced senescence",
            "senolytic", "senostatic", "clearance of senescent cells",
        ],
        "telomere": [
            "telomere", "telomerase", "TERT", "TERC",
            "telomere length", "telomere shortening", "telomere attrition",
            "ALT pathway", "shelterin complex", "telomere dysfunction",
        ],
        "epigenetics": [
            "epigenetic clock", "DNA methylation", "histone modification",
            "chromatin remodeling", "Horvath clock", "GrimAge", "PhenoAge",
            "H3K27me3", "H3K4me3", "HDAC", "HAT",
            "epigenetic reprogramming", "Yamanaka factors", "partial reprogramming",
        ],
        "proteostasis": [
            "proteostasis", "protein homeostasis", "unfolded protein response",
            "UPR", "heat shock protein", "HSP70", "HSP90",
            "autophagy", "mitophagy", "ubiquitin-proteasome system",
            "UPS", "aggresome", "chaperone", "protein aggregation",
        ],
        "mitochondria": [
            "mitochondria", "mitochondrial dysfunction", "ROS",
            "reactive oxygen species", "oxidative stress",
            "mitochondrial membrane potential", "NAD+", "NADH",
            "electron transport chain", "ETC", "ATP synthase",
            "mitochondrial biogenesis", "PGC-1alpha", "mitochondrial fission",
        ],
    },

    # ── 신호 전달 경로 ─────────────────────────────────────────────────────
    "signaling_pathways": {
        "mtor": [
            "mTOR", "mTORC1", "mTORC2", "rapamycin",
            "S6K1", "4EBP1", "AMPK", "PI3K", "AKT",
            "insulin signaling", "IGF-1", "IGF-1R",
        ],
        "sirtuin": [
            "sirtuin", "SIRT1", "SIRT2", "SIRT3", "SIRT6", "SIRT7",
            "NAD+", "NMN", "NR", "nicotinamide riboside",
            "caloric restriction", "CR mimetic",
        ],
        "inflammation": [
            "inflammaging", "chronic inflammation", "NF-κB",
            "TNF-alpha", "IL-6", "IL-1β", "CRP",
            "NLRP3 inflammasome", "cytokine storm",
            "anti-inflammatory", "resolvins", "maresins",
        ],
    },

    # ── 피부 고민 (Skin Concerns) ──────────────────────────────────────────
    "skin_concerns": {
        "wrinkles": [
            "wrinkle", "fine line", "expression line", "crow's feet",
            "forehead line", "nasolabial fold", "marionette line",
            "주름", "잔주름", "팔자주름", "눈가주름",
        ],
        "skin_aging_markers": [
            "skin aging", "photoaging", "UV-induced aging",
            "dermal fibroblast", "collagen degradation", "elastin loss",
            "matrix metalloproteinase", "MMP", "transepidermal water loss", "TEWL",
            "피부 노화", "광노화", "콜라겐 감소",
        ],
        "hyperpigmentation": [
            "hyperpigmentation", "dark spot", "age spot", "melasma",
            "melanin", "tyrosinase", "UV damage",
            "기미", "잡티", "색소침착", "멜라닌",
        ],
        "skin_barrier": [
            "skin barrier", "ceramide", "hyaluronic acid", "moisturization",
            "TEWL", "atopic", "sensitive skin",
            "피부 장벽", "세라마이드", "히알루론산",
        ],
        "sagging": [
            "skin laxity", "sagging skin", "loss of firmness",
            "facial contour", "jowl", "ptosis",
            "피부 처짐", "탄력 저하", "리프팅",
        ],
    },

    # ── 치료 유형 (Treatment Types) ────────────────────────────────────────
    "treatment_types": {
        "topical": [
            "retinol", "retinoid", "tretinoin", "vitamin C", "niacinamide",
            "peptide", "growth factor", "EGF", "FGF",
            "antioxidant serum", "sunscreen", "SPF",
            "레티놀", "비타민C", "나이아신아마이드",
        ],
        "injectable": [
            "botulinum toxin", "Botox", "filler", "hyaluronic acid filler",
            "collagen stimulator", "PLLA", "PCL",
            "exosome", "PRP", "platelet-rich plasma",
            "보톡스", "필러", "엑소좀", "PRP",
        ],
        "energy_based": [
            "laser resurfacing", "fractional laser", "IPL",
            "radiofrequency", "RF", "ultrasound", "HIFU",
            "microneedling", "photodynamic therapy", "PDT",
            "레이저", "고주파", "울쎄라", "써마지",
        ],
        "pharmacological": [
            "metformin", "rapamycin", "resveratrol", "quercetin",
            "dasatinib", "navitoclax", "NAD+ precursor", "spermidine",
            "fisetin", "alpha-ketoglutarate", "AKG",
        ],
        "dietary": [
            "caloric restriction", "intermittent fasting",
            "time-restricted eating", "fasting-mimicking diet",
            "Mediterranean diet", "protein restriction",
            "간헐적 단식", "소식", "지중해 식단",
        ],
    },

    # ── 노화 관련 질환 ─────────────────────────────────────────────────────
    "aging_diseases": {
        "neurodegenerative": [
            "Alzheimer's disease", "Parkinson's disease",
            "amyloid beta", "tau protein", "alpha-synuclein",
            "neuroinflammation", "cognitive decline",
            "알츠하이머", "파킨슨", "인지 저하",
        ],
        "cardiovascular": [
            "vascular aging", "arterial stiffness", "atherosclerosis",
            "cardiac fibrosis", "hypertension", "HFpEF",
            "혈관 노화", "동맥경화", "고혈압",
        ],
        "musculoskeletal": [
            "sarcopenia", "muscle atrophy", "osteoporosis",
            "bone density", "frailty", "근감소증", "골다공증",
        ],
        "metabolic": [
            "metabolic syndrome", "type 2 diabetes", "insulin resistance",
            "obesity", "visceral fat", "대사 증후군", "당뇨",
        ],
    },

    # ── 바이오마커 & 측정 ──────────────────────────────────────────────────
    "biomarkers": {
        "aging_clocks": [
            "biological age", "epigenetic age", "epigenetic clock",
            "Horvath clock", "GrimAge", "telomere length",
            "생물학적 나이", "에피제네틱 나이",
        ],
        "inflammatory": [
            "CRP", "IL-6", "TNF-alpha", "IL-1β", "GDF15",
            "growth differentiation factor", "klotho",
            "염증 마커", "클로토",
        ],
        "omics": [
            "transcriptomics", "proteomics", "metabolomics",
            "single-cell RNA-seq", "scRNA-seq", "spatial transcriptomics",
            "multi-omics", "systems biology",
        ],
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 2. 청킹 파라미터 (LangChain RecursiveCharacterTextSplitter 기준)
# ════════════════════════════════════════════════════════════════════════════

CHUNKING_CONFIG: dict[str, Any] = {
    "chunk_size":    500,   # 문자 수 기준 (LangChain 기본)
    "chunk_overlap": 50,    # 오버랩 문자 수
    "separators":    ["\n\n", "\n", ". ", "。", " ", ""],
    "length_function": len,
}

# ════════════════════════════════════════════════════════════════════════════
# 3. Milvus 컬렉션 스키마 (bge-m3, 768차원, IVF_FLAT/IP)
# ════════════════════════════════════════════════════════════════════════════

MILVUS_COLLECTION_SCHEMA: dict[str, Any] = {
    "collection_name": "manidata_anti_aging",
    "vector_dim":       768,              # BAAI/bge-m3
    "index_type":       "IVF_FLAT",
    "metric_type":      "IP",            # Inner Product (bge-m3 권장)
    "index_params":     {"nlist": 1024},
    "search_params":    {"nprobe": 16},
    "scalar_fields": [
        {"name": "chunk_id",     "dtype": "VARCHAR", "max_length": 64,   "is_primary": True},
        {"name": "source_file",  "dtype": "VARCHAR", "max_length": 256},
        {"name": "page_number",  "dtype": "INT32"},
        {"name": "chunk_index",  "dtype": "INT32"},
        {"name": "text",         "dtype": "VARCHAR", "max_length": 4096},
        {"name": "language",     "dtype": "VARCHAR", "max_length": 16},
        {"name": "domain",       "dtype": "VARCHAR", "max_length": 64},
        {"name": "category",     "dtype": "VARCHAR", "max_length": 64},
        {"name": "keywords",     "dtype": "VARCHAR", "max_length": 1024},  # JSON
        {"name": "year",         "dtype": "INT32"},
        {"name": "author",       "dtype": "VARCHAR", "max_length": 512},
        {"name": "token_count",  "dtype": "INT32"},
        # 통역 페어 (JSON 직렬화)
        {"name": "text_zh",      "dtype": "VARCHAR", "max_length": 4096},  # 중국어
        {"name": "text_ja",      "dtype": "VARCHAR", "max_length": 4096},  # 일본어
        {"name": "text_en",      "dtype": "VARCHAR", "max_length": 4096},  # 영어
    ],
}

# ════════════════════════════════════════════════════════════════════════════
# 4. 임베딩 & LLM 설정
# ════════════════════════════════════════════════════════════════════════════

EMBEDDING_CONFIG: dict[str, Any] = {
    "model_name":  "BAAI/bge-m3",
    "dimensions":  768,
    "batch_size":  32,
    "normalize":   True,             # IP 메트릭 사용 시 정규화 필수
    "device":      "cpu",            # "cuda" 로 변경 시 GPU 가속
}

LLM_ROUTER_CONFIG: dict[str, Any] = {
    "primary": {
        "provider":     "deepseek",
        "model":        "deepseek-chat",
        "api_base":     "https://api.deepseek.com",
        "temperature":  0.3,
        "max_tokens":   2048,
    },
    "secondary": {
        "provider":     "qwen",
        "model":        "qwen-max",
        "api_base":     "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "temperature":  0.3,
        "max_tokens":   2048,
    },
    "fallback": {
        "provider":     "anthropic",
        "model":        "claude-3-5-sonnet-20241022",
        "temperature":  0.3,
        "max_tokens":   2048,
    },
}

# 로케일별 시스템 프롬프트 접두어
LOCALE_SYSTEM_PROMPTS: dict[str, str] = {
    "ko": (
        "당신은 항노화·뷰티·피부 전문 AI 상담사입니다. "
        "제공된 참고 문서를 기반으로 정확하고 신뢰할 수 있는 정보를 제공하십시오. "
        "의학적 판단이 필요한 경우 반드시 전문의 상담을 권유하십시오."
    ),
    "zh": (
        "您是一位专业的抗衰老、美容和皮肤专家AI顾问。"
        "请基于提供的参考文献，提供准确可靠的信息。"
        "如需医学判断，请务必建议咨询专业医生。"
    ),
    "ja": (
        "あなたはアンチエイジング・美容・皮膚の専門AIアドバイザーです。"
        "提供された参考文書に基づいて、正確で信頼できる情報を提供してください。"
        "医学的判断が必要な場合は、必ず専門医への相談を勧めてください。"
    ),
    "en": (
        "You are a professional AI consultant specializing in anti-aging, beauty, and dermatology. "
        "Provide accurate and reliable information based on the reference documents provided. "
        "If medical judgment is required, always recommend consulting a specialist."
    ),
}

# ════════════════════════════════════════════════════════════════════════════
# 5. 헬퍼 함수
# ════════════════════════════════════════════════════════════════════════════

def _flatten_keywords() -> list[str]:
    """키워드 사전에서 모든 키워드를 평탄화합니다."""
    result: list[str] = []
    for cat in ANTI_AGING_KEYWORDS.values():
        for kws in cat.values():
            result.extend(kws)
    return list(set(result))


_ALL_KEYWORDS: list[str] = _flatten_keywords()


def get_domain_tags(text: str) -> list[str]:
    """텍스트에서 일치하는 카테고리 태그를 추출합니다."""
    text_lower = text.lower()
    matched: list[str] = []
    for cat_name, subcats in ANTI_AGING_KEYWORDS.items():
        for sub_name, kws in subcats.items():
            if any(kw.lower() in text_lower for kw in kws):
                matched.append(f"{cat_name}/{sub_name}")
    return list(set(matched))


def get_keywords(text: str, top_n: int = 15) -> list[str]:
    """텍스트에서 항노화 관련 키워드를 추출합니다 (길이 내림차순)."""
    text_lower = text.lower()
    matched = [kw for kw in _ALL_KEYWORDS if kw.lower() in text_lower]
    matched = sorted(set(matched), key=len, reverse=True)
    return matched[:top_n]


def tag_anti_aging_chunk(text: str) -> dict:
    """
    청크 텍스트를 분석하여 메타데이터 딕셔너리를 반환합니다.

    Returns:
        {
            "domain": "anti_aging",
            "category": "skin_concerns/wrinkles",   # 가장 높은 매칭 카테고리
            "domain_tags": ["skin_concerns/wrinkles", ...],
            "keywords": ["wrinkle", "collagen", ...],
        }
    """
    tags     = get_domain_tags(text)
    keywords = get_keywords(text)

    # 가장 구체적인 태그를 primary category로 사용
    primary_category = tags[0] if tags else "anti_aging/general"

    return {
        "domain":      "anti_aging",
        "category":    primary_category,
        "domain_tags": tags,
        "keywords":    keywords,
    }
