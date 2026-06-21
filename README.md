# ManiQuant — AI 에이전트 생태계 플랫폼 v1.0

> **비전**: "AI가 인간과 자연스럽게 공존하며 실질적인 가치를 창출하는 세상"

한국의 의료·뷰티·관광 도메인을 시작으로, 글로벌 사용자(중국어·일본어·영어권)에게
문화적으로 적응된 AI 에이전트를 제공하는 **AI 인력 생태계 플랫폼**입니다.

---

## 📐 3계층 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│  3층  ManiVerse (마니버스)  — B2C 마켓플레이스 & 포털 (Phase 2)    │
├──────────────────────────────────────────────────────────────────┤
│  2층  ManiAgent (마니에이전트) — 도메인 특화 AI 에이전트 서빙       │
│       RAG 엔진 + LLM 라우터(DeepSeek/Qwen/Claude) + REST API      │
├──────────────────────────────────────────────────────────────────┤
│  1층  ManiData (마니데이터) — 데이터 수집·정제·구조화              │
│       PDF/EPUB → 청킹 → 임베딩 → Milvus 벡터 DB                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 프로젝트 구조

```
maniquant/
├── manidata/
│   ├── __init__.py
│   ├── 01_pdf_to_jsonl.py          # PDF → JSONL (PyMuPDF + LangChain)
│   ├── 02_anti_aging_config.py     # 항노화 키워드·스키마·LLM 라우터 설정
│   └── 03_plastic_config.py        # 성형외과 도메인 설정 (Phase 1)
├── maniagent/
│   ├── __init__.py
│   ├── 04_upload_to_milvus.py      # JSONL → bge-m3 임베딩 → Milvus 적재
│   ├── 05_rag_engine.py            # ManiAgentRAG + LLM 라우터
│   └── 06_api_server.py            # FastAPI (/v1/chat, /health, SSE 스트리밍)
├── maniverse/
│   └── README.md                   # Phase 2 프론트엔드 계획
├── data/
│   ├── raw/                        # 원본 PDF 저장
│   └── processed/                  # JSONL 파일 저장
├── scripts/
│   └── run_full_pipeline.sh        # 전체 파이프라인 원스텝 실행
├── tests/
│   ├── test_rag_engine.py          # ManiAgentRAG 단위 테스트
│   └── test_api_server.py          # FastAPI 통합 테스트
├── requirements.txt
├── .env.example                    # 환경 변수 템플릿
├── docker-compose.yml              # Milvus + API 서버 컨테이너
└── README.md
```

---

## ⚡ 빠른 시작

### 1. 환경 준비

```bash
# 저장소 클론
git clone https://github.com/maniquant/maniquant.git
cd maniquant

# 가상환경 생성
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에 API 키 입력 (최소 DEEPSEEK_API_KEY 또는 QWEN_API_KEY 필요)
```

### 2. Milvus 실행 (Docker)

```bash
docker-compose up -d etcd minio milvus
# 헬스체크: curl http://localhost:9091/healthz
```

### 3. 전체 파이프라인 (원스텝)

```bash
chmod +x scripts/run_full_pipeline.sh
./scripts/run_full_pipeline.sh data/raw anti_aging ko zh ja
```

### 4. 단계별 실행

```bash
# Step 1: PDF → JSONL
python manidata/01_pdf_to_jsonl.py \
    --pdf data/raw/anti_aging_book.pdf \
    --output data/processed/manidata_anti_aging_v1.jsonl \
    --domain anti_aging \
    --locales ko zh

# Step 2: Milvus 적재
python maniagent/04_upload_to_milvus.py \
    --input data/processed/manidata_anti_aging_v1.jsonl \
    --collection manidata_anti_aging

# Step 3: API 서버
uvicorn maniagent.06_api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 5. API 테스트

```bash
# 한국어 질문 → 중국어 응답
curl -X POST http://localhost:8000/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "콜라겐이 피부에 좋은 이유는?", "locale": "zh"}'

# 영어 질문 → 일본어 응답
curl -X POST http://localhost:8000/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "What is cellular senescence?", "locale": "ja"}'

# Swagger UI
open http://localhost:8000/docs
```

---

## 🔌 API 명세

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET`  | `/health` | 서버 상태 확인 |
| `POST` | `/v1/chat` | RAG 질의응답 (locale 지원) |
| `POST` | `/v1/chat/stream` | SSE 스트리밍 응답 |
| `GET`  | `/v1/domains` | 지원 도메인·카테고리 목록 |
| `GET`  | `/v1/stats` | Milvus 컬렉션 통계 |

### POST `/v1/chat` 요청 예시

```json
{
  "query":           "콜라겐이 피부에 좋은 이유는?",
  "locale":          "zh",
  "top_k":           5,
  "score_threshold": 0.5,
  "filter_expr":     "year > 2020"
}
```

---

## 🛠 기술 스택

| 계층 | 기술 | 역할 |
|------|------|------|
| PDF 파싱 | PyMuPDF (fitz) | 텍스트·메타데이터 추출 |
| 청킹 | LangChain `RecursiveCharacterTextSplitter` | 의미 단위 분할 |
| 임베딩 | `BAAI/bge-m3` (768d) | 한/중/일/영 다국어 벡터 |
| 벡터 DB | Milvus (IVF_FLAT / IP) | 분산형 벡터 검색 |
| LLM 주력 | DeepSeek-V4 | RAG 응답 생성 (가성비 최적) |
| LLM 보조 | Qwen-max | DeepSeek 폴백 |
| LLM 폴백 | Claude Sonnet | 최종 폴백 |
| API 서버 | FastAPI + uvicorn | REST + SSE 스트리밍 |
| 컨테이너 | Docker Compose | Milvus + API 일괄 배포 |

---

## 🧬 지원 도메인

### 항노화 (Anti-Aging) — `anti_aging`
세포 노화·텔로미어·에피제네틱스·단백질 항상성·미토콘드리아·
mTOR 경로·시르투인·인플라메이징·심혈관 노화·신경퇴행성 질환·
근감소증·대사 증후군·피부 노화·약물 개입·식이 제한·유전자 치료·
마이크로바이옴·운동·바이오마커·오믹스

### 성형외과 (Plastic Surgery) — `plastic` (Phase 1)
시술 카테고리·시술 부위·회복·부작용·결과 평가·의료 환경 (한/중/일)

---

## 🧪 테스트

```bash
pytest tests/ -v

# 커버리지 측정
pytest tests/ --cov=manidata --cov=maniagent --cov-report=html
```

---

## 🗺 로드맵

| 단계 | 목표 | 기간 |
|------|------|------|
| **Phase 0** | 데이터 파이프라인 구축 ✅ | 2주 |
| **Phase 1** | ManiAgent RAG 고도화 (LLM 라우터·통역) | 4주 |
| **Phase 2** | ManiVerse 프론트엔드 MVP | 6주 |
| **Phase 3** | 병원 파일럿 연동 (강남 성형외과 5개사) | 8주 |
| **Phase 4** | 글로벌 확장 (음성·이미지·의료관광) | 12주+ |

---

## 📬 연락처

| 역할 | 연락처 |
|------|--------|
| CEO/PM | ooo@maniquant.ai |
| CTO | ooo@maniquant.ai |
