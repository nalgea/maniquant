# ManiQuant — AI 에이전트 생태계 플랫폼 v1.0

> **비전**: "AI가 인간과 자연스럽게 공존하며 실질적인 가치를 창출하는 세상"

한국의 의료·뷰티·관광 도메인을 시작으로, 글로벌 사용자(중국어·일본어·영어권)에게
문화적으로 적응된 AI 에이전트를 제공하는 **AI 인력 생태계 플랫폼**입니다.

[![Railway](https://img.shields.io/badge/Railway-배포중-brightgreen?logo=railway)](https://maniquant-production.up.railway.app)
[![Zilliz](https://img.shields.io/badge/Zilliz-Cloud-blue?logo=apachespark)](https://cloud.zilliz.com)
[![GitHub](https://img.shields.io/badge/GitHub-nalgea%2Fmaniquant-black?logo=github)](https://github.com/nalgea/maniquant)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)

---

## 🌐 배포 정보

| 항목 | 값 |
|------|-----|
| **API 서버** | https://maniquant-production.up.railway.app |
| **Swagger UI** | https://maniquant-production.up.railway.app/docs |
| **Redoc** | https://maniquant-production.up.railway.app/redoc |
| **Health Check** | https://maniquant-production.up.railway.app/health |
| **Metrics** | https://maniquant-production.up.railway.app/metrics |
| **벡터 DB** | Zilliz Cloud Free-01 (aws-eu-central-1) |
| **Collections** | `manidata_anti_aging`, `manidata_plastic_gangnm` |

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
│       PDF/EPUB → 청킹 → 임베딩 → Zilliz Cloud 벡터 DB             │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 프로젝트 구조

```
maniquant/
├── manidata/
│   ├── 01_pdf_to_jsonl.py          # PDF → JSONL (PyMuPDF + LangChain)
│   ├── 02_anti_aging_config.py     # 항노화 키워드·스키마·LLM 라우터 설정
│   ├── 03_plastic_config.py        # 성형외과 도메인 설정
│   ├── 04_naver_to_jsonl.py        # 네이버 블로그 → JSONL
│   ├── 05_homepage_to_jsonl.py     # 병원 홈페이지 크롤링 → JSONL
│   └── 06_banobagi_config.py       # 바노바기 전용 설정
├── maniagent/
│   ├── 05_upload_to_milvus.py      # JSONL → bge-m3 임베딩 → Zilliz 적재
│   ├── 06_rag_engine.py            # ManiAgentRAG + LLM 라우터
│   ├── 07_api_server.py            # FastAPI (/v1/chat, /health, /metrics, SSE)
│   └── 08_banobagi_agent.py        # 바노바기 특화 에이전트
├── maniverse/
│   ├── api_client.js               # JavaScript API 클라이언트 (프론트엔드용)
│   └── README.md                   # Phase 2 프론트엔드 계획
├── scripts/
│   ├── test_endpoints.py           # API 엔드포인트 통합 테스트
│   ├── upload_lite.py              # Milvus Lite 로컬 업로드
│   └── run_full_pipeline.sh        # 전체 파이프라인 원스텝 실행
├── tests/
│   ├── test_rag_engine.py          # ManiAgentRAG 단위 테스트
│   └── test_api_server.py          # FastAPI 통합 테스트
├── Dockerfile
├── railway.toml
├── requirements.txt
├── .env.example                    # 환경 변수 템플릿
└── README.md
```

---

## ⚡ 빠른 시작

### 1. 환경 준비

```bash
git clone https://github.com/nalgea/maniquant.git
cd maniquant

python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt

cp .env.example .env
# .env에 Zilliz 토큰 및 LLM API 키 입력
```

### 2. API 서버 로컬 실행

```bash
python -m uvicorn maniagent.07_api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 엔드포인트 테스트

```bash
# 로컬 서버 테스트
python scripts/test_endpoints.py --base-url http://localhost:8000

# Railway 배포 서버 테스트
python scripts/test_endpoints.py --base-url https://maniquant-production.up.railway.app

# JSON 리포트 저장
python scripts/test_endpoints.py \
  --base-url https://maniquant-production.up.railway.app \
  --output test_report.json
```

---

## 🔌 API 명세

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET`  | `/health` | 서버 상태 확인 |
| `GET`  | `/metrics` | 요청 수·오류율·응답시간 모니터링 |
| `POST` | `/v1/chat` | RAG 질의응답 (locale + domain 지원) |
| `POST` | `/v1/chat/stream` | SSE 스트리밍 응답 |
| `GET`  | `/v1/domains` | 지원 도메인·카테고리 목록 |
| `GET`  | `/v1/stats` | Zilliz 컬렉션 통계 |

### POST `/v1/chat` 요청 예시

```json
{
  "query":           "바노바기 코성형 가격은?",
  "locale":          "zh",
  "domain":          "plastic",
  "top_k":           5,
  "score_threshold": 0.5,
  "filter_expr":     ""
}
```

### 배포 서버 curl 테스트

```bash
# 헬스체크
curl https://maniquant-production.up.railway.app/health

# 한국어 질문 → 중국어 응답 (항노화)
curl -X POST https://maniquant-production.up.railway.app/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "콜라겐이 피부에 좋은 이유는?", "locale": "zh", "domain": "anti_aging"}'

# 성형외과 질문 → 일본어 응답
curl -X POST https://maniquant-production.up.railway.app/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "바노바기 성형외과 가격 알려줘", "locale": "ja", "domain": "plastic"}'

# 서버 메트릭 확인
curl https://maniquant-production.up.railway.app/metrics
```

---

## 🖥️ 프론트엔드 연동 (JavaScript)

```js
import { ManiAgentClient } from './maniverse/api_client.js';

const client = new ManiAgentClient();

// 일반 채팅
const res = await client.chat({
  query:  '바노바기 코성형 가격은?',
  locale: 'zh',
  domain: 'plastic',
});
console.log(res.answer);

// 스트리밍 채팅
await client.chatStream(
  { query: '콜라겐 효과', locale: 'ko' },
  {
    onDelta: (text) => process.stdout.write(text),
    onDone:  (meta) => console.log('\n완료:', meta.llm_provider),
  }
);
```

---

## 🛠 기술 스택

| 계층 | 기술 | 역할 |
|------|------|------|
| PDF 파싱 | PyMuPDF (fitz) | 텍스트·메타데이터 추출 |
| 청킹 | LangChain `RecursiveCharacterTextSplitter` | 의미 단위 분할 |
| 임베딩 | `BAAI/bge-m3` (768d) | 한/중/일/영 다국어 벡터 |
| 벡터 DB | **Zilliz Cloud** (IVF_FLAT / IP) | 관리형 Milvus 클라우드 |
| LLM 주력 | DeepSeek-V4 | RAG 응답 생성 (가성비 최적) |
| LLM 보조 | Qwen-max | DeepSeek 폴백 |
| LLM 폴백 | Claude Sonnet | 최종 폴백 |
| API 서버 | FastAPI + uvicorn | REST + SSE 스트리밍 |
| 배포 | **Railway** | 자동 GitHub 연동 CD |
| 모니터링 | `/metrics` 엔드포인트 | 요청 수·오류율·응답시간 |

---

## 🧬 지원 도메인

### 항노화 (Anti-Aging) — `anti_aging`
세포 노화·텔로미어·에피제네틱스·단백질 항상성·미토콘드리아·
mTOR 경로·시르투인·인플라메이징·심혈관 노화·신경퇴행성 질환·
근감소증·대사 증후군·피부 노화·약물 개입·식이 제한·유전자 치료·
마이크로바이옴·운동·바이오마커·오믹스

### 성형외과 (Plastic Surgery) — `plastic`
바노바기 성형외과·시술 카테고리·시술 부위·회복·부작용·결과 평가·의료 환경 (한/중/일)

---

## 🧪 테스트

```bash
# 단위 + 통합 테스트
pytest tests/ -v

# 커버리지 측정
pytest tests/ --cov=manidata --cov=maniagent --cov-report=html

# 배포 서버 엔드포인트 테스트
python scripts/test_endpoints.py \
  --base-url https://maniquant-production.up.railway.app \
  --output report.json
```

---

## ⚙️ 환경 변수

`.env.example`을 복사하여 `.env`를 생성하세요.

| 변수 | 설명 | 예시 |
|------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek LLM 키 | `sk-xxx` |
| `MILVUS_HOST` | Zilliz Cloud Endpoint | `in03-xxx.serverless.aws-eu-central-1.cloud.zilliz.com` |
| `MILVUS_PORT` | Zilliz 포트 | `443` |
| `MILVUS_TOKEN` | Zilliz API 토큰 | `xxxxxxxx` |
| `MILVUS_COLLECTION` | 기본 컬렉션 | `manidata_anti_aging` |
| `RAG_TOP_K` | 검색 청크 수 | `5` |
| `RAG_SCORE_THRESHOLD` | 유사도 임계값 | `0.5` |
| `RAILWAY_URL` | Railway 배포 URL | `https://maniquant-production.up.railway.app` |

---

## 🗺 로드맵

| 단계 | 목표 | 상태 |
|------|------|------|
| **Phase 0** | 데이터 파이프라인 구축 | ✅ 완료 |
| **Phase 1** | ManiAgent RAG + Railway + Zilliz 배포 | ✅ 완료 |
| **Phase 2** | ManiVerse 프론트엔드 MVP | 🔄 진행 중 |
| **Phase 3** | 병원 파일럿 연동 (강남 성형외과 5개사) | 📅 예정 |
| **Phase 4** | 글로벌 확장 (음성·이미지·의료관광) | 📅 예정 |

---

## 📬 연락처

| 역할 | 연락처 |
|------|--------|
| CEO/PM | nalgea@gmail.com |
| API 문의 | https://maniquant-production.up.railway.app/docs |
