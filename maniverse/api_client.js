/**
 * maniverse/api_client.js
 * ManiAgent API JavaScript 클라이언트
 *
 * Railway 배포 서버: https://maniquant-production.up.railway.app
 *
 * 사용법 (브라우저 / Node.js):
 *   import { ManiAgentClient } from './api_client.js';
 *   const client = new ManiAgentClient();
 *   const res = await client.chat({ query: '바노바기 코성형 가격', locale: 'zh' });
 */

// ── 기본 설정 ──────────────────────────────────────────────────────────────

const DEFAULT_BASE_URL =
  typeof process !== 'undefined' && process.env?.VITE_API_URL
    ? process.env.VITE_API_URL
    : 'https://maniquant-production.up.railway.app';

// ── 타입 정의 (JSDoc) ──────────────────────────────────────────────────────

/**
 * @typedef {'ko'|'zh'|'ja'|'en'} Locale
 * @typedef {'anti_aging'|'plastic'} Domain
 *
 * @typedef {Object} ChatRequest
 * @property {string}  query            - 자연어 질문
 * @property {Locale}  [locale='ko']    - 응답 언어
 * @property {Domain}  [domain='anti_aging'] - 도메인
 * @property {number}  [top_k=5]        - 검색 청크 수 (1~20)
 * @property {number}  [score_threshold=0.5] - 유사도 임계값
 * @property {string}  [filter_expr=''] - Milvus 필터 표현식
 *
 * @typedef {Object} SourceInfo
 * @property {string} source_file
 * @property {number} page_number
 * @property {number} score
 * @property {string} domain
 * @property {string} category
 * @property {number} year
 * @property {string} author
 *
 * @typedef {Object} ChatResponse
 * @property {string}       query
 * @property {Locale}       locale
 * @property {string}       answer
 * @property {SourceInfo[]} sources
 * @property {string}       llm_provider
 * @property {string}       model
 * @property {Object}       token_usage
 * @property {number}       elapsed_ms
 */


// ── ManiAgentClient 클래스 ─────────────────────────────────────────────────

export class ManiAgentClient {
  /**
   * @param {Object} [options]
   * @param {string} [options.baseUrl] - API 서버 베이스 URL
   * @param {number} [options.timeout] - fetch 타임아웃 (ms, 기본 60000)
   */
  constructor({ baseUrl = DEFAULT_BASE_URL, timeout = 60_000 } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.timeout = timeout;
  }

  // ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

  async _fetch(path, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);
    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          'Accept':       'application/json',
          ...options.headers,
        },
      });
      clearTimeout(timer);
      if (!res.ok) {
        const errBody = await res.text();
        throw new ManiAgentError(res.status, errBody);
      }
      return res;
    } catch (err) {
      clearTimeout(timer);
      if (err.name === 'AbortError') throw new ManiAgentError(408, 'Request timeout');
      throw err;
    }
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * 서버 상태 확인
   * @returns {Promise<{status: string, version: string, domains: string[]}>}
   */
  async health() {
    const res = await this._fetch('/health');
    return res.json();
  }

  /**
   * RAG 기반 다국어 질의응답
   * @param {ChatRequest} request
   * @returns {Promise<ChatResponse>}
   *
   * @example
   * const res = await client.chat({
   *   query: '콜라겐이 피부에 좋은 이유는?',
   *   locale: 'zh',
   *   domain: 'anti_aging',
   * });
   * console.log(res.answer);
   */
  async chat(request) {
    const body = {
      locale:          'ko',
      domain:          'anti_aging',
      top_k:           5,
      score_threshold: 0.5,
      filter_expr:     '',
      ...request,
    };
    const res = await this._fetch('/v1/chat', {
      method: 'POST',
      body:   JSON.stringify(body),
    });
    return res.json();
  }

  /**
   * SSE 스트리밍 채팅 — 청크 단위로 onDelta 콜백을 호출합니다.
   * @param {ChatRequest} request
   * @param {Object}      callbacks
   * @param {function(string): void} callbacks.onDelta   - 텍스트 청크 수신 콜백
   * @param {function(Object): void} callbacks.onDone    - 완료 메타데이터 콜백
   * @param {function(Error):  void} [callbacks.onError] - 오류 콜백
   *
   * @example
   * await client.chatStream(
   *   { query: '바노바기 성형외과', locale: 'ko', domain: 'plastic' },
   *   {
   *     onDelta: (text) => process.stdout.write(text),
   *     onDone:  (meta) => console.log('\n완료:', meta),
   *   }
   * );
   */
  async chatStream(request, { onDelta, onDone, onError } = {}) {
    const body = {
      locale:          'ko',
      domain:          'anti_aging',
      top_k:           5,
      score_threshold: 0.5,
      filter_expr:     '',
      ...request,
    };

    let res;
    try {
      res = await this._fetch('/v1/chat/stream', {
        method:  'POST',
        body:    JSON.stringify(body),
        headers: { Accept: 'text/event-stream' },
      });
    } catch (err) {
      if (onError) onError(err);
      return;
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let   buffer  = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const rawJson = line.slice(5).trim();
          if (!rawJson) continue;
          try {
            const data = JSON.parse(rawJson);
            if (data.event === 'done') {
              onDone?.(data);
            } else if (data.delta !== undefined) {
              onDelta?.(data.delta);
            } else if (data.error) {
              onError?.(new ManiAgentError(500, data.error));
            }
          } catch {
            // JSON 파싱 실패 — 무시
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * 지원 도메인 및 카테고리 목록
   * @returns {Promise<{anti_aging: Object, plastic: Object}>}
   */
  async getDomains() {
    const res = await this._fetch('/v1/domains');
    return res.json();
  }

  /**
   * Milvus 컬렉션 통계
   * @param {'anti_aging'|'plastic'} [domain='anti_aging']
   * @returns {Promise<{domain: string, collection: string, num_entities: number}>}
   */
  async getStats(domain = 'anti_aging') {
    const res = await this._fetch(`/v1/stats?domain=${domain}`);
    return res.json();
  }

  /**
   * 서버 운영 메트릭 (요청 수, 오류율, 엔드포인트별 응답시간)
   * @returns {Promise<Object>}
   */
  async getMetrics() {
    const res = await this._fetch('/metrics');
    return res.json();
  }
}


// ── 에러 클래스 ───────────────────────────────────────────────────────────

export class ManiAgentError extends Error {
  /**
   * @param {number} status  - HTTP 상태 코드
   * @param {string} message - 오류 메시지
   */
  constructor(status, message) {
    super(message);
    this.name   = 'ManiAgentError';
    this.status = status;
  }
}


// ── 기본 내보내기 ─────────────────────────────────────────────────────────

export default new ManiAgentClient();


// ── 사용 예시 (Node.js에서 직접 실행 시) ─────────────────────────────────

if (typeof process !== 'undefined' && process.argv[1]?.endsWith('api_client.js')) {
  const client = new ManiAgentClient({
    baseUrl: process.env.RAILWAY_URL || DEFAULT_BASE_URL,
  });

  console.log('🔍 ManiAgent API 클라이언트 데모\n');

  // 1. 헬스체크
  client.health()
    .then(h => console.log('✅ Health:', h))
    .then(() => client.chat({
      query:  '콜라겐이 피부에 좋은 이유는?',
      locale: 'zh',
      domain: 'anti_aging',
    }))
    .then(res => {
      console.log('\n💬 Chat (zh):');
      console.log('  Answer:', res.answer.slice(0, 100), '...');
      console.log('  LLM:   ', res.llm_provider, '/', res.model);
      console.log('  Time:  ', res.elapsed_ms, 'ms');
    })
    .catch(err => console.error('❌ Error:', err.message));
}
