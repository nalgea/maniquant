#!/usr/bin/env python3
"""
scripts/test_endpoints.py
ManiAgent API 엔드포인트 통합 테스트 스크립트

로컬 또는 Railway 배포 서버를 대상으로 모든 엔드포인트를 검증합니다.

실행:
    # 로컬 서버 테스트
    python scripts/test_endpoints.py --base-url http://localhost:8000

    # Railway 배포 서버 테스트
    python scripts/test_endpoints.py --base-url https://maniquant-production.up.railway.app

    # 결과를 JSON 파일로 저장
    python scripts/test_endpoints.py --base-url https://maniquant-production.up.railway.app --output report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    print("[ERROR] requests 패키지 없음: pip install requests")
    sys.exit(1)

# ── ANSI 색상 코드 ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg: str)   -> str: return f"{GREEN}✅ PASS{RESET}  {msg}"
def fail(msg: str) -> str: return f"{RED}❌ FAIL{RESET}  {msg}"
def warn(msg: str) -> str: return f"{YELLOW}⚠️  WARN{RESET}  {msg}"
def info(msg: str) -> str: return f"{CYAN}ℹ️  INFO{RESET}  {msg}"


# ── 테스트 결과 수집 ──────────────────────────────────────────────────────────
results: list[dict] = []

def record(name: str, passed: bool, elapsed_ms: float, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    results.append({"test": name, "status": status, "elapsed_ms": round(elapsed_ms, 1), "detail": detail})
    line   = ok(f"{name}  ({elapsed_ms:.0f}ms)") if passed else fail(f"{name}  — {detail}")
    print(line)


# ── 개별 테스트 함수 ──────────────────────────────────────────────────────────

def test_health(base: str, timeout: int) -> bool:
    """GET /health — 서버 상태 확인"""
    print(f"\n{BOLD}[1/6] GET /health{RESET}")
    t0 = time.perf_counter()
    try:
        r = requests.get(f"{base}/health", timeout=timeout)
        ms = (time.perf_counter() - t0) * 1000
        passed = r.status_code == 200 and r.json().get("status") == "ok"
        detail = json.dumps(r.json(), ensure_ascii=False) if not passed else ""
        record("/health", passed, ms, detail)
        if passed:
            data = r.json()
            print(info(f"  version={data.get('version')}  domains={data.get('domains')}"))
        return passed
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        record("/health", False, ms, str(e))
        return False


def test_chat(base: str, timeout: int) -> bool:
    """POST /v1/chat — 4개 locale 각각 테스트"""
    print(f"\n{BOLD}[2/6] POST /v1/chat  (한/중/일/영){RESET}")
    all_passed = True
    cases = [
        ("ko", "anti_aging", "콜라겐이 피부에 좋은 이유는?"),
        ("zh", "anti_aging", "胶原蛋白对皮肤有什么好处？"),
        ("ja", "plastic",    "バノバギ整形外科の評判は？"),
        ("en", "plastic",    "What procedures does Banobagi clinic offer?"),
    ]
    for locale, domain, query in cases:
        payload = {"query": query, "locale": locale, "domain": domain, "top_k": 3}
        t0 = time.perf_counter()
        try:
            r   = requests.post(f"{base}/v1/chat", json=payload, timeout=timeout)
            ms  = (time.perf_counter() - t0) * 1000
            ok_ = r.status_code == 200
            if ok_:
                data   = r.json()
                answer = data.get("answer", "")[:60]
                llm    = data.get("llm_provider", "?")
                src_n  = len(data.get("sources", []))
                detail = f"llm={llm}  sources={src_n}  answer={answer!r}..."
                record(f"/v1/chat [{locale}]", True, ms, "")
                print(info(f"  {detail}"))
            else:
                record(f"/v1/chat [{locale}]", False, ms, r.text[:120])
                all_passed = False
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            record(f"/v1/chat [{locale}]", False, ms, str(e))
            all_passed = False
    return all_passed


def test_chat_validation(base: str, timeout: int) -> bool:
    """POST /v1/chat — 잘못된 입력 422 응답 확인"""
    print(f"\n{BOLD}[3/6] POST /v1/chat  (입력 검증){RESET}")
    cases = [
        ("빈 query",        {"query": "",      "locale": "ko"}, 422),
        ("잘못된 locale",   {"query": "test",  "locale": "xx"}, 422),
        ("query 너무 긺",   {"query": "a"*1001,"locale": "ko"}, 422),
    ]
    all_passed = True
    for name, payload, expected_status in cases:
        t0 = time.perf_counter()
        try:
            r  = requests.post(f"{base}/v1/chat", json=payload, timeout=timeout)
            ms = (time.perf_counter() - t0) * 1000
            passed = r.status_code == expected_status
            record(f"validation [{name}]", passed, ms,
                   f"expected={expected_status} got={r.status_code}")
            if not passed:
                all_passed = False
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            record(f"validation [{name}]", False, ms, str(e))
            all_passed = False
    return all_passed


def test_stream(base: str, timeout: int) -> bool:
    """POST /v1/chat/stream — SSE 스트리밍 확인"""
    print(f"\n{BOLD}[4/6] POST /v1/chat/stream  (SSE){RESET}")
    payload = {"query": "바노바기 코성형 후기", "locale": "ko", "domain": "plastic"}
    t0 = time.perf_counter()
    try:
        with requests.post(f"{base}/v1/chat/stream", json=payload,
                           timeout=timeout, stream=True) as r:
            ms       = (time.perf_counter() - t0) * 1000
            ct       = r.headers.get("content-type", "")
            has_sse  = "text/event-stream" in ct
            chunks   = []
            for line in r.iter_lines(decode_unicode=True):
                if line.startswith("data:"):
                    chunks.append(line)
                if len(chunks) >= 3:
                    break
            total_ms = (time.perf_counter() - t0) * 1000
            passed   = r.status_code == 200 and has_sse and len(chunks) > 0
            detail   = f"status={r.status_code}  content-type={ct}  chunks={len(chunks)}"
            record("/v1/chat/stream", passed, total_ms, "" if passed else detail)
            if passed:
                print(info(f"  SSE chunks received: {len(chunks)}"))
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        record("/v1/chat/stream", False, ms, str(e))
        return False
    return passed


def test_domains(base: str, timeout: int) -> bool:
    """GET /v1/domains — 도메인 목록 확인"""
    print(f"\n{BOLD}[5/6] GET /v1/domains{RESET}")
    t0 = time.perf_counter()
    try:
        r  = requests.get(f"{base}/v1/domains", timeout=timeout)
        ms = (time.perf_counter() - t0) * 1000
        data   = r.json()
        passed = r.status_code == 200 and "anti_aging" in data
        record("/v1/domains", passed, ms, "" if passed else r.text[:100])
        if passed:
            domains = list(data.keys())
            print(info(f"  domains={domains}"))
        return passed
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        record("/v1/domains", False, ms, str(e))
        return False


def test_metrics(base: str, timeout: int) -> bool:
    """GET /metrics — 서버 메트릭 확인"""
    print(f"\n{BOLD}[6/6] GET /metrics{RESET}")
    t0 = time.perf_counter()
    try:
        r  = requests.get(f"{base}/metrics", timeout=timeout)
        ms = (time.perf_counter() - t0) * 1000
        data   = r.json()
        passed = r.status_code == 200 and "uptime_seconds" in data
        record("/metrics", passed, ms, "" if passed else r.text[:100])
        if passed:
            print(info(f"  uptime={data.get('uptime_seconds')}s"
                       f"  total_requests={data.get('total_requests')}"
                       f"  error_rate={data.get('error_rate_pct')}%"))
        return passed
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        record("/metrics", False, ms, str(e))
        return False


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="ManiAgent 엔드포인트 통합 테스트")
    p.add_argument(
        "--base-url", "-u",
        default=os.getenv("RAILWAY_URL", "http://localhost:8000"),
        help="테스트 대상 서버 URL (기본: RAILWAY_URL 환경변수 또는 localhost:8000)",
    )
    p.add_argument("--timeout", "-t", type=int, default=60,
                   help="요청 타임아웃 (초, 기본: 60)")
    p.add_argument("--output",  "-o", default="",
                   help="결과 JSON 저장 경로 (미입력 시 저장 안 함)")
    args = p.parse_args()

    base = args.base_url.rstrip("/")
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}ManiAgent API 엔드포인트 테스트{RESET}")
    print(f"  대상 서버: {CYAN}{base}{RESET}")
    print(f"  시각:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{BOLD}{'='*60}{RESET}")

    # 서버 연결 가능 여부 먼저 확인
    try:
        requests.get(f"{base}/health", timeout=args.timeout)
    except requests.exceptions.ConnectionError:
        print(f"\n{RED}[ERROR]{RESET} 서버에 연결할 수 없습니다: {base}")
        print("  → 서버가 실행 중인지 확인하세요.")
        sys.exit(1)

    # 순서대로 테스트 실행
    test_health(base, args.timeout)
    test_chat(base, args.timeout)
    test_chat_validation(base, args.timeout)
    test_stream(base, args.timeout)
    test_domains(base, args.timeout)
    test_metrics(base, args.timeout)

    # 최종 요약
    total  = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}최종 결과{RESET}: {GREEN}{passed} 통과{RESET} / {RED}{failed} 실패{RESET} / 총 {total}개")
    avg_ms = sum(r["elapsed_ms"] for r in results) / total if total else 0
    print(f"평균 응답시간: {avg_ms:.1f}ms")
    print(f"{BOLD}{'='*60}{RESET}\n")

    # JSON 리포트 저장
    if args.output:
        report = {
            "base_url":    base,
            "timestamp":   datetime.now().isoformat(),
            "summary":     {"total": total, "passed": passed, "failed": failed},
            "results":     results,
        }
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"📄 리포트 저장: {args.output}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
