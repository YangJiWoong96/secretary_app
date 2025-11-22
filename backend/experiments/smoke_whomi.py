"""
WHOMI Backend 스모크 테스트 (수동/자동화 가능)

시나리오:
1) Validator 타임아웃 강제(Fail-Open 확인)
   - 환경변수: VALIDATOR_TIMEOUT_S=0.01
   - 기대: 응답은 정상, 로그에 validator.timeout 경고만

2) Web 사후 평가 타임아웃
   - 환경변수: 유지(기본), 실제 호출 시 post-verify/web evaluator는 비차단/타임박스
   - 기대: 본 응답 OK, 로그에 web_evaluator_timeout 경고만

3) 동일 질의 2회 → 임베딩 캐시 히트 및 배치 확인
   - 기대: embeddings.cache_hit > 0, embeddings.batch_size >= 2 이벤트

실행:
    python -m backend.experiments.smoke_whomi
"""

from __future__ import annotations

import asyncio
import os

import websockets


async def _ws_echo(session_id: str, utter: str) -> None:
    url = f"ws://localhost:8000/ws/{session_id}"
    async with websockets.connect(url) as ws:
        await ws.send(utter)
        # 스트림 일부만 확인
        for _ in range(3):
            msg = await ws.recv()
            if not msg:
                break


async def run_smoke():
    # 1) Validator 강제 타임아웃
    os.environ["VALIDATOR_TIMEOUT_S"] = "0.01"
    await _ws_echo("smoke-1", "안녕! 오늘 하루 어땠어?")

    # 2) Web 사후 평가(강남역 국밥 질의)
    os.environ.pop("VALIDATOR_TIMEOUT_S", None)
    await _ws_echo("smoke-2", "강남역 국밥 맛집 추천해줘")

    # 3) 동일 질의 2회
    await _ws_echo("smoke-3", "강남역 국밥 맛집 추천해줘")
    await _ws_echo("smoke-3", "강남역 국밥 맛집 추천해줘")


if __name__ == "__main__":
    asyncio.run(run_smoke())
