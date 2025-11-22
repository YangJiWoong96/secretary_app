import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google.cloud import firestore
from pydantic import BaseModel, Field

# --- 1. 환경 변수 로드 및 Firestore 클라이언트 초기화 ---
load_dotenv()
try:
    db = firestore.Client()
    print("✅ Firestore 클라이언트 초기화 성공!")
except Exception as e:
    print(f"❌ Firestore 클라이언트 초기화 실패: {e}")
    db = None

# --- 2. FastAPI 앱 생성 ---
app = FastAPI(
    title="Timely Agent Backend",
    description="사용자 컨텍스트(위치, 캘린더, 선호도)를 수집하고 통합된 포맷으로 저장하는 API",
    version="1.0.0",
)


# --- 2-1. MeCab 동작 확인용 라우트 ---
@app.get("/health/mecab")
def health_mecab():
    """MeCab 사전/바인딩이 정상 동작하는지 간단 확인"""
    try:
        import MeCab

        parsed = MeCab.Tagger("").parse("삼성전자가 서울에 있다")
        return {"ok": True, "sample": parsed}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# --- 3. Flutter에서 보낼 데이터의 형태 정의 (Pydantic 모델) ---


# 3-1. 위치 데이터 수신 모델
class LocationInfo(BaseModel):
    latitude: float
    longitude: float
    timestamp: datetime


class LocationReceivePayload(BaseModel):
    userId: str
    recordTime: datetime
    location: LocationInfo


# 3-2. 캘린더 데이터 수신 모델
class CalendarEventInfo(BaseModel):
    title: Optional[str] = ""
    startTime: Optional[str] = None
    endTime: Optional[str] = None
    location: Optional[str] = ""


class CalendarReceivePayload(BaseModel):
    userId: str
    recordTime: datetime
    upcomingEvents: List[CalendarEventInfo]


# # 3-3. 선호도 데이터 수신 모델
# class PreferencesInfo(BaseModel):
#     food: str
#     hobby: str
#     interest: str


class PreferenceReceivePayload(BaseModel):
    userId: str
    updateTime: datetime
    preferences: Dict[str, str]  # 어떤 키와 값이든 받을 수 있는 딕셔너리 형태로 변경


# --- 4. 카카오 역지오코딩 호출 함수 ---
async def reverse_geocode(lat: float, lng: float) -> Optional[str]:
    """위도, 경도를 받아 카카오 API를 통해 주소 문자열로 변환합니다."""
    kakao_api_key = os.getenv("KAKAO_REST_API_KEY")
    if not kakao_api_key:
        print("⚠️ KAKAO_REST_API_KEY가 설정되지 않았습니다. 역지오코딩을 건너뜁니다.")
        return None

    endpoint = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    url = f"{endpoint}?x={lng}&y={lat}"  # 주의: Kakao는 x=경도, y=위도
    headers = {"Authorization": f"KakaoAK {kakao_api_key}"}

    try:
        from backend.utils.http_client import get_async_client

        client = get_async_client()
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        result = resp.json()

        documents = result.get("documents", [])
        if documents:
            address_info = documents[0].get("road_address") or documents[0].get(
                "address"
            )
            return address_info.get("address_name") if address_info else None
        return None
    except httpx.HTTPStatusError as e:
        print(f"❌ 카카오 API 호출 실패: {e.response.status_code}, {e.response.text}")
        return None
    except Exception as e:
        print(f"❌ 역지오코딩 중 알 수 없는 에러: {e}")
        return None


# --- 5. 데이터 저장 함수 ---
def save_to_firestore(user_id: str, data: Dict[str, Any]):
    """통합된 데이터를 Firestore에 저장합니다."""
    if not db:
        print("❌ Firestore 클라이언트가 없어 저장할 수 없습니다.")
        raise ConnectionError("Firestore client is not initialized.")

    # users/{userId}/unified_events/{자동생성ID} 경로에 저장
    doc_ref = (
        db.collection("users").document(user_id).collection("unified_events").document()
    )
    doc_ref.set(data)
    print(
        f"✅ Firestore 저장 완료 (dataType: {data.get('dataType')}, docId: {doc_ref.id})"
    )


# --- 6. 통합 API 엔드포인트 ---


@app.post("/api/v1/location", tags=["Unified Events"])
async def receive_location(payload: LocationReceivePayload):
    """(통합) 사용자의 위치 정보를 받아 주소 변환 후 Firestore에 저장"""
    try:
        # ① 위경도 → 주소 변환
        address = await reverse_geocode(
            payload.location.latitude,
            payload.location.longitude,
        )

        # ② 통합 데이터 포맷으로 변환
        data_to_store = {
            "userId": payload.userId,
            "recordTimestamp": payload.recordTime,
            "dataType": "LOCATION",
            "payload": {
                "latitude": payload.location.latitude,
                "longitude": payload.location.longitude,
                "address": address,  # 변환된 주소 추가
                "eventTimestamp": payload.location.timestamp,
            },
        }

        # ③ Firestore에 저장
        save_to_firestore(payload.userId, data_to_store)
        return {"status": "success", "dataType": "LOCATION", "data": data_to_store}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 에러: {e}")


@app.post("/api/v1/calendar", tags=["Unified Events"])
async def receive_calendar_events(payload: CalendarReceivePayload):
    """(통합) 사용자의 캘린더 정보를 받아 Firestore에 저장"""
    try:
        # ① 통합 데이터 포맷으로 변환
        data_to_store = {
            "userId": payload.userId,
            "recordTimestamp": payload.recordTime,
            "dataType": "CALENDAR_UPDATE",
            "payload": {
                # Pydantic 모델 리스트를 dict 리스트로 변환
                "events": [event.model_dump() for event in payload.upcomingEvents]
            },
        }

        # ② Firestore에 저장
        save_to_firestore(payload.userId, data_to_store)
        return {
            "status": "success",
            "dataType": "CALENDAR_UPDATE",
            "data": data_to_store,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 에러: {e}")


@app.post("/api/v1/preferences", tags=["Unified Events"])
async def receive_preferences(payload: PreferenceReceivePayload):
    """(통합) 사용자의 선호도 정보를 받아 Firestore 및 Milvus Profile Collection에 저장"""
    try:
        # ① 통합 데이터 포맷으로 변환
        data_to_store = {
            "userId": payload.userId,
            "recordTimestamp": payload.updateTime,  # recordTime 대신 updateTime 사용
            "dataType": "PREFERENCE_UPDATE",
            "payload": payload.preferences,
        }

        # ② Firestore에 저장
        save_to_firestore(payload.userId, data_to_store)

        # ③ Milvus Profile Collection에 explicit로 저장
        try:
            import asyncio as _aio

            from backend.rag.profile_writer import get_profile_writer

            writer = get_profile_writer()

            tasks = []
            for key_path, value in (payload.preferences or {}).items():
                norm_key = writer._normalize_key(str(key_path))
                tasks.append(
                    writer._upsert_chunk(
                        user_id=payload.userId,
                        category="preferences",
                        key_path=str(key_path),
                        norm_key=norm_key,
                        value=value,
                        source="explicit",
                        status="active",
                        confidence=1.0,
                        evidence_turn_ids=[],
                        extras=None,
                    )
                )

            if tasks:
                await _aio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"❌ Milvus explicit 저장 실패: {e}")

        return {
            "status": "success",
            "dataType": "PREFERENCE_UPDATE",
            "data": data_to_store,
            "milvus_stored": len(payload.preferences or {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 에러: {e}")
