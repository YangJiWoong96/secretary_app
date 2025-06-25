from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import firestore

# --- 1. 환경 변수 로드 및 Firestore 클라이언트 초기화 ---
load_dotenv()
db = firestore.Client()
print("Firestore 클라이언트 초기화 성공!")

# --- 2. FastAPI 앱 생성 ---
app = FastAPI()


# --- 3. Flutter에서 보낼 데이터의 형태를 미리 정의 (Pydantic 모델) ---
class LocationData(BaseModel):
    latitude: float
    longitude: float
    timestamp: datetime


class CalendarEvent(BaseModel):
    title: str
    startTime: Optional[str] = None  # Flutter에서 null일 수 있으므로 Optional
    endTime: Optional[str] = None
    location: Optional[str] = None


class ContextData(BaseModel):
    location: LocationData
    upcomingEvents: List[CalendarEvent]


class UserContextPayload(BaseModel):
    userId: str
    recordTime: datetime
    contextData: ContextData


# --- 4. Flutter 앱의 데이터를 받을 API 엔드포인트 생성 ---
@app.post("/api/v1/context")
async def receive_user_context(payload: UserContextPayload):
    """
    Flutter 앱으로부터 사용자의 위치 및 캘린더 데이터를 수신하여,
    Firestore에 저장하는 API입니다.
    """
    try:
        print(f"✅ [{payload.userId}]님으로부터 데이터 수신 성공!")

        # Pydantic 모델을 dict로 변환
        data_to_store = payload.dict()

        # Firestore에 데이터 저장
        doc_ref = (
            db.collection("users")
            .document(payload.userId)
            .collection("context_events")
            .document()  # ID 자동 생성
        )

        # 수정: await 제거
        doc_ref.set(data_to_store)

        print(f"✅ Firestore에 데이터 저장 완료 (ID: {doc_ref.id})")

        return {
            "status": "success",
            "message": "Data successfully stored in Firestore.",
        }

    except Exception as e:
        print(f"❌ 데이터 처리 또는 저장 중 에러 발생: {e}")
        return {"status": "error", "message": str(e)}
