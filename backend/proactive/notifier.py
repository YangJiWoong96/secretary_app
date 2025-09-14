# backend/proactive/notifier.py
import os, json, requests, datetime as dt
from google.oauth2 import service_account
from google.auth.transport.requests import Request

PROJECT_ID = os.getenv("GCP_PROJECT_ID")  # ex) my-firebase-proj
SA_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # service-account.json


def _access_token() -> str:
    creds = service_account.Credentials.from_service_account_file(
        SA_PATH, scopes=["https://www.googleapis.com/auth/firebase.messaging"]
    )
    creds.refresh(Request())
    return creds.token


def send_push(
    token: str, title: str, body: str, data: dict | None = None, ttl_sec: int = 3600
):
    """
    token: Flutter FCM registration token
    data:  {"session_id":"abc123","question":"오늘 저녁 운동 예정이야?" ...}  # 클릭 시 새 대화창 라우팅용
    """
    url = f"https://fcm.googleapis.com/v1/projects/{PROJECT_ID}/messages:send"
    headers = {
        "Authorization": f"Bearer {_access_token()}",
        "Content-Type": "application/json",
    }
    message = {
        "message": {
            "token": token,
            "notification": {"title": title, "body": body},
            "data": {**(data or {}), "click_action": "FLUTTER_NOTIFICATION_CLICK"},
            "android": {
                "priority": "HIGH",
                "ttl": f"{ttl_sec}s",
                "notification": {"channel_id": "high_importance_channel"},
            },
            "apns": {"headers": {"apns-priority": "10"}},
        }
    }
    resp = requests.post(url, headers=headers, data=json.dumps(message), timeout=5)
    resp.raise_for_status()
    return resp.json()
