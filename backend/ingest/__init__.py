# ingest/__init__.py
from .main import app as ingest_app

__all__ = ["ingest_app"]

# from ingest import ingest_app 로 활용

# 추후 각 수집 기능별 라우터 묶어버릴려면? 아래와 같이

# ingest/__init__.py
# from fastapi import APIRouter
# from .location import router as location_router
# from .calendar import router as calendar_router
# from .preferences import router as preferences_router

# router = APIRouter()
# router.include_router(location_router, prefix="/location")
# router.include_router(calendar_router, prefix="/calendar")
# router.include_router(preferences_router, prefix="/preferences")
