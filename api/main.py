# main.py
from fastapi import FastAPI

from .endpoints import feedback, health, logs, predict, tags  # 상대경로 import

app = FastAPI()

# 엔드포인트(라우터) 등록
app.include_router(predict.router)
app.include_router(health.router)
app.include_router(tags.router)
app.include_router(feedback.router)
app.include_router(logs.router)
