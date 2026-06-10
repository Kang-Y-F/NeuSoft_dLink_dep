from celery import Celery
import os

# 读取环境配置
BROKER_URL = "redis://localhost:6379/0"
RESULT_BACKEND = "redis://localhost:6379/0"

celery = Celery(
    "ct_task",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["app.api.endpoints"]
)

# Windows 专用配置
celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=False,
)