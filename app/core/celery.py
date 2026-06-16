# app/core/celery.py
# ============================================================
# Celery 实例配置
# broker  : Redis（消息队列）
# backend : Redis（短期任务状态缓存，持久结果在 MySQL）
# ============================================================

from celery import Celery
from app.core.config import settings

celery = Celery(
    "ct_task",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.infer_tasks"],   # ← 修正：指向实际任务模块
)

celery.conf.update(
    task_serializer   = "json",
    accept_content    = ["json"],
    result_serializer = "json",
    timezone          = "Asia/Shanghai",
    enable_utc        = False,
    # 任务结果在 Redis 中保留 1 小时（持久数据已写 MySQL，无需长期缓存）
    result_expires    = 3600,
    # Windows 兼容
    worker_pool       = "solo",
)
