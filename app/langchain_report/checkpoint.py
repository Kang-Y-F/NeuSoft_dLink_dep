# app/langchain_report/checkpoint.py  （新文件）
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

from app.core.config import settings

_pool: ConnectionPool | None = None
_checkpointer: PostgresSaver | None = None


def init_checkpointer() -> PostgresSaver:
    """FastAPI启动时调用一次。建议单独开一个PG库（比如 langgraph_checkpoint），
    跟你 CTvector 那个 pgvector 库分开，checkpointer会自动建 checkpoints /
    checkpoint_blobs / checkpoint_writes / checkpoint_migrations 这几张表。"""
    global _pool, _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    _pool = ConnectionPool(
        conninfo=settings.PG_CHECKPOINT_URI,
        max_size=10,
        kwargs={"autocommit": True, "row_factory": dict_row},  # PostgresSaver硬性要求这两个参数
    )
    _checkpointer = PostgresSaver(_pool)
    _checkpointer.setup()   # 只需要成功建表一次，重复调用是幂等的，但生产环境建议加个判断避免每次启动都跑DDL
    return _checkpointer


def get_checkpointer() -> PostgresSaver:
    if _checkpointer is None:
        raise RuntimeError("checkpointer 未初始化，请先在应用启动时调用 init_checkpointer()")
    return _checkpointer