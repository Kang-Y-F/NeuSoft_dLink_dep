"""
AI问诊记忆反思存储：PostgreSQL + pgvector

复用 ct_vector_db 已有的 pgvector 扩展，新增 chat_memory_reflection 表：
存储"被医生正向反馈验证过的诊疗经验"，用 embedding 做语义相似度判断，
替代掉 Java 那边"关键词重叠"的粗糙实现——这一步换成真正的语义融合。
"""

from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
import requests
from psycopg2 import Error as PgError

from app.db.pgvector_store import get_pg_conn

import requests

SIMILARITY_MERGE_THRESHOLD = 0.85   # 相似度超过这个值认为是"同一条经验"，做融合而不是新增
MAX_REFLECTIONS_PER_PATIENT = 8


DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_EMBEDDING_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"


def get_embedding(text: str) -> list:
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("环境变量 DASHSCOPE_API_KEY 未设置，请检查 .env 是否被正确加载")

    resp = requests.post(
        DASHSCOPE_EMBEDDING_URL,
        headers={
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "text-embedding-v2",
            "input": {"texts": [text]},
        },
        timeout=10,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Embedding接口调用失败: {resp.status_code} {resp.text}")

    data = resp.json()
    return data["output"]["embeddings"][0]["embedding"]


def init_reflection_table():
    conn = get_pg_conn()
    if not conn:
        print("[Reflection] 初始化跳过：无法连接数据库")
        return
    try:
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS chat_memory_reflection (
                id           BIGSERIAL PRIMARY KEY,
                patient_id   BIGINT NOT NULL,
                summary      TEXT NOT NULL,
                embedding    vector(1536) NOT NULL,
                reward_score DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                hit_count    INT NOT NULL DEFAULT 1,
                update_time  TIMESTAMP NOT NULL DEFAULT NOW()
            );
        ''')
        cur.execute('''
            CREATE INDEX IF NOT EXISTS chat_memory_reflection_hnsw_idx
            ON chat_memory_reflection USING hnsw (embedding vector_cosine_ops);
        ''')
        cur.execute('''
            CREATE INDEX IF NOT EXISTS chat_memory_reflection_patient_idx
            ON chat_memory_reflection (patient_id);
        ''')
        conn.commit()
        print("[Reflection] 初始化完成")
    except PgError as e:
        print(f"[Reflection] 初始化失败：{e}")
    finally:
        cur.close()
        conn.close()


def upsert_reflection(patient_id: int, summary: str, reward: float = 1.0) -> dict:
    """生成/融合一条反思经验：语义相似就加权融合，否则新增，超上限就淘汰权重最低的一条"""
    conn = get_pg_conn()
    if not conn:
        return {"action": "failed", "reason": "无法连接数据库"}
    try:
        vec = np.asarray(get_embedding(summary), dtype=np.float32)
        cur = conn.cursor()

        cur.execute(
            """
            SELECT id, reward_score, hit_count, 1 - (embedding <=> %s) AS similarity
            FROM chat_memory_reflection
            WHERE patient_id = %s
            ORDER BY embedding <=> %s LIMIT 1
            """,
            (vec, patient_id, vec)
        )
        row = cur.fetchone()

        if row and row[3] >= SIMILARITY_MERGE_THRESHOLD:
            rid, old_score, hit_count, similarity = row
            new_score = (old_score * hit_count + reward) / (hit_count + 1)
            cur.execute(
                "UPDATE chat_memory_reflection SET reward_score=%s, hit_count=hit_count+1, update_time=NOW() WHERE id=%s",
                (new_score, rid)
            )
            conn.commit()
            return {"action": "merged", "id": rid, "similarity": round(float(similarity), 4)}

        cur.execute(
            """
            INSERT INTO chat_memory_reflection (patient_id, summary, embedding, reward_score, hit_count, update_time)
            VALUES (%s, %s, %s, %s, 1, NOW()) RETURNING id
            """,
            (patient_id, summary, vec, reward)
        )
        new_id = cur.fetchone()[0]
        conn.commit()

        cur.execute("SELECT COUNT(*) FROM chat_memory_reflection WHERE patient_id=%s", (patient_id,))
        if cur.fetchone()[0] > MAX_REFLECTIONS_PER_PATIENT:
            cur.execute(
                """
                DELETE FROM chat_memory_reflection WHERE id = (
                    SELECT id FROM chat_memory_reflection WHERE patient_id=%s
                    ORDER BY reward_score ASC LIMIT 1
                )
                """,
                (patient_id,)
            )
            conn.commit()
            return {"action": "created_and_pruned", "id": new_id}

        return {"action": "created", "id": new_id}
    except PgError as e:
        print(f"[Reflection] upsert_reflection 失败：{e}")
        return {"action": "failed", "reason": str(e)}
    finally:
        cur.close()
        conn.close()


def query_top_reflections(patient_id: int, top_n: int = 3) -> list:
    conn = get_pg_conn()
    if not conn:
        return []
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT summary, reward_score FROM chat_memory_reflection
            WHERE patient_id=%s ORDER BY reward_score DESC LIMIT %s
            """,
            (patient_id, top_n)
        )
        return [{"summary": r[0], "rewardScore": round(float(r[1]), 3)} for r in cur.fetchall()]
    except PgError as e:
        print(f"[Reflection] query_top_reflections 失败：{e}")
        return []
    finally:
        cur.close()
        conn.close()