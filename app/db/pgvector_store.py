# app/db/pgvector_store.py
"""
CT 特征向量存储：PostgreSQL + pgvector

设计原则：
- 只负责 ct_patient_features 这一张表（患者CT特征向量）
- 其余所有业务数据（患者基本信息、检查单、报告等）继续留在 MySQL，
  两边通过 patient_id 关联（应用层保证一致，不建跨库物理外键）
- 相似度检索、聚类分析读特征时，统一从这里读，不再依赖 mysql.py 里的
  feature_upsert / feature_load_all 等函数
"""
import numpy as np
import psycopg2
from psycopg2 import Error as PgError
from pgvector.psycopg2 import register_vector
from datetime import datetime

from app.core.config import settings


def get_pg_conn():
    try:
        conn = psycopg2.connect(
            host=settings.PGVECTOR_HOST,
            port=settings.PGVECTOR_PORT,
            user=settings.PGVECTOR_USER,
            password=settings.PGVECTOR_PASSWORD,
            dbname=settings.PGVECTOR_DB,
        )
        register_vector(conn)  # 让 psycopg2 能直接收发 numpy 数组 <-> vector 类型
        return conn
    except PgError as e:
        print(f"[PGVector] 连接失败：{e}")
        return None


def init_pgvector_db():
    """
    启动时调用：确保扩展和表存在。
    注意：pgvector 的 vector(N) 维度写死在建表语句里，
    如果你的模型特征维度不是 512，请把下面两处 512 一起改掉，
    或者手动建表后把这里的 CREATE TABLE 语句去掉，避免维度不一致报错。
    """
    conn = get_pg_conn()
    if not conn:
        print("[PGVector] 初始化跳过：无法连接数据库")
        return
    try:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute('''
            CREATE TABLE IF NOT EXISTS ct_patient_features (
                patient_id  BIGINT PRIMARY KEY,
                feature     vector(960) NOT NULL,
                model_type  VARCHAR(32) NOT NULL,
                updated_at  TIMESTAMP NOT NULL DEFAULT NOW()
            );
        ''')
        # HNSW 索引：近似最近邻，余弦距离
        cur.execute('''
            CREATE INDEX IF NOT EXISTS ct_patient_features_hnsw_idx
            ON ct_patient_features
            USING hnsw (feature vector_cosine_ops);
        ''')
        conn.commit()
        print("[PGVector] 初始化完成（扩展 + 表 + HNSW索引）")
    except PgError as e:
        print(f"[PGVector] 初始化失败：{e}")
    finally:
        cur.close()
        conn.close()


def feature_upsert_vec(patient_id: int, feature_vector: np.ndarray, model_type: str) -> bool:
    conn = get_pg_conn()
    if not conn:
        return False
    try:
        vec = np.asarray(feature_vector).squeeze().astype(np.float32)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ct_patient_features (patient_id, feature, model_type, updated_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (patient_id)
            DO UPDATE SET feature = EXCLUDED.feature,
                          model_type = EXCLUDED.model_type,
                          updated_at = NOW()
            """,
            (patient_id, vec, model_type)
        )
        conn.commit()
        return True
    except PgError as e:
        print(f"[PGVector] feature_upsert_vec 失败：{e}")
        return False
    finally:
        cur.close()
        conn.close()


def find_similar_vec(patient_id: int, top_k: int = 5):
    """
    返回 None 表示目标患者本身无特征数据（由上层接口转成404）。
    返回 [] 表示有目标特征，但库里没有其他可比对的患者。
    走 HNSW 索引的近似最近邻检索，不是暴力遍历。
    """
    conn = get_pg_conn()
    if not conn:
        return []
    try:
        cur = conn.cursor()
        cur.execute("SELECT feature FROM ct_patient_features WHERE patient_id = %s", (patient_id,))
        row = cur.fetchone()
        if row is None:
            return None

        target_vec = row[0]

        # <=> 是 pgvector 提供的余弦距离运算符，值域 [0, 2]，0表示完全相同
        # 相似度 = 1 - 余弦距离，跟原来 numpy 版本的语义保持一致
        cur.execute(
            """
            SELECT patient_id, 1 - (feature <=> %s) AS similarity
            FROM ct_patient_features
            WHERE patient_id != %s
            ORDER BY feature <=> %s
            LIMIT %s
            """,
            (target_vec, patient_id, target_vec, top_k)
        )
        rows = cur.fetchall()
        return [
            {
                "patientId": r[0],
                "similarity": round(float(r[1]), 4),
                "distance": round(1 - float(r[1]), 4),
            }
            for r in rows
        ]
    except PgError as e:
        print(f"[PGVector] find_similar_vec 失败：{e}")
        return []
    finally:
        cur.close()
        conn.close()


def feature_delete_vec(patient_id: int) -> bool:
    conn = get_pg_conn()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM ct_patient_features WHERE patient_id = %s", (patient_id,))
        conn.commit()
        return cur.rowcount > 0
    except PgError as e:
        print(f"[PGVector] feature_delete_vec 失败：{e}")
        return False
    finally:
        cur.close()
        conn.close()


def feature_list_patients_vec() -> list:
    """对应原来 mysql.py 里的 feature_list_patients，供 /patients/list 使用"""
    conn = get_pg_conn()
    if not conn:
        return []
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT patient_id, model_type, updated_at FROM ct_patient_features ORDER BY updated_at DESC"
        )
        rows = cur.fetchall()
        result = []
        for pid, model_type, updated_at in rows:
            result.append({
                "patient_id": pid,
                "model_type": model_type,
                "updated_at": updated_at.strftime("%Y-%m-%d %H:%M:%S")
                              if isinstance(updated_at, datetime) else updated_at,
            })
        return result
    except PgError as e:
        print(f"[PGVector] feature_list_patients_vec 失败：{e}")
        return []
    finally:
        cur.close()
        conn.close()


def feature_load_all_vec() -> dict:
    """
    对应原来 mysql.py 里的 feature_load_all，供 /cluster/analysis 做 t-SNE + KMeans 使用。
    注意：聚类分析本身没有走向量索引（需要全量特征做降维），这里仍是全表读取，
    这是合理的——索引解决的是"相似度检索"这一单点查询问题，聚类分析天然需要全量数据。
    """
    conn = get_pg_conn()
    if not conn:
        return {}
    try:
        cur = conn.cursor()
        cur.execute("SELECT patient_id, feature FROM ct_patient_features")
        rows = cur.fetchall()
        return {pid: np.asarray(feat, dtype=np.float32) for pid, feat in rows}
    except PgError as e:
        print(f"[PGVector] feature_load_all_vec 失败：{e}")
        return {}
    finally:
        cur.close()
        conn.close()