"""
独立迁移脚本：直接从 MySQL 读出 ct_patient_features 的数据，
反序列化后写入 PostgreSQL + pgvector。

不依赖项目里的 app.db.mysql / app.db.pgvector_store，
所有连接参数直接写在这个文件顶部，方便你改完直接跑，也方便照着理解每一步。

用法：
    1. 把下面 MYSQL_* 和 PG_* 几个变量改成你自己的实际配置
    2. 确认 VECTOR_DIM 是你模型真实的特征维度
    3. pip install mysql-connector-python psycopg2-binary pgvector numpy
    4. python migrate_direct.py
"""
import pickle
import numpy as np
import mysql.connector
import psycopg2
from pgvector.psycopg2 import register_vector

# ── 1. 连接参数：改成你自己的 ─────────────────────────────────
MYSQL_HOST     = "127.0.0.1"
MYSQL_PORT     = 3306
MYSQL_USER     = "root"
MYSQL_PASSWORD = "kyf050313"
MYSQL_DB       = "ct"

PG_HOST     = "127.0.0.1"
PG_PORT     = 5433
PG_USER     = "postgres"
PG_PASSWORD = "kyf050313"
PG_DB       = "ct_vector_db"

VECTOR_DIM  = 960   # ← 改成你模型实际输出的特征维度


# ── 2. 从 MySQL 读出所有旧数据 ────────────────────────────────
def read_from_mysql():
    conn = mysql.connector.connect(
        host=MYSQL_HOST, port=MYSQL_PORT,
        user=MYSQL_USER, password=MYSQL_PASSWORD,
        database=MYSQL_DB,
    )
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT patient_id, feature, model_type FROM ct_patient_features")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # feature 字段是 pickle 序列化过的 numpy 数组，这里反序列化回来
    result = []
    for r in rows:
        feat = pickle.loads(r["feature"])
        result.append((r["patient_id"], feat, r["model_type"]))
    return result


# ── 3. 建好 pgvector 的目标表（如果还没建过）────────────────────
def ensure_pg_table():
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        user=PG_USER, password=PG_PASSWORD,
        dbname=PG_DB,
    )
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f'''
        CREATE TABLE IF NOT EXISTS ct_patient_features (
            patient_id  BIGINT PRIMARY KEY,
            feature     vector({VECTOR_DIM}) NOT NULL,
            model_type  VARCHAR(32) NOT NULL,
            updated_at  TIMESTAMP NOT NULL DEFAULT NOW()
        );
    ''')
    cur.execute('''
        CREATE INDEX IF NOT EXISTS ct_patient_features_hnsw_idx
        ON ct_patient_features
        USING hnsw (feature vector_cosine_ops);
    ''')
    conn.commit()
    cur.close()
    conn.close()


# ── 4. 把数据逐条写入 pgvector ─────────────────────────────────
def write_to_pgvector(rows):
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        user=PG_USER, password=PG_PASSWORD,
        dbname=PG_DB,
    )
    register_vector(conn)  # 让 psycopg2 认识 numpy 数组 <-> pgvector 的 vector 类型
    cur = conn.cursor()

    success, failed = 0, []
    for patient_id, feat, model_type in rows:
        try:
            vec = np.asarray(feat).squeeze().astype(np.float32)
            if vec.shape[0] != VECTOR_DIM:
                raise ValueError(f"特征维度是 {vec.shape[0]}，但 VECTOR_DIM 设置的是 {VECTOR_DIM}，两者要一致")

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
            success += 1
            print(f"  ✓ 患者 {patient_id} 迁移成功（维度 {vec.shape[0]}）")
        except Exception as e:
            conn.rollback()
            failed.append(patient_id)
            print(f"  ✗ 患者 {patient_id} 迁移失败：{e}")

    cur.close()
    conn.close()
    return success, failed


# ── 5. 主流程 ─────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("[1/3] 从 MySQL 读取旧数据 ...")
    rows = read_from_mysql()
    print(f"共读取到 {len(rows)} 条记录")

    if not rows:
        print("没有数据需要迁移，结束。")
        return

    print("\n[2/3] 确保 pgvector 表已建好 ...")
    ensure_pg_table()

    print("\n[3/3] 写入 pgvector ...")
    success, failed = write_to_pgvector(rows)

    print("=" * 50)
    print(f"迁移完成：成功 {success} 条，失败 {len(failed)} 条")
    if failed:
        print(f"失败的患者ID：{failed}")
    print("=" * 50)


if __name__ == "__main__":
    main()