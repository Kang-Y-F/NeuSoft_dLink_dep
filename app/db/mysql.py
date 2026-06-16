# app/db/mysql.py
# ============================================================
# MySQL 数据访问层
# 对应表：tasks（内部任务追踪）、patient_features（特征向量）、
#         check_report（PACS影像报告，来自 HIS 数据库）
# ============================================================

import pickle
import numpy as np
import mysql.connector
from mysql.connector import Error
from datetime import datetime

from app.core.config import settings


# ── 连接 ──────────────────────────────────────────────────────

def get_conn():
    """获取 MySQL 连接，失败时返回 None 并打印错误。"""
    try:
        conn = mysql.connector.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DB
        )
        return conn
    except Error as e:
        print(f"[MySQL] 连接失败：{e}")
        return None


# ── 建表（服务启动时执行一次）────────────────────────────────────

def init_db():
    """
    创建微服务自用的补充表。
    HIS 原有表（pmi_patient、check_report 等）由 HIS 系统维护，
    这里只建微服务独立管理的两张表。
    """
    conn = get_conn()
    if not conn:
        return
    cur = conn.cursor()

    # 推理任务追踪表（Celery 异步任务专用）
    cur.execute('''
    CREATE TABLE IF NOT EXISTS ct_tasks (
        task_id      VARCHAR(64)  PRIMARY KEY COMMENT 'Celery task_id',
        patient_id   VARCHAR(64)  NOT NULL    COMMENT '患者编号',
        status       VARCHAR(20)  NOT NULL DEFAULT 'pending'
                         COMMENT 'pending / running / success / failure',
        model_type   VARCHAR(32)  NOT NULL    COMMENT '使用的模型',
        mask_path    TEXT                     COMMENT '掩码文件本地路径',
        feature_path TEXT                     COMMENT '特征文件本地路径',
        error        TEXT                     COMMENT '失败原因',
        created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
        finished_at  DATETIME                 COMMENT '完成时间'
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='CT推理任务追踪表';
    ''')

    # 患者特征向量表（BLOB 存 pickle 序列化的 numpy 向量）
    cur.execute('''
    CREATE TABLE IF NOT EXISTS ct_patient_features (
        patient_id   VARCHAR(64)  PRIMARY KEY COMMENT '患者编号',
        feature      LONGBLOB     NOT NULL    COMMENT 'pickle(np.ndarray) 特征向量',
        model_type   VARCHAR(32)  NOT NULL    COMMENT '提取特征使用的模型',
        updated_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                         ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='患者CT特征向量表';
    ''')

    conn.commit()
    cur.close()
    conn.close()
    print("[MySQL] 微服务补充表初始化完成")


# ══════════════════════════════════════════════════════════════
#  ct_tasks CRUD
# ══════════════════════════════════════════════════════════════

def task_create(task_id: str, patient_id: str, model_type: str) -> bool:
    """新建任务记录，状态为 pending。"""
    conn = get_conn()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ct_tasks (task_id, patient_id, model_type) VALUES (%s, %s, %s)",
            (task_id, patient_id, model_type)
        )
        conn.commit()
        return True
    except Error as e:
        print(f"[MySQL] task_create 失败：{e}")
        return False
    finally:
        cur.close(); conn.close()


def task_update(
    task_id: str,
    status: str,
    mask_path: str = None,
    feature_path: str = None,
    error: str = None
) -> bool:
    """更新任务状态及结果路径。"""
    conn = get_conn()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            '''UPDATE ct_tasks
               SET status=%s, mask_path=%s, feature_path=%s,
                   error=%s, finished_at=NOW()
               WHERE task_id=%s''',
            (status, mask_path, feature_path, error, task_id)
        )
        conn.commit()
        return True
    except Error as e:
        print(f"[MySQL] task_update 失败：{e}")
        return False
    finally:
        cur.close(); conn.close()


def task_get(task_id: str) -> dict | None:
    """查询单条任务记录。"""
    conn = get_conn()
    if not conn:
        return None
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM ct_tasks WHERE task_id=%s", (task_id,))
        return cur.fetchone()
    except Error as e:
        print(f"[MySQL] task_get 失败：{e}")
        return None
    finally:
        cur.close(); conn.close()


# ══════════════════════════════════════════════════════════════
#  ct_patient_features CRUD
# ══════════════════════════════════════════════════════════════

def feature_upsert(patient_id: str, feature_vector: np.ndarray, model_type: str) -> bool:
    """
    保存或更新患者特征向量（存在则覆盖）。
    特征向量以 pickle 序列化存入 LONGBLOB。
    """
    conn = get_conn()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        blob = pickle.dumps(feature_vector)
        cur.execute(
            '''INSERT INTO ct_patient_features (patient_id, feature, model_type)
               VALUES (%s, %s, %s)
               ON DUPLICATE KEY UPDATE feature=%s, model_type=%s, updated_at=NOW()''',
            (patient_id, blob, model_type, blob, model_type)
        )
        conn.commit()
        return True
    except Error as e:
        print(f"[MySQL] feature_upsert 失败：{e}")
        return False
    finally:
        cur.close(); conn.close()


def feature_load_all() -> dict[str, np.ndarray]:
    """
    加载所有患者特征，返回 {patient_id: np.ndarray}。
    供 FAISS 索引构建使用。
    """
    conn = get_conn()
    if not conn:
        return {}
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT patient_id, feature FROM ct_patient_features")
        rows = cur.fetchall()
        return {r["patient_id"]: pickle.loads(r["feature"]) for r in rows}
    except Error as e:
        print(f"[MySQL] feature_load_all 失败：{e}")
        return {}
    finally:
        cur.close(); conn.close()


def feature_list_patients() -> list[dict]:
    """返回所有患者编号及更新时间，用于前端列表。"""
    conn = get_conn()
    if not conn:
        return []
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT patient_id, model_type, updated_at FROM ct_patient_features ORDER BY updated_at DESC"
        )
        rows = cur.fetchall()
        # updated_at 是 datetime 对象，转为字符串方便 JSON 序列化
        for r in rows:
            if isinstance(r.get("updated_at"), datetime):
                r["updated_at"] = r["updated_at"].strftime("%Y-%m-%d %H:%M:%S")
        return rows
    except Error as e:
        print(f"[MySQL] feature_list_patients 失败：{e}")
        return []
    finally:
        cur.close(); conn.close()


def feature_delete(patient_id: str) -> bool:
    """删除指定患者特征。"""
    conn = get_conn()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM ct_patient_features WHERE patient_id=%s", (patient_id,)
        )
        conn.commit()
        return cur.rowcount > 0
    except Error as e:
        print(f"[MySQL] feature_delete 失败：{e}")
        return False
    finally:
        cur.close(); conn.close()


# ══════════════════════════════════════════════════════════════
#  check_report CRUD（对接 HIS PACS 影像报告表）
# ══════════════════════════════════════════════════════════════

def report_upsert(
    patient_id: int,
    order_id: int,
    mask_path: str,
    artifact_pixel_count: int,
    feature_shape: str,
    report_text: str = "CT伪影分割完成"
) -> bool:
    """
    将推理结果写入 HIS check_report 表。
    - image_url      存掩码文件的服务端路径
    - artifact_result 存伪影统计摘要（JSON字符串）
    - report_text    存报告正文
    只有当 patient_id 和 order_id 能关联到 HIS 数据时才写入。
    若 order_id 为 None，则跳过（微服务独立测试场景）。
    """
    if not order_id:
        return False
    conn = get_conn()
    if not conn:
        return False
    try:
        import json
        artifact_result = json.dumps({
            "artifact_pixel_count": artifact_pixel_count,
            "feature_shape": feature_shape
        }, ensure_ascii=False)

        cur = conn.cursor()
        cur.execute(
            '''INSERT INTO check_report
               (order_id, patient_id, img_type, image_url, artifact_result, report_text)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE
                   image_url=%s, artifact_result=%s,
                   report_text=%s, create_time=NOW()''',
            (
                order_id, patient_id, "CT", mask_path,
                artifact_result, report_text,
                mask_path, artifact_result, report_text
            )
        )
        conn.commit()
        return True
    except Error as e:
        print(f"[MySQL] report_upsert 失败：{e}")
        return False
    finally:
        cur.close(); conn.close()
