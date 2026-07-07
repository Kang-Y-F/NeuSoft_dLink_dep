# app/db/mysql.py
import pickle
import numpy as np
import mysql.connector
from mysql.connector import Error
from datetime import datetime

from app.core.config import settings


def get_conn():
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


def init_db():
    conn = get_conn()
    if not conn:
        return
    cur = conn.cursor()

    cur.execute('''
    CREATE TABLE IF NOT EXISTS ct_tasks (
        task_id      VARCHAR(64)  PRIMARY KEY,
        patient_id   VARCHAR(64)  NOT NULL,
        status       VARCHAR(20)  NOT NULL DEFAULT 'pending',
        model_type   VARCHAR(32)  NOT NULL,
        mask_path    TEXT,
        feature_path TEXT,
        error        TEXT,
        created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
        finished_at  DATETIME
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS ct_patient_features (
        patient_id   BIGINT       NOT NULL PRIMARY KEY,
        feature      LONGBLOB     NOT NULL,
        model_type   VARCHAR(32)  NOT NULL,
        updated_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                         ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    ''')

    conn.commit()
    cur.close()
    conn.close()
    print("[MySQL] 微服务补充表初始化完成")


# ══════════════════════════════════════════════════════════════
#  ct_tasks CRUD
# ══════════════════════════════════════════════════════════════

def task_create(task_id: str, patient_id: str, model_type: str) -> bool:
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


def task_update(task_id, status, mask_path=None, feature_path=None, error=None) -> bool:
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


def task_get(task_id: str):
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


def feature_load_all() -> dict:
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


def feature_list_patients() -> list:
    conn = get_conn()
    if not conn:
        return []
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT patient_id, model_type, updated_at FROM ct_patient_features ORDER BY updated_at DESC"
        )
        rows = cur.fetchall()
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
    conn = get_conn()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM ct_patient_features WHERE patient_id=%s", (patient_id,))
        conn.commit()
        return cur.rowcount > 0
    except Error as e:
        print(f"[MySQL] feature_delete 失败：{e}")
        return False
    finally:
        cur.close(); conn.close()


# ══════════════════════════════════════════════════════════════
#  check_report CRUD
# ══════════════════════════════════════════════════════════════

def report_upsert(
    patient_id: int,
    order_id,
    mask_path: str,
    artifact_pixel_count: int,
    feature_shape: str,
    ct_path: str = "",
    dearti_ct_path: str = "",
    report_text: str = "CT伪影分割完成"
) -> bool:
    conn = get_conn()
    if not conn:
        return False
    cur = None
    try:
        import json
        artifact_result = json.dumps({
            "artifact_pixel_count": artifact_pixel_count,
            "feature_shape": feature_shape
        }, ensure_ascii=False)

        # 通过 check_order.id 查出 medical_order.id
        medical_order_id = order_id
        if order_id:
            cur = conn.cursor()
            cur.execute("SELECT order_id FROM check_order WHERE id=%s", (order_id,))
            row = cur.fetchone()
            print(f"[DEBUG report_upsert] check_order.id={order_id} -> medical_order.id={row[0] if row else 'None'}")
            if row and row[0]:
                medical_order_id = row[0]
            cur.close()
            cur = None

        cur = conn.cursor()
        cur.execute(
            '''INSERT INTO check_report
               (order_id, patient_id, img_type, image_url, ct_url,
                dearti_ct_url, artifact_result, report_text)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)''',
            (medical_order_id, patient_id, "CT", mask_path, ct_path,
             dearti_ct_path, artifact_result, report_text)
        )
        conn.commit()
        print(f"[DEBUG report_upsert] 写入成功 patient_id={patient_id} order_id={medical_order_id}")
        return True
    except Exception as e:
        print(f"[MySQL] report_upsert 失败：{e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if cur:
            cur.close()
        conn.close()

# ══════════════════════════════════════════════════════════════
#  查询患者待执行CT检查单
# ══════════════════════════════════════════════════════════════

def get_pending_ct_orders(patient_id: int) -> list:
    conn = get_conn()
    if not conn:
        return []
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            '''SELECT
                co.id           AS order_id,
                co.record_id    AS record_id,
                ci.name         AS item_name,
                ci.item_type    AS item_type,
                d.name          AS doctor_name,
                co.create_time  AS create_time
               FROM check_order co
               LEFT JOIN check_item ci ON co.item_id = ci.id
               LEFT JOIN doctor      d  ON co.doctor_id = d.id
               WHERE co.user_id    = %s
                 AND co.order_type = 1
                 AND co.status     = 1
               ORDER BY co.create_time DESC''',
            (patient_id,)
        )
        rows = cur.fetchall()
        for r in rows:
            if isinstance(r.get("create_time"), datetime):
                r["create_time"] = r["create_time"].strftime("%Y-%m-%d %H:%M:%S")
        return rows
    except Error as e:
        print(f"[MySQL] get_pending_ct_orders 失败：{e}")
        return []
    finally:
        cur.close(); conn.close()


def complete_ct_order(order_id: int) -> bool:
    """
    CT推理完成后：
    1. check_order.status = 4（已完成）
    2. medical_order.exec_status = 2（已完成）
    """
    conn = get_conn()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        # 更新 check_order 状态为已完成(4)
        cur.execute("UPDATE check_order SET status=4 WHERE id=%s", (order_id,))

        # 查出关联的 medical_order.id（check_order.order_id 就是 medical_order.id）
        cur.execute("SELECT order_id FROM check_order WHERE id=%s", (order_id,))
        row = cur.fetchone()
        if row and row[0]:
            cur.execute("UPDATE medical_order SET exec_status=2 WHERE id=%s", (row[0],))

        conn.commit()
        return cur.rowcount > 0
    except Error as e:
        print(f"[MySQL] complete_ct_order 失败：{e}")
        return False
    finally:
        cur.close()
        conn.close()
