import mysql.connector
from mysql.connector import Error
import numpy as np
import pickle
from app.core.config import settings

# ---------------------- 连接 ----------------------
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
        print("MySQL 连接错误：", e)
        return None

# ---------------------- 建表（首次自动执行）----------------------
def init_db():
    conn = get_conn()
    if not conn:
        return
    cur = conn.cursor()

    # 任务表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS tasks (
        task_id VARCHAR(64) PRIMARY KEY,
        patient_id VARCHAR(64),
        status VARCHAR(20) DEFAULT 'pending',
        model_type VARCHAR(32),
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        finished_at DATETIME NULL,
        mask_path TEXT,
        feature_path TEXT,
        error TEXT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    ''')

    # 患者特征表（BLOB 存 pickle 向量）
    cur.execute('''
    CREATE TABLE IF NOT EXISTS patient_features (
        patient_id VARCHAR(64) PRIMARY KEY,
        feature BLOB NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    ''')

    conn.commit()
    cur.close()
    conn.close()

# ---------------------- 任务 CRUD ----------------------
def save_task(task_id, patient_id, model_type):
    conn = get_conn()
    if not conn:
        return
    cur = conn.cursor()
    sql = '''
    INSERT INTO tasks (task_id, patient_id, model_type)
    VALUES (%s, %s, %s)
    '''
    cur.execute(sql, (task_id, patient_id, model_type))
    conn.commit()
    cur.close()
    conn.close()

def update_task(task_id, status, mask_path=None, feature_path=None, error=None):
    conn = get_conn()
    if not conn:
        return
    cur = conn.cursor()
    sql = '''
    UPDATE tasks
    SET status=%s, mask_path=%s, feature_path=%s, error=%s, finished_at=NOW()
    WHERE task_id=%s
    '''
    cur.execute(sql, (status, mask_path, feature_path, error, task_id))
    conn.commit()
    cur.close()
    conn.close()

def get_task(task_id):
    conn = get_conn()
    if not conn:
        return None
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM tasks WHERE task_id=%s", (task_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

# ---------------------- 患者特征 CRUD ----------------------
def save_patient_feature(patient_id, feature_vector: np.ndarray):
    """向量用 pickle 存 BLOB"""
    conn = get_conn()
    if not conn:
        return
    cur = conn.cursor()
    blob = pickle.dumps(feature_vector)
    sql = '''
    INSERT INTO patient_features (patient_id, feature)
    VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE feature=%s
    '''
    cur.execute(sql, (patient_id, blob, blob))
    conn.commit()
    cur.close()
    conn.close()

def load_all_features():
    """加载所有患者特征：dict{pid: vec}"""
    conn = get_conn()
    if not conn:
        return {}
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT patient_id, feature FROM patient_features")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    result = {}
    for r in rows:
        vec = pickle.loads(r["feature"])
        result[r["patient_id"]] = vec
    return result

def delete_patient(patient_id):
    conn = get_conn()
    if not conn:
        return
    cur = conn.cursor()
    cur.execute("DELETE FROM patient_features WHERE patient_id=%s", (patient_id,))
    conn.commit()
    cur.close()
    conn.close()