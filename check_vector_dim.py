"""
小工具：查看 MySQL 里已存的CT特征向量，实际维度是多少。
不需要跑模型，直接读现成数据反推。

用法：
    1. 改下面 MYSQL_* 几个连接参数
    2. python check_vector_dim.py
"""
import pickle
import numpy as np
import mysql.connector

MYSQL_HOST     = "127.0.0.1"
MYSQL_PORT     = 3306
MYSQL_USER     = "root"
MYSQL_PASSWORD = "kyf050313"
MYSQL_DB       = "ct"


def main():
    conn = mysql.connector.connect(
        host=MYSQL_HOST, port=MYSQL_PORT,
        user=MYSQL_USER, password=MYSQL_PASSWORD,
        database=MYSQL_DB,
    )
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT patient_id, feature, model_type FROM ct_patient_features LIMIT 5")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print("表里没有数据，没法反推维度，需要先跑一次推理产生特征。")
        return

    print(f"读取到 {len(rows)} 条样本，逐条打印维度：\n")
    for r in rows:
        feat = pickle.loads(r["feature"])
        arr = np.asarray(feat)
        print(f"患者ID: {r['patient_id']:>4}  model_type: {r['model_type']:<15}  "
              f"原始shape: {arr.shape}  squeeze后shape: {arr.squeeze().shape}")

    print("\n把上面 'squeeze后shape' 里的数字，填到 migrate_direct.py 的 VECTOR_DIM 就行。")
    print("如果所有行的维度都一样，那就是唯一正确的值；")
    print("如果不同患者维度不一样，说明用了不同模型/不同版本提取的特征，需要额外处理（先告诉我）。")


if __name__ == "__main__":
    main()