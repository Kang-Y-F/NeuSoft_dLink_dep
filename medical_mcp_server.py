"""
医疗RAG MCP Server
提供数据库查询、文件读取、Python服务调用等工具
运行命令: python medical_mcp_server.py
"""
from mcp.server.fastmcp import FastMCP
import psycopg2
import json
import os
from typing import List, Dict
import requests

import dashscope
from dashscope import TextEmbedding

# 建议用环境变量存key，不要硬编码
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")

# 初始化MCP Server
server = FastMCP("Medical RAG MCP Server", host="0.0.0.0", port=8081)


# ═══════════════════════════════════════════════════════════
#  辅助函数：调用Embedding接口，把文本转成向量
# ═══════════════════════════════════════════════════════════
def get_embedding(text: str) -> list:
    """调用通义千问 embedding 接口，把 query 转成 1536 维向量"""
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v2,
        input=text
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Embedding接口调用失败: {resp.code} {resp.message}")
    return resp.output["embeddings"][0]["embedding"]


# ═══════════════════════════════════════════════════════════
#  Tool 1: 查询医疗知识库（从PostgreSQL向量库检索，真实向量相似度搜索）
# ════════════════════════════════════════════════════════════
@server.tool()
def query_medical_knowledge(query: str, top_k: int = 5) -> str:
    """
    从医疗知识向量库中检索相关知识（基于embedding的语义相似度检索）

    Args:
        query: 查询文本（症状、疾病名称等）
        top_k: 返回最相关的K条结果

    Returns:
        JSON格式的检索结果，包含content、metadata和相似度距离
    """
    try:
        # 1. 把查询文本转成向量
        query_vector = get_embedding(query)

        # 2. 连接PostgreSQL（ct_vector_db），用pgvector做余弦距离排序
        conn = psycopg2.connect(
            host="localhost",
            port=5433,
            database="ct_vector_db",
            user="postgres",
            password="kyf050313"  # 修改为你的实际密码
        )
        cursor = conn.cursor()

        # <=> 是pgvector的余弦距离操作符，距离越小越相似
        cursor.execute(
            """
            SELECT content, metadata, embedding <=> %s::vector AS distance
            FROM medical_knowledge_vector
            ORDER BY distance
            LIMIT %s
            """,
            (query_vector, top_k)
        )

        results = []
        for row in cursor.fetchall():
            results.append({
                "content": row[0],
                "metadata": row[1],
                "distance": float(row[2])
            })

        cursor.close()
        conn.close()

        return json.dumps({
            "success": True,
            "count": len(results),
            "results": results
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════
#  Tool 2: 读取华佗数据集文件
# ═══════════════════════════════════════════════════════════
@server.tool()
def read_huatuo_dataset(file_path: str) -> str:
    """
    读取华佗医疗数据集JSON文件

    Args:
        file_path: JSON文件的绝对路径

    Returns:
        JSON格式的文件内容
    """
    try:
        if not os.path.exists(file_path):
            return json.dumps({
                "success": False,
                "error": f"文件不存在: {file_path}"
            }, ensure_ascii=False)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return json.dumps({
            "success": True,
            "file_path": file_path,
            "record_count": len(data) if isinstance(data, list) else 1,
            "preview": data[:3] if isinstance(data, list) else data
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# ══════════════════════════════════════════════════════════
#  Tool 3: 调用Python预测服务（趋势分析）
# ═══════════════════════════════════════════════════════════
@server.tool()
def call_prediction_service(indicator: str, patient_id: int, history_data: List[Dict]) -> str:
    """
    调用Python预测服务进行指标趋势预测

    Args:
        indicator: 指标名称（如"动态血糖"）
        patient_id: 患者ID
        history_data: 历史数据列表

    Returns:
        JSON格式的预测结果
    """
    try:
        response = requests.post(
            "http://localhost:8000/predict/trend",
            json={
                "indicator": indicator,
                "patientId": str(patient_id),
                "history": history_data,
                "steps": 3,
                "granularity": "auto"
            },
            timeout=10
        )

        if response.status_code == 200:
            return json.dumps({
                "success": True,
                "data": response.json()
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# ════════════════════════════════════════════════════════════
#  Tool 4: 查询患者病历（从MySQL）
# ═══════════════════════════════════════════════════════════
@server.tool()
def query_patient_record(patient_id: int) -> str:
    """
    查询患者的病历信息（从MySQL ct数据库）

    Args:
        patient_id: 患者ID

    Returns:
        JSON格式的病历信息
    """
    try:
        import pymysql

        conn = pymysql.connect(
            host="localhost",
            port=3306,
            database="ct",
            user="root",
            password="kyf050313"
        )
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # 查询患者基本信息
        cursor.execute("SELECT * FROM pmi_patient WHERE id = %s", (patient_id,))
        patient = cursor.fetchone()

        # 查询最近3次就诊记录
        cursor.execute("""
                       SELECT *
                       FROM medical_record
                       WHERE user_id = %s
                       ORDER BY create_time DESC LIMIT 3
                       """, (patient_id,))
        records = cursor.fetchall()

        cursor.close()
        conn.close()

        return json.dumps({
            "success": True,
            "patient": patient,
            "recent_records": records
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# ════════════════════════════════════════════════════════════
# 启动MCP Server
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🚀 Medical RAG MCP Server starting...")
    print("Available tools:")
    print("  - query_medical_knowledge (真实向量检索)")
    print("  - read_huatuo_dataset")
    print("  - call_prediction_service")
    print("  - query_patient_record")

    # Java端已改用 streamable-http，这里保持一致
    server.run(transport="streamable-http")