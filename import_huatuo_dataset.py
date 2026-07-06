"""
华佗医疗数据集 -> 向量知识库 导入脚本
用法: python import_huatuo_dataset.py

功能：
1. 读取 llama_data.json（华佗数据集）
2. 调用通义千问 embedding 接口，把每条知识转成 1536 维向量
3. 写入 PostgreSQL 的 medical_knowledge_vector 表（需已开启 pgvector 插件）

前置条件：
- pip install dashscope psycopg2-binary tqdm
- 设置环境变量 DASHSCOPE_API_KEY（你的通义千问 API Key）
- medical_knowledge_vector 表已存在，结构大致为：
    id SERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(1536),
    created_at TIMESTAMP DEFAULT now()
"""

import json
import os
import time
import psycopg2
from tqdm import tqdm

import dashscope
from dashscope import TextEmbedding

# ═══════════════════════════════════════════
# 配置区（按需修改）
# ═══════════════════════════════════════════
DATASET_PATH = r"E:\ai_data\huatuo_8k\llama_data.json"   # 你的数据集路径

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "ct_vector_db",
    "user": "postgres",
    "password": "kyf050313"
}

# 建议用环境变量存key，不要硬编码在代码里
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-08c402da503e494c9d05b16962d70f1d")

BATCH_SIZE = 10          # 每批处理多少条（避免一次性打爆内存/接口限速）
MAX_RECORDS = None       # 测试时可以设成比如 200，先跑一小批验证没问题；正式导入设为 None（全部导入）
SLEEP_BETWEEN_BATCH = 0.5  # 避免触发 QPS 限制


def load_dataset(path: str) -> list:
    """读取华佗数据集文件（JSONL格式：每行一个独立的JSON对象）"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ 第{line_num}行解析失败，跳过: {e}")
    return data


def extract_qa_text(item: dict) -> tuple[str, str]:
    """
    从一条原始数据中提取 (查询内容, 完整知识文本)
    兼容常见的几种字段命名，如果都不匹配会抛异常并打印这条数据方便排查
    """
    # 常见格式1：alpaca风格 instruction/input/output
    if "instruction" in item and "output" in item:
        question = (item.get("instruction", "") + " " + item.get("input", "")).strip()
        answer = item.get("output", "")
    # 常见格式2：query/response
    elif "query" in item and "response" in item:
        question = item["query"]
        answer = item["response"]
    # 常见格式3：question/answer
    elif "question" in item and "answer" in item:
        question = item["question"]
        answer = item["answer"]
    else:
        raise KeyError(f"无法识别的数据字段，原始数据: {json.dumps(item, ensure_ascii=False)[:300]}")

    full_text = f"问：{question}\n答：{answer}"
    return question, full_text


def get_embedding_batch(texts: list[str]) -> list[list[float]]:
    """调用通义千问 embedding 接口，批量获取向量"""
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v2,
        input=texts
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Embedding接口调用失败: {resp.code} {resp.message}")

    # 按 text_index 排序，保证和输入顺序一致
    embeddings = sorted(resp.output["embeddings"], key=lambda x: x["text_index"])
    return [e["embedding"] for e in embeddings]


def main():
    if not dashscope.api_key:
        raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY")

    print(f"📖 正在读取数据集: {DATASET_PATH}")
    raw_data = load_dataset(DATASET_PATH)
    if MAX_RECORDS:
        raw_data = raw_data[:MAX_RECORDS]
    print(f"共读取到 {len(raw_data)} 条原始数据")

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    success_count = 0
    fail_count = 0

    for i in tqdm(range(0, len(raw_data), BATCH_SIZE), desc="导入进度"):
        batch = raw_data[i:i + BATCH_SIZE]

        texts_to_embed = []
        full_texts = []
        metadatas = []

        for item in batch:
            try:
                question, full_text = extract_qa_text(item)
                texts_to_embed.append(question)   # 用问题去做embedding，检索时匹配度更高
                full_texts.append(full_text)
                metadatas.append(json.dumps({"source": "huatuo_8k"}, ensure_ascii=False))
            except KeyError as e:
                print(f"\n⚠️ 跳过一条无法解析的数据: {e}")
                fail_count += 1

        if not texts_to_embed:
            continue

        try:
            embeddings = get_embedding_batch(texts_to_embed)
        except Exception as e:
            print(f"\n❌ 第{i}批 embedding 调用失败: {e}")
            fail_count += len(texts_to_embed)
            time.sleep(2)
            continue

        for full_text, meta, emb in zip(full_texts, metadatas, embeddings):
            try:
                cursor.execute(
                    """
                    INSERT INTO medical_knowledge_vector (content, metadata, embedding, created_at)
                    VALUES (%s, %s, %s::vector, now())
                    """,
                    (full_text, meta, emb)
                )
                success_count += 1
            except Exception as e:
                print(f"\n❌ 写入数据库失败: {e}")
                fail_count += 1

        conn.commit()
        time.sleep(SLEEP_BETWEEN_BATCH)

    cursor.close()
    conn.close()

    print(f"\n✅ 导入完成！成功 {success_count} 条，失败 {fail_count} 条")


if __name__ == "__main__":
    main()