#!/usr/bin/env python3
"""
构建 ChromaDB 向量数据库
从知识库 JSON 文件读取数据，进行嵌入后存储到 ChromaDB
"""
import json
import sys
sys.path.insert(0, '.')

from pathlib import Path
from api_client import DoubaoAPIClient
import chromadb
from chromadb.config import Settings

DATA_DIR = Path(__file__).parent / "data" / "knowledge_base"
CHROMA_DIR = DATA_DIR / "chroma_db"

def load_json_file(filepath):
    """加载 JSON 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_embedding(client, text):
    """获取文本嵌入"""
    try:
        result = client.get_embedding(text)
        if isinstance(result, dict) and 'response' in result:
            resp = result['response']
            # resp 是 MultimodalEmbedding 对象
            if hasattr(resp, 'embedding'):
                return resp.embedding
            elif hasattr(resp, 'data'):
                data = resp.data
                if hasattr(data, 'embedding'):
                    return data.embedding
        return None
    except Exception as e:
        print(f"  嵌入错误: {e}")
        return None

def build_regulations_collection(client, client_persistent):
    """构建法规集合"""
    print("\n构建 regulations 集合...")

    filepath = DATA_DIR / "regulations.json"
    if not filepath.exists():
        print("  regulations.json 不存在，跳过")
        return

    data = load_json_file(filepath)
    if not data:
        print("  法规数据为空，跳过")
        return

    collection = client_persistent.get_or_create_collection(
        name="regulations",
        metadata={"description": "金融法规知识库"}
    )

    documents = []
    metadatas = []
    ids = []

    for i, item in enumerate(data):
        content = item.get('content', '') or item.get('description', '')
        if content:
            documents.append(content)
            metadatas.append({
                "title": item.get('title', f'regulation_{i}'),
                "category": "regulations"
            })
            ids.append(f"reg_{i}")

    if documents:
        print(f"  准备嵌入 {len(documents)} 条法规...")

        embeddings = []
        for i, doc in enumerate(documents):
            print(f"  嵌入进度: {i+1}/{len(documents)}", end='\r')
            emb = get_embedding(client, doc)
            if emb:
                embeddings.append(emb)
            else:
                embeddings.append([0] * 2048)

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        print(f"\n  已添加 {len(documents)} 条法规到 regulations 集合")

def build_industry_knowledge_collection(client, client_persistent):
    """构建行业知识集合"""
    print("\n构建 industry_knowledge 集合...")

    filepath = DATA_DIR / "industry_knowledge.json"
    if not filepath.exists():
        print("  industry_knowledge.json 不存在，跳过")
        return

    data = load_json_file(filepath)
    if not data:
        print("  行业知识数据为空，跳过")
        return

    collection = client_persistent.get_or_create_collection(
        name="industry_knowledge",
        metadata={"description": "金融风控行业知识库"}
    )

    documents = []
    metadatas = []
    ids = []

    for i, item in enumerate(data):
        content = item.get('content', '')
        if content:
            documents.append(content)
            metadatas.append({
                "category": item.get('category', 'unknown'),
                "knowledge_id": item.get('knowledge_id', f'ik_{i}')
            })
            ids.append(f"ik_{i}")

    if documents:
        print(f"  准备嵌入 {len(documents)} 条行业知识...")

        embeddings = []
        for i, doc in enumerate(documents):
            print(f"  嵌入进度: {i+1}/{len(documents)}", end='\r')
            emb = get_embedding(client, doc)
            if emb:
                embeddings.append(emb)
            else:
                embeddings.append([0] * 2048)

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        print(f"\n  已添加 {len(documents)} 条行业知识到 industry_knowledge 集合")

def build_risk_cases_collection(client, client_persistent):
    """构建风险案例集合"""
    print("\n构建 risk_cases 集合...")

    filepath = DATA_DIR / "risk_cases.json"
    if not filepath.exists():
        print("  risk_cases.json 不存在，跳过")
        return

    data = load_json_file(filepath)
    if not data:
        print("  风险案例数据为空，跳过")
        return

    collection = client_persistent.get_or_create_collection(
        name="risk_cases",
        metadata={"description": "金融风控风险案例库"}
    )

    documents = []
    metadatas = []
    ids = []

    for i, item in enumerate(data):
        content = item.get('description', '') or item.get('case_summary', '')
        if content:
            documents.append(content)
            metadatas.append({
                "case_type": item.get('case_type', 'unknown'),
                "risk_level": item.get('risk_level', 'unknown')
            })
            ids.append(f"rc_{i}")

    if documents:
        print(f"  准备嵌入 {len(documents)} 条风险案例...")

        embeddings = []
        for i, doc in enumerate(documents):
            print(f"  嵌入进度: {i+1}/{len(documents)}", end='\r')
            emb = get_embedding(client, doc)
            if emb:
                embeddings.append(emb)
            else:
                embeddings.append([0] * 2048)

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        print(f"\n  已添加 {len(documents)} 条风险案例到 risk_cases 集合")

def main():
    print("=" * 60)
    print("ChromaDB 向量数据库构建")
    print("=" * 60)

    # 初始化 ChromaDB
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client_persistent = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    # 清空现有集合（可选）
    try:
        existing = client_persistent.list_collections()
        if existing:
            print(f"\n发现 {len(existing)} 个已有集合，将清空重建...")
            for col in existing:
                client_persistent.delete_collection(col.name)
    except Exception as e:
        print(f"  清空集合时出错: {e}")

    # 初始化 API 客户端
    print("\n初始化 API 客户端...")
    try:
        client = DoubaoAPIClient()
        test_result = client.call_llm("测试")
        print("  API 连接成功")
    except Exception as e:
        print(f"  API 连接失败: {e}")
        print("  将使用占位符嵌入（0向量）")
        client = None

    # 构建各集合
    build_regulations_collection(client, client_persistent)
    build_industry_knowledge_collection(client, client_persistent)
    build_risk_cases_collection(client, client_persistent)

    # 验证结果
    print("\n" + "=" * 60)
    print("验证构建结果")
    print("=" * 60)

    collections = client_persistent.list_collections()
    for col in collections:
        data = col.get()
        print(f"\n集合: {col.name}")
        print(f"  文档数量: {len(data.get('documents', []))}")

    print("\n" + "=" * 60)
    print("构建完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
