# test_api.py
from api_client import DoubaoAPIClient

client = DoubaoAPIClient()

# 测试LLM调用
print("=== 测试 LLM 调用 ===")
result = client.call_llm("你好，请简单介绍一下自己")
print(f"响应: {result}")

# 测试Embedding调用
print("\n=== 测试 Embedding 调用 ===")
embedding_result = client.get_embedding("天很蓝，海很深")
print(f"Embedding 响应: {embedding_result}")

print("\n=== API 客户端测试完成 ===")
