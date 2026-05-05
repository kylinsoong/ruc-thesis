#!/usr/bin/env python3
"""
知识库测试代码
用于查看和验证知识库状态
"""
import json
import chromadb
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
KB_DIR = DATA_DIR / "knowledge_base"

def test_knowledge_files():
    """查看知识库文件"""
    print("=" * 60)
    print("知识库文件")
    print("=" * 60)

    if not KB_DIR.exists():
        print("知识库目录不存在")
        return

    for f in KB_DIR.iterdir():
        if f.is_file() and f.suffix in ['.json', '.txt', '.md']:
            size = f.stat().st_size / 1024
            print(f"\n{f.name}: {size:.1f} KB")

def test_industry_knowledge():
    """测试行业知识库"""
    print("\n" + "=" * 60)
    print("行业知识库 (industry_knowledge.json)")
    print("=" * 60)

    kb_file = KB_DIR / "industry_knowledge.json"
    if not kb_file.exists():
        print("文件不存在")
        return

    with open(kb_file, 'r', encoding='utf-8') as f:
        knowledge = json.load(f)

    print(f"\n知识条目数量: {len(knowledge)}")

    # 按类别统计
    categories = {}
    for item in knowledge:
        cat = item.get('category', '未知')
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n类别分布:")
    for cat, count in categories.items():
        print(f"  - {cat}: {count} 条")

    # 显示示例
    print(f"\n示例知识 (前2条):")
    for item in knowledge[:2]:
        print(f"\n  ID: {item['knowledge_id']}")
        print(f"  类别: {item['category']}")
        content = item['content'][:100] + "..." if len(item['content']) > 100 else item['content']
        print(f"  内容: {content}")

def test_regulations():
    """测试法规知识库"""
    print("\n" + "=" * 60)
    print("法规知识库 (regulations.json)")
    print("=" * 60)

    reg_file = KB_DIR / "regulations.json"
    if not reg_file.exists():
        print("文件不存在")
        return

    with open(reg_file, 'r', encoding='utf-8') as f:
        regulations = json.load(f)

    print(f"\n法规条目数量: {len(regulations)}")

    if regulations:
        print(f"\n第一条法规:")
        print(f"  标题: {regulations[0].get('title', 'N/A')}")
        print(f"  内容: {regulations[0].get('content', '')[:150]}...")

def test_risk_cases():
    """测试风险案例库"""
    print("\n" + "=" * 60)
    print("风险案例库 (risk_cases.json)")
    print("=" * 60)

    case_file = KB_DIR / "risk_cases.json"
    if not case_file.exists():
        print("文件不存在")
        return

    with open(case_file, 'r', encoding='utf-8') as f:
        cases = json.load(f)

    print(f"\n案例数量: {len(cases)}")

    if cases:
        print(f"\n第一个案例:")
        case = cases[0]
        for key, value in case.items():
            if isinstance(value, str) and len(value) > 80:
                value = value[:80] + "..."
            print(f"  {key}: {value}")

def test_text_cases():
    """测试文本案例库"""
    print("\n" + "=" * 60)
    print("文本案例库 (test_cases/text_cases.json)")
    print("=" * 60)

    test_dir = DATA_DIR / "test_cases"
    if not test_dir.exists():
        print("测试案例目录不存在")
        return

    case_file = test_dir / "text_cases.json"
    if not case_file.exists():
        print("文件不存在")
        return

    with open(case_file, 'r', encoding='utf-8') as f:
        text_cases = json.load(f)

    print(f"\n文本案例数量: {len(text_cases)}")

    if text_cases:
        print(f"\n第一个案例:")
        case = text_cases[0]
        for key, value in case.items():
            if isinstance(value, str) and len(value) > 80:
                value = value[:80] + "..."
            print(f"  {key}: {value}")

def test_chroma_db():
    """测试 ChromaDB 向量数据库"""
    print("\n" + "=" * 60)
    print("ChromaDB 向量数据库")
    print("=" * 60)

    chroma_dir = KB_DIR / "chroma_db"
    if not chroma_dir.exists():
        print("ChromaDB 目录不存在")
        return

    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))

        # 获取所有集合
        collections = client.list_collections()
        print(f"\n集合数量: {len(collections)}")

        for col in collections:
            print(f"\n集合名称: {col.name}")
            try:
                data = col.get()
                print(f"  文档数量: {len(data.get('documents', []))}")
            except Exception as e:
                print(f"  读取错误: {e}")

    except Exception as e:
        print(f"ChromaDB 连接错误: {e}")

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("知识库状态检查")
    print("=" * 60)

    test_knowledge_files()
    test_industry_knowledge()
    test_regulations()
    test_risk_cases()
    test_text_cases()
    test_chroma_db()

    print("\n" + "=" * 60)
    print("知识库检查完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
