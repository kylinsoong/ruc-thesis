import graphviz

# Create a new directed graph
g = graphviz.Digraph('tech_integration_architecture', format='png')

# Set graph attributes for better appearance
g.attr(rankdir='TB', fontname='Noto Sans CJK JP', fontsize='16', bgcolor='white', margin='0.5')

# Define node styles
g.attr('node', shape='box', style='rounded,filled', fillcolor='#f5f5f5', 
       fontname='Noto Sans CJK JP', fontsize='14', margin='0.2', height='0.6')

# Define edge styles
g.attr('edge', fontname='Noto Sans CJK JP', fontsize='12')

# Create clusters for different components
with g.subgraph(name='cluster_traditional', graph_attr={'label': '传统金融风控专家系统', 'style': 'rounded,filled', 'fillcolor': '#e6f3ff', 'margin': '16'}) as c:
    c.node('rule_engine', '规则引擎')
    c.node('knowledge_base', '专家知识库')
    c.node('decision_system', '决策系统')
    c.node('risk_models', '风险模型')
    
    # Add edges within traditional system
    c.edge('knowledge_base', 'rule_engine', label='提供规则')
    c.edge('rule_engine', 'decision_system', label='执行规则')
    c.edge('risk_models', 'decision_system', label='风险评估')

with g.subgraph(name='cluster_llm', graph_attr={'label': '大语言模型(LLM)技术', 'style': 'rounded,filled', 'fillcolor': '#fff2e6', 'margin': '16'}) as c:
    c.node('llm_core', 'LLM核心模型')
    c.node('reasoning', '推理能力')
    c.node('nlp', '自然语言处理')
    c.node('multi_modal', '多模态融合')
    
    # Add edges within LLM system
    c.edge('llm_core', 'reasoning', label='提供基础')
    c.edge('llm_core', 'nlp', label='支持')
    c.edge('llm_core', 'multi_modal', label='扩展')

with g.subgraph(name='cluster_rag', graph_attr={'label': '检索增强生成(RAG)技术', 'style': 'rounded,filled', 'fillcolor': '#e6ffe6', 'margin': '16'}) as c:
    c.node('vector_db', '向量数据库')
    c.node('retrieval', '检索模块')
    c.node('knowledge_graph', '知识图谱')
    c.node('fact_verification', '事实验证')
    
    # Add edges within RAG system
    c.edge('vector_db', 'retrieval', label='提供索引')
    c.edge('knowledge_graph', 'retrieval', label='结构化知识')
    c.edge('retrieval', 'fact_verification', label='提供证据')

# Create integration nodes
g.node('data_integration', '数据集成层', shape='box', style='rounded,filled', fillcolor='#f0e6ff')
g.node('fusion_layer', '融合决策层', shape='box', style='rounded,filled', fillcolor='#f0e6ff')
g.node('api_layer', '应用接口层', shape='box', style='rounded,filled', fillcolor='#f0e6ff')
g.node('user_interface', '用户界面', shape='box', style='rounded,filled', fillcolor='#ffe6e6')

# Connect traditional system to integration
g.edge('rule_engine', 'fusion_layer', label='规则决策')
g.edge('knowledge_base', 'data_integration', label='领域知识')
g.edge('risk_models', 'fusion_layer', label='风险评分')

# Connect LLM to integration
g.edge('reasoning', 'fusion_layer', label='复杂推理')
g.edge('nlp', 'data_integration', label='非结构化数据处理')
g.edge('multi_modal', 'data_integration', label='多模态数据融合')

# Connect RAG to integration
g.edge('retrieval', 'fusion_layer', label='相关知识检索')
g.edge('fact_verification', 'fusion_layer', label='事实核验')
g.edge('knowledge_graph', 'data_integration', label='知识结构化')

# Connect integration layers
g.edge('data_integration', 'fusion_layer', label='提供数据支持')
g.edge('fusion_layer', 'api_layer', label='决策结果')
g.edge('api_layer', 'user_interface', label='数据展示')

# Add external data sources
g.node('external_data', '外部数据源', shape='cylinder', style='filled', fillcolor='#e6e6e6')
g.edge('external_data', 'data_integration', label='数据输入')

# Add final output
g.node('risk_decision', '风控决策', shape='box', style='rounded,filled,bold', fillcolor='#ffcccc')
g.edge('user_interface', 'risk_decision', label='人机协作')

# Save the graph to a file
g.render(filename='tech_integration_architecture', directory='./output', cleanup=True)

print("技术融合架构图已生成: ./output/tech_integration_architecture.png")
