from transformers import AutoModel, AutoTokenizer

model_name = "GiteeAI/bge-m3"  
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModel.from_pretrained(model_name)  

text = "示例文本"
inputs = tokenizer(f"为这个句子生成嵌入：{text}", return_tensors="pt")  

with torch.no_grad():
    outputs = model(**inputs)  
    embeddings = outputs.last_hidden_state[:, 0, :]  
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # 归一化
