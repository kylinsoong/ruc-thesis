from FlagEmbedding import FlagModel

model = FlagModel('BAAI/bge-base-en-v1.5')

sentences = ["Hello world", "I am inevitable"]
embeddings = model.encode(sentences)

print(embeddings)
