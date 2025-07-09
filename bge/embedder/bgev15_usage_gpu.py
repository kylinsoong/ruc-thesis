from FlagEmbedding import FlagModel
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model = FlagModel('BAAI/bge-base-en-v1.5', devices=0)

sentences = ["Hello world", "I am inevitable"]
embeddings = model.encode(sentences)

print(embeddings)
