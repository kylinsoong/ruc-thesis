from FlagEmbedding import FlagAutoModel

model = FlagAutoModel.from_finetuned('BAAI/bge-base-en-v1.5',
                                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                      use_fp16=True)

sentences_1 = ["I love NLP", "I love machine learning"]
sentences_2 = ["I love BGE", "I love text retrieval"]
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)

print(embeddings_1)
print(embeddings_2)

similarity = embeddings_1 @ embeddings_2.T
print(similarity)

print("-------------")


sentences_1 = ["I love BGE"]
sentences_2 = ["I love BGE"]
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)

print(embeddings_1)
print(embeddings_2)

similarity = embeddings_1 @ embeddings_2.T
print(similarity)

print("-------------")


sentences_1 = ["I love you"]
sentences_2 = ["我爱你"]
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)

print(embeddings_1)
print(embeddings_2)

similarity = embeddings_1 @ embeddings_2.T
print(similarity)


print("-------------")

sentences_1 = ["苹果", "香蕉"]
sentences_2 = ["狮子", "老虎"]
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)

print(embeddings_1)
print(embeddings_2)

similarity = embeddings_1 @ embeddings_2.T
print(similarity)
