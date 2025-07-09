from FlagEmbedding import FlagAutoModel

model = FlagAutoModel.from_finetuned('BAAI/bge-base-en-v1.5')

sentence_1 = "watch a play"
sentence_2 = "play with a watch"

embedding_1 = model.encode([sentence_1])
embedding_2 = model.encode([sentence_2])

similarity = embedding_1 @ embedding_2.T

print(f"Similarity between '{sentence_1}' and '{sentence_2}': {similarity.item():.4f}")

