from FlagEmbedding import FlagReranker

# Setting use_fp16 to True speeds up computation with a slight performance degradation
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

score = reranker.compute_score(['query', 'passage'])
# or set "normalize=True" to apply a sigmoid function to the score for 0-1 range
score = reranker.compute_score(['query', 'passage'], normalize=True)

print(score)
