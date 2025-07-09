from FlagEmbedding import FlagReranker

reranker = FlagReranker(
    'BAAI/bge-reranker-base',
    query_max_length=256,
    use_fp16=True,
    devices=['cuda:1'],
)

score = reranker.compute_score(['I am happy to help', 'Assisting you is my pleasure'])
print(score)
