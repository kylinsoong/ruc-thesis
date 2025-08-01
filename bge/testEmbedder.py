from FlagEmbedding import FlagAutoModel

sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]

model = FlagAutoModel.from_finetuned('BAAI/bge-large-zh-v1.5', 
                                     query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                     use_fp16=True,
                                     devices=['cuda:0']) # Setting use_fp16 to True speeds up computation with a slight performance degradation

embeddings_1 = model.encode_corpus(sentences_1)
embeddings_2 = model.encode_corpus(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
# corpus in retrieval task can still use encode_corpus(), since they don't need instruction
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode_corpus(passages)
scores = q_embeddings @ p_embeddings.T
print(scores)
