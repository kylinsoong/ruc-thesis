from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3')
sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

output = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=True)
dense, sparse, multiv = output['dense_vecs'], output['lexical_weights'], output['colbert_vecs']

print(dense)
print(sparse)
print(multiv)
