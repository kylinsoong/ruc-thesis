from FlagEmbedding import FlagModel

corpus = [
    "Michael Jackson was a legendary pop icon known for his record-breaking music and dance innovations.",
    "Fei-Fei Li is a professor in Stanford University, revolutionized computer vision with the ImageNet project.",
    "Brad Pitt is a versatile actor and producer known for his roles in films like 'Fight Club' and 'Once Upon a Time in Hollywood.'",
    "Geoffrey Hinton, as a foundational figure in AI, received Turing Award for his contribution in deep learning.",
    "Eminem is a renowned rapper and one of the best-selling music artists of all time.",
    "Taylor Swift is a Grammy-winning singer-songwriter known for her narrative-driven music.",
    "Sam Altman leads OpenAI as its CEO, with astonishing works of GPT series and pursuing safe and beneficial AI.",
    "Morgan Freeman is an acclaimed actor famous for his distinctive voice and diverse roles.",
    "Andrew Ng spread AI knowledge globally via public courses on Coursera and Stanford University.",
    "Robert Downey Jr. is an iconic actor best known for playing Iron Man in the Marvel Cinematic Universe.",
]

query = "Who could be an expert of neural network?"

model = FlagModel('BAAI/bge-base-en-v1.5',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)


corpus_embeddings = model.encode(corpus)
query_embedding = model.encode(query)

print("type of the query embedding:  ", type(query_embedding))
print("type of the corpus embeddings:", type(corpus_embeddings))

print(query_embedding)
print(corpus_embeddings)

print("shape of the query embedding:  ", query_embedding.shape)
print("shape of the corpus embeddings:", corpus_embeddings.shape)

print(query_embedding[:10])

sim_scores = query_embedding @ corpus_embeddings.T
print(sim_scores)

sorted_indices = sorted(range(len(sim_scores)), key=lambda k: sim_scores[k], reverse=True)
print(sorted_indices)

print(corpus[3])

for i in sorted_indices:
    print(f"Score of {sim_scores[i]:.3f}: \"{corpus[i]}\"")
