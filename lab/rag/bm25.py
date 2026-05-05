import math
import re
from collections import Counter


class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = {}
        self.N = 0

    def _tokenize(self, text):
        tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', text)
        result = []
        for token in tokens:
            if re.search(r'[\u4e00-\u9fff]', token):
                for i in range(len(token)):
                    for j in range(i + 2, min(i + 5, len(token) + 1)):
                        result.append(token[i:j])
            else:
                result.append(token.lower())
        return result

    def fit(self, documents):
        self.documents = documents
        self.N = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_doc_length = sum(self.doc_lengths) / self.N if self.N > 0 else 0

    def _get_idf(self, term):
        freq = self.doc_freqs.get(term, 0)
        if freq == 0:
            return 0
        idf = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)
        return max(idf, 0.1)

    def score(self, query, doc_idx):
        doc = self.documents[doc_idx]
        doc_tokens = self._tokenize(doc)
        doc_len = self.doc_lengths[doc_idx]
        freq = Counter(doc_tokens)

        query_tokens = self._tokenize(query)
        score = 0.0

        for term in query_tokens:
            tf = freq.get(term, 0)
            if tf > 0:
                idf = self._get_idf(term)
                tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_length, 1)))
                score += idf * tf_component

        return score

    def retrieve(self, query, top_k=5):
        scores = []
        for i in range(self.N):
            scores.append((i, self.score(query, i)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
