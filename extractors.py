import math
from collections import Counter
import networkx as nx


class TextRankExtractor:
    def get_keywords(self, doc, top_n=10):
        # Будуємо граф: вузли - слова, ребра - спільна поява в межах вікна (window=4)
        graph = nx.Graph()
        window_size = 4

        for i in range(len(doc) - window_size + 1):
            window = doc[i: i + window_size]
            for word1 in window:
                for word2 in window:
                    if word1 != word2:
                        graph.add_edge(word1, word2)

        # Обчислюємо вагу вузлів
        scores = nx.pagerank(graph)
        sorted_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_n]

class TFIDFExtractor:
    def __init__(self, all_documents):
        self.all_documents = all_documents
        self.num_docs = len(all_documents)

    def _calculate_idf(self, word):
        # Рахуємо, у скількох документах зустрічається слово
        docs_with_word = sum(1 for doc in self.all_documents if word in doc)
        return math.log(self.num_docs / (1 + docs_with_word))

    def get_keywords(self, doc, top_n=10):
        # doc — список слів одного тексту
        word_counts = Counter(doc)
        total_words = len(doc)

        tfidf_scores = {}
        for word in set(doc):
            tf = word_counts[word] / total_words
            idf = self._calculate_idf(word)
            tfidf_scores[word] = tf * idf

        # Сортуємо за вагою
        sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_n]