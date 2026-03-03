from preprocessor import TextPreprocessor
from extractors import TFIDFExtractor, TextRankExtractor
import os


def run_lab():
    files = [f for f in os.listdir('data/') if f.endswith('.txt')]
    raw_texts = []
    processed_docs = []

    for f_name in files:
        with open(f'data/{f_name}', 'r', encoding='utf-8') as f:
            content = f.read()
            lang = 'uk' if 'ukr' in f_name else 'en'
            prep = TextPreprocessor(lang=lang)
            clean_words = prep.process(content)
            processed_docs.append(clean_words)
            raw_texts.append(f_name)

    # Ініціалізація обох методів
    tfidf = TFIDFExtractor(processed_docs)
    trank = TextRankExtractor()

    print(f"{'Видання':<15} | {'TF-IDF (Статистика)':<40} | {'TextRank (Графи)':<40}")
    print("-" * 100)

    for i, name in enumerate(raw_texts):
        kw_tfidf = [w for w, s in tfidf.get_keywords(processed_docs[i], 5)]
        kw_trank = [w for w, s in trank.get_keywords(processed_docs[i], 5)]

        print(f"{name:<15} | {', '.join(kw_tfidf):<40} | {', '.join(kw_trank):<40}")


if __name__ == "__main__":
    run_lab()