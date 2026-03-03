import re
import pymorphy3
import spacy
from nltk.corpus import stopwords

class TextPreprocessor:
    def __init__(self, lang='uk'):
        self.lang = lang
        # Завантажуємо стоп-слова
        try:
            self.stop_words = set(stopwords.words('ukrainian' if lang == 'uk' else 'english'))
        except:
            self.stop_words = {"the", "and", "with", "for", "that", "this", "і", "та", "що", "але", "це"}

        if lang == 'uk':
            self.morph = pymorphy3.MorphAnalyzer(lang='uk')
            self.allowed_proper = {'Name', 'Surn', 'Patr', 'Geox'}
        else:
            # Для англійської завантажуємо spaCy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("Модель не знайдено! Виконай: python -m spacy download en_core_web_sm")
                self.nlp = None

    def process(self, text):
        # 1. Попередня очистка
        text = re.sub(r'[^a-zA-Zа-яА-ЯіїєґІЇЄҐ\s]', ' ', text).lower()

        result = []

        if self.lang == 'uk':
            words = text.split()
            for word in words:
                if word not in self.stop_words and len(word) > 2:
                    parsed = self.morph.parse(word)[0]
                    gram = parsed.tag.grammemes
                    # Фільтрація іменників та власних назв
                    if 'NOUN' in gram or not self.allowed_proper.isdisjoint(gram):
                        result.append(parsed.normal_form)
        else:
            # Обробка англійської через spaCy
            if self.nlp:
                doc = self.nlp(text)
                for token in doc:
                    # Фільтруємо стоп-слова, знаки пунктуації та залишаємо тільки іменники/власні назви
                    if not token.is_stop and not token.is_punct and len(token.text) > 2:
                        if token.pos_ in ['NOUN', 'PROPN']:
                            result.append(token.lemma_)
            else:
                # План Б, якщо spaCy не завантажився
                result = [w for w in text.split() if w not in self.stop_words and len(w) > 2]

        return result