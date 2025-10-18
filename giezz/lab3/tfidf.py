import numpy as np
from collections import defaultdict
import math


class TfidfVectorizer:
    def __init__(self, max_features=None):
        """
        Инициализация TF-IDF векторизатора

        Args:
            max_features (int): Максимальный размер словаря терминов
        """
        self.max_features = max_features
        self.vocabulary = {}
        self.idf = {}
        self.fitted = False

    def fit(self, documents):
        """
        Построение словаря и расчет IDF значений

        Args:
            documents (list): Список текстовых документов
        """
        # Подсчет частоты терминов во всех документах
        term_freq = defaultdict(int)
        doc_freq = defaultdict(int)

        for doc in documents:
            # Токенизация и базовая предобработка
            tokens = self._preprocess(doc)
            unique_tokens = set(tokens)

            for token in tokens:
                term_freq[token] += 1

            for token in unique_tokens:
                doc_freq[token] += 1

        # Ограничение размера словаря
        if self.max_features:
            # Сортировка терминов по частоте и выбор top-N
            sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
            selected_terms = [term for term, freq in sorted_terms[:self.max_features]]
        else:
            selected_terms = list(term_freq.keys())

        # Создание словаря
        self.vocabulary = {term: idx for idx, term in enumerate(selected_terms)}

        # Расчет IDF
        total_docs = len(documents)
        for term in self.vocabulary:
            if doc_freq[term] > 0:
                self.idf[term] = math.log(total_docs / doc_freq[term])
            else:
                self.idf[term] = 0

        self.fitted = True
        return self

    def transform(self, documents):
        """
        Преобразование документов в TF-IDF матрицу

        Args:
            documents (list): Список текстовых документов

        Returns:
            numpy.ndarray: TF-IDF матрица
        """
        if not self.fitted:
            raise ValueError("Векторизатор должен быть сначала обучен (fit)")

        tfidf_matrix = np.zeros((len(documents), len(self.vocabulary)))

        for doc_idx, doc in enumerate(documents):
            tokens = self._preprocess(doc)
            doc_length = len(tokens)

            if doc_length == 0:
                continue

            # Подсчет TF
            tf = defaultdict(int)
            for token in tokens:
                if token in self.vocabulary:
                    tf[token] += 1

            # Расчет TF-IDF
            for token, count in tf.items():
                if token in self.vocabulary:
                    term_idx = self.vocabulary[token]
                    tf_val = count / doc_length
                    tfidf_matrix[doc_idx, term_idx] = tf_val * self.idf[token]

        return tfidf_matrix

    def fit_transform(self, documents):
        """
        Обучение и преобразование в одном методе

        Args:
            documents (list): Список текстовых документов

        Returns:
            numpy.ndarray: TF-IDF матрица
        """
        return self.fit(documents).transform(documents)

    def _preprocess(self, text):
        """
        Базовая предобработка текста

        Args:
            text (str): Входной текст

        Returns:
            list: Список токенов
        """
        if not isinstance(text, str):
            return []

        # Приведение к нижнему регистру и удаление пунктуации
        text = text.lower()
        tokens = text.split()

        # Простая фильтрация: удаление коротких слов и цифр
        filtered_tokens = []
        for token in tokens:
            if len(token) > 2 and not token.isdigit():
                # Удаление пунктуации
                token = ''.join(char for char in token if char.isalnum())
                if len(token) > 2:
                    filtered_tokens.append(token)

        return filtered_tokens

    def get_feature_names(self):
        """
        Получение списка терминов словаря

        Returns:
            list: Список терминов
        """
        return list(self.vocabulary.keys())


# Тестирование реализации
if __name__ == "__main__":
    # Пример документов для тестирования
    documents = [
        "Машинное обучение это интересно и полезно",
        "Глубокое обучение является частью машинного обучения",
        "Нейронные сети используются в глубоком обучении",
        "Машинное обучение и глубокое обучение развиваются быстро"
    ]

    # Тест 1: Без ограничения размера словаря
    print("Тест 1: Без ограничения размера словаря")
    vectorizer1 = TfidfVectorizer()
    tfidf_matrix1 = vectorizer1.fit_transform(documents)

    print("Размер словаря:", len(vectorizer1.vocabulary))
    print("Термины:", vectorizer1.get_feature_names())
    print("TF-IDF матрица:")
    print(tfidf_matrix1)
    print()

    # Тест 2: С ограничением размера словаря
    print("Тест 2: С ограничением размера словаря (max_features=5)")
    vectorizer2 = TfidfVectorizer(max_features=5)
    tfidf_matrix2 = vectorizer2.fit_transform(documents)

    print("Размер словаря:", len(vectorizer2.vocabulary))
    print("Термины:", vectorizer2.get_feature_names())
    print("TF-IDF матрица:")
    print(tfidf_matrix2)
    print()

    # Тест 3: Преобразование новых документов
    print("Тест 3: Преобразование новых документов")
    new_documents = [
        "Обучение машинное важно",
        "Нейронные сети сложны"
    ]

    new_tfidf = vectorizer2.transform(new_documents)
    print("TF-IDF для новых документов:")
    print(new_tfidf)