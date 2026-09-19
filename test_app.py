"""
Unit-тесты для ML-продукта.
Запуск: pytest test_app.py -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import mask_phone, mask_name, predict_sentiment, compute_metrics, \
    top_products_by_positive_share, top_negative_words


# ─── Test 1: Маскировка телефона ──────────────────────────────────────────────

class TestMaskPhone:
    def test_standard_phone(self):
        assert mask_phone("+79001234567") == "+790***4567"

    def test_short_phone(self):
        assert mask_phone("12345") == "12345"  # < 10 chars, no masking

    def test_exact_10_chars(self):
        result = mask_phone("1234567890")
        assert result == "1234***7890"
        assert "***" in result


# ─── Test 2: Маскировка имени ─────────────────────────────────────────────────

class TestMaskName:
    def test_standard_name(self):
        result = mask_name("Иванов")
        assert result.startswith("Ив")
        assert result.endswith("ов")
        assert "*" in result

    def test_short_name(self):
        result = mask_name("Ян")
        assert result == "Я*"

    def test_empty_name(self):
        assert mask_name("") == "***"

    def test_three_char_name(self):
        result = mask_name("Bob")
        assert result == "B**"


# ─── Test 3: Предсказание тональности ────────────────────────────────────────

class TestPredictSentiment:
    @pytest.fixture(autouse=True)
    def setup(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        texts = (["great product love it"] * 50 +
                 ["terrible awful bad broken"] * 50)
        labels = [1] * 50 + [0] * 50

        self.vectorizer = TfidfVectorizer(max_features=100)
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(X, labels)

    def test_returns_tuple(self):
        result = predict_sentiment("great product", self.model, self.vectorizer)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_positive_text(self):
        label, conf = predict_sentiment("great product love it", self.model, self.vectorizer)
        assert label == "позитивный"
        assert conf > 50

    def test_negative_text(self):
        label, conf = predict_sentiment("terrible awful bad", self.model, self.vectorizer)
        assert label == "негативный"
        assert conf > 50

    def test_confidence_range(self):
        _, conf = predict_sentiment("some text", self.model, self.vectorizer)
        assert 0 <= conf <= 100


# ─── Test 4: Загрузка данных ──────────────────────────────────────────────────

class TestLoadData:
    def test_csv_exists(self):
        csv_path = os.path.join(os.path.dirname(__file__), 'Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv')
        assert os.path.exists(csv_path)

    def test_csv_columns(self):
        csv_path = os.path.join(os.path.dirname(__file__), 'Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv')
        df = pd.read_csv(csv_path, nrows=5)
        required = ['name', 'reviews.text', 'reviews.rating']
        for col in required:
            assert col in df.columns

    def test_csv_has_data(self):
        csv_path = os.path.join(os.path.dirname(__file__), 'Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv')
        df = pd.read_csv(csv_path)
        assert len(df) >= 200  # Минимум 200 записей


# ─── Test 5: Вычисление метрик ────────────────────────────────────────────────

class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics['accuracy'] == 1.0
        assert metrics['f1'] == 1.0

    def test_all_wrong(self):
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 1, 1]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics['accuracy'] == 0.0

    def test_returns_dict(self):
        metrics = compute_metrics([1, 0], [1, 0])
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics


# ─── Test 6: Рекомендации ─────────────────────────────────────────────────────

class TestRecommendations:
    def test_popular_products(self):
        from recommendations import get_popular_products

        df = pd.DataFrame({
            'reviewer_id': ['u1']*5 + ['u2']*5,
            'product_id': ['p1', 'p2', 'p3', 'p1', 'p2'] * 2,
            'overall': [5, 4, 3, 5, 4] * 2,
            'summary': ['A', 'B', 'C', 'A', 'B'] * 2,
        })
        recs = get_popular_products(df, n=3)
        assert isinstance(recs, list)
        assert len(recs) <= 3

    def test_recommendation_format(self):
        from recommendations import get_recommendations

        df = pd.DataFrame({
            'reviewer_id': ['u1']*10 + ['u2']*10,
            'product_id': (['p1', 'p2', 'p3', 'p4', 'p5'] * 2) * 2,
            'overall': ([5, 4, 3, 5, 4] * 2) * 2,
            'summary': (['A', 'B', 'C', 'D', 'E'] * 2) * 2,
        })
        recs = get_recommendations('u1', df, n_recommendations=3)
        assert isinstance(recs, list)
        if recs:
            assert 'product_id' in recs[0]
            assert 'predicted_rating' in recs[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
