import re
import numpy as np
import pandas as pd
from collections import Counter


# ─── Защита персональных данных ───────────────────────────────────────────────

def mask_phone(phone: str) -> str:
    if len(phone) >= 10:
        return phone[:4] + "***" + phone[-4:]
    return phone


def mask_name(name: str) -> str:
    if not name or len(name) == 0:
        return "***"
    if len(name) <= 3:
        return name[0] + "*" * (len(name) - 1)
    return name[:2] + "*" * (len(name) - 4) + name[-2:]


# ─── Предсказание тональности ─────────────────────────────────────────────────

def predict_sentiment(text: str, model, vectorizer) -> tuple:
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    pred_int = int(pred)
    proba = model.predict_proba(X)[0][pred_int] #Массив вероятностей
    label = "позитивный" if pred_int == 1 else "негативный"
    confidence = round(proba * 100, 1)
    return label, confidence


# ─── Аналитические функции ────────────────────────────────────────────────────

def top_products_by_positive_share(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    df_copy = df.copy()
    # Поддержка обоих названий столбцов
    rating_col = 'reviews.rating' if 'reviews.rating' in df_copy.columns else 'overall'
    product_col = 'name' if 'name' in df_copy.columns else 'asin'
    df_copy['is_positive'] = (df_copy[rating_col] >= 4).astype(int)

    product_stats = df_copy.groupby(product_col).agg(
        total_reviews=(rating_col, 'count'),
        positive_share=('is_positive', 'mean'),
        avg_rating=(rating_col, 'mean')
    ).reset_index()
    product_stats = product_stats.rename(columns={product_col: 'product'})

    # Фильтруем товары с минимум 3 отзывами
    product_stats = product_stats[product_stats['total_reviews'] >= 3]
    product_stats = product_stats.sort_values('positive_share', ascending=False).head(n)
    product_stats['positive_share'] = (product_stats['positive_share'] * 100).round(1)
    product_stats['avg_rating'] = product_stats['avg_rating'].round(2)

    return product_stats


def top_negative_words(df: pd.DataFrame, n: int = 5) -> list:
    rating_col = 'reviews.rating' if 'reviews.rating' in df.columns else 'overall'
    text_col = 'reviews.text' if 'reviews.text' in df.columns else ('reviewText' if 'reviewText' in df.columns else 'review_text')
    negative_df = df[df[rating_col] <= 2]
    texts = negative_df[text_col].dropna().str.lower()

    stop_words = {
        'the', 'a', 'an', 'is', 'it', 'and', 'or', 'but', 'in', 'on', 'at',
        'to', 'for', 'of', 'with', 'by', 'from', 'this', 'that', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'not', 'no',
        'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they', 'them',
        'its', 'his', 'her', 'their', 'so', 'if', 'than', 'as', 'just', 'about',
        'up', 'out', 'all', 'very', 'what', 'when', 'how', 'which', 'who', 'whom',
        'there', 'here', 'are', 'am', 'get', 'got', 'one', 'two', 'also', 'more',
        'after', 'before', 'then', 'these', 'those', 'some', 'any', 'each', 'only',
        'into', 'over', 'such'
    }

    all_words = []
    for text in texts:
        words = re.findall(r'[a-zA-Zа-яА-Я]{3,}', text)
        words = [w for w in words if w not in stop_words]
        all_words.extend(words)

    counter = Counter(all_words)
    return counter.most_common(n)


def compute_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
