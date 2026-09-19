import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from utils import top_products_by_positive_share, top_negative_words

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv')


def generate_report():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['reviews.text'])
    df = df.rename(columns={
        'reviews.text': 'reviewText',
        'reviews.rating': 'overall',
        'reviews.username': 'reviewerID',
        'reviews.title': 'summary',
        'reviews.date': 'reviewDate',
    })
    df['date'] = pd.to_datetime(df['reviewDate'], errors='coerce')

    df_binary = df[df['overall'].isin([1, 2, 4, 5])].copy()
    df_binary['sentiment'] = (df_binary['overall'] >= 4).astype(int)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_vec = vectorizer.fit_transform(df_binary['reviewText'].values)
    y = df_binary['sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Student Report — ML-анализ отзывов (Amazon Products Reviews)',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. Распределение оценок
    ax = axes[0, 0]
    ratings = df['overall'].value_counts().sort_index()
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
    ax.bar(ratings.index, ratings.values, color=colors, edgecolor='white', width=0.7)
    ax.set_title('Распределение оценок', fontweight='bold')
    ax.set_xlabel('Оценка'); ax.set_ylabel('Количество')
    for i, v in enumerate(ratings.values):
        ax.text(ratings.index[i], v + 50, str(v), ha='center', fontsize=9)

    # 2. Динамика отзывов
    ax = axes[0, 1]
    monthly = df.groupby(df['date'].dt.to_period('M')).size()
    monthly.index = monthly.index.astype(str)
    ax.plot(range(len(monthly)), monthly.values, 'b-o', linewidth=1.5, markersize=3)
    step = max(1, len(monthly) // 10)
    ax.set_xticks(range(0, len(monthly), step))
    ax.set_xticklabels([monthly.index[i] for i in range(0, len(monthly), step)], rotation=45, fontsize=8)
    ax.set_title('Динамика отзывов по месяцам', fontweight='bold'); ax.set_ylabel('Количество')

    # 3. Матрица ошибок
    ax = axes[0, 2]
    ax.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_title('Матрица ошибок (LogReg)', fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Негатив', 'Позитив']); ax.set_yticklabels(['Негатив', 'Позитив'])

    # 4. Облако слов
    ax = axes[1, 0]
    all_text = ' '.join(df['reviewText'].dropna().values)
    wc = WordCloud(width=400, height=300, background_color='white', max_words=100, colormap='viridis')
    wc.generate(all_text)
    ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
    ax.set_title('Облако слов', fontweight='bold')

    # 5. Топ товаров по позитиву
    ax = axes[1, 1]
    top_products = top_products_by_positive_share(df, n=8)
    if not top_products.empty:
        labels = [pid[:25] for pid in top_products['product'].values]
        ax.barh(labels, top_products['positive_share'].values, color='#2ecc71', edgecolor='white')
        ax.set_title('Топ товаров по % позитива', fontweight='bold')
        ax.set_xlabel('% позитивных отзывов'); ax.invert_yaxis()

    # 6. Топ слов в негативных отзывах
    ax = axes[1, 2]
    neg_words = top_negative_words(df, n=10)
    if neg_words:
        words, counts = zip(*neg_words)
        ax.barh(list(words), list(counts), color='#e74c3c', edgecolor='white')
        ax.set_title('Частые слова в негативных отзывах', fontweight='bold')
        ax.set_xlabel('Частота'); ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(BASE_DIR, 'student_report.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Отчёт сохранён: {output_path}")
    return output_path


if __name__ == '__main__':
    generate_report()
