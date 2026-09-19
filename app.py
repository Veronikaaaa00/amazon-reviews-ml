import os
import io
import uuid
import logging
from datetime import datetime

from flask import (Flask, render_template, request, jsonify, redirect, url_for,
                   flash, make_response, send_file)
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from database import init_db, load_csv_to_db, add_review, get_product_stats, \
    get_all_reviews_df, get_top_products, get_daily_review_counts, get_user_reviews
from ml_models import load_saved_models
from recommendations import get_recommendations
from utils import predict_sentiment, top_negative_words
from generate_report import generate_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = 'ml-reviews-secret-key-2024'

log_path = os.path.join(BASE_DIR, 'app.log')
file_handler = logging.FileHandler(log_path, encoding='utf-8')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
))
app_logger = logging.getLogger('app_logger')
app_logger.setLevel(logging.INFO)
app_logger.addHandler(file_handler)


def log_action(user_id, action, result='OK'):
    app_logger.info(f'user={user_id[:8]} | action={action} | result={result}')


ml_data = {}


def ensure_models():
    global ml_data
    if not ml_data or ml_data.get('best_model') is None:
        ml_data = load_saved_models()


def get_or_create_user_id():
    user_id = request.cookies.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())
    return user_id


@app.before_request
def before_req():
    ensure_models()


@app.route('/')
def index():
    user_id = get_or_create_user_id()
    log_action(user_id, 'view_index')

    df = get_all_reviews_df()
    total_reviews = len(df)
    total_products = df['product_id'].nunique() if not df.empty else 0
    avg_rating = round(df['overall'].mean(), 2) if not df.empty else 0

    best_model_name = ml_data.get('best_model_name', 'N/A')

    recommendations = []
    if not df.empty:
        recommendations = get_recommendations(user_id, df, n_recommendations=5)

    products = get_top_products(30)

    ratings_dist = df['overall'].value_counts().sort_index() if not df.empty else pd.Series()
    ratings_data = {
        'labels': [str(int(x)) for x in ratings_dist.index.tolist()],
        'values': ratings_dist.values.tolist()
    }

    daily = get_daily_review_counts()
    if not daily.empty:
        daily['month'] = daily['date'].dt.to_period('M').astype(str)
        monthly = daily.groupby('month')['reviews_count'].sum().reset_index()
        dynamics_data = {
            'dates': monthly['month'].tolist(),
            'counts': monthly['reviews_count'].tolist()
        }
    else:
        dynamics_data = {'dates': [], 'counts': []}

    resp = make_response(render_template('index.html',
        active_page='index',
        user_id=user_id,
        total_reviews=total_reviews,
        total_products=total_products,
        avg_rating=avg_rating,
        best_model_name=best_model_name or 'N/A',
        recommendations=recommendations,
        products=products,
        ratings_data=ratings_data,
        dynamics_data=dynamics_data,
    ))
    resp.set_cookie('user_id', user_id, max_age=365*24*3600)
    return resp


@app.route('/predict', methods=['POST'])
def predict():
    user_id = get_or_create_user_id()
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({'error': 'Пустой текст'}), 400

    model = ml_data.get('best_model')
    vectorizer = ml_data.get('vectorizer')

    if model is None or vectorizer is None:
        return jsonify({'error': 'Модели не обучены'}), 500

    label, confidence = predict_sentiment(text, model, vectorizer)
    log_action(user_id, 'predict_sentiment', f'{label} ({confidence}%)')

    return jsonify({'sentiment': label, 'confidence': confidence})


@app.route('/submit_review', methods=['POST'])
def submit_review():
    user_id = get_or_create_user_id()
    product_id = request.form.get('product_id', '')
    review_text = request.form.get('review_text', '')
    rating = float(request.form.get('rating', 3))

    if not review_text.strip():
        flash('Введите текст отзыва', 'error')
        return redirect(url_for('index'))

    add_review(user_id, product_id, review_text, rating, reviewer_name='Web User')
    log_action(user_id, f'submit_review product={product_id[:10]} rating={rating}')
    flash('Отзыв успешно добавлен! ✓', 'success')

    resp = make_response(redirect(url_for('index')))
    resp.set_cookie('user_id', user_id, max_age=365*24*3600)
    return resp


@app.route('/statistics')
def statistics():
    user_id = get_or_create_user_id()
    log_action(user_id, 'view_statistics')

    stats = get_product_stats()
    df = get_all_reviews_df()

    avg_positive = 0
    avg_all_rating = 0
    total_reviews = len(df)

    if stats:
        avg_positive = sum(s['positive_share'] for s in stats) / len(stats)
    if not df.empty:
        avg_all_rating = df['overall'].mean()

    neg_words = top_negative_words(df, n=10) if not df.empty else []

    return render_template('statistics.html',
        active_page='statistics',
        stats=stats,
        avg_positive=avg_positive,
        avg_all_rating=avg_all_rating,
        total_reviews=total_reviews,
        neg_words=neg_words,
    )


@app.route('/models')
def models():
    user_id = get_or_create_user_id()
    log_action(user_id, 'view_models')

    results = ml_data.get('comparison_results')

    return render_template('models.html',
        active_page='models',
        results=results,
    )


@app.route('/logs')
def logs():
    user_id = get_or_create_user_id()
    log_action(user_id, 'view_logs')

    lines = []
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            lines = [l.strip() for l in all_lines[-50:]]
            lines.reverse()

    return render_template('logs.html',
        active_page='logs',
        logs=lines,
    )


@app.route('/generate_report')
def report():
    user_id = get_or_create_user_id()
    log_action(user_id, 'generate_report')
    try:
        path = generate_report()
        return send_file(path, mimetype='image/png', as_attachment=True,
                         download_name='student_report.png')
    except Exception as e:
        flash(f'Ошибка генерации отчёта: {e}', 'error')
        return redirect(url_for('models'))


@app.route('/wordcloud')
def wordcloud_image():
    df = get_all_reviews_df()
    texts = df['review_text'].dropna()

    if texts.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.axis('off')
    else:
        all_text = ' '.join(texts.values)
        wc = WordCloud(
            width=800, height=400,
            background_color='#0f0f1a',
            colormap='cool',
            max_words=150,
            contour_width=0,
        )
        wc.generate(all_text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        fig.patch.set_facecolor('#0f0f1a')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


if __name__ == '__main__':
    print("Инициализация базы данных...")
    init_db()
    count = load_csv_to_db()
    print(f"Загружено записей: {count}")

    print("Запуск веб-приложения...")
    app.run(debug=True, host='0.0.0.0', port=5001)