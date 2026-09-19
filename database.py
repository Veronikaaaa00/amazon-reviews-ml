import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reviews.db')
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv')


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # Строки в виде словарей
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reviewer_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            reviewer_name TEXT,
            review_text TEXT,
            overall REAL NOT NULL,
            summary TEXT,
            review_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            asin TEXT PRIMARY KEY,
            name TEXT
        )
    ''')

    conn.commit()
    conn.close()


def load_csv_to_db():
    conn = get_connection()
    cursor = conn.cursor()

    count = cursor.execute('SELECT COUNT(*) FROM reviews').fetchone()[0]
    if count > 0:
        conn.close()
        return count

    df = pd.read_csv(CSV_PATH)
    df = df.fillna({'reviews.username': 'Anonymous', 'reviews.text': '', 'reviews.title': ''})

    product_names = df[['name']].drop_duplicates()
    for _, row in product_names.iterrows():
        pname = str(row['name'])[:150]
        cursor.execute(
            'INSERT OR IGNORE INTO products (asin, name) VALUES (?, ?)',
            (pname, pname)
        )

    for _, row in df.iterrows():
        try:
            review_date = pd.to_datetime(row['reviews.date']).strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            review_date = ''

        reviewer_id = str(row['reviews.username'])
        product_name = str(row['name'])[:150]

        cursor.execute('''
            INSERT INTO reviews (reviewer_id, product_id, reviewer_name, review_text, overall, summary, review_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            reviewer_id,
            product_name,
            reviewer_id,
            str(row['reviews.text']),
            int(row['reviews.rating']),
            str(row['reviews.title']),
            review_date
        ))

    conn.commit()
    total = cursor.execute('SELECT COUNT(*) FROM reviews').fetchone()[0]
    conn.close()
    return total


def add_review(reviewer_id: str, product_id: str, review_text: str, overall: float,
               reviewer_name: str = 'Web User', summary: str = ''):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO reviews (reviewer_id, product_id, reviewer_name, review_text, overall, summary, review_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (reviewer_id, product_id, reviewer_name, review_text, overall, summary,
          datetime.now().strftime('%Y-%m-%d')))
    conn.commit()
    conn.close()


def get_user_reviews(reviewer_id: str) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    rows = cursor.execute(
        'SELECT * FROM reviews WHERE reviewer_id = ? ORDER BY created_at DESC',
        (reviewer_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_product_stats() -> list:
    conn = get_connection()
    cursor = conn.cursor()
    rows = cursor.execute('''
        SELECT
            r.product_id,
            COALESCE(p.name, r.product_id) AS product_name, -- коалесц - если пустое имя, берет айди
            COUNT(*) AS review_count,
            ROUND(AVG(r.overall), 2) AS avg_rating,
            ROUND(100.0 * SUM(CASE WHEN r.overall >= 4 THEN 1 ELSE 0 END) / COUNT(*), 1) AS positive_share
        FROM reviews r
        LEFT JOIN products p ON r.product_id = p.asin
        GROUP BY r.product_id
        HAVING review_count >= 3
        ORDER BY avg_rating DESC
        LIMIT 50
    ''').fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_reviews_df() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query('SELECT * FROM reviews', conn)
    conn.close()
    return df


def get_top_products(n: int = 20) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    rows = cursor.execute('''
        SELECT r.product_id, COALESCE(p.name, r.product_id) AS product_name, COUNT(*) AS cnt
        FROM reviews r
        LEFT JOIN products p ON r.product_id = p.asin
        GROUP BY r.product_id
        ORDER BY cnt DESC
        LIMIT ?
    ''', (n,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_daily_review_counts() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query('''
        SELECT review_date as date, COUNT(*) as reviews_count
        FROM reviews
        WHERE review_date IS NOT NULL AND review_date != ''
        GROUP BY review_date
        ORDER BY review_date
    ''', conn)
    conn.close()
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
    return df
