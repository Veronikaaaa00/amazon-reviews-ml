import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_curve, auc)

from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
CSV_PATH = os.path.join(BASE_DIR, 'Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv')

# === ОГРАНИЧЕНИЯ ДЛЯ СКОРОСТИ ===
MAX_SAMPLES = 8000          # берём только 8000 отзывов (балансируем классы)
MAX_FEATURES = 3000         # TF-IDF признаков


def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_prepare_data():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['reviews.text'])
    df = df.rename(columns={
        'reviews.text': 'reviewText',
        'reviews.rating': 'overall',
        'reviews.username': 'reviewerID',
        'reviews.title': 'summary',
        'reviews.date': 'reviewDate',
    })

    df_binary = df[df['overall'].isin([1, 2, 4, 5])].copy()
    df_binary['sentiment'] = (df_binary['overall'] >= 4).astype(int)

    # === БАЛАНСИРОВКА КЛАССОВ ===
    # у вас 25545 позитив и 1581 негатив — сильно несбалансировано
    # берём все негативные + столько же позитивных (случайно)
    neg = df_binary[df_binary['sentiment'] == 0]
    pos = df_binary[df_binary['sentiment'] == 1]

    n = min(len(neg), len(pos), MAX_SAMPLES // 2)
    pos_sampled = pos.sample(n=n, random_state=42)
    neg_sampled = neg.sample(n=n, random_state=42)

    df_balanced = pd.concat([pos_sampled, neg_sampled]).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    print(f"   После балансировки: {len(df_balanced)} "
          f"(позитив: {df_balanced['sentiment'].sum()}, "
          f"негатив: {len(df_balanced) - df_balanced['sentiment'].sum()})")

    return df, df_balanced


def train_tfidf_vectorizer(texts, max_features=MAX_FEATURES):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                                  stop_words='english')
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


def get_model_configs():
    """Сетки параметров СИЛЬНО сокращены для скорости."""
    return {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=500, random_state=42),
            'params': {
                'C': [1.0, 10.0],          # было 3 значения
                'solver': ['liblinear'],
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100],   # было [50,100,200]
                'max_depth': [20],           # было [10,20,None]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(eval_metric='logloss', random_state=42,
                                    verbosity=0, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100],   # было [50,100,200]
                'max_depth': [3, 5],         # было [3,5,7]
            }
        },
    }


def train_and_compare_models(X_train, X_test, y_train, y_test):
    configs = get_model_configs()
    results = {}

    for name, config in configs.items():
        print(f"  Обучение {name}...")

        model = config['model']
        model.fit(X_train, y_train)
        y_pred_before = model.predict(X_test)
        y_proba_before = model.predict_proba(X_test)[:, 1]

        metrics_before = {
            'accuracy': round(accuracy_score(y_test, y_pred_before), 4),
            'precision': round(precision_score(y_test, y_pred_before, zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred_before, zero_division=0), 4),
            'f1': round(f1_score(y_test, y_pred_before, zero_division=0), 4),
        }

        print(f"    GridSearchCV для {name}...")
        grid = GridSearchCV(
            config['model'].__class__(**{k: v for k, v in config['model'].get_params().items()
                                          if k in config['model'].__class__().get_params()}),
            config['params'],
            cv=2, scoring='f1', n_jobs=-1, refit=True   # cv=2 вместо 3
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred_after = best_model.predict(X_test)
        y_proba_after = best_model.predict_proba(X_test)[:, 1]

        metrics_after = {
            'accuracy': round(accuracy_score(y_test, y_pred_after), 4),
            'precision': round(precision_score(y_test, y_pred_after, zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred_after, zero_division=0), 4),
            'f1': round(f1_score(y_test, y_pred_after, zero_division=0), 4),
        }

        fpr, tpr, _ = roc_curve(y_test, y_proba_after)
        roc_auc = round(auc(fpr, tpr), 4)

        results[name] = {
            'metrics_before': metrics_before,
            'metrics_after': metrics_after,
            'best_params': grid.best_params_,
            'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc},
            'model': best_model
        }

        print(f"    {name}: F1 {metrics_before['f1']} → {metrics_after['f1']} "
              f"(best: {grid.best_params_})")

    return results


def train_all_and_save():
    ensure_models_dir()

    print("=" * 60)
    print("ОБУЧЕНИЕ ML-МОДЕЛЕЙ")
    print("=" * 60)

    print("\n1. Загрузка данных...")
    df, df_binary = load_and_prepare_data()
    print(f"   Всего отзывов: {len(df)}")

    print("\n2. TF-IDF векторизация...")
    texts = df_binary['reviewText'].values
    y = df_binary['sentiment'].values
    vectorizer, X_vec = train_tfidf_vectorizer(texts)
    print(f"   Признаков: {X_vec.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    print("\n3. Обучение моделей с GridSearchCV...")
    results = train_and_compare_models(X_train, X_test, y_train, y_test)

    best_name = max(results, key=lambda k: results[k]['metrics_after']['f1'])
    best_model = results[best_name]['model']
    print(f"\n   Лучшая модель: {best_name} "
          f"(F1 = {results[best_name]['metrics_after']['f1']})")

    print("\n4. Сохранение моделей и результатов...")
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'vectorizer.pkl'))
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
    joblib.dump(best_name, os.path.join(MODELS_DIR, 'best_model_name.pkl'))

    results_serializable = {}
    for name, res in results.items():
        results_serializable[name] = {
            'metrics_before': res['metrics_before'],
            'metrics_after': res['metrics_after'],
            'best_params': {str(k): str(v) for k, v in res['best_params'].items()},
            'roc': res['roc'],
        }
    joblib.dump(results_serializable, os.path.join(MODELS_DIR, 'comparison_results.pkl'))

    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)

    return vectorizer, best_model, best_name, results_serializable


def load_saved_models():
    data = {}
    files = {
        'vectorizer': 'vectorizer.pkl',
        'best_model': 'best_model.pkl',
        'best_model_name': 'best_model_name.pkl',
        'comparison_results': 'comparison_results.pkl',
    }

    for key, filename in files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            data[key] = joblib.load(path)
        else:
            data[key] = None

    return data


if __name__ == '__main__':
    train_all_and_save()