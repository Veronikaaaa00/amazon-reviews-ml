# Amazon Reviews AI — ML-продукт для анализа отзывов

Практическая работа №10. Разработка полноценного ML-продукта для анализа отзывов клиентов интернет-магазина.

## Что делает проект

- Классификация тональности отзывов (позитив / негатив) — 3 модели: LogisticRegression, RandomForest, XGBoost
- Защита персональных данных (маскировка имён и телефонов)
- Персонализированные рекомендации (user-based collaborative filtering)
- Веб-интерфейс на Flask с 5 страницами
- Хранение отзывов в SQLite
- Визуализация: графики Plotly, облако слов, ROC-кривые
- Кластеризация отзывов (KMeans, DBSCAN, t-SNE)
- Логирование действий пользователя
- Генерация отчёта `student_report.png`

## Стек

Python 3.10+, Flask, pandas, numpy, scikit-learn, xgboost, plotly, wordcloud, matplotlib, joblib, pytest.

## Структура проекта

```
Prac10/
├── app.py                      # Flask-приложение
├── ml_models.py                # Обучение моделей
├── database.py                 # SQLite
├── recommendations.py          # Рекомендательная система
├── utils.py                    # Защита ПД, метрики, тональность
├── generate_report.py          # Генерация student_report.png
├── test_app.py                 # Unit-тесты (pytest)
├── templates/                  # HTML-шаблоны
├── static/style.css            # Стили
├── Datafiniti_...May19.csv     # Датасет
└── requirements.txt
```

## Установка и запуск

### 1. Клонировать репозиторий

```bash
git clone <ссылка-на-репозиторий>
cd Prac10
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Обучить модели

```bash
python ml_models.py
```

После этого появится папка `saved_models/` с обученными моделями.

### 4. Запустить веб-приложение

```bash
python app.py
```

Открыть в браузере: **http://localhost:5001**

### 5. (Опционально) Запустить тесты

```bash
pytest test_app.py -v
```

### 6. (Опционально) Сгенерировать отчёт

```bash
python generate_report.py
```

Файл `student_report.png` появится в корне проекта.

## Датасет

Datafiniti Amazon Consumer Reviews of Amazon Products (May 2019) — 28 332 отзыва на товары Amazon.

**Целевая переменная:** `sentiment` — 1 (оценки 4–5), 0 (оценки 1–2).

## Автор

Тепловодская Вероника