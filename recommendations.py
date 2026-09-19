import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


#Единая таблица чисел
def build_user_item_matrix(df: pd.DataFrame) -> tuple:
    #Сводоная таблица
    pivot = df.pivot_table(
        index='reviewer_id',
        columns='product_id',
        values='overall',
        aggfunc='mean'
    ).fillna(0)

    return pivot.values, pivot.index.tolist(), pivot.columns.tolist()


def get_recommendations(user_id: str, df: pd.DataFrame, n_recommendations: int = 5,
                         n_neighbors: int = 10) -> list:
    if df.empty or len(df) < 10:
        return []

    # Строим user-item матрицу
    matrix, user_ids, product_ids = build_user_item_matrix(df)

    if user_id not in user_ids:
        return get_popular_products(df, n_recommendations)

    user_idx = user_ids.index(user_id)
    user_vector = matrix[user_idx].reshape(1, -1) #Вектор оценок этого пользователя

    # Косинусное сходство с другими пользователями
    similarities = cosine_similarity(user_vector, matrix)[0] #Угол между векторами пол. и других пол.
    similarities[user_idx] = 0

    # Находим k ближайших соседей
    neighbor_indices = np.argsort(similarities)[::-1][:n_neighbors]
    neighbor_sims = similarities[neighbor_indices]

    # Товары, которые пользователь уже оценил
    rated_mask = matrix[user_idx] > 0

    #Рейтинг для неоценённых товаров
    predictions = []
    for item_idx in range(len(product_ids)):
        if rated_mask[item_idx]:
            continue 

        # Взвешенное среднее рейтингов соседей
        numerator = 0
        denominator = 0
        for n_idx, sim in zip(neighbor_indices, neighbor_sims):
            if matrix[n_idx, item_idx] > 0 and sim > 0:
                numerator += sim * matrix[n_idx, item_idx]
                denominator += abs(sim)

        if denominator > 0:
            pred_rating = numerator / denominator
            predictions.append((product_ids[item_idx], pred_rating))

    # Сортируем по предсказанному рейтингу
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Получаем имена товаров
    product_names = get_product_name_map(df)

    result = []
    for product_id, pred_rating in predictions[:n_recommendations]:
        result.append({
            'product_id': product_id,
            'product_name': product_names.get(product_id, product_id),
            'predicted_rating': round(pred_rating, 2)
        })

    return result if result else get_popular_products(df, n_recommendations)


def get_popular_products(df: pd.DataFrame, n: int = 5) -> list:
    product_names = get_product_name_map(df) #Возвращаем популярные товары если нет соседей

    popular = df.groupby('product_id').agg(
        avg_rating=('overall', 'mean'),
        count=('overall', 'count')
    ).reset_index()

    popular = popular[popular['count'] >= 3]
    popular = popular.sort_values(['avg_rating', 'count'], ascending=[False, False]).head(n)

    result = []
    for _, row in popular.iterrows():
        result.append({
            'product_id': row['product_id'],
            'product_name': product_names.get(row['product_id'], row['product_id']),
            'predicted_rating': round(row['avg_rating'], 2)
        })

    return result


def get_product_name_map(df: pd.DataFrame) -> dict:
    names = {}
    for _, row in df.drop_duplicates('product_id').iterrows():
        pid = row['product_id']
        names[pid] = str(pid)
    return names
