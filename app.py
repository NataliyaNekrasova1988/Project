import pandas as pd, numpy as np, logging
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import time, timeit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import timeit
from collaborative_filtering import ClusterCollaborativeFiltering
   
# Загрузка данных и подготовка:
# Считывание двух CSV файлов: "rec-libimseti-dir.edges" и "info.csv" с помощью Pandas:
df_ratings = pd.read_csv("rec-libimseti-dir.edges", sep='\s+', skiprows=1, names=["id_from", "id_to", "rating"]).fillna(0)
df_info = pd.read_csv('info.csv')
    
# Предобработка данных:
df_info = df_info.rename(columns={' gender': 'gender'})
    
#Преобразование категориальных данных в числовой формат с помощью LabelEncoder:
label_encoder = LabelEncoder()
df_info['gender'] = label_encoder.fit_transform(df_info['gender'])

#объединим два датасета по столбцам 'id_from' и 'id'
df_ratings = pd.merge(df_ratings, df_info, left_on='id_from', right_on='id')
#удалим из обновлённой таблицы лишний столбец 'id':
df_ratings.drop('id', axis=1, inplace=True)
                
# Вычисление рейтинга
# Предполагается, что рейтинг >= 6 означает лайк, иначе - дизлайк:
threshold = 6
    
#Преобразование рейтингов: если рейтинг больше или равен порогу, то рейтинг равен 1, иначе -1:
LIKE = 1
DISLIKE = -1
df_ratings['rating'] = np.where(df_ratings['rating'] >= threshold, LIKE, DISLIKE)

#Преобразуем столбцы
df_ratings['rating'] = df_ratings['rating'].astype(int)
df_ratings['id_from'] = df_ratings['id_from'].astype(int)
df_ratings['id_to'] = df_ratings['id_to'].astype(int)
    
# Создание разреженной матрицы:
user_item_matrix_sparse = coo_matrix((df_ratings['rating'], (df_ratings['id_from'], df_ratings['id_to']))).tocsr()
    
# Использование коллаборативной фильтрации:
user_ids = [98447, 111944, 111930, 82091, 125299, 125298, 17335, 60356]
k_clusters = 8
k_neighbors = 10
    
#Инициализация объекта ClusterCollaborativeFiltering с созданной разреженной матрицей, количеством кластеров и числом соседей:
clustered_collab_filter = ClusterCollaborativeFiltering(user_item_matrix_sparse, k_clusters, k_neighbors)

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/recommended_users/<int:user_id>')
def get_recommended_users(user_id):
    max_liked_users = clustered_collab_filter.find_max_liked_users(user_id)
    return jsonify({'user_id': user_id, 'recommended_users': max_liked_users})

@app.route('/potential_matches/<int:user_id>')
def get_potential_matches(user_id):
    potential_matches = clustered_collab_filter.find_potential_matches(user_id)
    return jsonify({'user_id': user_id, 'potential_matches': potential_matches})

if __name__ == '__main__':
    app.run(debug=True)
