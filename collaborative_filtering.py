import pandas as pd, numpy as np, logging
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import time, timeit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logging.basicConfig(filename='collaborative_filtering.log', level=logging.INFO)

class ClusterCollaborativeFiltering:
    '''
        Модель коллаборативной фильтрации на основе кластеризации пользователей:

        Параметры конструктора:
           - train_data (scipy.sparse.csr_matrix): Разреженная матрица взаимодействия пользователь-объект.
           - k_clusters (int): Количество кластеров для кластеризации пользователей.
           - k_neighbors (int): Количество рассматриваемых ближайших соседей.
           - batch_size (int, по умолчанию 100): Размер партии данных для Mini-Batch KMeans.

        Методы:
           __init__(self, train_data, k_clusters, k_neighbors, batch_size=100):
               Инициализирует модель коллаборативной фильтрации.
               Обучает KMeans для кластеризации пользователей и подгоняет модели NearestNeighbors для каждого кластера.

           fit(self, X, y=None):
               Обучает модель, обновляя модели NearestNeighbors для каждого кластера на основе обучающих данных.
               Возвращает обученную модель.

           find_max_liked_users(self, user_id):
               Находит пользователей с наибольшей вероятностью понравиться указанному пользователю в том же кластере.
               Возвращает список рекомендованных пользователей.

           find_potential_matches(self, user_id):
               Находит потенциальные совпадения для указанного пользователя на основе ближайших соседей.
               Возвращает список потенциальных совпадений.

            В приведенном коде использованы блоки try-except, чтобы перехватывать и обрабатывать возможные исключения при 
        инициализации объекта, обучении модели, поиске пользователей и при запуске основной части программы. 
        Это поможет улучшить надежность и устойчивость нашей программы, а также предостеречь от внезапных сбоев
    '''

        
    def __init__(self, train_data, k_clusters, k_neighbors, batch_size=100):
        
        '''
            Модель кластеризации пользователей базируется на их поведении в датасете взаимодействий. 
            Алгоритм KMeans пытается разделить пользователей на кластеры (группы) на основе сходства их взаимодействий с объектами. 
            Подобные пользователи (например, с похожими предпочтениями или поведением) оказываются в одном кластере.

            Кластеризация проводится на основе разреженной матрицы взаимодействий пользователь-объект. 
            Какой пользователь какие объекты оценил, как оценил (лайк/дизлайк) - все это используется 
            для определения сходства пользователей и формирования кластеров.
            
            Таким образом, модель кластеризации пользователей  работает на 
            основе их взаимодействий с объектами и создает группы пользователей с похожими паттернами взаимодействия.
        '''
        try:
            #Устанавливает количество кластеров, которые будут использоваться для кластеризации пользователей.
            self.k_clusters = k_clusters
    
            #Устанавливает количество ближайших соседей, которые будут рассматриваться при поиске подходящих пользователей.
            self.k_neighbors = k_neighbors
    
            #Сохраняет разреженную матрицу взаимодействий между пользователями и объектами.
            self.train_data = train_data
    
            #Создает объект KMeans с заданным количеством кластеров и устанавливает начальное состояние генератора случайных чисел.
            self.kmeans = KMeans(n_clusters=self.k_clusters, random_state=0)
    
            #Производит кластеризацию пользователей на k_clusters кластеров и сохраняет присвоенные кластеры.
            self.cluster_assignments = self.kmeans.fit_predict(train_data)
    
            #Создает словарь, где каждому кластеру присваивается объект NearestNeighbors для поиска ближайших соседей.
            self.cluster_models = {cluster_id: NearestNeighbors(n_neighbors=self.k_neighbors+1, algorithm='brute', n_jobs=-1)
                                   for cluster_id in range(self.k_clusters)}
            #Для каждого кластера выполняется обучение модели NearestNeighbors на данных, принадлежащих кластеру.
            for cluster_id in range(self.k_clusters):
                cluster_data = train_data[self.cluster_assignments == cluster_id]
                self.cluster_models[cluster_id].fit(cluster_data)

        except Exception as e:
            logging.error(f"An error __init__ occurred during initialization: {e}")

    

    def fit(self, X, y=None):
        '''
            Функция для обучения модели на основе новых данных или обновления существующей модели    
        '''
        
        try:
            #Логгирование информационного сообщения о начале обучения модели.
            logging.info('Обучение модели начато...') 
    
            #Для каждого кластера в диапазоне от 0 до self.k_clusters:
            for cluster_id in range(self.k_clusters):
    
                #Выбираются данные кластера из обучающей выборки.
                cluster_data = self.train_data[self.cluster_assignments == cluster_id]
    
                #Модель NearestNeighbors для текущего кластера обучается на этих данных.
                self.cluster_models[cluster_id].fit(cluster_data)
    
            #Логгирование информационного сообщения об окончании обучения модели
            logging.info('Обучение модели завершено.')
    
            #Возвращается обученная модель self
            return self
            
        except Exception as e:
            logging.error(f"An error fit occurred during fitting the model: {e}")


    
    def find_max_liked_users(self, user_id):
        '''
            Функция для поиска пользователей с наивысшей вероятностью понравиться указанному пользователю в том же кластере.
        
            Parameters:
                - user_id: Идентификатор пользователя, для которого ищутся наиболее предпочтительные пользователи в том же кластере.
        
            Returns:
                - liked_users: Список идентификаторов пользователей с наивысшей вероятностью понравиться пользователю
           
        '''
        try:
            #Определение идентификатора кластера, к которому принадлежит указанный пользователь
            cluster_id = self.cluster_assignments[user_id]
    
            #Получение модели ближайших соседей для данного кластера.
            cluster_model = self.cluster_models[cluster_id]
    
            #Поиск ближайших соседей указанного пользователя с помощью модели.
            distances, indices = cluster_model.kneighbors(self.train_data[user_id].reshape(1, -1), n_neighbors=10)
    
            #Фильтрация найденных пользователей: сохранение только тех, кто не является текущим пользователем и имеет вероятность понравиться.
            liked_users = [user for user in indices.flatten()[1:] if user not in self.train_data[user_id].indices and user != 0]
    
            #Логирование информации о найденных пользователях с наивысшей вероятностью понравиться указанному пользователю
            logging.info(f'Найдены пользователи с максимальной вероятностью понравиться для пользователя с ID {user_id}: {liked_users}')
    
            #Возвращение списка liked_users.
            return liked_users

        except Exception as e:
            logging.error(f"An error find_max_liked_users occurred during finding max liked users: {e}")


    
    def find_potential_matches(self, user_id):
        '''
            Функция для поиска потенциальных совпадений для указанного пользователя на основе ближайших соседей в том же кластере.
        
            Parameters:
                - user_id: Идентификатор пользователя, для которого выполняется поиск потенциальных совпадений.
        
            Returns:
                - potential_matches: Список идентификаторов потенциальных совпадений для указанного пользователя.
        '''
        try:
            #Определение идентификатора кластера, к которому принадлежит указанный пользователь
            cluster_id = self.cluster_assignments[user_id]
    
            #Получение модели ближайших соседей для данного кластера
            cluster_model = self.cluster_models[cluster_id]
    
            #Поиск ближайших соседей указанного пользователя с помощью модели
            distances, indices = cluster_model.kneighbors(self.train_data[user_id].reshape(1, -1), n_neighbors=10)
    
            #Фильтрация найденных пользователей: сохранение только тех, для которых нет взаимодействия с указанным 
            # пользователем и они не представляют собой самого пользователя.
            potential_matches = [user for user in indices.flatten()[1:] if user not in self.train_data.indices and self.train_data[user_id, user] == 0 and user != 0]
            
            #Логирование информации о найденных потенциальных совпадениях для указанного пользователя.
            logging.info(f'Найдены потенциальные совпадения для пользователя с ID {user_id}: {potential_matches}')
    
            #Возвращение списка potential_matches
            return potential_matches

        except Exception as e:
            logging.error(f"An error find_potential_matches occurred during finding potential matches: {e}")



