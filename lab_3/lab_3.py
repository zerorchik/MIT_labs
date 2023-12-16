import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Параметри для генерації датасету
n_samples = 500
n_features = 2
n_clusters = 5
names = ['фіксовані центри', 'рандомні центри']
# Вибір режиму зчитування даних
print('Оберіть режим генерації центрів:')
for i in range(len(names)):
    print(i + 1, '-', names[i])
data_mode = int(input('mode:'))
# Якщо джерело даних існує
if data_mode in range(1, len(names) + 1):
    # Фіксовані центри
    if (data_mode == 1):
        centers = [[3, 2], [4, 5], [5, 3], [7, 4], [8, 6]]
        centers = np.array(centers)
    # Рандомні центри
    else:
        centers = np.random.rand(n_clusters, n_features) * 10
random_state = 42

# Параметри для фазифікованої кластеризації
print('Оберіть ступінь фазифікації (2):')
q = float(input('q = '))
print('Оберіть зміну центрів кластерівміж ітераціями (0.00001):')
error = float(input('error = '))
print('Оберіть максимальну кількість ітерацій (100):')
max_iter = int(input('max_iter = '))

# Функція генерації датасету
def create_customers_dataset(n_samples, n_features, centers, random_state):
    dataset, mark = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=random_state)
    return dataset, mark

# Функція фазифікованої кластеризації
def fuzzy_clustering(dataset, n_clusters, q, error, max_iter):
    centers, u, _, _, jm, _, _ = fuzz.cluster.cmeans(dataset.T, n_clusters, q, error, max_iter)
    return centers, u, jm

# Функція виведення центрів кластеризації
def print_centers(centers, title):
    print(title)
    print(centers)

# Функція виведення графіків точок датасету
def plot(dataset, title, clustering):
    # Якщо результати кластеризації
    if (clustering):
        plt.scatter(dataset[:, 0], dataset[:, 1], marker='x', color='black')
    # Якщо згенерований датасет
    if not clustering:
        plt.scatter(dataset[:, 0], dataset[:, 1])
    plt.title(title)
    plt.xlabel('Витрати')
    plt.ylabel('Частота покупок')
    plt.show()

# Функція виведення графіку кластеризації
def plot_clustering(dataset, membership, title):
    max_membership = np.argmax(membership, axis=0)
    # Візуалізація значень в кластерах
    for cluster in range(len(centers)):
        cluster_points = dataset[max_membership == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1])
    # Візуалізація центрів кластерів
    plot(centers, title, True)

# Функція виведення графіку цільової функції
def plot_jm(jm):
    plt.plot(jm)
    plt.title('Цільова функція')
    plt.xlabel('Кількість ітерацій')
    plt.ylabel('Значення цільової функції')
    plt.grid(True)
    plt.show()

# Датасет
customers, mark = create_customers_dataset(n_samples, n_features, centers, random_state)
plot(customers, 'Дані покупців', False)
# Кластеризація
customers_centers, membership, jm = fuzzy_clustering(customers, n_clusters, q, error, max_iter)
plot_clustering(customers, membership, 'Отримані кластери та їх центри')
# Центри
print_centers(centers, '\nЗгенеровані центри')
print_centers(customers_centers, '\nОтримані центри')
# Цільова функція
plot_jm(jm)