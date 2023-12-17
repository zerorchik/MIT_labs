import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Клас генетичного алгоритму
class Genetic_Algorithm:
    def __init__(self, function, variables, bounds, max_search=True):
        self.function = function
        self.variables = variables
        self.bounds = bounds
        self.max_search = max_search

    # Створення початкової популяції
    def create_population(self):
        return [[random.uniform(*bound) for bound in self.bounds] for i in range(20)]

    # Обчислення значення функції Y для конкретного індивіда
    def calculate_y(self, individual):
        return self.function(*individual)

    # Вибір індивіда з популяції на основі значень функції
    def select(self, population):
        ys = [self.calculate_y(individual) for individual in population]
        # Якщо максимізація (функція Z)
        if self.max_search:
            return max(zip(population, ys), key=lambda x: x[1])[0]
        # Якщо мінімізація (функція Y)
        else:
            return min(zip(population, ys), key=lambda x: x[1])[0]

    # Операція кросовера для створення нащадка
    def crossover(self, parent_1, parent_2):
        child = []
        # Вибирається випадково половина значень від кожного з батьків
        for i in range(self.variables):
            if random.random() < 0.5:
                child.append(parent_1[i])
            else:
                child.append(parent_2[i])
        return child

    # Операція мутації
    def mutate(self, сhromosom):
        # Змінює значення окремих генів індивіда з невеликою ймовірністю
        for i in range(self.variables):
            if random.random() < 0.1:
                сhromosom[i] = random.uniform(*self.bounds[i])
        return сhromosom

    # Оновлення поколінь
    def update_population(self, current_population, offspring):
        population_size = len(current_population)
        combined_population = current_population + offspring
        sorted_population = sorted(combined_population, key=lambda x: self.calculate_y(x), reverse=self.max_search)
        return sorted_population[:population_size]

    # Графік функції оптимальності алгоритму
    def plot_fitness(self, fitness_history, generations):
        # Якщо максимізація
        if self.max_search == True:
            text = 'Max Z'
            label = 'Z'
        # Якщо мінімізація
        else:
            text = 'Min Y'
            label = 'Y'

        plt.plot(range(generations), fitness_history, label='Генетичний алгоритм')
        plt.title('Функція оптимальності - ' + text)
        plt.xlabel('Покоління')
        plt.ylabel(label)
        plt.legend()
        plt.show()

    # Головний цикл алгоритму
    def run(self, generations, children):
        # Створення популяції
        population = self.create_population()

        # Створення поколінь
        # Значення функції оптимальності для кожного покоління
        fitness_history = []
        for generation in range(generations):
            offspring = []
            for j in range(20):
                # Створення дітей
                child = [None] * children
                for i in range(children):
                    child[i] = self.crossover(random.choice(population), random.choice(population))
                    # Можлива мутація
                    child[i] = self.mutate(child[i])
                    offspring.append(child[i])

            # Оновлення поколінь
            population = self.update_population(population, offspring)
            # Обчислення функції оптимальності
            best_individual = self.select(population)
            fitness_history.append(self.calculate_y(best_individual))

        # Виведення графіка функції оптимальності
        self.plot_fitness(fitness_history, generations)

        return self.select(population)

# Функція Y
def y_func(x):
    return 0.2 * np.sin(3 * x) * x**2

# Функція Z
def z_func(x, y):
    return np.sin(np.abs(x)) * np.sin(x + y)

# Знаходження min для функції Y
def find_min():
    # Вбудований метод
    y_min = np.min(y_values)

    # Використання генетичного алгоритму
    y_genetic = Genetic_Algorithm(y_func, variables=1, bounds=[(-10, 10)], max_search=False)
    y_min_genetic = y_genetic.run(200, 10)

    return y_min, y_func(*y_min_genetic)

# Знаходження max для функції Z
def find_max():
    # Вбудований метод
    z_max = np.max(z_values)

    # Використання генетичного алгоритму
    z_genetic = Genetic_Algorithm(z_func, variables=2, bounds=[(-10, 10), (-10, 10)])
    z_max_genetic = z_genetic.run(50, 10)

    return z_max, z_func(*z_max_genetic)

# Графік функції Y
def plot_2d(x, y_values):
    plt.plot(x, y_values)
    plt.title('Функція Y')
    plt.grid()
    plt.show()

# Графік функції Z
def plot_3d(x_grid, y_grid, z_values):
    ax = plt.subplot(projection='3d')
    ax.plot_surface(x_grid, y_grid, z_values, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('z_func(x, y)')
    ax.view_init(azim=30, elev=10)
    plt.title('Функція Z')
    plt.tight_layout()
    plt.show()

# Головні виклики
if __name__ == "__main__":
    # Діапазон зміни X
    x = np.linspace(-10, 10, 400)
    # Значення функції Y
    y_values = y_func(x)

    # Діапазон зміни Y
    y = np.linspace(-10, 10, 400)
    # Створення простору XYZ
    x_grid, y_grid = np.meshgrid(x, y)
    # Значення функції Z
    z_values = z_func(x_grid, y_grid)

    # Знаходження min та max функцій
    y_min, y_min_genetic = find_min()
    z_max, z_max_genetic = find_max()

    # Створення таблиці порівняння результатів
    data = {
        'Min Y': [y_min_genetic, y_min, abs(y_min_genetic - y_min)],
        'Max Z': [z_max_genetic, z_max, abs(z_max_genetic - z_max)]
    }
    df = pd.DataFrame(data, index=['Генетичний алгоритм', 'Реальне значення', 'Похибка'])
    # Виведення таблиці
    print('\nРезультат алгоритму:')
    print(df)

    # Виведення графіків
    plot_2d(x, y_values)
    plot_3d(x_grid, y_grid, z_values)