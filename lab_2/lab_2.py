import pandas as pd
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

# Функція для виведення графіків вхідних даних
def plot(df, df_1, title, label, label_1, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    if label != label_1:
        plt.plot(df_1, label=label_1)
    plt.plot(df, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()

# Функція для виведення графіків приналежності
def plot_data(title):
    plt.title(title)
    plt.grid(True)
    plt.show()

# Функція z
def z_func(x, y):
    return np.sin(np.abs(x)) * np.sin(x + y)

# Функція розбиття змінних на інтервали
def calculate_intervals(values, n_intervals):
    start = min(values)
    end = max(values)
    intervals = []
    val = np.linspace(start, end, n_intervals + 1)
    for i in range(n_intervals):
        interval_start = val[i]
        interval_end = val[i + 1]
        ser = (interval_end + interval_start) / 2
        intervals.append([interval_start, ser, interval_end])
    diff = (intervals[0][1] - intervals[0][0]) / 2
    return intervals, diff

# Функція для виведення проміжків
def print_intervals(intervals, text):
    print('\nМежі функцій приналежності ' + str(text) + ':')
    for koord in intervals:
        print(koord)

# Функція створення функцій приналежності Гауса
def create_gauss_mf(variable, intervals, names):
    for i, (interval, name) in enumerate(zip(intervals, names)):
        universe = variable.universe
        variable[name] = fuzz.gaussmf(universe, interval[1], 0.1)

# Функція створення функцій приналежності Узагальненого дзвону
def create_gbell(variable, intervals, names):
    for i, (interval, name) in enumerate(zip(intervals, names)):
        universe = variable.universe
        variable[name] = fuzz.gbellmf(universe, 0.1, 5, interval[1])

# Функція для обчислення коефіцієнта детермінації R^2
def mae_r2_score(actual, predicted):
    # Розрахунок MAE
    mae = np.mean(np.abs(actual - predicted))
    # Розрахунок R^2
    mean_actual = np.mean(actual)
    sst = np.sum((actual - mean_actual) ** 2)
    ssr = np.sum((actual - predicted) ** 2)
    r2 = 1 - (ssr / sst)
    print('\nЯкість симуляції:')
    print('Середньо-абсолютна помилка =', mae)
    print('Коефіцієнт детермінації =', r2)

names = ['Гауса', 'Узагальнений дзвін']

# Вибір режиму створення функцій приналежності
print('Оберіть функцію приналежності:')
for i in range(len(names)):
    print(i + 1, '-', names[i])
data_mode = int(input('mode:'))

# Діапазон значень x
x_values = np.arange(0, 6, 0.1)
# Значення y
y_values = 0.2 * np.sin(3 * x_values) * x_values**2
# Значення z
z_values = np.sin(np.abs(x_values)) * np.sin(x_values + y_values)

# Графік функції у
plot(y_values, y_values, 'Графік функції y', 'y = 0.2 * sin(3x) * x^2', 'y = 0.2 * sin(3x) * x^2', 'x', 'y')
# Графік функції z
plot(z_values, z_values, 'Графік функції z', 'z = sin(|x|) * sin(x + y)', 'z = sin(|x|) * sin(x + y)', 'x', 'z')

# Створення консеквентів
x = ctrl.Antecedent(x_values, 'x')
y = ctrl.Antecedent(np.arange(min(y_values), max(y_values), 0.1), 'y')
z = ctrl.Consequent(np.arange(min(z_values), max(z_values), 0.1), 'z')

# Розбиваємо x на проміжки та зберігаємо межі
x_intervals, x_diff = calculate_intervals(x_values, 6)
print_intervals(x_intervals, 'x')
# Розбиваємо y на проміжки та зберігаємо межі
y_intervals, y_diff = calculate_intervals(y_values, 6)
print_intervals(y_intervals, 'y')
# Розбиваємо z на проміжки та зберігаємо межі
z_intervals, z_diff = calculate_intervals(z_values, 9)
print_intervals(z_intervals, 'z')

# Якщо Гаус
if data_mode == 1:
    # Створення функцій приналежності Гауса для x
    x_intervals, x_diff = calculate_intervals(x_values, 6)
    x_names = ['mx1', 'mx2', 'mx3', 'mx4', 'mx5', 'mx6']
    create_gauss_mf(x, x_intervals, x_names)
    # Створення функцій приналежності Гауса для y
    y_intervals, y_diff = calculate_intervals(y_values, 6)
    y_names = ['my1', 'my2', 'my3', 'my4', 'my5', 'my6']
    create_gauss_mf(y, y_intervals, y_names)
    # Створення функцій приналежності Гауса для z
    z_intervals, z_diff = calculate_intervals(z_values, 9)
    z_names = ['mf1', 'mf2', 'mf3', 'mf4', 'mf5', 'mf6', 'mf7', 'mf8', 'mf9']
    create_gauss_mf(z, z_intervals, z_names)
# Якщо Дзвін
else:
    # Створення функцій приналежності Узагальнений дзвін для x
    x_intervals, x_diff = calculate_intervals(x_values, 6)
    x_names = ['mx1', 'mx2', 'mx3', 'mx4', 'mx5', 'mx6']
    create_gbell(x, x_intervals, x_names)
    # Створення функцій приналежності Узагальнений дзвін для y
    y_intervals, y_diff = calculate_intervals(y_values, 6)
    y_names = ['my1', 'my2', 'my3', 'my4', 'my5', 'my6']
    create_gbell(y, y_intervals, y_names)
    # Створення функцій приналежності Узагальнений дзвін для z
    z_intervals, z_diff = calculate_intervals(z_values, 9)
    z_names = ['mf1', 'mf2', 'mf3', 'mf4', 'mf5', 'mf6', 'mf7', 'mf8', 'mf9']
    create_gbell(z, z_intervals, z_names)

# Таблиця приналежності

# Створення масивів назв термінів
labels_x = x.terms.keys()
labels_y = y.terms.keys()
# Створення простору універсуму для оцінки приналежності
universe = np.arange(-7, 6, 0.1)
# Таблиця для значень функцій
table_values = pd.DataFrame(index=labels_y, columns=labels_x)
# Таблиця для назв функцій
table_names = pd.DataFrame(index=labels_y, columns=labels_x)

for label_x in labels_x:
    for label_y in labels_y:
        membership_x = fuzz.interp_membership(x.universe, x[label_x].mf, universe)
        membership_y = fuzz.interp_membership(y.universe, y[label_y].mf, universe)
        max_x = universe[np.argmax(membership_x)]
        max_y = universe[np.argmax(membership_y)]
        # Отримання значення функції по максимуму вхідних функцій приналежності
        value_at_maximum = z_func(max_x, max_y)
        if (value_at_maximum > z.universe.max()):
            value_at_maximum = z.universe.max()
        elif (value_at_maximum < z.universe.min()):
            value_at_maximum = z.universe.min()
        # Таблиця значень функції
        table_values.at[label_y, label_x] = value_at_maximum

        # Знаходимо, до якої функції відноситься це значення
        found_membership = False
        i = 0
        for min_range, _, max_range in z_intervals:
            i += 1
            if min_range <= value_at_maximum <= max_range:
                table_names.at[label_y, label_x] = 'mf' + str(i)
                found_membership = True
                break

# Виведення таблиць
print('\nТаблиця значень функцій по максимумам вхідних функцій приналежності:')
print(table_values)
print('\nТаблиця імен функцій:')
print(table_names)

# Графік функцій приналежності x
x.view()
plot_data('Графік функцій приналежності x')
# Графік функцій приналежності y
y.view()
plot_data('Графік функцій приналежності y')
# Графік функцій приналежності z
z.view()
plot_data('Графік функцій приналежності z')

# Створюємо список для правил
rules = []
# Ітеруємо правила через всі комбінації назв колонок та рядків таблиці
print('\nПравила:')
i = 0
for col_name in table_names.columns:
    for row_name in table_names.index:
        i += 1
        label_z = table_names.at[row_name, col_name]  # Отримуємо ім'я відповідної mf з таблиці
        antecedent = (x[col_name] & y[row_name])  # Визначаємо антецедент
        consequent = z[label_z]  # Визначаємо консеквент
        rule = ctrl.Rule(antecedent=antecedent, consequent=consequent)
        rules.append(rule)
        print(str(i) + '. if(x is ' + str(col_name) + ') and (y is ' + str(row_name) + ') then (f is ' + str(label_z) + ')')

# Додаємо всі правила до системи
system = ctrl.ControlSystem(rules)
# Створюємо симуляцію
simulation = ctrl.ControlSystemSimulation(system)

# Зберігаємо результати симуляції
results = []
# Виконуємо симуляцію для кожної комбінації значень x та y
for x_val, y_val in zip(x_values, y_values):
    simulation.input['x'] = x_val
    simulation.input['y'] = y_val
    simulation.compute()
    # Отримуємо вивід z для поточних значень x та y
    output = simulation.output['z']
    results.append(output)

# Аналіз результатів
mae_r2_score(z_values, results)
# Графік результату симуляції z
plot(z_values, results, 'Порівняння графіків функції та симуляції', 'Графік функції z', 'Графік симуляції z', 'x', 'z')