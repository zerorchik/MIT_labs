import numpy as np
import torch
from matplotlib import pyplot as plt

# Визначення функцій
def get_y(x):
    return 0.2 * np.sin(3 * x) * x**2

def get_z(x, y):
    return np.sin(np.abs(x)) * np.sin(x + y)

# Вхідні дані
def load_data():
    x_values_train = np.linspace(7.05, 7.65, 80)
    x_values_test = np.linspace(8.15, 8.25, 20)
    y_values_train = get_y(x_values_train)
    y_values_test = get_y(x_values_test)
    z_values_train = get_z(x_values_train, y_values_train)
    z_values_test = get_z(x_values_test, y_values_test)
    return np.vstack((x_values_train, y_values_train)).T, np.vstack((x_values_test, y_values_test)).T, z_values_train, z_values_test

# Навчання моделі
def train_model(num_epochs, inputs, targets, model, optimizer, criterion):
    for epoch in range(num_epochs):
        # Прохід вперед
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # Прохід назад
        loss.backward()
        # Оптимізація вагових коефіцієнтів
        optimizer.step()
        if (epoch + 1) % 1000 == 0:
            print('Епоха [' + str(epoch + 1) + '/' + str(num_epochs) + '], Втрати: ' + str(loss.item()))
    return

# Критерій оцінки якості нейронної мережі - середня відносна помилка моделювання
def eval_model(test_inputs, test_targets, model):
    # Передбачення на тестовому наборі
    predictions = model(test_inputs)
    # Розрахунок Relative Error
    abs_relative_error = torch.abs((test_targets - predictions) / test_targets)
    # Розрахунок Mean Relative Error
    mean_relative_error = torch.mean(abs_relative_error) * 100
    return mean_relative_error

# Графіки даних
def plot_data(data, title):
    plt.plot(data, label=title)
    plt.title('Графік функції Z')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# Графіки функції
def plot(X_test, y_test, y_pred, title):
    plt.plot(X_test, y_test, label='Справжні значення')
    plt.plot(X_test, y_pred, label='Прогнозовані значення')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()