import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anfis
import membership.membershipfunction
import warnings
# Ігнорувати RuntimeWarning: divide by zero encountered in scalar divide
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Функція завантаження даних
def load_data(file_name, end_date):
    # Завантаження даних з CSV-файлу
    dataset = pd.read_csv(file_name)
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset = dataset[dataset['Date'] <= end_date]

    # Розбиття тестових даних
    X_test = dataset[['BO', 'BH', 'BL']].tail(1).to_numpy()
    y_test = dataset['BC'].tail(1).to_numpy()

    # Розбиття тренувальних даних
    dataset.drop(dataset.index[-1], inplace=True)
    X_train = dataset[['BO', 'BH', 'BL']].to_numpy()
    y_train = dataset['BC'].to_numpy()

    # Візуалізація даних на графіках
    plot_data(dataset)

    return X_train, X_test, y_train, y_test

# Гаусівські функції приналежності
def gaussian_mf(X_train):
    mf = [
        [
            ['gaussmf', {'mean': np.mean(X_train[:, 0] - 1000), 'sigma': np.std(X_train[:, 0])}],
            ['gaussmf', {'mean': np.mean(X_train[:, 0]), 'sigma': np.std(X_train[:, 0])}],
            ['gaussmf', {'mean': np.mean(X_train[:, 0] + 1000), 'sigma': np.std(X_train[:, 0])}],
        ],
        [
            ['gaussmf', {'mean': np.mean(X_train[:, 1] - 1000), 'sigma': np.std(X_train[:, 1])}],
            ['gaussmf', {'mean': np.mean(X_train[:, 1]), 'sigma': np.std(X_train[:, 1])}],
            ['gaussmf', {'mean': np.mean(X_train[:, 1] + 1000), 'sigma': np.std(X_train[:, 1])}],
        ],
        [
            ['gaussmf', {'mean': np.mean(X_train[:, 2] - 1000), 'sigma': np.std(X_train[:, 2])}],
            ['gaussmf', {'mean': np.mean(X_train[:, 2]), 'sigma': np.std(X_train[:, 2])}],
            ['gaussmf', {'mean': np.mean(X_train[:, 2] + 1000), 'sigma': np.std(X_train[:, 2])}],
        ],
    ]

    return mf

# Функція візуалізації датасету
def plot_data(dataset):
    plt.plot(dataset.index, dataset['BC'])
    plt.title('Курс EUR до USD')
    plt.xlabel('Екземпляр')
    plt.ylabel('Курс')
    plt.grid(True)
    plt.show()

# Головні виклики
if __name__ == "__main__":
    file_name = 'eur_usd_hour.csv'
    end_date = '2006-02-01'
    epochs = 5

    # Завантаження даних
    X_train, X_test, y_train, y_test = load_data(file_name, end_date)

    # Візуалізація вхідних даних
    df_X_train = pd.DataFrame(X_train, columns=['Open', 'High', 'Low'])
    df_y_train = pd.DataFrame(y_train, columns=['Close'])
    df_train = pd.concat([df_X_train, df_y_train], axis=1)
    print('\nТренувальні дані:')
    print(df_train)

    df_X_test = pd.DataFrame(X_test, columns=['Open', 'High', 'Low'])
    df_y_test = pd.DataFrame(y_test, columns=['Close'])
    df_test = pd.concat([df_X_test, df_y_test], axis=1)
    print('\nТестові дані:')
    print(df_test)

    # Створення і навчання нечіткої нейромережі ANFIS
    print('\nНавчання:')
    mf = gaussian_mf(X_train)
    mfc = membership.membershipfunction.MemFuncs(mf)
    anf = anfis.ANFIS(X_train, y_train, mfc)
    anf.trainHybridJangOffLine(epochs=5)

    # # Виведення початкової похибки
    # error = np.sum((anf.Y - anf.fittedValues.T) ** 2)
    # print('Початкова похибка:', error)
    #
    # # Тренування моделі
    # for epoch in range(epochs):
    #     anf.trainHybridJangOffLine(epochs=1)
    #     # Виведення похибки на кожній епохі
    #     error = np.sum((anf.Y - anf.fittedValues.T) ** 2)
    #     print(f'Епоха {epoch + 1}, похибка: {error}')

    # Візуалізація графіка функції втрат та результатів навчання
    # Графік втрат
    anf.plotErrors()
    # Результати навчання
    anf.plotResults()

    # Передбачення та порівняння результатів
    y_pred = anfis.predict(anf, X_test)[0][0]
    y_real = y_test[0]
    df_combined = pd.DataFrame(np.column_stack((y_real, y_pred)), columns=['Real', 'Predicted'])

    print('\nРезультати передбачення:')
    print(df_combined)
