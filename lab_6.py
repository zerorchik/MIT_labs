import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

# Вимкнення варнінгів PyTorch
warnings.filterwarnings("ignore", category=UserWarning)

# Функція завантаження даних
def load_data(file_name, end_date):
    # Завантаження даних з CSV-файлу
    dataset = pd.read_csv(file_name)
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset = dataset[dataset['Date'] <= end_date]

    # Розділення даних на ознаки та мітки
    features = dataset[['BO', 'BH', 'BL']]
    labels = dataset['BC']

    # Конвертація у тензори PyTorch
    X = torch.tensor(features.values).float()
    y = torch.tensor(labels.values).float()

    # Розділення даних на тренувальні та тестувальні набори
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Візуалізація даних на графіках
    plot_data(dataset)

    return X_train, X_test, y_train, y_test

# Функція візуалізації датасету
def plot_data(dataset):
    plt.plot(dataset.index, dataset['BC'])
    plt.title('Курс EUR до USD')
    plt.xlabel('Екземпляр')
    plt.ylabel('Курс')
    plt.grid(True)
    plt.show()

# Клас гібридної нейронної мережі
class HybridNN(nn.Module):
    def __init__(self):
        super(HybridNN, self).__init__()
        self.train_loss = []
        self.test_loss = []
        self.layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    # Функція прямого проходу
    def forward(self, x):
        x = self.layers(x)
        return x

    # Функція тренування моделі
    def fit(self, X_train, y_train, X_test, y_test, epochs, learning_rate):
        criteria = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        history = {'train_loss': [], 'test_loss': []}

        print('\nНавчання нейромережі:')
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            train_loss = self.epoch_train(X_train, y_train, criteria)
            optimizer.step()
            self.eval()

            test_loss = self.epoch_test(X_test, y_test, criteria)

            history['train_loss'].append(train_loss.item())
            history['test_loss'].append(test_loss.item())

            print(f'Epoch {epoch + 1}/{epochs},\tTrain Loss: {train_loss.item()}, \tTest Loss: {test_loss.item()}')

        return self, history

    # Функція оновлення історії втрат
    def update_history(self, train_loss, test_loss):
        self.train_loss.append(train_loss.item())
        self.test_loss.append(test_loss.item())

    # Функція отримання тренувальних втрат
    def epoch_train(self, X_train, y_train, criteria):
        train_output = self(X_train)
        train_loss = criteria(train_output, y_train)
        train_loss.backward()

        return train_loss

    # Функція отримання тестових втрат
    def epoch_test(self, X_test, y_test, criteria):
        with torch.no_grad():
            test_output = self(X_test)
            test_loss = criteria(test_output, y_test)

        return test_loss

    # Функція отримання загальних втрат
    def get_loss(self, X_test, y_test):
        self.eval()
        with torch.no_grad():
            y_pred = self(X_test)
            test_loss = nn.MSELoss()(y_pred, y_test.unsqueeze(1))

        return y_pred, test_loss.item()

# Функція відображення процесу навчання та графіка функції втрат
def plot_training_process(train_losses, test_losses):
    plt.plot(train_losses, label='Тренувальні втрати')
    plt.plot(test_losses, label='Тестові втрати')
    plt.title('Втрати навчання та тестування протягом епох')
    plt.xlabel('Епохи')
    plt.ylabel('Втрати')
    plt.legend()
    plt.show()

# Функція візуалізації реальних та передбачених значень
def plot_pred_compare(y_test, y_pred):
    plt.plot(y_test, label='Реальна ціна')
    plt.plot(y_pred, label='Передбачена ціна')
    plt.title('Прогнозований курс VS Реальний')
    plt.xlabel('Екземпляр')
    plt.ylabel('Курс')
    plt.grid(True)
    plt.show()

# Головні виклики
if __name__ == "__main__":
    file_name = 'eur_usd_hour.csv'
    end_date = '2006-02-01'

    # Завантаження даних

    X_train, X_test, y_train, y_test = load_data(file_name, end_date)

    # Візуалізація вхідних даних

    # Конвертація train_data у DataFrame
    df_X_train = pd.DataFrame(X_train.numpy(), columns=['Open', 'High', 'Low'])
    df_y_train = pd.DataFrame(y_train.numpy(), columns=['Close'])
    # Об'єднання DataFrames
    df_train = pd.concat([df_X_train, df_y_train], axis=1)
    print('\nТренувальні дані:')
    print(df_train)

    # Конвертація test_data у DataFrame
    df_X_test = pd.DataFrame(X_test.numpy(), columns=['Open', 'High', 'Low'])
    df_y_test = pd.DataFrame(y_test.numpy(), columns=['Close'])
    # Об'єднання DataFrames
    df_test = pd.concat([df_X_test, df_y_test], axis=1)
    print('\nТестові дані:')
    print(df_test)

    # Створення і навчання нейромережі

    model = HybridNN()
    print('\nАрхітектура гібридної нейромережі:')
    print(model)

    model, history = model.fit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        learning_rate=0.001,
        epochs=100
    )

    # Візуалізація процесу навчання та графіка функції втрат

    plot_training_process(history['train_loss'], history['test_loss'])

    # Порівняння результатів

    y_pred, _ = model.get_loss(X_test, y_test)
    y_real = y_test.squeeze()
    # Конвертація у DataFrame
    df_real = pd.DataFrame(y_real.numpy(), columns=['Real'])
    df_pred = pd.DataFrame(y_pred.numpy(), columns=['Predicted'])
    # Об'єднання DataFrames
    df_combined = pd.concat([df_real, df_pred], axis=1)

    print('\nРезультати тестування:')
    print(df_combined)

    plot_pred_compare(y_real.numpy(), y_pred.detach().numpy())