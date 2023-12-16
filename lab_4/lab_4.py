import torch
import torch.nn as nn
import torch.optim as optim
import lab_4_func

# Вхідні дані
X_train, X_test, y_train, y_test = lab_4_func.load_data()
print('Розмір вхідних даних:')
print('X_train shape:', X_train.shape, 'y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape, 'y_test shape:', y_test.shape)
input_size = X_train.shape[1]
output_size = 1
lab_4_func.plot_data(y_train, 'Тренувальні дані')
lab_4_func.plot_data(y_test, 'Тестувальні дані')

# Нейромережа типу 'Feedforward'
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    # Прохід вперед через мережу
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Нейромережа типу 'Cascadeforward'
class CascadeForwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(CascadeForwardNN, self).__init__()
        self.layers = nn.ModuleList()
        in_features = input_size
        # Каскадне додавання шарів
        for hidden_size in hidden_sizes:
            # Лінійний шар
            self.layers.append(nn.Linear(in_features, hidden_size))
            # Активаційний шар
            self.layers.append(nn.ReLU())
            in_features = hidden_size
        # Вихідний шар
        self.layers.append(nn.Linear(in_features, output_size))
    # Прохід вперед
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Нейромережа типу 'Elman'
class ElmanNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ElmanNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.elman_layers = nn.ModuleList()
        self.elman_layers.append(nn.RNN(input_size, hidden_size, batch_first=True))
        # RNN шари
        for i in range(num_layers - 1):
            self.elman_layers.append(nn.RNN(hidden_size, hidden_size, batch_first=True)) # Additional hidden RNN layers
        # Вихідний шар
        self.fc = nn.Linear(hidden_size, output_size)
    # Прямий прохід
    def forward(self, x):
        batch_size = x.size(0)
        # Приховані стани
        h = [torch.zeros(1, batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        out = x
        # Прямий прохід через шари RNN, оновлення прихованих станів
        for i in range(self.num_layers):
            out, h[i] = self.elman_layers[i](out, h[i])
        out = self.fc(out[:, -1, :])
        return out

# Використання 'Feedforward' з 1 шаром, N нейронами
def try_model_FeedForward(hidden_size):
    # Архітектура нейронної мережі
    print('\n----------------------------------')
    print('Модель - Feedforward')
    print('Шарів - 1')
    print('Нейронів у шарі -', hidden_size)
    print('----------------------------------')
    model = FeedForwardNN(input_size, hidden_size, output_size)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # Навчання моделі
    print('Навчання моделі:')
    num_epochs = 10000
    inputs = torch.Tensor(X_train)
    targets = torch.Tensor(y_train).view(-1, 1)
    lab_4_func.train_model(num_epochs, inputs, targets, model, optimizer, criterion)
    # Оцінка якості моделі
    print('\nОцінка якості моделі:')
    model.eval()
    test_inputs = torch.Tensor(X_test)
    test_targets = torch.Tensor(y_test).view(-1, 1)
    mean_relative_error = lab_4_func.eval_model(test_inputs, test_targets, model).item()
    print('Mean Relative Percentage Error = ' + str(round(mean_relative_error, 2)) + '%')
    # Графіки
    lab_4_func.plot(X_test[:, 0], y_test, model(test_inputs).detach().numpy(), 'Feedforward 1 шар,' + str(hidden_size) + ' нейронів у шарі')

# Використання 'Cascadeforward' з N шарами по N нейронів
def try_model_CascadeForward(hidden_size):
    # Архітектура нейронної мережі
    print('\n----------------------------------')
    print('Модель - Cascadeforward')
    print('Шарів -', len(hidden_size))
    print('Нейронів у шарі -', hidden_size[0])
    print('----------------------------------')
    model = CascadeForwardNN(input_size, hidden_size, output_size)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # Навчання моделі
    print('Навчання моделі:')
    num_epochs = 10000
    inputs = torch.Tensor(X_train)
    targets = torch.Tensor(y_train).view(-1, 1)
    lab_4_func.train_model(num_epochs, inputs, targets, model, optimizer, criterion)
    # Оцінка якості моделі
    print('\nОцінка якості моделі:')
    model.eval()
    test_inputs = torch.Tensor(X_test)
    test_targets = torch.Tensor(y_test).view(-1, 1)
    mean_relative_error = lab_4_func.eval_model(test_inputs, test_targets, model).item()
    print('Mean Relative Percentage Error = ' + str(round(mean_relative_error, 2)) + '%')
    # Графіки
    lab_4_func.plot(X_test[:, 0], y_test, model(test_inputs).detach().numpy(), 'Cascadeforward ' + str(len(hidden_size)) + ' шарів, ' + str(hidden_size[0]) + ' нейронів у шарі')

# Використання 'Elman' з N шарами по N нейронів
def try_model_Elman(num_layers, hidden_size):
    # Архітектура нейронної мережі
    print('\n----------------------------------')
    print('Модель - Elman')
    print('Шарів -', num_layers)
    print('Нейронів у шарі -', hidden_size)
    print('----------------------------------')
    model = ElmanNN(input_size, hidden_size, num_layers, output_size)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # Навчання моделі
    print('Навчання моделі:')
    num_epochs = 10000
    inputs = torch.Tensor(X_train).unsqueeze(1)
    targets = torch.Tensor(y_train).view(-1, 1)
    lab_4_func.train_model(num_epochs, inputs, targets, model, optimizer, criterion)
    # Оцінка якості моделі
    print('\nОцінка якості моделі:')
    model.eval()
    test_inputs = torch.Tensor(X_test).unsqueeze(1)
    test_targets = torch.Tensor(y_test).view(-1, 1)
    mean_relative_error = lab_4_func.eval_model(test_inputs, test_targets, model).item()
    print('Mean Relative Percentage Error = ' + str(round(mean_relative_error, 2)) + '%')
    # Графіки
    lab_4_func.plot(X_test[:, 0], y_test, model(test_inputs).detach().numpy(), 'Elman ' + str(num_layers) + ' шарів, ' + str(hidden_size) + ' нейронів у шарі')

# Навчання та оцінка для моделі з 10 нейронами
try_model_FeedForward(10)
# Навчання та оцінка для моделі з 20 нейронами
try_model_FeedForward(20)

# Навчання та оцінка для моделі з 20 нейронами
try_model_CascadeForward([20])
# Навчання та оцінка для моделі з 2 шарами по 10 нейронів
try_model_CascadeForward([10, 10])

# Навчання та оцінка для моделі з 15 нейронами
try_model_Elman(1, 15)
# Навчання та оцінка для моделі з 3 шарами по 5 нейронів
try_model_Elman(3, 5)