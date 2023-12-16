def hebbian_network(letters, expected_result, neurons_number):
    for index in range(neurons_number):
        letters[index] = [1] + letters[index]
    weights = [[0] * len(letters[0]) for _ in range(neurons_number)]
    # Розрахунок вагових коефіцієнтів
    for index_of_letter in range(neurons_number):
        for index_of_neuron in range(neurons_number):
            for index_of_weight in range(len(weights[index_of_neuron])):
                weights[index_of_neuron][index_of_weight] += letters[index_of_letter][index_of_weight] * expected_result[index_of_letter][index_of_neuron]
    # Розрахунок результату розпізнавання
    actual_result = calculate_output(letters, weights, neurons_number)
    # Перевірка виконання правила для завершення
    if actual_result == expected_result:
        return weights
    # Якщо нерозв'язна проблема адаптації ваг
    raise Exception('Виникла нерозв\'язна проблема адаптації ваг зв\'язків нейромережі. Ваги: ' + str(weights))

def calculate_output(letters, weights, neurons_number): 
    actual_result = []
    for index_of_letter in range(len(letters)):
        letter_result = []
        for index_of_neuron in range(neurons_number):
            s = 0
            for index_of_weight in range(len(weights[index_of_neuron])):
                s += weights[index_of_neuron][index_of_weight] * letters[index_of_letter][index_of_weight]
            if s > 0:
                letter_result += [1]
            else:
                letter_result += [-1]
        actual_result += [letter_result]
    return actual_result

# Правильні літери
# Sofiia Pavlova
I_letter = [-1,  1, -1,
            -1,  1, -1,
            -1,  1, -1]
P_letter = [ 1,  1,  1,
             1,  1,  1,
             1, -1, -1]
L_letter = [ 1, -1, -1,
             1, -1, -1,
             1,  1,  1]
O_letter = [ 1,  1,  1,
             1, -1,  1,
             1,  1,  1]

# Очікуваний результат
expected_result = [[ 1, -1, -1, -1],
                   [-1,  1, -1, -1],
                   [-1, -1,  1, -1],
                   [-1, -1, -1,  1]]

# Навчання моделі
train_letters = [I_letter, P_letter, L_letter, O_letter]
number_of_neurons = len(train_letters) 
final_weights = hebbian_network(train_letters, expected_result, number_of_neurons)
print('Натреновані вагові коефіцієнти:')
for weights in final_weights:
    print(weights)

# Тестування моделі
P_mistake = [ 1,  1,  1,
              1, -1, -1,
              1, -1, -1]
O_mistake = [-1,  1,  1,
              1, -1,  1,
              1,  1,  1]

# Правильні дані
letters_no_mistakes = [I_letter, P_letter, L_letter, O_letter]
for index in range(len(letters_no_mistakes)):
    letters_no_mistakes[index] = [1] + letters_no_mistakes[index]
actual_result = calculate_output(letters_no_mistakes, final_weights, number_of_neurons)
print('\nРезультат для "I", "P", "L", "O":')
for res in actual_result:
    print(res)

# Дані з помилками
letters_with_mistakes = [P_mistake, O_mistake]
for index in range(len(letters_with_mistakes)):
    letters_with_mistakes[index] = [1] + letters_with_mistakes[index]
actual_result = calculate_output(letters_with_mistakes, final_weights, number_of_neurons)
print('\nРезультат для "P" та "O" з помилками:')
for res in actual_result:
    print(res)