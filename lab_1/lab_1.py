import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Діапазон значень x
x = np.linspace(0, 10, 1000)

# Трикутна функція приналежності
trimf_values = fuzz.trimf(x, [2, 5, 6])
# Графік
plt.figure(figsize=(8, 6))
plt.plot(x, trimf_values, label='(2, 5, 6)')
plt.xlabel('Значення')
plt.ylabel('Приналежність')
plt.legend()
plt.grid(True)
plt.title('Графік трикутної функції приналежності')
plt.show()

# Трапецієподібна функція приналежності
trapmf_values = fuzz.trapmf(x, [3, 4, 7, 9])
# Графік
plt.figure(figsize=(8, 6))
plt.plot(x, trapmf_values, label='(3, 4, 7, 9)')
plt.xlabel('Значення')
plt.ylabel('Приналежність')
plt.legend()
plt.grid(True)
plt.title('Графік трапецієподібної функції приналежності')
plt.show()

# Проста функція приналежності Гаусса
gaussmf_values = fuzz.gaussmf(x, 6, 1)
# Графік
plt.figure(figsize=(8, 6))
plt.plot(x, gaussmf_values, label='(6, 1)')
plt.xlabel('Значення')
plt.ylabel('Приналежність')
plt.legend()
plt.grid(True)
plt.title('Графік простої Гауссівської ФП')
plt.show()

# Двостороння функція приналежності Гаусса
gauss2mf_values = fuzz.gauss2mf(x, 5, 1.5, 6, 1)
gaussmf_1_values = fuzz.gaussmf(x, 5, 1.5)
# Графік
plt.figure(figsize=(8, 6))
plt.plot(x, gauss2mf_values, label='(5, 1.5, 6, 1)')
plt.plot(x, gaussmf_values, linestyle='dotted', label='(6, 1)')
plt.plot(x, gaussmf_1_values, linestyle='dotted', label='(5, 1.5)')
plt.xlabel('Значення')
plt.ylabel('Приналежність')
plt.legend()
plt.grid(True)
plt.title('Графік двосторонньої Гауссівської ФП')
plt.show()

# Функція приналежності "Узагальнений дзвін"
gbellmf_values = fuzz.gbellmf(x, 2, 3, 5)
# Графік
plt.figure(figsize=(8, 6))
plt.plot(x, gbellmf_values, label='(2, 3, 5)')
plt.xlabel('Значення')
plt.ylabel('Приналежність')
plt.legend()
plt.grid(True)
plt.title('Графік функції приналежності "Узагальнений дзвін"')
plt.show()

# Основна одностороння сигмоїдна функція приналежності
sigmf_values = fuzz.sigmf(x, 2, 4)
# Графік
plt.figure(figsize=(8, 6))
plt.plot(x, sigmf_values, label='(2, 4)')
plt.xlabel('Значення')
plt.ylabel('Приналежність')
plt.legend()
plt.grid(True)
plt.title('Графік основної односторонньої сигмоїдної ФП')
plt.show()

# Додаткова двостороння сигмоїдна функція приналежності
dsigmf_values = fuzz.dsigmf(x, 2, 4, 6, 2)
sigmf_1_values = fuzz.sigmf(x, 6, 2)
# Графік
plt.figure(figsize=(8, 6))
plt.plot(x, dsigmf_values, label='(2, 4, 6, 2)')
plt.plot(x, sigmf_values, linestyle='dotted', label='(2, 4)')
plt.plot(x, sigmf_1_values, linestyle='dotted', label='(6, 2)')
plt.xlabel('Значення')
plt.ylabel('Приналежність')
plt.legend()
plt.grid(True)
plt.title('Графік додаткової двосторонньої сигмоїдної ФП')
plt.show()

# Додаткова несиметрична сигмоїдна функція приналежності
psigmf_values = fuzz.psigmf(x, 2, 4, 6, 2)
# Графік
plt.figure(figsize=(8, 6))
plt.plot(x, psigmf_values, label='(2, 4, 6, 2)')
plt.plot(x, sigmf_values, linestyle='dotted', label='(2, 4)')
plt.plot(x, sigmf_1_values, linestyle='dotted', label='(6, 2)')
plt.xlabel('Значення')
plt.ylabel('Приналежність')
plt.legend()
plt.grid(True)
plt.title('Графік додаткової несиметричної сигмоїдної ФП')
plt.show()

# Поліноміальна Z-функція приналежності
zmf_values = fuzz.zmf(x, 2, 4)
# Графік
plt.figure(figsize=(8, 6))
plt.plot(x, zmf_values, label='(2, 4)')
plt.xlabel('Значення')
plt.ylabel('Приналежність')
plt.legend()
plt.grid(True)
plt.title('Графік поліноміальної Z-функції приналежності')
plt.show()

# Поліноміальна S-функція приналежності
smf_values = fuzz.smf(x, 5, 8)
# Графік
plt.figure(figsize=(8, 6))
plt.plot(x, smf_values, label='(5, 8)')
plt.xlabel('Значення')
plt.ylabel('Приналежність')
plt.legend()
plt.grid(True)
plt.title('Графік поліноміальної S-функції приналежності')
plt.show()

# Поліноміальна PI-функція приналежності
pimf_values = fuzz.pimf(x, 2, 4, 5, 8)
# Графік
plt.figure(figsize=(8, 6))
plt.plot(x, pimf_values, label='(2, 4, 5, 8)')
plt.plot(x, zmf_values, linestyle='dotted', label='(2, 4)')
plt.plot(x, smf_values, linestyle='dotted', label='(5, 8)')
plt.xlabel('Значення')
plt.ylabel('Приналежність')
plt.legend()
plt.grid(True)
plt.title('Графік поліноміальної PI-функції приналежності')
plt.show()

# Мінімаксна інтерпретація логічного оператора AND (кон'юнкція)
min_values = np.fmin(gaussmf_values, gaussmf_1_values)
# Графік
plt.figure(figsize=(8, 6))
plt.plot(x, min_values, label='Мінімум')
plt.plot(x, gaussmf_values, linestyle='dotted', label='(6, 1)')
plt.plot(x, gaussmf_1_values, linestyle='dotted', label='(5, 1.5)')
plt.xlabel('Значення')
plt.ylabel('Приналежність')
plt.legend()
plt.grid(True)
plt.title('Графік мінімаксної інтерпретації логічного оператора AND')
plt.show()

# Мінімаксна інтерпретація логічного оператора OR (диз'юнкція)
max_values = np.fmax(gaussmf_values, gaussmf_1_values)
# Графік
plt.figure(figsize=(8, 6))
plt.plot(x, max_values, label='Максимум')
plt.plot(x, gaussmf_values, linestyle='dotted', label='(6, 1)')
plt.plot(x, gaussmf_1_values, linestyle='dotted', label='(5, 1.5)')
plt.xlabel('Значення')
plt.ylabel('Приналежність')
plt.legend()
plt.grid(True)
plt.title('Графік мінімаксної інтерпретації логічного оператора OR')
plt.show()

# Імовірнісна інтерпретація логічного оператора AND (кон'юнкція)
i_min_values = gaussmf_values * gaussmf_1_values
# Графік
plt.figure(figsize=(8, 6))
plt.plot(x, i_min_values, label='Мінімум')
plt.plot(x, gaussmf_values, linestyle='dotted', label='(6, 1)')
plt.plot(x, gaussmf_1_values, linestyle='dotted', label='(5, 1.5)')
plt.xlabel('Значення')
plt.ylabel('Приналежність')
plt.legend()
plt.grid(True)
plt.title('Графік імовірнісної інтерпретації логічного оператора AND')
plt.show()

# Імовірнісна інтерпретація логічного оператора OR (диз'юнкція)
i_max_values = gaussmf_values + gaussmf_1_values - i_min_values
# Графік
plt.figure(figsize=(8, 6))
plt.plot(x, i_max_values, label='Максимум')
plt.plot(x, gaussmf_values, linestyle='dotted', label='(6, 1)')
plt.plot(x, gaussmf_1_values, linestyle='dotted', label='(5, 1.5)')
plt.xlabel('Значення')
plt.ylabel('Приналежність')
plt.legend()
plt.grid(True)
plt.title('Графік імовірнісної інтерпретації логічного оператора OR')
plt.show()

# Доповнення нечіткої множини, інтерпретація логічного оператора NOT (заперечення)
not_values = 1 - gaussmf_values
# Графік
plt.figure(figsize=(8, 6))
plt.plot(x, not_values, label='Доповнення')
plt.plot(x, gaussmf_values, linestyle='dotted', label='(6, 1)')
plt.xlabel('Значення')
plt.ylabel('Приналежність')
plt.legend()
plt.grid(True)
plt.title('Графік інтерпретації логічного оператора NOT')
plt.show()