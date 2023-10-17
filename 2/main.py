import numpy as np

# Функція активації (сигмоїдна функція)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Похідна сигмоїдної функції
def sigmoid_derivative(x):
    return x * (1 - x)

# Вхідні дані (операція XOR)
initial_layer = np.array(
  [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ]
)

# Вихідні дані (результат XOR)
reference = np.array(
  [
    [0],
    [1],
    [1],
    [0]
  ])

# Ініціалізація ваг та зміщення
np.random.seed(0) # базове значення для випадкових чисел = 0
weights0 = 2 * np.random.random((2, 2)) - 1
weights1 = 2 * np.random.random((2, 1)) - 1

# Коефіцієнт навчання
learning_rate = 0.9

# Кількість ітерацій навчання
iterations = 19_500

# Історія помилок
history = []

# Навчання нейромережі
for i in range(iterations):
    # Пряме поширення (обчислення прогнозувань)
    hidden_layer = sigmoid(np.dot(initial_layer, weights0))
    output = sigmoid(np.dot(hidden_layer, weights1))

    # Обчислення помилки
    error_output = reference - output
    error_hidden = np.dot(error_output, weights1.T)


    # Зворотнє поширення (оновлення ваг)
    d_output = error_output * sigmoid_derivative(output)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer)

    weights1 += np.dot(hidden_layer.T, d_output) * learning_rate
    weights0 += np.dot(initial_layer.T, d_hidden) * learning_rate

    # Додаємо помилку до історії для подальшого аналізу
    history.append(np.mean(abs(error_output)))

# Виведення прогнозувань
print("Прогнозування після навчання:")
print(output)

testArray = [
  int(input('Введіть значення:')),
  int(input('Введіть значення:'))]

# Приклади прогнозувань
new_input = np.array(testArray)
hidden_layer = sigmoid(np.dot(new_input, weights0))
prediction = sigmoid(np.dot(hidden_layer, weights1))
print(f"Прогноз для {testArray}: {prediction[0]:.{2}}")

import matplotlib.pyplot as plt

# Графік історії точності
plt.plot(history)
plt.legend([f'min:{min(history)}; max:{max(history)}'])
plt.title('Історія точності')
plt.xlabel('Покоління')
plt.ylabel('Помилка')
plt.show()
