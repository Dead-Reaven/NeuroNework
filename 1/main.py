import numpy as np

# Функция активации (сигмоидная функция)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная сигмоидной функции
def sigmoid_derivative(x):
    return x * (1 - x)

# Входные данные (XOR операция)
initial_layer = np.array(
  [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ]
)

# Выходные данные (результат XOR)
reference = np.array(
  [
    [0],
    [1],
    [1],
    [0]
  ])

# Инициализация весов и смещения
np.random.seed(0) # базовое значение для случайных чисел = 0
weights0 = 2 * np.random.random((2, 2)) - 1
weights1 = 2 * np.random.random((2, 1)) - 1

# Коэффициент обучения
learning_rate = 0.9

# Количество итераций обучения
iterations = 19_500

# История ошибок
history = []

# Обучение нейросети
for i in range(iterations):
    # Прямое распространение (вычисление предсказаний)
    hidden_layer = sigmoid(np.dot(initial_layer, weights0))
    output = sigmoid(np.dot(hidden_layer, weights1))

    # Вычисление ошибки
    error_output = reference - output
    error_hidden = np.dot(error_output, weights1.T)


    # Обратное распространение (обновление весов)
    d_output = error_output * sigmoid_derivative(output)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer)

    weights1 += np.dot(hidden_layer.T, d_output) * learning_rate
    weights0 += np.dot(initial_layer.T, d_hidden) * learning_rate

    # Добавляем ошибку в историю для последующего анализа
    history.append(np.mean(abs(error_output)))

# Вывод предсказаний
print("Предсказания после обучения:")
print(output)

# testArray = [
#   int(input('Введите значение:')),
#   int(input('Введите значение:'))]

# # Примеры предсказаний
# new_input = np.array(testArray)
# hidden_layer = sigmoid(np.dot(new_input, weights0))
# prediction = sigmoid(np.dot(hidden_layer, weights1))
# print(f"Предсказание для {testArray}: {prediction[0]:.{2}}")


# График истории точности
import matplotlib.pyplot as plt
plt.plot(history)
plt.legend([f'min:{min(history)}; max:{max(history)}'])
plt.title('История точности')
plt.xlabel('Поколения')
plt.ylabel('error')
plt.show()


