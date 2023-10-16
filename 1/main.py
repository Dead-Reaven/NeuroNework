import numpy as np

# Функция активации (сигмоидная функция)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная сигмоидной функции
def sigmoid_derivative(x):
    return x * (1 - x)

# Входные данные (XOR операция)
# два входных нейронна

"""
  В данной нейросети топология следующая: два входных нейрона,
  один скрытый слой с двумя нейронами и один выходной нейрон
"""
initial_layer = np.array(
  [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ]
)

# Выходные данные (результат XOR)
# виходной нейрон
output_set = np.array(
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
iterations = 10_000

# Обучение нейросети
for i in range(iterations):
    # Прямое распространение (вычисление предсказаний)

    hidden_layer = sigmoid(np.dot(initial_layer, weights0))
    output = sigmoid(np.dot(hidden_layer, weights1))

    # Вычисление ошибки
    error_output = output_set - output
    error_hidden = np.dot(error_output, weights1.T)

    # Обратное распространение (обновление весов)
    d_output = error_output * sigmoid_derivative(output)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer)

    weights1 += np.dot(hidden_layer.T, d_output) * learning_rate
    weights0 += np.dot(initial_layer.T, d_hidden) * learning_rate


# Вывод предсказаний
print("Предсказания после обучения:")
print(output)

testArray = [
  int(input('Введите значение:')),
  int(input('Введите значение:'))]

# Примеры предсказаний
new_input = np.array(testArray)  # Пример входных данных для предсказания
hidden_layer = sigmoid(np.dot(new_input, weights0))
prediction = sigmoid(np.dot(hidden_layer, weights1))
print(f"Предсказание для {testArray}: {prediction[0]:.{2}}")
