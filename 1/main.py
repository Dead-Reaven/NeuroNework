import numpy as np

# Функция активации (сигмоидная функция)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Входные данные (XOR операция)
data_set = np.array(
  [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ]
)

# Выходные данные (результат XOR)
output_set = np.array(
  [
    0,
    1,
    1,
    0
  ])

# Инициализация весов и смещения
np.random.seed(0)
weights = 2 * np.random.random((2, 1)) - 1
bias = 2 * np.random.random(1) - 1

# Коэффициент обучения
learning_rate = 0.0001

# Для отслеживания точности и ошибки
accuracy = []
loss = []

# Обучение нейросети
for i in range(4):
    # Прямое распространение (вычисление предсказаний)
    input_layer = data_set
    output = sigmoid(np.dot(input_layer, weights) + bias)

    # Вычисление ошибки
    error = output_set.reshape(-1, 1) - output

    # Средняя ошибка (loss)
    mean_loss = np.mean(np.abs(error))
    loss.append(mean_loss)

    # Точность (accuracy)
    correct_predictions = np.sum(np.round(output) == output_set)
    acc = correct_predictions / len(output_set)
    accuracy.append(acc)

    # Обратное распространение (обновление весов)
    d_output = error * (output * (1 - output))
    weights += np.dot(input_layer.T, d_output) * learning_rate
    bias += np.sum(d_output) * learning_rate



# Вывод предсказаний
print("Предсказания после обучения:")
print(output)

val1 = input('enter value:')
val2 = input('enter value:')
testArray = [int(val1),int(val2)]

# Примеры предсказаний
new_input = np.array(testArray)  # Пример входных данных для предсказания
prediction = sigmoid(np.dot(new_input, weights) + bias)
print(f"Предсказание для {testArray}: {prediction[0]}")









import matplotlib.pyplot as plt

# Визуализация точности и ошибки
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(accuracy)
plt.title("Точность обучения")
plt.xlabel("Итерации")
plt.ylabel("Точность")

plt.subplot(1, 2, 2)
plt.plot(loss)
plt.title("Ошибка обучения")
plt.xlabel("Итерации")
plt.ylabel("Ошибка")

plt.show()
