import numpy as np
import utils

class Neuron:
  def __init__(self, name):
    # Ініціалізуємо стан нейрона значенням 1
    self.state = 1
    self.name = name

  def update(self, weights):
    self.state = np.sign(np.dot(weights, self.state.T))
    # print( f"Neuron:{self.name}, state = {self.state}")

class HopfieldNetwork:
  def __init__(self, size):
    # Створюємо мережу з заданою кількістю нейронів
    self.neurons = [Neuron(i) for i in range(size)]
    # Ініціалізуємо ваги нейронів нулями
    self.weights = np.zeros((size, size))

  def train(self, patterns):
    # print('train called')
    # Тренуємо мережу на заданих шаблонах
    for pattern in patterns:
      # Збільшуємо ваги на зовнішнє добуток шаблону
      self.weights += np.outer(pattern, pattern)

  def recall(self, patterns, steps=35):
    # Використовуємо мережу для відтворення шаблонів
    recalled = []
    # print('recal called')

    for pattern in patterns:
      # Встановлюємо початковий стан кожного нейрона відповідно до шаблону
      for neuron, state in zip(self.neurons, pattern):
        neuron.state = state
      # Оновлюємо стан кожного нейрона задану кількість кроків
      for _ in range(steps):
        for i, neuron in enumerate(self.neurons):
          neuron.update(self.weights[i])
      # Додаємо остаточний стан кожного нейрона до списку вихідних шаблонів
      recalled.append([neuron.state for neuron in self.neurons])
    return recalled

# Визначаємо вхідні шаблони
patterns = np.array([[1,1,1 ,-1], [-1, -1 , -1, 1], ]) #utils.load_dataset()

# Створюємо об'єкт мережі Хопфілда
hopfield_net = HopfieldNetwork(4)

# Тренуємо мережу
hopfield_net.train(patterns)

# Перевіряємо вихідні шаблони
# test_patterns , correct_answer= utils.load_testdataset()
patterns = np.array([[1,1, 0, 0]]) #utils.load_dataset()

output_patterns = hopfield_net.recall(patterns)

print('result:')
for i, p in enumerate(output_patterns):
  print(f"pattern {i}")
  for j, el in enumerate(p):
      print(f"Neuron({j}) = {el}\n")
      pass
