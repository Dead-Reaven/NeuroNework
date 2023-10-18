import numpy as np
import matplotlib.pyplot as plt
import utils

# Функція активації
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Пряме поширення для навчання
def forward_propagation(image, weight_input_to_hidden, bias_input_to_hidden, weight_hidden_to_output, bias_hidden_to_output):
    hidden_raw = bias_input_to_hidden + weight_input_to_hidden @ image
    hidden = sigmoid(hidden_raw)
    output_raw = bias_hidden_to_output + weight_hidden_to_output @ hidden
    output = sigmoid(output_raw)
    return hidden, output

# Зворотнє поширення для коригування вагів
def backward_propagation(label, hidden, output, weight_hidden_to_output):
    delta_output = output - label
    delta_hidden = np.transpose(weight_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
    return delta_output, delta_hidden

# Оновлення ваг і зсувів
def update_weights_biases(learning_rate, delta_output, delta_hidden, hidden, image):
    global weight_hidden_to_output, bias_hidden_to_output
    global weight_input_to_hidden, bias_input_to_hidden

    weight_hidden_to_output += -learning_rate * np.dot(delta_output, hidden.T)
    bias_hidden_to_output += -learning_rate * delta_output
    weight_input_to_hidden += -learning_rate * np.dot(delta_hidden, image.T)
    bias_input_to_hidden += -learning_rate * delta_hidden

# Завантаження набору даних
# зображення в форматі вектора / еталон
images, labels = utils.load_dataset()


input_neurons = 35  # initial layer
hidden_neurons = 35 # or 18
output_neurons = 1  # result neuron

epochs = 50 # or 250
learning_rate = 0.3 # or 0.1

neurons_input_to_hidden =  (hidden_neurons, input_neurons)
neurons_hidden_to_output = (output_neurons, hidden_neurons)

# Ініціалізація ваг. генеруємо випадковими числами від -0.5 до 1.5
weight_input_to_hidden = np.random.uniform(-0.5, 1.5, neurons_input_to_hidden )
weight_hidden_to_output = np.random.uniform(-0.5, 1.5, neurons_hidden_to_output )

# нейрони зміщення
bias_input_to_hidden = np.zeros((hidden_neurons,1))
bias_hidden_to_output = np.zeros((output_neurons,output_neurons))

# Створення списку для зберігання точності на кожній епохі
accuracy_list = []

for epoch in range(epochs):
    correct_predictions = 0
    for image, label in zip(images, labels):
        #перетворює зображення/метку в двовимірний масив з одним стовпцем
        image = np.reshape(image, (-1,1))
        label = np.reshape(label, (-1,1))

        # Пряме поширення
        hidden, output = forward_propagation(
          image,
          weight_input_to_hidden,
          bias_input_to_hidden,
          weight_hidden_to_output,
          bias_hidden_to_output
        )

        # Зворотнє поширення помилки
        delta_output, delta_hidden = backward_propagation(
          label,
          hidden,
          output,
          weight_hidden_to_output
        )

        # Оновлення ваг і зсувів
        update_weights_biases(learning_rate, delta_output, delta_hidden, hidden, image)

        # Підрахунок правильних прогнозів
        if np.round(output) == label:
            correct_predictions += 1

    # Розрахунок точності для поточної епохи та додавання до списку точності
    accuracy = correct_predictions / len(images)
    accuracy_list.append(accuracy)

def checkNN():
  # Нові дані для тестування
  stop = False

  test_dataset = utils.load_test_dataset()
  while not stop:

    index = int(input("Введіть номер:"))

    test_image = np.reshape(test_dataset[index], (-1,1))

    # Пряме поширення
    hidden, output = forward_propagation(
      test_image,
      weight_input_to_hidden,
      bias_input_to_hidden,
      weight_hidden_to_output,
      bias_hidden_to_output
    )

    # Прогнозування
    prediction = 'Є коренем числа 2' if np.round(output[0][0]) == 1 else 'Не є коренем числа 2'

    # Створення першого підграфа
    plt.subplot(2, 1, 1) # (кількість рядків, кількість стовпців, індекс підграфа)
    plt.imshow(test_image.reshape(7,5), cmap='gray')
    plt.text(5.5, 1.5, f"Вірогідність: {output[0][0]}", fontsize=12)
    plt.text(5.5, 2.5,f"Відповідь: {prediction}", fontsize=12)

    # Створення другого підграфа
    plt.subplot(2, 1, 2)
    plt.plot(range(epochs), accuracy_list)
    plt.title('Точність навчання за епохами')
    plt.xlabel('Епохи')
    plt.ylabel('Точність')

    # Відображення обох графіків
    plt.show()

checkNN()
