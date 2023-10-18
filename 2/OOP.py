import numpy as np
import matplotlib.pyplot as plt
import utils

class NeuralNetwork:
    def __init__(self, input_neurons, hidden_neurons, output_neurons, learning_rate):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.learning_rate = learning_rate

        self.weight_input_to_hidden = np.random.uniform(-0.5, 1.5, (hidden_neurons, input_neurons))
        self.weight_hidden_to_output = np.random.uniform(-0.5, 1.5, (output_neurons, hidden_neurons))

        self.bias_input_to_hidden = np.zeros((hidden_neurons, 1))
        self.bias_hidden_to_output = np.zeros((output_neurons, output_neurons))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, image):
        hidden_raw = self.bias_input_to_hidden + self.weight_input_to_hidden @ image
        hidden = self.sigmoid(hidden_raw)
        output_raw = self.bias_hidden_to_output + self.weight_hidden_to_output @ hidden
        output = self.sigmoid(output_raw)
        return hidden, output

    def backward_propagation(self, label, hidden, output):
        delta_output = output - label
        delta_hidden = np.transpose(self.weight_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
        return delta_output, delta_hidden

    def update_weights_biases(self, delta_output, delta_hidden, hidden, image):
        self.weight_hidden_to_output += -self.learning_rate * np.dot(delta_output, hidden.T)
        self.bias_hidden_to_output += -self.learning_rate * delta_output
        self.weight_input_to_hidden += -self.learning_rate * np.dot(delta_hidden, image.T)
        self.bias_input_to_hidden += -self.learning_rate * delta_hidden

    def train(self, images, labels, epochs):
        self.epochs = epochs
        self.accuracy_list = []

        for epoch in range(epochs):
            correct_predictions = 0
            for image, label in zip(images, labels):
                image = np.reshape(image, (-1, 1))
                label = np.reshape(label, (-1, 1))

                # Пряме поширення
                hidden, output = self.forward_propagation(image)

                # Зворотнє поширення помилки
                delta_output, delta_hidden = self.backward_propagation(label, hidden, output)

                # Оновлення ваг і зсувів
                self.update_weights_biases(delta_output, delta_hidden, hidden, image)

                # Підрахунок правильних прогнозів
                if np.round(output) == label:
                    correct_predictions += 1

            # Розрахунок точності для поточної епохи та додавання до списку точності
            accuracy = correct_predictions / len(images)
            self.accuracy_list.append(accuracy)


    def checkNN(self, test_dataset):
        stop = False
        while not stop:
            index = input("Введіть номер:")
            test_image = np.reshape(test_dataset[int(index)], (-1,1))

            images, labels, = utils.load_dataset()
            # надає можливість перенавчатися заново кожну ітерацію
            self.__init__(
                self.input_neurons,self.hidden_neurons,self.output_neurons,self.learning_rate ,
            )
            self.train(images, labels, epochs=50)

            # Пряме поширення
            hidden, output = self.forward_propagation(test_image)

            # Прогнозування
            prediction = 'Є коренем числа 2' if np.round(output[0][0]) == 1 else 'Не є коренем числа 2'

            # Створення першого підграфа
            plt.subplot(2, 1, 1) # (кількість рядків, кількість стовпців, індекс підграфа)
            plt.imshow(test_image.reshape(7,5), cmap='gray')
            plt.text(5.5, 1.5, f"Вірогідність: {output[0][0]}", fontsize=12)
            plt.text(5.5, 2.5,f"Відповідь: {prediction}", fontsize=12)

            # Створення другого підграфа
            plt.subplot(2, 1, 2)
            plt.plot(range(self.epochs), self.accuracy_list )
            plt.title('Точність навчання за епохами')
            plt.xlabel('Епохи')
            plt.ylabel('Точність')

            # Відображення обох графіків
            plt.show()


# Завантаження набору даних
# зображення в форматі вектора / еталон
images_train_set , labels_train_set  = utils.load_dataset()

# Создание экземпляра нейронной сети и обучение её на тренировочном наборе данных
nn = NeuralNetwork(input_neurons=35 , hidden_neurons=18, output_neurons=1 , learning_rate=0.3)
nn.train(images_train_set , labels_train_set , epochs=50)

# Загрузка тестового набора данных и проверка работы нейронной сети на тестовых данных
test_dataset = utils.load_test_dataset()
nn.checkNN(test_dataset)

