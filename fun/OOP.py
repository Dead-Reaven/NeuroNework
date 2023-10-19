import numpy as np
import matplotlib.pyplot as plt
import utils

class Neuron:
    def __init__(self, input_values: int, learning_rate: int, index: int, layer_name="hidden"):
        self.learning_rate = learning_rate
        self.layer_name = layer_name
        self.index = index
        #! as [[floats]]
        self.weights = np.random.uniform(-0.5, 1.5, (input_values, 1)).T
        self.bias = np.zeros((input_values, 1)).T

    @staticmethod
    def sigmoid(x: float)-> float:
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, image: list[list[float]]):
        # scalar mult img on weight + bias
        # print( "bias:", self.bias)
        # raw_output = self.bias + np.dot(self.weights , image)
        # output = self.sigmoid(raw_output)

        print(f" {self.layer_name}-Neuron-({self.index}) - ",self.weights, image  )

        return 0

    def backward_propagation(self, label: list[float], output: list[float])-> list[list[float]]:
        delta_output = output - label
        return delta_output

    def update_weights_biases(self, delta_output: list[list[float]], image: list[list[float]]) -> None:
        # print("delta - " , delta_output)
        self.weights += -self.learning_rate * np.dot(image, delta_output)
        self.bias += -self.learning_rate * delta_output

class NeuralNetwork:
    def __init__(self, input_neurons, hidden_neurons, output_neurons, learning_rate):
        self.hidden_layer = [Neuron(input_neurons, learning_rate, index, "hidden") for index in range(hidden_neurons)]
        self.output_layer = [Neuron(hidden_neurons, learning_rate,index,  "__output") for index in range(output_neurons)]
    def train(self, images, labels, epochs):
        self.epochs = epochs

        for epoch in range(epochs):
            for image, label in zip(images, labels):
                hidden = [neuron.forward_propagation(image) for neuron in self.hidden_layer][0]
                self.res = [neuron.forward_propagation(hidden) for neuron in self.output_layer][0]





# Load the dataset
# Images in vector format / reference
images_train_set , labels_train_set  = utils.load_dataset()

# Create an instance of the neural network and train it on the training dataset
nn = NeuralNetwork(input_neurons=35, hidden_neurons=2, output_neurons=1, learning_rate=0.3)
nn.train(images_train_set , labels_train_set , epochs=1)
# print(nn.res)





# # Завантаження набору даних
# # зображення в форматі вектора / еталон
# images_train_set , labels_train_set  = utils.load_dataset()

# # Создание экземпляра нейронной сети и обучение её на тренировочном наборе данных
# nn = NeuralNetwork(input_neurons=35 , hidden_neurons=18, output_neurons=1 , learning_rate=0.3)
# nn.train(images_train_set , labels_train_set , epochs=50)

# # Загрузка тестового набора данных и проверка работы нейронной сети на тестовых данных
# test_dataset = utils.load_test_dataset()
# nn.checkNN(test_dataset)

