import numpy as np

class NeuroNetwork:
    def __init__(self, initial, reference, learning_rate = 0.9) -> None:
        self.initial = np.array(initial)
        self.reference = np.array(reference)
        self.learning_rate = learning_rate
        self.history = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def init_weight(self, seed = 0):
        # Ініціалізація ваг та зміщення
        np.random.seed(seed) # базове значення для випадкових чисел = 0
        self.weights0 = 2 * np.random.random((2, 2)) - 1
        self.weights1 = 2 * np.random.random((2, 1)) - 1



    def learn(self, iterations):
        self.init_weight()
        for _ in range(iterations):
            # Пряме поширення (обчислення прогнозувань)
            self.hidden_layer = self.sigmoid(np.dot(self.initial, self.weights0))
            self.result = self.sigmoid(np.dot(self.hidden_layer, self.weights1))

            # Обчислення помилки
            error_result = self.reference - self.result
            error_hidden = np.dot(error_result, self.weights1.T)


            # Зворотнє поширення (оновлення ваг)
            d_result = error_result * self.sigmoid_derivative(self.result)
            d_hidden = error_hidden * self.sigmoid_derivative(self.hidden_layer)

            self.weights1 += np.dot(self.hidden_layer.T, d_result) * self.learning_rate
            self.weights0 += np.dot(self.initial.T, d_hidden) * self.learning_rate

            # Додаємо помилку до історії для подальшого аналізу
            self.history.append(np.mean(abs(error_result)))


    def show_predication(self):
        print("Результат навчання:\n", self.result)

    def check_network(self):
        testArray = [
            int(input('Введіть значення:')),
            int(input('Введіть значення:'))]

        # Приклади прогнозувань
        new_input = np.array(testArray)
        hidden_layer = self.sigmoid(np.dot(new_input, self.weights0))
        prediction = self.sigmoid(np.dot(hidden_layer, self.weights1))
        print(f"Прогноз для {testArray}: {prediction[0]:.{2}}")

    def draw_graph(self):
        import matplotlib.pyplot as plt

        # Графік історії точності
        plt.plot(self.history)
        plt.legend([f'min:{min(self.history)}; max:{max(self.history)}'])
        plt.title('Історія точності')
        plt.xlabel('Покоління')
        plt.ylabel('Помилка')
        plt.show()
