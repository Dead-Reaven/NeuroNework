import numpy as np
import utils

class SOM:
    def __init__(self, weights):
        self.weights = np.array(weights)  # Ініціалізація ваг

    def learn(self, data, epochs, alpha):
        for _ in range(epochs):
            for sample in data:
                # Визначаємо переможний вектор ваги для цього прикладу
                winner = self.winner(sample)
                # Оновлюємо переможний вектор ваги
                self.update(sample, winner, alpha)
        pass

    def winner(self, sample):
        # Розрахунок евклідової відстані між прикладом та кожним вектором ваги
        # print(f"--sample: \n{sample},\n--weights: \n{self.weights}")
        distances = np.linalg.norm(self.weights - sample, axis=1)
        # print("distances:\n", distances)
        # Повертаємо індекс вектора ваги з найменшою відстанню
        return np.argmin(distances)

    def update(self, sample, J, alpha):
        # Оновлюємо переможний вектор ваги
        self.weights[J] += alpha * (sample - self.weights[J])

def main():

    epochs = 35
    alpha = 0.1
    accuracity = 0

    iterations = 1000
    stat = []

    for _ in range(iterations):
        data, weight = utils.load_dataset()
        #ініціалізація мережі та її обучення обучення
        ob = SOM(weight)
        ob.learn(data, epochs, alpha )
        # тестовий датасет, та мітка
        test_data, correct_answer = utils.load_testdataset()
        test_winner = ob.winner(test_data)

        accuracity += 1 if test_winner in correct_answer else 0

        stat.append(test_winner)
        counts = np.bincount(stat)
        # Создание словаря для хранения результатов
        stat_dict = {str(i): counts[i] for i in range(np.max(stat) + 1)}

    # print(f"winner №{i}: {test_winner}")
    print(f"accuracity: {accuracity}/{iterations}")
    print(f"Right answers: {correct_answer}\n")
    print(stat_dict)


main()


