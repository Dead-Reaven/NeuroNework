#main.py
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

    def winner(self, sample):
        # Розрахунок евклідової відстані між прикладом та кожним вектором ваги
        distances = np.linalg.norm(self.weights - sample, axis=1)
        # Повертаємо індекс вектора ваги з найменшою відстанню
        return np.argmin(distances)

    def update(self, sample, J, alpha):
        # Оновлюємо переможний вектор ваги
        self.weights[J] += alpha * (sample - self.weights[J])

def main():

    #гіперпараметри епоха/коеф. навчання
    epochs = 50
    alpha = 0.1

    # аналіз
    accuracity = 0
    stat = []

    #повсторення тестування
    iterations = 1000

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
        #рахує скільки разів був обран кожен варіант
        counts = np.bincount(stat)
        # створюжмо словарь що виводить статистику по кожному обранному варіанту
        stat_dict = {str(i): counts[i] for i in range(np.max(stat) + 1)}

    print(f"accuracity: {accuracity}/{iterations}")
    print(f"Right answers: {correct_answer}\n")
    print(stat_dict)


main()


