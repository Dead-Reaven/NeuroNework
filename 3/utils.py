#utils.py

import numpy as np


dim = 35

def load_dataset():
    initial_layer = [
      [
          [0, 1, 1, 1, 0],
          [1, 0, 0, 0, 1],
          [1, 0, 0, 0, 1],
          [1, 0, 0, 0, 1],
          [1, 0, 0, 0, 1],
          [1, 0, 0, 0, 1],
          [0, 1, 1, 1, 0]
      ],
      [
          [0, 0, 1, 0, 0],
          [0, 1, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 1, 1, 1, 0]
      ],
    ]

    weights = np.zeros((1, dim))
    images = [np.array(image).ravel() for image in initial_layer]
    weights = [np.array(weight).ravel() for weight in weights]

    return np.array(images), weights

# тестові вхідні данні, яких не було в датасеті
zero = [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0]
    ]

one = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0]
]


def load_testdataset():
    #обираємо правильну відповідь, та генеруємо вектори "шуму"
    #щоб заплутати SOM
    correct_answer = [5]
    initial_layer = [
        np.random.uniform(0,1,(1, dim)),
        np.random.uniform(0,1,(1, dim)),
        np.random.uniform(0,1,(1, dim)),
        np.random.uniform(0,1,(1, dim)),
        np.random.uniform(0,1,(1, dim)),
        zero,
        np.random.uniform(0,1,(1, dim)),
        np.random.uniform(0,1,(1, dim)),
        np.random.uniform(0,1,(1, dim)),


    ]
    images = [np.array(image).ravel().astype(np.float64) for image in initial_layer]

    return images , correct_answer
