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
    images = [np.array(image).ravel() for image in initial_layer]
    print(f"patterns:" , images)
    return np.array(images)

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
    correct_answer = [5]
    initial_layer = [
        one,
        zero,
    ]
    images = [np.array(image).ravel() for image in initial_layer]

    print(f"test:" , images)

    return images , correct_answer
