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
    # print(weights)

    return np.array(images), weights

zero = [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0]
    ]

one = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0]
]

# random = np.random.uniform(0,1,(1, dim))


def load_testdataset():
    correct_answer = [6]
    initial_layer = [
        np.random.uniform(0,1,(1, dim)),
        # one,
        np.random.uniform(0,1,(1, dim)),
        np.random.uniform(0,1,(1, dim)),
        np.random.uniform(0,1,(1, dim)),
        np.random.uniform(0,1,(1, dim)),
        np.random.uniform(0,1,(1, dim)),
        one,
    ]
    images = [np.array(image).ravel().astype(np.float64) for image in initial_layer]

    return images , correct_answer
