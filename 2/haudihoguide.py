import numpy as np
import utils

images, labels = utils.load_dataset()

weight_input_to_hidden =  np.random.uniform(-0.5, 0.5 , (20, 784) )
weight_hidden_to_output = np.random.uniform(-0.5, 0.5 , (10, 20) )

# нейрон зміщення
bias_input_to_hidden =  np.zeros((20,1))
bias_hidden_to_output = np.zeros((10,1))

epochs = 3
e_loss = 0
e_correct = 0
learning_rate = 0.1

for epoch in range(epochs):
  print(f"№{epoch}")

  for image, label in zip(images, labels):
    image  = np.reshape(image, (-1,1))
    label = np.reshape(label, (-1,1))

    #forward propagation
    hidden_raw = bias_input_to_hidden + weight_input_to_hidden @ image
    hidden = 1 / (1+ np.exp(-hidden_raw))

    #forward propagation to output layer
    output_raw = bias_hidden_to_output + weight_hidden_to_output @ hidden
    output = 1 / (1+ np.exp(-output_raw))

    # Loss / Error calc
    #formula MSE
    e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
    e_correct += int(np.argmax(output) == np.argmax(label))

    # backpropagation (output layer)
    delta_output = output - label
    weight_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
    bias_hidden_to_output += -learning_rate * delta_output

    # backpropagation (hidden layer)
    delta_hidden = np.transpose(weight_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
    weight_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
    bias_input_to_hidden += -learning_rate * delta_hidden

    #DONE

  # print(output)
  print(f"Loss:{round((e_loss[0] / images.shape[0]) * 100, 3 )}%")
  print(f"Accuracy:{round((e_correct / images.shape[0]) * 100, 3 )}%")
  e_loss, e_correct = 0, 0



import random

test_image = random.choice(images)

image  = np.reshape(test_image, (-1,1))

#forward propagation
hidden_raw = bias_input_to_hidden + weight_input_to_hidden @ image
hidden = 1 / (1+ np.exp(-hidden_raw))

#forward propagation to output layer
output_raw = bias_hidden_to_output + weight_hidden_to_output @ hidden
output = 1 / (1+ np.exp(-output_raw))


import matplotlib.pyplot as plt

plt.imshow(test_image.reshape(28,28), cmap="Greys")
plt.title(f"NN suggest is number :{output.argmax()}")
plt.show()

