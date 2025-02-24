import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load .h5 file of arbitrary name for testing (last if more than one)
print(os.getcwd())
for file in os.listdir(os.getcwd()):
    if file.endswith(".h5"):
        print(file)
        net = load_model(file, compile=False)
        net.summary()

# Determine what type of network this is
input_dims = net.input_shape
netType = 'CNN' if len(input_dims) > 2 else 'MLP'

# Test with MNIST data
from tensorflow.keras.datasets import mnist

(x_train, labels_train), (x_test, labels_test) = mnist.load_data()
x_test = x_test.astype('float32') / 255

if netType == 'MLP':
    x_test = x_test.reshape(10000, 784)
else:
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Evaluate
outputs = net.predict(x_test)
labels_predicted = np.argmax(outputs, axis=1)
correct_classified = np.sum(labels_predicted == labels_test)
print('Percentage correctly classified MNIST =', 100 * correct_classified / labels_test.size)

# Find and plot some misclassified samples
misclassified_indices = np.where(labels_predicted != labels_test)[0]
n_plot = min(len(misclassified_indices), 8)

if n_plot > 0:
    fig, axes = plt.subplots(2, n_plot, figsize=(12, 3))
    fig.subplots_adjust(hspace=0.5)
    for i in range(n_plot):
        idx = misclassified_indices[i]
        axes[0, i].imshow(x_test[idx].reshape(28, 28), cmap='gray_r')
        axes[0, i].set_title(f'True: {labels_test[idx]}', y=1.05)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)
        if netType == 'MLP':
            output = net.predict(x_test[idx:idx + 1].reshape(1, 784))
        else:
            output = net.predict(x_test[idx:idx + 1].reshape(1, 28, 28, 1))
        output = output[0, 0:]
        axes[1, i].bar(range(10), output)
        axes[1, i].set_xticks(range(10))
        axes[1, i].set_title(f'Pred: {np.argmax(output)}', y=1.05)
        axes[1, i].get_yaxis().set_visible(False)
        if i == 0:
            axes[1, i].get_yaxis().set_visible(True)
            axes[1, 0].set_ylabel('Probability')
            axes[1, 0].set_yticks([0, 0.25, 0.5, 0.75, 1])
    fig.subplots_adjust(bottom=0.2)
    fig.text(0.51, 0.05, 'Classes', ha='center', va='center')
    plt.savefig("misclassified.pdf")
else:
    print("No misclassified samples found. No plot generated.")