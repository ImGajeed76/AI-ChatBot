import os
import pickle

import numpy
import numpy as np


class Layer:
    def __init__(self, input_size=0, output_size=0):
        self.inputs = None
        self.outputs = None
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, inputs):
        pass

    def backward(self, output_gradient, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.weights, self.inputs) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.inputs.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, inputs):
        self.inputs = inputs
        return self.activation(self.inputs)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.inputs))


class TanH(Activation):
    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        return 1 - np.tanh(x) ** 2

    def __init__(self):
        super().__init__(self.tanh, self.tanh_prime)


class Sigmoid(Activation):
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_prime)


class Softmax(Layer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, inputs):
        tmp = np.exp(inputs)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


class NeuralNetwork:
    def __init__(self, network: list[Layer] = None, learning_rate=0.1):
        self.min_error = None
        self.network = network
        self.learning_rate = float(learning_rate)

    def setup_min_error(self, min_error):
        self.min_error = min_error

    def load_network(self, network: list[Layer]):
        self.network = network

    def load_network_from_file(self, file_path: str, log=True):
        if os.stat(file_path).st_size == 0:
            return

        if log:
            print("Loading..", end='\r')
        with open(file_path, 'rb') as file:
            network = pickle.load(file)
            self.network = network
        if log:
            print(f"Loaded from {file_path}")

    def save_network_to_file(self, file_path: str, log=True):
        if log:
            print("Saving..", end='\r')
        with open(file_path, 'wb') as file:
            pickle.dump(self.network, file, protocol=pickle.HIGHEST_PROTOCOL)
        if log:
            print(f"Saved in {file_path}")

    def network_check(self, inputs=None, outputs=None):
        if self.network is None:
            exit("Error: Network not loaded!")

        if inputs is not None and self.network[0].input_size != len(inputs):
            exit("Error: Input size not matching!")

        last_out = 0
        last_dense_layer = None
        for layer in self.network:
            if type(layer) == Dense:
                last_dense_layer = layer
                if last_out == layer.input_size or last_out == 0:
                    last_out = layer.output_size
                else:
                    exit("Error: Layer inputs/outputs not matching!")

        if last_dense_layer is not None and outputs is not None and last_dense_layer.output_size != len(outputs):
            exit("Error: Output size not matching!")

    def train(self, training_inputs: numpy.ndarray, training_outputs: numpy.ndarray, epochs=1000):
        self.network_check(training_inputs[0], training_outputs[0])
        print("Training..")

        for e in range(epochs):
            error = 0

            for x, y in zip(training_inputs, training_outputs):
                output = self.forward(x)
                error += mse(y, output)
                self.backward(mse_prime(y, output), self.learning_rate)

            error /= len(training_inputs)
            print('', end='\r')
            print(
                f'{round((e + 1) / epochs * 100, 2)}% of {epochs} epochs, '
                f'error = {round(error, 8)}, '
                f'lr = {round(self.learning_rate, 8)}',
                end='')

            if self.min_error is not None and error <= self.min_error:
                print('\r')
                print("Network landed on min error and stopped training.")
                print(f"error = {round(error, 8)}, lr = {round(self.learning_rate, 8)}")
                return

        print('\r')
        print('Finished the training')

    def epoch(self, inputs: numpy.ndarray, expected_outputs: numpy.ndarray):
        self.network_check(inputs)

        output = self.forward(inputs)
        self.backward(mse_prime(expected_outputs, output), self.learning_rate)
        return mse(expected_outputs, output)

    def forward(self, inputs):
        self.network_check(inputs)

        output = inputs

        for layer in self.network:
            output = layer.forward(output)

        return output

    def backward(self, error, learning_rate):

        grad = error

        for layer in reversed(self.network):
            grad = layer.backward(grad, learning_rate)
