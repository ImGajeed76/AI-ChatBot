import numpy
import numpy as np


class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        pass

    def backward(self, output_gradient, learning_rate):
        pass


class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
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


class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, inputs):
        self.inputs = inputs
        return self.activation(self.inputs)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.inputs))


class TanH(ActivationLayer):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


class Sigmoid(ActivationLayer):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


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
    def __init__(self, network: list[Layer]):
        self.network = network

    def train(self, training_inputs: numpy.ndarray, training_outputs: numpy.ndarray, epochs=1000, learning_rate=0.1):
        for e in range(epochs):
            error = 0

            for x, y in zip(training_inputs, training_outputs):
                output = self.forward(x)
                error += mse(y, output)
                self.backward(mse_prime(y, output), learning_rate)

            error /= len(training_inputs)
            print('', end='\r')
            print(f'{round((e + 1) / epochs * 100, 2)}% of {epochs} epochs, error = {error}', end='')

        print('\n')

    def forward(self, inputs):
        output = inputs

        for layer in self.network:
            output = layer.forward(output)

        return output

    def backward(self, error, learning_rate):
        grad = error

        for layer in reversed(self.network):
            grad = layer.backward(grad, learning_rate)
