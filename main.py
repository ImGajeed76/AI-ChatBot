from neural_network import *

# reshape the inputs because we expect a column vector
# reshape to (example count, input size, height of matrix)
training_inputs = np.reshape([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], (8, 3, 1))
training_outputs = np.reshape([[0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7]], (8, 1, 1))

nn = NeuralNetwork([
    DenseLayer(3, 5),
    TanH(),
    DenseLayer(5, 3),
    TanH(),
    DenseLayer(3, 1),
    TanH()
])

nn.train(training_inputs, training_outputs, epochs=50000)

while True:
    a = int(input("A: "))
    b = int(input("B: "))
    c = int(input("C: "))

    inputs = np.array([[a], [b], [c]])
    output = nn.forward(inputs)

    print(output)
