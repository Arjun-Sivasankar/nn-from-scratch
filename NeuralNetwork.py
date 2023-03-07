import numpy as np

class NeuralNetwork:
    ## Initializing weights:
    def __init__(self, input_size, hidden_size, output_size):
        # initialize the weights randomly
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)

    # Activation function to calculate the output of the hidden/output layers
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid activation function
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    ## Forward Prop.:
    def forward(self, inputs):
        # feed the inputs through the network
        self.hidden = np.dot(inputs, self.weights1)
        self.hidden_activation = self.sigmoid(self.hidden)
        self.output = np.dot(self.hidden_activation, self.weights2)
        self.output_activation = self.sigmoid(self.output)
        return self.output_activation

    ## Back Prop.:
    def backward(self, inputs, expected_output, output):
        # calculate the error
        error = expected_output - output
        d_output = error * self.sigmoid_derivative(output)

        # backpropagate the error to the hidden layer
        error_hidden = np.dot(d_output, self.weights2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.hidden_activation)

        # update the weights
        self.weights2 = self.weights2 + np.dot(self.hidden_activation.T, d_output)
        self.weights1 = self.weights1 + np.dot(inputs.T, d_hidden)

    ## Training:
    def train(self, inputs, expected_output, epochs):
        # train the network for a certain number of epochs
        for epoch in range(epochs):
            # forward pass
            output = self.forward(inputs)

            # backward pass
            self.backward(inputs, expected_output, output)

            # print the error every 1000 epochs
            if epoch % 1000 == 0:
                error = np.mean(np.abs(expected_output - output))
                print("Epoch {0}: error {1}".format(epoch, error))


# create a neural network with 2 inputs, 3 hidden neurons, and 1 output
nn = NeuralNetwork(2, 3, 1)

# create some training data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])

# train the network for 10000 epochs
nn.train(inputs, expected_output, 10000)

# test the network with some new input
new_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

print("Expected Output: \n", expected_output)
print("Actual Output: \n", nn.forward(new_input))

