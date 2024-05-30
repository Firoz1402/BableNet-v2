import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


class SimpleANN:
    
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_input_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden_output = np.zeros((1, output_size))
        self.activation = activation
        self.loss_history = []

    def forward(self, inputs):
       
        self.hidden_output = np.dot(inputs, self.weights_input_hidden) + self.bias_input_hidden
        if self.activation == 'sigmoid':
            self.hidden_activation = sigmoid(self.hidden_output)
        elif self.activation == 'relu':
            self.hidden_activation = relu(self.hidden_output)
        else:
            raise ValueError("Activation function not supported.")
        self.output = np.dot(self.hidden_activation, self.weights_hidden_output) + self.bias_hidden_output
        self.output_probs = softmax(self.output)
        return self.output_probs

    def backward(self, inputs, targets, learning_rate):
        
        batch_size = inputs.shape[0]
        d_output = self.output_probs - targets
        d_weights_hidden_output = np.dot(self.hidden_activation.T, d_output) / batch_size
        d_bias_hidden_output = np.sum(d_output, axis=0, keepdims=True) / batch_size
        d_hidden_activation = np.dot(d_output, self.weights_hidden_output.T)
        if self.activation == 'sigmoid':
            d_hidden_output = d_hidden_activation * self.hidden_activation * (1 - self.hidden_activation)
        elif self.activation == 'relu':
            d_hidden_output = d_hidden_activation * (self.hidden_output > 0)
        else:
            raise ValueError("Activation function not supported.")
        d_weights_input_hidden = np.dot(inputs.T, d_hidden_output) / batch_size
        d_bias_input_hidden = np.sum(d_hidden_output, axis=0, keepdims=True) / batch_size

        # Update weights and biases
        self.weights_input_hidden -= learning_rate * d_weights_input_hidden
        self.bias_input_hidden -= learning_rate * d_bias_input_hidden
        self.weights_hidden_output -= learning_rate * d_weights_hidden_output
        self.bias_hidden_output -= learning_rate * d_bias_hidden_output

    def train(self, X_train, y_train, epochs, learning_rate):
        num_classes = len(np.unique(y_train))
        y_train_encoded = np.eye(num_classes)[y_train]

        for epoch in range(epochs):
            output = self.forward(X_train)
            self.backward(X_train, y_train_encoded, learning_rate)
            if epoch % 100 == 0:
                loss = -np.sum(y_train_encoded * np.log(output + 1e-9)) / len(y_train)
                self.loss_history.append(loss)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
