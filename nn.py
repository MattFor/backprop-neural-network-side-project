import random
import math
from collections import Counter

MAX_VAL = 10
NON_ZERO = 1e-10

class Layer:
    def __init__(self):
        self.perceptrons = []
        self.predictions = []
        self.pred_errors = []

class Perceptron:
    def __init__(self, weights: float, sigmoid: bool, always_active: bool, lambda_reg: float = 0.01):
            self.weights = [random.uniform(-1, 1) for i in range(weights + 1)]
            self.sigmoid = sigmoid
            self.always_activate = always_active
            self.learning_rate = 0.05
            self.lambda_reg = lambda_reg  # L2 regularization parameter

    @staticmethod
    def dot_product(X, Y):
        return sum(X[k] * Y[k] for k in range(len(X)))

    @staticmethod
    def sigmoid_function(x):
        # If x is too large, return 1
        if x > MAX_VAL:
            return 1
        # If x is too small, return 0
        elif x < -MAX_VAL:
            return 0
        # Otherwise, compute the sigmoid function as usual
        else:
            return 1 / (1 + math.exp(-x))

    def predict(self, arr):
        val = self.dot_product(arr + [-1], self.weights)
        if self.sigmoid:
            return self.sigmoid_function(val)
        return 0 if val < 0 else 1

    def resolve_stalemate(self, arr):
        return self.dot_product(arr + [-1], self.weights)

    def learn(self, y_model, arr):
        y_pred = self.predict(arr)
        arr = arr + [-1]
        gradients = []

        for j in range(len(self.weights)):
            gradient = self.learning_rate * ((y_model - y_pred) * arr[j] + 2 * self.lambda_reg * self.weights[j])
            gradients.append(gradient)

        gradients = [max(min(gradient, 1), -1) for gradient in gradients]

        for j in range(len(self.weights)):
            self.weights[j] += gradients[j]


def subtract_vectors(v1, v2):
    return [v1[i] - v2[i] for i in range(len(v1))]

def vector_by_scalar(k, arr):
    return [arr[i] * k for i in range(len(arr))]


def add_vectors(v1, v2):
    return [v1[i] + v2[i] for i in range(len(v1))]


class NeuralNetwork:
    def __init__(self, sigmoid: bool, layer_num: int, output_num: int, learning_set: list[list[list[int]], list[int]]):
        self.layers = []
        self.layer_num = layer_num
        self.output_num = output_num
        self.learning_set = learning_set
        self.bias = 0.2
        self.sigmoid = sigmoid
        self.learning_rate = 0.05

    def adjust_learning_rate(self):
        pass
        # error = sum(layer.pred_errors[-1] for layer in self.layers[1:])
        # self.learning_rate = 0.1 * error  # adjust the learning rate based on error

    def predict(self, arr):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[0].predictions = arr[:]  # [perceptron.predict(arr[j]) for j, perceptron in enumerate(self.layers[0].perceptrons)]
            else:
                self.layers[i].predictions = [perceptron.predict(self.layers[i - 1].predictions) for perceptron in self.layers[i].perceptrons]

        if self.layers[-1].predictions.count(1) == 1:
            for x in self.layers[-1].predictions:
                if x == 1:
                    return x

        final_predictions = [perceptron.resolve_stalemate(self.layers[-2].predictions) for perceptron in self.layers[-1].perceptrons]
       
        return final_predictions.index(max(final_predictions))

    def learn(self, x_train, y_train, passes, epochs):
        for epoch in range(epochs):
            combined = list(zip(x_train, y_train))
            random.shuffle(combined)
            x_train[:], y_train[:] = zip(*combined)

            for _ in range(passes):
                for i in range(len(x_train)):
                    y_pred = self.predict(x_train[i])

                    self.layers[-1].pred_errors = subtract_vectors(
                        [1 if y_train[i] == z else 0 for z in range(self.output_num)],
                        [1 if y_pred == z else 0 for z in range(self.output_num)]
                    )

                    for j in range(len(self.layers) - 2, 0, -1):  
                        for k, perceptron in enumerate(self.layers[j].perceptrons):  
                            self.layers[j].pred_errors.append(0)
                            for p, neuron in enumerate(self.layers[j + 1].perceptrons):  
                                self.layers[j].pred_errors[-1] = self.layers[j + 1].pred_errors[p] * perceptron.weights[p]

                    # self.adjust_learning_rate()

                    for m in range(1, len(self.layers)):
                        for j in range(len(self.layers[m].pred_errors)):
                            self.layers[m].perceptrons[j].weights = add_vectors(self.layers[m].perceptrons[j].weights, vector_by_scalar(self.layers[m].pred_errors[j] * self.bias, self.layers[m - 1].predictions + [-1]))

                    for j in range(len(self.layers)):
                        self.layers[j].pred_errors = []
                        self.layers[j].predictions = []

            print(f'Epoch {epoch + 1}/{epochs} finished')

    def display(self):
        print('Displaying Neural Network Information:')
        for i, layer in enumerate(self.layers):
            print(f'Layer {i + 1} - Number of perceptrons: {len(layer.perceptrons)}')
            for j, perceptron in enumerate(layer.perceptrons):
                print(f'Perceptron {j + 1} weights: {perceptron.weights}')
            print('\n')

    def initialize(self, perceptron_count: int, automatic=False):
        for i in range(self.layer_num):
            layer = Layer()

            layer.perceptrons = [Perceptron(len(self.layers[-1].perceptrons) if len(self.layers) != 0 else 1, self.sigmoid, i == 0) for i in range(
                ((perceptron_count - 2 * i if perceptron_count - 2 * i > self.output_num + 2 else self.output_num + 2) if i != self.layer_num - 1 else self.output_num)
                if automatic else perceptron_count
            )]

            layer.predictions = [0] * len(layer.perceptrons)
            self.layers.append(layer)

        return self

def prepare_data():
    processed_dataset = [[], []]
    
    try:
        with open('zoo.txt') as file:
            for line in file.readlines():
                arguments = [float(x.replace('\n', '')) for x in line.split(',')[1:]]

                processed_dataset[0].append([arguments[i] for i in range(len(arguments) - 1)])
                processed_dataset[1].append(int(arguments[-1]))

        max_val = [0 for _ in range(len(processed_dataset[0][0]))]  # Initialize max_val

        for i in range(len(processed_dataset[0][0])):
            for j in range(len(processed_dataset[0])):
                if processed_dataset[0][j][i] > max_val[i]:
                    max_val[i] = processed_dataset[0][j][i]

        for i in range(len(processed_dataset[0][0])):
            for j in range(len(processed_dataset[0])):
                processed_dataset[0][j][i] = processed_dataset[0][j][i] / (max_val[i] + NON_ZERO)
    except FileNotFoundError:
        print('File not found!')
        return None

    return processed_dataset

def main():
    processed_dataset = prepare_data()
    
    split_point = int(len(processed_dataset[0]) * 0.8)
    train_data = [processed_dataset[0][:split_point], processed_dataset[1][:split_point]]
    test_data = [processed_dataset[0][split_point:], processed_dataset[1][split_point:]]

    neural_network = NeuralNetwork(True, 3, len(Counter(processed_dataset[1])), train_data)
    neural_network.initialize(16, True)

    neural_network.learn(train_data[0], train_data[1], 500, 1)
    
    predictions = list(map(neural_network.predict, test_data[0]))
    correct_predictions = sum(predicted == actual for predicted, actual in zip(predictions, test_data[1]))
    total_predictions = len(test_data[1])
    accuracy = correct_predictions / total_predictions * 100
    print(f'Accuracy: {accuracy}%')

    # Calculate confoosion matrix
    items = len(Counter(processed_dataset[1]))
    confusion_matrix = [[0 for _ in range(items)] for _ in range(items)]
    for predicted, actual in zip(predictions, test_data[1]):
        confusion_matrix[predicted][actual - 1] += 1
    
    # Calculate  F-measure for each item
    for i in range(items):
        true_positive = confusion_matrix[i][i]
        false_positive = sum(confusion_matrix[i]) - true_positive
        false_negative = sum(row[i] for row in confusion_matrix) - true_positive
        precision = true_positive / (true_positive + false_positive + NON_ZERO)
        recall = true_positive / (true_positive + false_negative + NON_ZERO)
        fmeasure = 2 * precision * recall / (precision + recall + NON_ZERO)
        print(f'Item {i + 1} - Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F-Measure: {fmeasure * 100:.2f}%')

    # neural_network.display()

if __name__ == '__main__':
    main()