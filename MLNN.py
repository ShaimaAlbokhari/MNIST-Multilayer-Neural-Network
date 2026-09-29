"""_____________________________________________________________
     Homework 2: MLNN
     by: Shaima Nabeel Albokhari     44100014
_____________________________________________________________"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

#load data
data_train = pd.read_csv("mnist_train.csv")
data_test = pd.read_csv("mnist_test.csv")

#preprocess data
X_train = data_train.iloc[:, 1:].values / 255.0
X_test = data_test.iloc[:, 1:].values / 255.0
y_train = data_train.iloc[:, 0].values
y_test = data_test.iloc[:, 0].values

#convert labels to one-hot encoding and scale
y_train = np.eye(10)[y_train] * 0.8 + 0.1
y_test = np.eye(10)[y_test] * 0.8 + 0.1

input_size = 784
output_size = 10

#define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#class of MLNN
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, momentum_term=0.9):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum_term = momentum_term

        #initialize weights and biases
        self.w1 = np.random.uniform(-0.05, 0.05, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.uniform(-0.05, 0.05, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

        #initialize momentum velocities
        self.velocity_w1 = np.zeros_like(self.w1)
        self.velocity_b1 = np.zeros_like(self.b1)
        self.velocity_w2 = np.zeros_like(self.w2)
        self.velocity_b2 = np.zeros_like(self.b2)
    #forward process
    def forward(self, X):
        self.hidden_input = np.dot(X, self.w1) + self.b1
        self.hidden_output = sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.w2) + self.b2
        self.output = sigmoid(self.output_input)
        return self.output
    #backward process
    def backward(self, x, y):
        error = y - self.output

        #calculate deltas
        d_output = error * sigmoid_derivative(self.output)
        error_hidden = d_output.dot(self.w2.T)
        d_hidden = error_hidden * sigmoid_derivative(self.hidden_output)

        #update weights with momentum
        self.velocity_w2 = self.momentum_term * self.velocity_w2 + self.hidden_output.T.dot(d_output) * self.learning_rate
        self.velocity_b2 = self.momentum_term * self.velocity_b2 + np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        self.velocity_w1 = self.momentum_term * self.velocity_w1 + x.T.dot(d_hidden) * self.learning_rate
        self.velocity_b1 = self.momentum_term * self.velocity_b1 + np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate

        self.w2 += self.velocity_w2
        self.b2 += self.velocity_b2
        self.w1 += self.velocity_w1
        self.b1 += self.velocity_b1

    def evaluate(self, X, y):
        predictions = self.forward(X)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        return np.mean(predicted_classes == true_classes)

    def iteration(self, X_train, y_train, X_test, y_test, epochs):
        train_accuracies = []
        test_accuracies = []

        #initial evaluation
        train_acc = self.evaluate(X_train, y_train)
        test_acc = self.evaluate(X_test, y_test)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        for epoch in range(epochs):
            X_train, y_train = shuffle(X_train, y_train)
            for i in range(X_train.shape[0]):
                x = X_train[i].reshape(1, -1)
                y = y_train[i].reshape(1, -1)
                self.forward(x)
                self.backward(x, y)

            #evaluate after each epoch
            train_acc = self.evaluate(X_train, y_train)
            test_acc = self.evaluate(X_test, y_test)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

        return train_accuracies, test_accuracies
    
#plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, size):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({size})')
    plt.show()

#plot accuracy curves
def plot_accuracy_curves(size, train_acc, test_acc):
    plt.figure(figsize=(10,5))
    plt.plot(range(len(train_acc)), train_acc, label='Train Accuracy')
    plt.plot(range(len(test_acc)), test_acc, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Training VS Test Accuracy ({size})')
    plt.grid(True)
    plt.show()

#__________________________________________________________________
# Exp 1: Vary number of hidden units. 
results = {}
epochs = 1
hidden_sizes = [20, 50, 100] # List of hidden layer sizes to test

print('=' * 40)
print('Exp 1: Different Hidden Layers')
print('=' * 40)

for hidden_size in hidden_sizes:
    print(f"\nTraining model with hidden size: {hidden_size}")
    model = NeuralNetwork(input_size, hidden_size, output_size)
    train_acc, test_acc = model.iteration(X_train, y_train, X_test, y_test, epochs)
    #store results
    results[hidden_size] = {
        'train_accuracies': train_acc,
        'test_accuracies': test_acc
    }
    #print accuracies
    final_train_acc = results[hidden_size]['train_accuracies'][-1]
    final_test_acc = results[hidden_size]['test_accuracies'][-1]
    print(f"Training Accuracy: {final_train_acc * 100:.2f}%")
    print(f"Test Accuracy: {final_test_acc * 100:.2f}%")
    print("_" * 40)

    #generate confusion matrix
    y_test_pred = np.argmax(model.forward(X_test), axis=1)
    y_test_true = np.argmax(y_test, axis=1)
    plot_confusion_matrix(y_test_true, y_test_pred, f'Hidden Layers = {hidden_size}')
    #plot accuracy curves for this hidden size
    plot_accuracy_curves(f'Hidden Layers = {hidden_size}', train_acc, test_acc)

print('\n')

#__________________________________________________________________
# Exp 2: Vary Momentum
results_momentum = {}
hidden_size_fixed = 100 
momentum_terms = [0, 0.25, 0.5]  #0.9 from Exp 1

print('=' * 40)
print('Exp 2: Vary Momentum')
print('=' * 40)

for momentum in momentum_terms:
    print(f"\nTraining model with momentum term: {momentum}")
    model_m = NeuralNetwork(input_size, hidden_size_fixed, output_size, momentum_term=momentum)
    train_acc_m, test_acc_m = model_m.iteration(X_train, y_train, X_test, y_test, epochs)  
    #store results
    results_momentum[momentum] = {
        'train_accuracies': train_acc_m,
        'test_accuracies': test_acc_m
    }

    #print accuracies
    final_train_acc_m = results_momentum[momentum]['train_accuracies'][-1]
    final_test_acc_m = results_momentum[momentum]['test_accuracies'][-1]
    print(f"Training Accuracy: {final_train_acc_m * 100:.2f}%")
    print(f"Test Accuracy: {final_test_acc_m * 100:.2f}%")
    print("_" * 40)

    #generate confusion matrix
    y_test_pred = np.argmax(model_m.forward(X_test), axis=1) 
    plot_confusion_matrix(y_test_true, y_test_pred, f'Momentum = {momentum}')
    plot_accuracy_curves(f'Momentum = {momentum}', train_acc_m, test_acc_m)

print('\n')

#__________________________________________________________________
# Exp 3: Vary Training Data Size

print('=' * 40)
print('Exp 3: Vary Training Data Size')
print('=' * 40)

def create_balanced_subset(X, y, subset_size):
    X_subset, _, y_subset, _ = train_test_split(
        X, y, train_size=subset_size, 
        stratify=np.argmax(y, axis=1), 
        random_state=42
    )
    return X_subset, y_subset

#create subsets
X_train_25, y_train_25 = create_balanced_subset(X_train, y_train, 0.25)
X_train_50, y_train_50 = create_balanced_subset(X_train, y_train, 0.5)

#train models
#training model with 25% of training data:
model_25 = NeuralNetwork(input_size, 100, output_size, momentum_term=0.9)
train_acc_25, test_acc_25 = model_25.iteration(X_train_25, y_train_25, X_test, y_test, epochs)  

#training model with 50% of training data:
model_50 = NeuralNetwork(input_size, 100, output_size, momentum_term=0.9)
train_acc_50, test_acc_50 = model_50.iteration(X_train_50, y_train_50, X_test, y_test, epochs)  

#store results
results_subsets = {
    '25%': {'train_acc': train_acc_25, 'test_acc': test_acc_25},
    '50%': {'train_acc': train_acc_50, 'test_acc': test_acc_50}
}

#print accuracies
for subset_size, data in results_subsets.items():
    print(f"\nSubset Size: {subset_size}")
    print(f"Training Accuracy: {data['train_acc'][-1] * 100:.2f}%")
    print(f"Test Accuracy: {data['test_acc'][-1] * 100:.2f}%")
    print("_" * 40)

    #plot curves and confusion matrices
    y_test_pred = np.argmax(model_25.forward(X_test), axis=1) if subset_size == '25%' else np.argmax(model_50.forward(X_test), axis=1)
    plot_confusion_matrix(y_test_true, y_test_pred, f'Subset Size = {subset_size}')
    plot_accuracy_curves(f'Subset Size = {subset_size}', data['train_acc'], data['test_acc'])
    
print('\n')