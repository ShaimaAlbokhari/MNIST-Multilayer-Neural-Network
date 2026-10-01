# 🧠 MNIST Handwritten Digit Classification using MLNN

A **Multilayer Neural Network (MLNN)** implemented from scratch using NumPy to classify handwritten digits from the MNIST dataset.

The project explores how different neural network configurations affect classification performance by experimenting with the number of hidden units, momentum values, and training dataset size.

---

## 🎯 Project Objective

The objective of this project is to build a multilayer neural network from scratch and use it to classify handwritten digits from **0 to 9**.

The network performs multiclass classification using a hidden layer and an output layer with ten neurons representing the ten digit classes.

---

## 📊 Dataset

The project uses the **MNIST handwritten digit dataset**.

Each image represents a handwritten digit from **0 to 9** using 784 pixel features.

The dataset files used by the project are:

```text
mnist_train.csv
mnist_test.csv
```

---

## ⚙️ Data Preprocessing

Before training the neural network:

1. Labels are separated from the image pixel features.
2. Pixel values are normalized from `0–255` to `0–1`.
3. Class labels are converted into one-hot encoded vectors.
4. The encoded target values are scaled using `0.1` and `0.9`.

The network uses:

```text
Input neurons:  784
Hidden neurons: Variable
Output neurons: 10
```

---

## 🧠 Neural Network Architecture

The Multilayer Neural Network consists of:

```text
784 Input Neurons
        ↓
Hidden Layer
        ↓
10 Output Neurons
```

The neural network is implemented from scratch using NumPy rather than using a pre-built neural network model.

The implementation includes:

- Weight and bias initialization
- Forward propagation
- Sigmoid activation function
- Backpropagation
- Momentum-based weight updates
- Training and test accuracy evaluation

---

## 🔄 Forward Propagation

During forward propagation, the input data is passed through the hidden layer and then through the output layer.

The **Sigmoid activation function** is used in both layers:

```text
sigmoid(x) = 1 / (1 + e^-x)
```

The output layer contains ten neurons corresponding to digits `0–9`.

The class with the highest output activation is selected as the predicted digit.

---

## 🔙 Backpropagation

The network learns by calculating the difference between the expected output and the predicted output.

The error is propagated backward through the network to update the weights and biases.

Momentum is also incorporated into the weight update process to study its effect on network training and classification performance.

---

## 🧪 Experiments

Three experiments were conducted to evaluate how different configurations affect the neural network.

### Experiment 1 — Hidden Layer Size

The first experiment evaluates different numbers of hidden units:

```text
20
50
100
```

The reported results were:

| Hidden Units | Training Accuracy | Test Accuracy |
|---:|---:|---:|
| 20 | 96.23% | 94.38% |
| 50 | 98.91% | 96.12% |
| 100 | 99.61% | 96.95% |

Increasing the number of hidden units improved the model's capacity and test performance.

However, the larger difference between training and test accuracy when using 100 hidden units indicated increased overfitting.

---

### Experiment 2 — Momentum

The second experiment studies the effect of different momentum values while using a fixed hidden layer size.

The tested momentum values were:

```text
0
0.25
0.5
```

The reported results were:

| Momentum | Training Accuracy | Test Accuracy |
|---:|---:|---:|
| 0 | 99.53% | **97.70%** |
| 0.25 | 99.66% | 97.62% |
| 0.5 | 99.74% | 97.60% |

The highest reported test accuracy in this experiment was:

```text
97.70%
```

with:

```text
Momentum = 0
```

The results showed only small differences in test accuracy between the tested momentum values.

---

### Experiment 3 — Training Data Size

The third experiment investigates the effect of changing the amount of training data.

Two subsets of the original training dataset were evaluated:

```text
25% of training data
50% of training data
```

The experiment showed that increasing the amount of training data improved test performance and reduced the gap between training and test accuracy.

This indicates better generalization when the model is trained using more examples.

---

## 📈 Model Evaluation

The models are evaluated using:

- Training accuracy
- Test accuracy
- Confusion matrices
- Accuracy curves across epochs

Confusion matrices are used to analyze how well the model distinguishes between the ten handwritten digit classes.

Accuracy curves are also generated to compare training and test performance throughout the training process.

---

## 🛠️ Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

The neural network itself is implemented from scratch using **NumPy**.

`scikit-learn` is used for supporting operations such as data shuffling, confusion matrix calculation, and creating balanced training subsets.

---

## 📁 Project Files

```text
MNIST-Multilayer-Neural-Network/
│
├── MNIST-Digit-Classification-using-MLNN.py
├── Description_HW.pdf
├── report.pdf
├── dataset.zip
└── README.md
```

### File Description

- `MNIST-Digit-Classification-using-MLNN.py` — Neural network implementation, experiments, training, and evaluation.
- `Description_HW.pdf` — Project assignment description.
- `report.pdf` — Experimental results, visualizations, and analysis.
- `dataset.zip` — Dataset files used by the project.
- `README.md` — Project documentation.

---

## ▶️ Running the Project

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Make sure the required dataset files are available in the working directory:

```text
mnist_train.csv
mnist_test.csv
```

Then run:

```bash
python MNIST-Digit-Classification-using-MLNN.py
```

The program trains the neural network under different experimental configurations and generates accuracy results, confusion matrices, and training/test accuracy curves.

---

## 📄 Project Report

Detailed experimental results, confusion matrices, accuracy curves, and analysis are available in:

```text
report.pdf
```

The original project requirements are available in:

```text
Description_HW.pdf
```

---

## 👩‍💻 Developer

**Shaima Nabeel Albokhari**  
B.Sc. Computer Science — Taif University
