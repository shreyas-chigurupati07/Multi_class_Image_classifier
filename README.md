# Multi-Class Image Classification with Convolutional Neural Networks

## Table of Contents ##

1. Introduction
2. Dependencies
3. Getting Started
4. Dataset
5. Preprocessing
6. Model Architecture
7. Training
8. Evaluation
9. Results
10. License
----
### 1. Introduction ###

This project aims to classify images into multiple classes using a Convolutional Neural Network (CNN). The model takes grayscale, normalized images as input and outputs the class labels, achieving a test accuracy of 80%.

### 2. Dependencies ###

Python 3.X
TensorFlow 2.X
Pandas
NumPy
Matplotlib
Seaborn
scikit-learn
### 3. Getting Started ###

Clone this repository.
Run pip install -r requirements.txt to install the dependencies.
Run your_script.py to execute the model.
### 4. Dataset ###

The dataset consists of labeled images in .p (pickle) format, divided into training, validation, and test sets.

Training set: train.p
Validation set: valid.p
Test set: test.p
### 5. Preprocessing ###

Grayscale Conversion: Each image is converted to grayscale to reduce computational complexity.
Normalization: The pixel intensity is normalized to the range [-1, 1] to improve model convergence speed.
### 6. Model Architecture ###

The CNN model was implemented using TensorFlow and Keras, and consists of the following layers:

Convolutional Layer with 6 filters (5x5 kernel, ReLU activation)
Average Pooling Layer
Dropout Layer with a rate of 0.2
Convolutional Layer with 16 filters (5x5 kernel, ReLU activation)
Average Pooling Layer
Flatten Layer
Fully Connected Layer with 120 units (ReLU activation)
Fully Connected Layer with 84 units (ReLU activation)
Output Layer with 43 units (Softmax activation)
### 7. Training ###

The model is trained on the grayscale and normalized training images using the following parameters:

Optimizer: Adam
Loss Function: Sparse Categorical Cross-Entropy
Metrics: Accuracy
Batch Size: 500
Epochs: 5
### 8. Evaluation ###

The model performance is evaluated using the test set, and the results are as follows:

Test Loss: 80.64
Test Accuracy: 81.3%
### 9. Results ###

Training and validation loss curves
Training and validation accuracy curves
Confusion matrix
Sample predictions on test images
You can find the visualizations in the results/ folder.

### 10. License ###

This project is licensed under the MIT License - see the LICENSE.md file for details.
