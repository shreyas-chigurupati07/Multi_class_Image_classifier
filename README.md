# Multi-Class Image Classification with Convolutional Neural Networks

# Table of Contents

Introduction
Dependencies
Getting Started
Dataset
Preprocessing
Model Architecture
Training
Evaluation
Results
License
Introduction

This project aims to classify images into multiple classes using a Convolutional Neural Network (CNN). The model takes grayscale, normalized images as input and outputs the class labels, achieving a test accuracy of XX%.

Dependencies

Python 3.X
TensorFlow 2.X
Pandas
NumPy
Matplotlib
Seaborn
scikit-learn
Getting Started

Clone this repository.
Run pip install -r requirements.txt to install the dependencies.
Run your_script.py to execute the model.
Dataset

The dataset consists of labeled images in .p (pickle) format, divided into training, validation, and test sets. The labels correspond to XX different classes.

Training set: XX images
Validation set: XX images
Test set: XX images
Preprocessing

Grayscale Conversion: Each image is converted to grayscale to reduce computational complexity.
Normalization: The pixel intensity is normalized to the range [-1, 1] to improve model convergence speed.
Model Architecture

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
Training

The model is trained on the grayscale and normalized training images using the following parameters:

Optimizer: Adam
Loss Function: Sparse Categorical Cross-Entropy
Metrics: Accuracy
Batch Size: 500
Epochs: 5
Evaluation

The model performance is evaluated using the test set, and the results are as follows:

Test Loss: XX
Test Accuracy: XX%
Results

Training and validation loss curves
Training and validation accuracy curves
Confusion matrix
Sample predictions on test images
You can find the visualizations in the results/ folder.

License

This project is licensed under the MIT License - see the LICENSE.md file for details.
