# CIFAR10 Image classification.

## Convolutional Neural Network (CNN) model for classifying CIFAR10 dataset images

### Using Tensorflow


This repo contains codes that classify images in the CIFAR10 dataset. The model classifies the images into ten different classes (i.e. airplane, person), using the typical softmax function. Loss is calculated via cross entropy and optimizer used here is AdamOptimizer.

### Run
    sh trainer.sh

## Following is the list of steps we are taking to post-process the received data and classify them using our trained network


	1. Pre-processing & Data augmentation
	2. Feed the training data into the neural network
	3. Backpropagate using optimization algorithms provided by Tensorflow (i.e. Adagrad, RMSProp)
	4. Continue training until the loss is reasonably low and performance shows a promising result
	5. Use the validation data to check for overfitting
	6. Use the test data to evaluate the model


