from keras.datasets import mnist
from keras.utils import to_categorical
from Layers.Conv import Conv2D
from Layers.MaxPool import MaxPool2D
from Layers.Dense import Dense10

import numpy as np
import cv2

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

Conv = Conv2D(8)
Max = MaxPool2D()
Dense = Dense10((8,13,13), 10)

totalLoss = 0
numCorrect = 0

def forward(image, label):
    convout = Conv.convolve((image/255) - 0.5)
    poolout = Max.pool(convout)
    out = Dense.predict(poolout)

    loss = -np.log(out[label])
    accuracy = 1 if np.argmax(out) == label else 0

    return out, loss, accuracy

def backprop(image, label, alpha=0.005):
    out, loss, accuracy = forward(image, label)

    grad = np.zeros(10)
    grad[label] = -1 / out[label]

    grad = Dense.train(grad, alpha)
    grad = Max.train(grad)
    Conv.train(grad, alpha)

    return loss, accuracy



for epoch in range(3):
    print("----- Epoch %d -----" % (epoch+1))

    randorder = np.random.permutation(len(X_train))
    traindat = X_train[randorder]
    trainlabels = Y_train[randorder]

    totalLoss = 0
    numCorrect = 0


    for i, (im, label) in enumerate(zip(X_train, Y_train)):
        if i%100 == 99:
            print("Step %d | Avg Loss: %.5f | Accuracy: %d%%" %(i+1, totalLoss/100, numCorrect))
            totalLoss = 0
            numCorrect = 0
        
        loss, accuracy = backprop(im, label)
        totalLoss+=loss
        numCorrect+=accuracy


print("Testing...")
totalLoss = 0
numCorrect = 0


for im, label in zip(X_test, Y_test):
    
    _, loss, accuracy = forward(im, label)
    totalLoss+=loss
    numCorrect+=accuracy

nTests = len(X_test)
print("Loss:", loss/nTests)
print("Accuracy:", numCorrect / nTests)