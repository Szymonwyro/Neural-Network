import numpy as np

W1 = np.load("W1.npy")
b1 = np.load("b1.npy")
W2 = np.load("W2.npy")
b2 = np.load("b2.npy")

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / np.sum(expZ)

def predict(x):
    x = x.reshape(784, 1) / 255.0
    Z1 = W1 @ x + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    probs = softmax(Z2)
    digit = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return digit, confidence