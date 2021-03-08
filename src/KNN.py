import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def accuracy(y, y_hat):
    return np.mean(y, y_hat)

def random_generator(D:int, C:int, random_seed:int=123):
    np.random.seed(random_seed)
    N=int(C*1e3)

    X0 = np.random.randn((N//C),D)+np.array([2,2])
    X1 = np.random.randn((N//C),D)+np.array([0,-2])
    X2 = np.random.randn((N//C),D)+np.array([-2,2])

    X =np.vstack((X0,X1,X2))

    y = np.array([0]*(N//C)+[1]*(N//C)+[2]*(N//C))
    return (X, y)

class KNNClassifier:
    def fit(self, X:np.array, y:np.array) -> None:
        self.X = X
        self.y = y

    def predict(self, X:np.array, K:int, epsilon:int=1e-3) -> np.array:
        N = len(X)
        y_hat = np.zeros(N)

        for i in range(N):
            dist2 = np.sum((self.X-X[i])**2, axis=1)
            idx = np.argsort(dist2)[:K] # getting smallest distances indices
            gamma_k = 1 / (np.sqrt(dist2[idx]) + epsilon) # computing the weights 
            y_hat[i] = np.bincount(self.y[idx], weights=gamma_k).argmax()
        return y_hat