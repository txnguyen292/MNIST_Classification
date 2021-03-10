from typing import Optional, List
import heapq
import numpy as np
from numpy.linalg import norm
from numpy import dot
import pandas as pd
import matplotlib.pyplot as plt
from logzero import setup_logger
import logging
import functools
from sklearn.metrics.pairwise import cosine_similarity

logger = setup_logger(level=logging.INFO)

Matrix = List[List[float]]


import multiprocessing as mp

def accuracy(y: np.array, y_hat: np.array) -> float:
    """Compute rate of correction

    Args:
        y (np.array): ground truth
        y_hat (np.array): prediction vector

    Returns:
        float: [description]
    """
    assert len(y) == len(y_hay), "Vectors must be of same length"
    return np.mean(y, y_hat)

def random_generator(D:int, C:int, random_seed:Optional[int]=123) -> Matrix:
    """randomly generating data with normal distribution

    Args:
        D (int): ?
        C (int): ?
        random_seed (int, optional): [description]. Defaults to 123.

    Returns:
        Matrix: [description]
    """
    np.random.seed(random_seed)
    N=int(C*1e3)

    X0 = np.random.randn((N//C),D)+np.array([2,2])
    X1 = np.random.randn((N//C),D)+np.array([0,-2])
    X2 = np.random.randn((N//C),D)+np.array([-2,2])

    X =np.vstack((X0,X1,X2))

    y = np.array([0]*(N//C)+[1]*(N//C)+[2]*(N//C))
    return (X, y)

class KNNClassifier:
    """KNN classifier using euclidean distance"""
    def fit(self, X:np.array, y:np.array) -> None:
        self.X = X
        self.y = y
        self._pool = mp.cpu_count()
    def predict(self, X:np.array, K:int, epsilon:Optional[int]=1e-3) -> np.array:
        """Making predictions with KNN 

        Args:
            X (np.array): input data
            K (int): number of nearest neighbors
            epsilon (int, optional): value to smoothen division. Defaults to 1e-3.

        Returns:
            np.array: [description]
        """
        N = len(X)
        y_hat = np.zeros(N)

        for i in range(N):
            dist2 = np.sum((self.X-X[i])**2, axis=1)
            idx = np.argsort(dist2)[:K] # getting smallest distances indices
            gamma_k = 1 / (np.sqrt(dist2[idx]) + epsilon) # computing the weights 
            y_hat[i] = np.bincount(self.y[idx], weights=gamma_k).argmax()
        return y_hat

    def predict_single(self, K:int, epsilon:int, X: np.array) -> List[float]:
        """Make prediction with Knn on 1 single observation

        Args:
            K (int): number of nearest neighbors
            epsilon (int): value to smoothen division
            X (np.array): input data

        Returns:
            List[float]: prediction
        """
        dist2 = np.sum((self.X - X)**2, axis=1)
        idx = np.argsort(dist2)[:K]
        gamma_k = 1 / (np.sqrt(dist2[idx]) + epsilon)
        y_hat = np.bincount(self.y[idx], weights=gamma_k).argmax()
        return y_hat

    def predict_multi(self, K:int, epsilon:int, num_cpu: int, X:np.array) -> np.array:
        """Making predictions using knn with multi cores

        Args:
            K (int): number of nearest neighbors
            epsilon (int): value to smoothen division
            num_cpu (int): number of cores to be used
            X (np.array): input data

        Returns:
            np.array: vector of predictions
        """
        predictSingle = functools.partial(self.predict_single, K, epsilon)
        if num_cpu:
            cpus = num_cpu
        else:
            cpus = self._pool - 1
        logger.debug(f"training with {cpus} cpus")
        pool = mp.Pool(cpus)
        y_hat = pool.map(predictSingle, X)
        return y_hat

class KNNClassifierCosineSim:
    """KNN classifier using cosine similarity"""
    def fit(self, X:np.array, y:np.array) -> None:
        self.X = X
        self.y = y
    
    def cos_sim(self, a:np.array, b:np.array) -> float:
        cos_sim = dot(a, b) / (norm(a) * norm(b))
        return cos_sim
    def predict(self, X: np.array, K:int, epsilon:int=1e-3) -> np.array:
        """KNN with cosine_similiarity

        Args:
            X (np.array): Input data
            K (int): number of nearest neighbors
            epsilon (int, optional): value to smoothen division. Defaults to 1e-3.

        Returns:
            np.array: vector of predictions
        """

        cosim = cosine_similarity(self.X, X)
        top = [(heapq.nlargest((k+1), range(len(i)), i.take)) for i in cosim]

        # convert indices to numbers
        top = [[y_train[j] for j in i[:k]] for i in top]
        # votes and return prediction for every input data
        pred = [max(set(i), key=i.count) for i in top]

        pred = np.array(pred)

        return pred
        