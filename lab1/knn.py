import mnist
import numpy as np
import time
import math


class knn(object):
    # Constructor
    def __init__(self, k=3, weighted=True, classify=True):
        self.k = k
        self.weighted = weighted
        self.classify = classify
        self.neighbor = None    # the list of the distances
        self.sorted = None  # the sorted locations for distances

    def fit(self, x, y):
        self.x_train = x.astype(np.float32)
        self.y_train = y
        # fixme: remove when done testing
        skip = 50
        self.x_train = self.x_train[::skip]
        self.y_train = self.y_train[::skip]

    # test the model
    def predict(self, x_test):
        self.buildNeighborList(x_test)
        # x_test = x_test[::self.skip]    # fixme: remove when done testing
        # if self.classify:
        #     return self.classifier(x_test)
        # else:
        #     return self.regression(x_test)

    # KNN classifier: weighted or unweighted?
    def classifier(self, test):
        neighbors = self.neighbor[:self.k]
        weight = np.ones(neighbors.shape[0])    # weights for each distance pair
        pred = np.zeros(test.shape[0])
        V = np.unique(self.y_train)
        for x in range(test.shape[0]):
            if self.weighted:   # if weighted, w=1/d(x,xi)
                weight[x] = 1/self.neighbor[x]
            for v in range(V):
                pred[x] += weight*self.delta(v,self.y_train[x])
        return pred

    # determine whether the values are equal or not
    def delta(self, a, b):
        if a == b:
            return 1
        return 0

    # KNN regression: weighted or unweighted?
    def regression(self, test):
        neighbors = self.neighbor[:self.k]
        mean = np.zeros(test.shape[0])
        weight = np.ones(neighbors.shape[0])    # weights for each distance pair
        for x in range(test.shape[0]):  # for each test example
            if self.weighted:   # if weighted, w=1/d(x,xi)
                weight[x] = 1/self.neighbor[x]
            mean[x] = np.sum(neighbors[:self.k]*weight[:self.k])    # get the sum for the neighbors, weighted/not
        if self.weighted:   # if weighted, divide by the sum of the weights
            mean = mean/np.sum(weight)
        else:   # if not weighted, divide by k
            mean = mean/self.k
        return mean

    # calculate the distances from each training point to other training points
    def buildNeighborList(self, test):
        self.neighbor = np.zeros(test.shape[0])
        print(self.neighbor.shape)
        for x1 in range(test.shape[0]):
            for x2 in range(self.x_train.shape[0]):
                self.neighbor[x1] = (self.distance(test[x1], self.x_train[x2]))
                print(x1)
                print(x2)
        self.sorted = np.sort(self.neighbor)  # save indices of sorted neighbors
        print(self.sorted.shape)

    # calculate the euclidean distance between point x1 and x2
    def distance(self, x1, x2):
        d = 0
        for f in range(len(x1)):
            d += pow((x1[f] - x2[f]), 2)
        return math.sqrt(d)


if __name__ == "__main__":  
    print('MNIST dataset')
    x_train, y_train, x_test, y_test = mnist.load()
    model = knn(weighted=False)
    
    start = time.time()
    model.fit(x_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(x_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    
    print('Accuracy:', np.sum(pred == y_test)/len(y_test))
    
    print('\nSolar particle dataset')
    dir = 'D:\Google Drive\skool\CS 5361\datasets\lab1\\'
    x_train = np.load(dir + 'x_ray_data_train.npy')
    y_train = np.load(dir + 'x_ray_target_train.npy')
    x_test = np.load(dir + 'x_ray_data_test.npy')
    y_test = np.load(dir + 'x_ray_target_test.npy')
    model = knn(classify=False)
        
    start = time.time()
    model.fit(x_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(x_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    
    print('Mean square error:', np.mean(np.square(pred-y_test)))
