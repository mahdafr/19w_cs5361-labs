import mnist
import cupy as np
import time
import math


class knn(object):
    # Constructor
    def __init__(self, k=3, weighted=True, classify=True):
        self.k = k
        self.weighted = weighted
        self.classify = classify
        # @author mmafr
        self.n_matrix = None    # the list of the distances
        self.n_weights = None
        self.n_labels = None

    # train the model
    def fit(self, x, y):
        self.x_train = x.astype(np.float32)
        self.y_train = y
        # fixme: remove when done testing
        skip = 100
        self.x_train = self.x_train[::skip]
        self.y_train = self.y_train[::skip]

    # test the model
    # @author mmafr
    def predict(self, x_test):
        s = ' (weighted)' if self.weighted else ''
        print('Classification' + s if self.classify else 'Regression' + s)
        self.buildNeighborList(x_test)
        if self.classify:
            return self.classifier(x_test)
        else:
            return self.regression(x_test)

    # KNN classifier: weighted or unweighted?
    # @author mmafr
    def classifier(self, test):
        guess = np.zeros(test.shape[0])
        V = np.unique(self.y_train)     # every possible class
        for x in range(test.shape[0]):
            for v in range(V.shape[0]):     # for each label
                votes = np.zeros(V.shape[0])  # sums up to k
                for i in range(self.k):     # for each neighbor
                    d = self.delta(v, int(self.n_labels[x][i]))
                    w = self.n_weights[x][i]
                    votes[V[int(self.n_labels[x][i])]] += w*d
            guess[x] = np.amax(votes)
        return guess

    # @author mmafr
    def delta(self, a, b):
        return 1 if a == b else 0

    # KNN regression: weighted or unweighted?
    # @author mmafr
    def regression(self, test):
        mean = np.zeros(test.shape[0])
        for x in range(test.shape[0]):  # for each test example
            mean[x] = np.sum(self.n_weights[x]*self.n_labels[x])
        if self.weighted:   # divide by sum of the weights
            mean = mean/np.sum(self.n_weights)
        else:   # divide by k
            mean = mean/self.k
        return mean

    # calculate the distances from each test point to other points
    # @author mmafr
    def buildNeighborList(self, test):
        n_tmp = np.zeros(shape=(test.shape[0], self.x_train.shape[0]))
        for x1 in range(test.shape[0]):     # foreach test sample
            for x2 in range(self.x_train.shape[0]):     # for each train point
                n_tmp[x1][x2] = (self.distance(test[x1], self.x_train[x2]))
        srtd = np.argsort(n_tmp, axis=1)    # sort list of neighbors
        # build the knn matrix and weights
        self.k_neighbors(n_tmp, srtd)
        self.weights()
        print("Built neighbor matrix.")

    # calculate the euclidean distance between point x1 and x2
    # @author mmafr
    def distance(self, x1, x2):
        d = 0
        for f in range(len(x1)):
            d += pow((x1[f] - x2[f]), 2)
        return math.sqrt(d)

    # determines the list of the k nearest-neighbors
    # @author mmafr
    def k_neighbors(self, neighbors, nearest):
        # each test sample has k nearest neighbors
        self.n_matrix = np.zeros(shape=(neighbors.shape[0], self.k))
        self.n_labels = np.zeros(self.n_matrix.shape)
        for i in range(self.n_matrix.shape[0]):
            for j in range(self.k):
                self.n_matrix[i][j] = neighbors[i][nearest[i][j]]
                self.n_labels[i][j] = self.y_train[int(nearest[i][j])]

    # get the weights, list of ones if not
    #  @author mmafr
    def weights(self):
        self.n_weights = np.ones(self.n_matrix.shape)
        if self.weighted:  # if weighted, w=1/d(x,xi)
            self.n_weights = 1/self.n_matrix
        # mask = np.isin(self.sorted, self.neighbor)
        # weight = 1/weight[mask]


if __name__ == "__main__":
    TESTS = 20
    print('MNIST dataset')
    x_train, y_train, x_test, y_test = mnist.load()
    x_test = x_test[:TESTS]
    y_test = y_test[:TESTS]
    print('Training size= ' + str(x_train.shape[0]))
    print('Testing size= ' + str(x_test.shape[0]))
    model = knn(classify=True, weighted=True)

    start = time.time()
    model.fit(x_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(x_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   

    y_test = np.asarray(y_test)     # for CuPy
    print('Accuracy:', np.sum(pred == y_test)/len(y_test))
    
    print('\nSolar particle dataset')
    dir = 'D:\Google Drive\skool\CS 5361\datasets\lab1\\'
    x_train = np.load(dir + 'x_ray_data_train.npy')
    y_train = np.load(dir + 'x_ray_target_train.npy')
    x_test = np.load(dir + 'x_ray_data_test.npy')
    y_test = np.load(dir + 'x_ray_target_test.npy')
    y_test = np.asarray(y_test)  # for CuPy
    x_test = x_test[:TESTS]
    y_test = y_test[:TESTS]
    print('Training size= ' + str(x_train.shape[0]))
    print('Testing size= ' + str(x_test.shape[0]))
    model = knn(classify=False, weighted=True)

    start = time.time()
    model.fit(x_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))

    start = time.time()
    pred = model.predict(x_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))

    print('Mean square error:', np.mean(np.square(pred-y_test)))
