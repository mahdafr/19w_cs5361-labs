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

    # test the model
    # @author mmafr
    def predict(self, test):
        s = ' (weighted)' if self.weighted else ''
        print('Classification' + s if self.classify else 'Regression' + s)
        test = self.attr_selection(test)  # for optimization
        self.buildNeighborList(test)
        if self.classify:
            return self.classifier(test)
        else:
            return self.regression(test)

    # KNN classifier: weighted or unweighted?
    # @author mmafr
    def classifier(self, test):
        guess = np.zeros(test.shape[0])
        label = np.unique(self.y_train)     # every possible class
        for x in range(test.shape[0]):
            votes = np.zeros(label.shape[0])
            for l in range(label.shape[0]):     # for each label
                d = self.delta(l, self.n_labels[x])
                votes[l] = d*np.sum(self.n_weights[x])
                # votes = np.around(votes, 3)
            guess[x] = np.argmax(votes*np.sum(self.n_weights[x]))
            print("Prediction for Test=" + str(x) + " is " + str(guess[x]))
        return guess

    # @author mmafr
    def delta(self, a, b):
        return (b==a).sum()

    # KNN regression: weighted or unweighted?
    # @author mmafr
    def regression(self, test):
        mean = np.zeros(test.shape[0])
        for x in range(test.shape[0]):  # for each test example
            mean[x] = np.sum(self.n_weights[x]*self.n_labels[x])
            if self.weighted:   # divide by sum of the weights
                mean[x] = mean[x]/np.sum(self.n_weights[x])
            else:   # divide by k
                mean[x] = mean[x]/self.k
            # print("Prediction for Test=" + str(x) + " is " + str(mean[x]))
        mean = np.around(mean, 0)   # round for picking a class
        return mean

    # calculate the distances from each test point to other points
    # @author mmafr
    def buildNeighborList(self, test):
        n_mat = np.sqrt(np.asarray(np.dot(x_test, x_train.T), dtype=np.float64))
        print("Built neighbor matrix.")
        srtd = np.argsort(n_mat, axis=1)[:, :self.k]
        # build the knn matrix and weights
        self.k_neighbors(n_mat, srtd)
        self.weights()

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
        self.n_matrix = np.zeros(shape=nearest.shape)
        self.n_labels = np.zeros(nearest.shape)
        for i in range(self.n_matrix.shape[0]):
            for j in range(self.k):
                self.n_matrix[i][j] = neighbors[i][nearest[i][j]]
                self.n_labels[i][j] = self.y_train[int(nearest[i][j])]
        print("DATA: " + str(self.n_matrix[0]) + ", " + str(self.n_labels[0]))

    # drop the features with the lowest variance
    # @author mmafr
    def attr_selection(self, test):
        start = self.x_train.shape[1]
        mean = np.mean(self.x_train, axis=0)   # means of features
        mask = (self.x_train > 0.25*np.mean(mean))
        self.x_train = np.asarray(self.x_train[:, mask.any(axis=0)])
        x_test = test[:, mask.any(axis=0)]
        print("Dropped " + str(start - self.x_train.shape[1]) + " features.")
        return x_test

    # get the weights, list of ones if not weighted
    #  @author mmafr
    def weights(self):
        if self.classify:
            self.n_weights = np.zeros(self.n_labels.shape)
            self.n_weights[: , 0] = 1   # all others are 0
        else:
            self.n_weights = np.ones(self.n_matrix.shape)
        if self.weighted:  # if weighted, w=1/d(x,xi)
            self.n_weights = 1/self.n_matrix


if __name__ == "__main__":
    TESTS = 100
    TRAIN = 10000
    THEK = 3
    print('MNIST dataset')
    x_train, y_train, x_test, y_test = mnist.load()
    x_train = x_train[:TRAIN:]
    y_train = y_train[:TRAIN:]
    x_test = x_test[:TESTS:]
    y_test = y_test[:TESTS:]
    print('Training size = ' + str(x_train.shape[0]))
    print('Testing size = ' + str(x_test.shape[0]))
    print("K = " + str(THEK))
    model = knn(k=THEK, classify=True, weighted=True)

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
    
    # print('\nSolar particle dataset')
    # dir = 'D:\Google Drive\skool\CS 5361\datasets\lab1\\'
    # x_train = np.load(dir + 'x_ray_data_train.npy')
    # y_train = np.load(dir + 'x_ray_target_train.npy')
    # x_test = np.load(dir + 'x_ray_data_test.npy')
    # y_test = np.load(dir + 'x_ray_target_test.npy')
    # y_test = np.asarray(y_test)  # for CuPy
    # x_train = x_train[:TRAIN]
    # y_train = y_train[:TRAIN]
    # x_test = x_test[:TESTS:]
    # y_test = y_test[:TESTS:]
    # print('Training size = ' + str(x_train.shape[0]))
    # print('Testing size = ' + str(x_test.shape[0]))
    # print("K = " + str(THEK))
    # model = knn(k=THEK, classify=True, weighted=True)
    #
    # start = time.time()
    # model.fit(x_train, y_train)
    # elapsed_time = time.time()-start
    # print('Elapsed_time training  {0:.6f} '.format(elapsed_time))
    #
    # start = time.time()
    # pred = model.predict(x_test)
    # elapsed_time = time.time()-start
    # print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))
    #
    # print('Mean square error:', np.mean(np.square(pred-y_test)))
