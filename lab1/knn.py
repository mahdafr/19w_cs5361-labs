import mnist
import numpy as np
import time
import math


class knn(object):
    # Constructor
    def __init__(self,k=3,weighted=True,classify=True):  
        self.training_set = None
        self.k = k
        self.weighted = weighted
        self.classify = classify
        
    def fit(self,x,y):
        self.x_train = x.astype(np.float32)
        self.y_train = y
        # use every skip-th ex; for development uses
        skip = 50
        self.x_train = self.x_train[::skip]
        self.y_train = self.y_train[::skip]
        self.training_set = np.zeros(shape=self.x_train.shape)
        print(self.training_set.shape)

    def predict(self,x_test):
        if self.classify:
            if self.weighted:
                print("hi")
            else:
                print("bye")
        else:
            if self.weighted:
                print("ok")
            else:
                print("no")
        return np.random.rand(len(x_test))

    # calculate the euclidean distance between point x1 and x2
    def distance(self,x1, x2):
        d = 0
        for x in range(x1.shape):
            d += pow((x1[x] - x2[x]), 2)
        return math.sqrt(d)
    
if __name__ == "__main__":  
    print('MNIST dataset')
    x_train, y_train, x_test, y_test = mnist.load()
    model = knn()
    
    start = time.time()
    model.fit(x_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(x_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    
    print('Accuracy:',np.sum(pred==y_test)/len(y_test))
    
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
    
    print('Mean square error:',np.mean(np.square(pred-y_test)))
    
    