import mnist
import numpy as np
import time

class zeroR(object):
    # Constructor
    def __init__(self,classify=True):  
        # classify is True for classification tasks, false for regression tasks
        self.classify = classify
        
    def fit(self,x_train,y_train):  
        if self.classify:
            n_classes = np.max(y_train)+1    
            count = np.zeros(n_classes,dtype=int)
            for y in y_train:
                count[y] +=1
            self.pred = np.argmax(count)
        else:
            self.pred = np.mean(y_train)

    def predict(self,x_test):
        if self.classify:
            pred = np.zeros(len(x_test),dtype=int)+self.pred
        else:
            pred = np.zeros(len(x_test),dtype=np.float32)+self.pred
        return pred
    
if __name__ == "__main__":  
    print('MNIST dataset')
    x_train, y_train, x_test, y_test = mnist.load()
    model = zeroR()
    
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
    x_train = np.load(dir+'x_ray_data_train.npy')
    y_train = np.load(dir+'x_ray_target_train.npy')
    x_test = np.load(dir+'x_ray_data_test.npy')
    y_test = np.load(dir+'x_ray_target_test.npy')
    model = zeroR(classify=False)
        
    start = time.time()
    model.fit(x_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(x_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    
    print('Mean square error:',np.mean(np.square(pred-y_test)))
    