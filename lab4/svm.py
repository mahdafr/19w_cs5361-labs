from sklearn.svm import SVC, SVR
import itertools as it
import numpy as np

C = [1]
degree = [2,3,4,5]
gamma = ['scale','auto']
coef0 = [0,0.25,0.5,0.75,1]
shrinking = [True,False]
tol = [1e-2,1e-3,1e-4]
cache_size = [0.5,1.0,1.5]
verbose = [False]  # not Spyder
max_iter = [-1,1,2,3,4,5]

def classification(file,X,Y,x,y):
    param = []
    acc = []
    kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    probability = [True,False]
    class_weight = ['balanced']
    decision_function_shape = ['ovo','ovr']
    random_state = [23]
    for i in it.product(C,kernel,degree,gamma,coef0,shrinking,probability,tol,cache_size,
                        class_weight,verbose,max_iter,decision_function_shape,random_state):
        # print(*i)
        svm = SVC(*i)
        svm.fit(X, Y)
        # print('Accuracy: ' + str(svm.score(x, y)) + '\n')
        acc.append(svm.score(x,y))
        param.append([*i])
    _results(file,acc,param)

def regressor(file,X,Y,x,y):
    acc = []
    param = []
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    epsilon = [0.1,0.15,0.2]
    for i in it.product(kernel,degree,gamma,coef0,tol,C,epsilon,shrinking,
                        cache_size,verbose,max_iter):
        # print(*i)
        svm = SVR(*i)
        svm.fit(X, Y)
        acc.append(svm.score(x,y))
        param.append([*i])
    _results(file,acc,param)

def _results(file,acc,params):
    file.write('SVM')
    for i in range(len(acc)):
        file.write('\n'+str(acc[i]))
        file.write('\n'+str(params[i]))
    file.write('\nHighest_acc='+str(np.max(acc))+'params='+str(params[np.argmax(acc)]))
    file.write('\nLowest_acc='+str(np.min(acc))+'params='+str(params[np.argmin(acc)]))
