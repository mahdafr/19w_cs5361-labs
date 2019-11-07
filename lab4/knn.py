from sklearn.neighbors import KNeighborsClassifier as knnc
from sklearn.neighbors import KNeighborsRegressor as knnr
import itertools as it
import sklearn.preprocessing as prep
import numpy as np

n_neighbors = [3, 4, 5, 6]
weights = ['uniform', 'distance']
algorithm = ['ball_tree', 'kd_tree', 'brute']
metric = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
leaf_size = [20,30,50,70]
p = [1, 2]

def classification(file,X,Y,x,y):
    # lab_enc = prep.LabelEncoder()
    # Y = lab_enc.fit_transform(Y)
    # y = lab_enc.fit_transform(y)
    param = []
    acc = []
    for i in it.product(n_neighbors,weights,algorithm,leaf_size):
        for m in metric:
            # print(*i,m)
            if m=='minkowski':
                for j in p:
                    # print('p=',j)
                    knn = knnc(*i,p=j,metric=m)
                    knn.fit(X,Y)
                    acc.append(knn.score(x,y))
                    param.append([*i,m,j])
            else:
                knn = knnc(*i,metric=m)
                knn.fit(X, Y)
                acc.append(knn.score(x,y))
                param.append([*i,m])
    _results(file,acc,param)

def regressor(file,X,Y,x,y):
    param = []
    acc = []
    for i in it.product(n_neighbors,weights,algorithm,leaf_size):
        for m in metric:
            # print(*i,m)
            if m=='minkowski':
                for j in p:
                    # print('p=',j)
                    knn = knnr(*i,p=j,metric=m)
                    knn.fit(X,Y)
                    acc.append(knn.score(x,y))
                    param.append([*i,m,j])
            else:
                knn = knnr(*i,metric=m)
                knn.fit(X, Y)
                acc.append(knn.score(x,y))
                param.append([*i,m])
    _results(file,acc,param)

def _results(file,acc,params):
    file.write('KNN')
    for i in range(len(acc)):
        file.write('\n'+str(acc[i]))
        file.write('\n'+str(params[i]))
    file.write('\nHighest_acc='+str(np.max(acc))+'params='+str(params[np.argmax(acc)]))
    file.write('\nLowest_acc='+str(np.min(acc))+'params='+str(params[np.argmin(acc)]))
