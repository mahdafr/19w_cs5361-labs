from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.tree import DecisionTreeRegressor as dtr
import itertools as it
import numpy as np

splitter = ['best', 'random']
max_depth = [7, 8, 9, 10]
min_samples_split = [0.25,0.5,0.75,1.0]
min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8]
min_weight_fraction_leaf = [0.1, 0.25, 0.4]
max_features = [1, 2, 3, 4, 5]
random_state = [23]
max_leaf_nodes = [100,200,300]
min_impurity_decrease = [0]
min_impurity_split = [1e-7,0.0]
class_weight = ['balanced', None]
presort = [True, False]

def classification(file,X,Y,x,y):
    param = []
    acc = []
    criterion = ['gini', 'entropy']
    for i in it.product(criterion,splitter,max_depth,min_samples_split,min_samples_leaf,
                        min_weight_fraction_leaf,max_features,random_state,max_leaf_nodes,
                        min_impurity_decrease,min_impurity_split,class_weight,presort):
        # print(*i)
        dtree = dtc(*i)
        dtree.fit(X,Y)
        # print('Accuracy: ' + str(dtree.score(x,y)) + '\n')
        acc.append(dtree.score(x,y))
        param.append([*i])
    _results(file,acc,param)

def regressor(file,X,Y,x,y):
    param = []
    acc = []
    criterion = ['mse', 'friedman_mse', 'mae']
    for i in it.product(criterion,splitter,max_depth,min_samples_split,min_samples_leaf,
                        min_weight_fraction_leaf,max_features,random_state,max_leaf_nodes,presort):
        # print(*i)
        dtree = dtr([*i])
        dtree.fit(X,Y)
        # print('Accuracy: ' + str(dtree.score(x,y)) + '\n')
        acc.append(dtree.score(x,y))
        param.append([*i])
    _results(file,acc,param)

def _results(file,acc,params):
    file.write('DecTree')
    for i in range(len(acc)):
        file.write('\n'+str(acc[i]))
        file.write('\n'+str(params[i]))
    file.write('\nHighest_acc='+str(np.max(acc))+'params='+str(params[np.argmax(acc)]))
    file.write('\nLowest_acc='+str(np.min(acc))+'params='+str(params[np.argmin(acc)]))
