from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
import itertools as it
import numpy as np

n_estimators = [10,15,20,25]
criterion = []
max_depth = [None,6,8,10,12]
min_samples_split = [2,4,6]
min_samples_leaf = [1,2,3,4]
min_weight_fraction_leaf = [0,0.25,0.5]
max_features = [2,5,7,10,'auto','sqrt','log2',None]
max_leaf_nodes = [50,100,100,None]
min_impurity_decrease = [0,0.25,0.5,0.75]
min_impurity_split = [1e-7]
bootstrap = [True]
oob_score = [True]
n_jobs = [None]
random_state = [23]
verbose = [0]
warm_start = [True,False]

def classificaation(file,X,Y,x,y):
    param = []
    acc = []
    criterion = ['gini','entropy']
    class_weight = ['balanced','balanced_subsample',None]
    for i in it.product(n_estimators,criterion,max_depth,min_samples_split,
                        min_samples_leaf,min_weight_fraction_leaf,max_features,
                        max_leaf_nodes,min_impurity_decrease,min_impurity_split,
                        bootstrap,oob_score,n_jobs,random_state,verbose,
                        warm_start,class_weight):
        # print(*i)
        forest = rfc(*i)
        forest.fit(X,Y)
        # print('Accuracy: ' + str(forest.score(x, y)) + '\n')
        acc.append(forest.score(x,y))
        param.append([*i])
    _results(file,acc,param)

def regressor(file,X,Y,x,y):
    param = []
    acc = []
    criterion = ['mse','mae']
    for i in it.product(n_estimators,criterion,max_depth,min_samples_split,
                        min_samples_leaf,min_weight_fraction_leaf,max_features,
                        max_leaf_nodes,min_impurity_decrease,min_impurity_split,
                        bootstrap,oob_score,n_jobs,random_state,verbose,warm_start):
        # print(*i)
        forest = rfr(*i)
        forest.fit(X,Y)
        # print('Accuracy: ' + str(forest.score(x, y)) + '\n')
        acc.append(forest.score(x,y))
        param.append([*i])
    _results(file,acc,param)

def _results(file,acc,params):
    file.write('Forest')
    for i in range(len(acc)):
        file.write('\n'+str(acc[i]))
        file.write('\n'+str(params[i]))
    file.write('\nHighest_acc='+str(np.max(acc))+'params='+str(params[np.argmax(acc)]))
    file.write('\nLowest_acc='+str(np.min(acc))+'params='+str(params[np.argmin(acc)]))
