from sklearn.linear_model import LogisticRegression as lr
import itertools as it
import numpy as np
import sklearn.preprocessing as prep
import numpy as np

penalty = ['l1','l2','elasticnet']
dual = [False]
tol = [1e-2,1e-3,1e-4]
C = [0.5, 0.75,1]
fit_intercept = [False,True]
intercept_scaling = [1]
class_weight = ['balanced']
random_state = [23]
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
max_iter = [75,90,100,115]
multi_class = ['ovr','multinomial','auto']
verbose = [0]
warm_start = [True,False]
n_jobs = [None]
l1_ratio = [0.25,0.5,1.0]

def classification(file,X,Y,x,y):
    # lab_enc = prep.Label'Encoder()
    # X = lab_enc.fit_transform(X)
    # x = lab_enc.fit_transform(x)
    param = []
    acc = []
    for i in it.product(penalty,dual,tol,C,fit_intercept,intercept_scaling,
                        class_weight,random_state,solver,max_iter,multi_class,
                        verbose,warm_start,n_jobs,l1_ratio):
        if cannot(i):
            continue
        # print(*i)
        logreg = lr(*i)
        logreg.fit(X,Y)
        # print('Accuracy: ' + str(logreg.score(x, y)) + '\n')
        acc.append(logreg.score(x,y))
        param.append([*i])
    _results(file,acc,param)

def regressor(file,X,Y,x,y):
    param = []
    acc = []
    for i in it.product(penalty,dual,tol,C,fit_intercept,intercept_scaling,
                        class_weight,random_state,solver,max_iter,multi_class,
                        verbose,warm_start,n_jobs,l1_ratio):
        if cannot(i):
            continue
        # print(*i)
        logreg = lr(*i)
        logreg.fit(X,Y)
        # print('Accuracy: ' + str(np.mean(np.square(logreg.predict_proba(x)-y))) + '\n')
        p = logreg.predict_proba(x)
        print(p[0])
        acc.append(np.mean(np.square(p-y)))
        param.append([*i])
    _results(file,acc,param)

def cannot(s):
    return {'newton-cg', 'l1'}.issubset(set(s)) \
            or {'lbfgs', 'l1'}.issubset(set(s)) \
            or {'lbfgs', 'elasticnet'}.issubset(set(s)) \
            or {'liblinear', 'multinomial'}.issubset(set(s)) \
            or {'sag', 'elasticnet'}.issubset(set(s)) \
            or {'sag', 'l1'}.issubset(set(s)) \
            or {'saga', 'l1'}.issubset(set(s)) \
            or {'saga', 'l2'}.issubset(set(s)) \
            or {'newton-cg', 'elasticnet'}.issubset(set(s)) \
            or {'liblinear', 'elasticnet'}.issubset(set(s))
            # or {'l1'}.issubset(set(s)) \
            # or {'l2',None}.issubset(set(s))

def _results(file,acc,params):
    file.write('LogReg')
    for i in range(len(acc)):
        file.write('\n'+str(acc[i]))
        file.write('\n'+str(params[i]))
    file.write('\nHighest_acc='+str(np.max(acc))+'params='+str(params[np.argmax(acc)]))
    file.write('\nLowest_acc='+str(np.min(acc))+'params='+str(params[np.argmin(acc)]))
