from lab4.dataset import Data
from lab4 import knn, dectree, forest, logreg, svm

def classification(file,X, Y, x, y):
    file.write('\n==========Classification predictor model==========')
    # knn.classification(file,X,Y,x,y)
    # print('=')
    # dectree.classification(file,X,Y,x,y)
    # print('==')
    # forest.classificaation(file,X,Y,x,y)
    # print('===')
    logreg.classification(file,X,Y,x,y)
    print('====')
    # svm.classification(file,X,Y,x,y)
    # print('=====')

def regressor(file,X, Y, x, y):
    file.write('\n=============Regressor predictor model============')
    # knn.regressor(file,X,Y,x,y)
    # print('=')
    # dectree.regressor(file,X,Y,x,y)
    # print('==')
    # forest.regressor(file,X,Y,x,y)
    # print('===')
    # logreg.regressor(file,X,Y,x,y)
    # print('====')
    # svm.regressor(file,X,Y,x,y)
    # print('=====')

if __name__ == "__main__":
    data = Data()
    f = open("res.txt", "a")
    for i in range(data.sets()):
        f.write('\n\nDataset: ' + str(data.using(i)))
        x_train, y_train, x_test, y_test = data.get(i)
        classification(f,x_train, y_train, x_test, y_test)
        regressor(f,x_train, y_train, x_test, y_test)
    f.close()
