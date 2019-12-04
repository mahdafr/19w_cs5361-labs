import numpy as np
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from project import dataset, doc2vec as emb
from sklearn.preprocessing import MinMaxScaler

# fixme: change to dataset location
d = "D:\\Google Drive\\skool\\CS 5361\\datasets\\project\\"
title = 'main'

""" To preprocess the data """
def second_run(data):
    data.of_each()
    X, Y, x, y = _preprocess(*data.train(), *data.test())
    data.set(X,Y,x,y,name='_main')
    data.of_each()

""" One-time use: preprocess the data """
def _preprocess(X,Y,x,y):
    X, Y = _urls_lower_stem(X,Y)
    x, y = _urls_lower_stem(x,y)
    return X, Y, x, y

""" Removes samples with URLs, and makes samples lowercase """
def _urls_lower_stem(X, Y):
    nwX = []; nwY = []
    ps = PorterStemmer()
    for i in range(X.shape[0]):
        if 'http' in X[i]:  # chop the URL from the string instead
            X[i] = re.sub(r"(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*", '', X[i], re.MULTILINE)
        nwX.append(str([w for w in word_tokenize(ps.stem(X[i].lower()))]))
        nwY.append(Y[i])
    return np.array(nwX), np.array(nwY)

""" Naive Bayes' Models """
def build_model_gauss(data):
    X, Y, x, y = emb.train(data, title+'_', first_time=True)
    # todo change the parameters for tests
    # pred(X, x, data, GaussianNB(), name='Gaussian')
    pred(*rescale(X,x), data, MultinomialNB(), name='Multinomial')

""" Train and report on results """
def pred(X, x, data, model, name=''):
    print('Classifier:', name)
    score = []
    for i in range(4):
        score.append(model.fit(X,data.get_train_target(i)).score(x,data.get_test_target(i)))
        print("\tTarget="+str(i),"Score:\t%f" % score[i])
    print("Overall:",str(np.average(score)))

""" Rescales the features for MNB """
def rescale(X, x):
    s = MinMaxScaler()
    s.fit(np.append(X,x))
    return s.transform(X), s.transform(x)

if __name__=="__main__":
    data = dataset.Dataset(to_load=title+'.', chop=0.0005)
    # second_run(data)
    build_model_gauss(data)
