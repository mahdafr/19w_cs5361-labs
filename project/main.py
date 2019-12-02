import numpy as np
import re
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from project import dataset, word2vec as emb

# fixme: change to dataset location
d = "D:\\Google Drive\\skool\\CS 5361\\datasets\\project\\"
glove = "D:\\Projects\\cs5361-labs\\_dataset\\glove.6B-50d.txt"

""" To preprocess the data """
def second_run(data):
    data.of_each()
    X, Y, x, y = _preprocess(*data.train(), *data.test())
    data.set(X,Y,x,y,name='_main')
    data.of_each(preproc=True)

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
        nwX.append(ps.stem(X[i].lower()))
        nwY.append(ps.stem(Y[i].lower()))
    return np.array(nwX), np.array(nwY)

""" Naive Bayes' Models """
def build_model_gauss(data):
    # todo change the parameters for tests
    vec = emb.train(data, title, first_time=True)
    X, Y = data.train()
    x, y, = data.test()
    X = vec[:Y.shape[0]]; x = vec[Y.shape[0]+1:]
    pred(X,Y,x,y, GaussianNB(), name='Gaussian')
    pred(X,Y,x,y, MultinomialNB(), name='Multinomial')

def pred(X,Y,x,y, model, name=''):
    print('Classifier:', name)
    print("Score: %f" % model.fit(X,Y).score(x,y))

if __name__=="__main__":
    data = dataset.Dataset()
    # second_run(data)

    build_model_gauss(data)
