import numpy as np
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from project import dataset, doc2vec as emb

# fixme: change to dataset location
d = "D:\\Google Drive\\skool\\CS 5361\\datasets\\project\\"
title = 'baseline'

""" To preprocess the data """
def second_run(data):
    data.of_each()
    X, Y, x, y = _preprocess(*data.train(), *data.test())
    data.set(X,Y,x,y,name='_baseline')
    data.of_each(preproc=True)

""" One-time use: preprocess the data """
def _preprocess(X,Y,x,y):
    X, Y = _urls_lower_lem_stem(X,Y)
    x, y = _urls_lower_lem_stem(x,y)
    return X, Y, x, y

""" Removes samples with URLs, and makes samples lowercase """
def _urls_lower_lem_stem(X, Y):
    nwX = []; nwY = []
    ps = PorterStemmer()
    lem = WordNetLemmatizer()
    for i in range(X.shape[0]):
        if 'http' not in X[i]:
            nwX.append(ps.stem(lem.lemmatize(X[i].lower())))
            nwY.append(ps.stem(lem.lemmatize(Y[i].lower())))
    return np.array(nwX), np.array(nwY)

""" Logistic regression """
def build_model(data):
    vec = emb.train(data, title, first_time=True)
    print(vec)
    print(vec.shape)
    X, Y = data.train()
    x, y, = data.test()
    X = vec[:Y.shape[0]]; x = vec[Y.shape[0]+1:]
    model = LogisticRegression(solver='lbfgs', max_iter=1000,multi_class='multinomial')
    print("Score: %f" % model.fit(X,Y).score(x,y))

if __name__=="__main__":
    data = dataset.Dataset(to_load=title+'.')
    # second_run(data)
    build_model(data)
