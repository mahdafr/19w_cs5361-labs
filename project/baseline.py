import numpy as np
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from project import dataset, doc2vec as emb

# fixme: change to dataset location
d = "D:\\Google Drive\\skool\\CS 5361\\datasets\\project\\"
title = 'baseline'

""" To preprocess the data """
def second_run(data):
    data.of_each()
    X, Y, x, y = _preprocess(*data.train(), *data.test())
    data.set(X,Y,x,y,name='_'+title)
    data.of_each()

""" One-time use: preprocess the data """
def _preprocess(X,Y,x,y):
    X, Y = _urls_lower_lem_stem_stop(X,Y)
    x, y = _urls_lower_lem_stem_stop(x,y)
    return X, Y, x, y

""" Removes samples with URLs, lemmatizes, stems, then removes stopwords """
def _urls_lower_lem_stem_stop(X, Y):
    nwX = []; nwY = []
    ps = PorterStemmer()
    lem = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # for each sample
    for i in range(X.shape[0]):
        if 'http' not in X[i]:
            filt = ps.stem(lem.lemmatize(X[i].lower()))
            nwX.append(str([w for w in word_tokenize(filt) if not w in stop_words]))
            nwY.append(Y[i])
    return np.array(nwX), np.array(nwY)

""" Logistic regression """
def build_model(data):
    X, Y, x, y = emb.train(data, title+'_')
    model = LogisticRegression(solver='lbfgs', max_iter=100,multi_class='multinomial')
    print('Classifier:\tLogistic Regression')
    for i in range(4):
        print("Score:\t%f" % model.fit(X,Y[i]).score(x,y[i]))

if __name__=="__main__":
    data = dataset.Dataset(to_load=title+'.')
    # second_run(data)
    build_model(data)
