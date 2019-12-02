import numpy as np
import pandas as pd
import os, math

# fixme: change to dataset location
d = "D:\\Google Drive\\skool\\CS 5361\\datasets\\project\\"
f = "mbti.csv"
rand = np.random.randint(0,100000)
chop = 0.01

class Dataset(object):
    def __init__(self, first_time=False, lo=0.71, hi=0.77, to_load=''):
        self.X = None; self.Y = None
        self.x = None; self.y = None
        self.lo = lo; self.hi = hi
        if first_time:
            self._load()    # read from csv (for first run only)
            self._save()    # save the data as npy for faster future runs
        else:
            self._read(to_load)    # read from npy

    """ FIRST RUN ONLY Read the data to be saved for faster runs """
    def _load(self):
        data = np.asarray(pd.read_csv(d + f, sep=',', header=0))   # load from file
        x, y = self._break(data)   # get the data into samples and target values
        self._split(x, y)   # get training and test sets

    """ Read the data to be stored """
    def _read(self, to_load):
        # find the npy files
        lst = os.listdir(d)
        # and ('baseline' not in s) and ('main' not in s) for non-spec use
        file = [i for i, s in enumerate(lst) if (to_load+'npy' in s)]
        # todo randomly load from list of test/train sets
        self.X = np.load(d+lst[file[1]], allow_pickle=True)
        self.Y = np.load(d+lst[file[3]], allow_pickle=True)
        self.x = np.load(d+lst[file[0]], allow_pickle=True)
        self.y = np.load(d+lst[file[2]], allow_pickle=True)
        print('Loaded',self.X.shape[0],'train, ',self.x.shape[0],'test samples.')

    """ Break the data into target values and training samples """
    def _break(self, data):
        X = []; Y = []
        for i in range(data.shape[0]):
            for s in np.array(data[i][1].split('|||')):
                X.append(s)     # sample
                Y.append(data[i][0])    # target value
        return np.asarray(X), np.asarray(Y)

    """ Split the data into training and test sets """
    def _split(self, x, y):
        # randomly choosing a value between lo%-hi% of dataset for train/test split
        breakpoint = np.random.randint(self.lo*x.shape[0], self.hi*x.shape[0])
        self.X = x[:breakpoint]
        self.Y = y[:breakpoint]
        # saving the test data
        self.x = x[breakpoint+1:]
        self.y = y[breakpoint+1:]
        print('Using',self.X.shape[0],'train, ',self.x.shape[0],'test samples.')

    """ Saves the training and test data into npy for efficient access """
    def _save(self, name=''):
        np.save(d+ 'X_' +str(self.X.shape[0]) + name,self.X)
        np.save(d+ 'Y_' +str(self.X.shape[0]) + name,self.Y)
        np.save(d+ 'x_' +str(self.x.shape[0]) + name,self.x)
        np.save(d+ 'y_' +str(self.x.shape[0]) + name,self.y)
        print('Saved npy files to ' +d+name)

    """ Getter methods: testing and training data """
    def train(self):
        print('Using',math.ceil(chop*self.X.shape[0]),'training samples.')
        return self.X[:rand+math.ceil(chop*self.X.shape[0])], self.Y[:rand+math.ceil(chop*self.Y.shape[0])]

    def test(self):
        print('Using',math.floor(0.25*chop*self.X.shape[0]),'test samples.')
        return self.x[rand+math.floor(0.25*chop*self.X.shape[0]):], self.y[rand+math.floor(0.25*chop*self.y.shape[0]):]

    """ Setter method: testing and training data after preprocessing """
    def set(self, X, Y, x, y, name=''):
        self.X = X
        self.Y = Y
        self.x = x
        self.y = y
        self._save(name)

    """ Get sample count of each personality type """
    def of_each(self):
        classes = ['INTJ', 'INTP', 'INFJ', 'INFP', 'ISTJ', 'ISTP', 'ISFJ',
                   'ISFP', 'ENTJ', 'ENTP', 'ENFJ', 'ENFP', 'ESTJ', 'ESTP',
                   'ESFJ', 'ESFP']
        for c in classes:
            print(str(np.sum(self.Y == c)), c, 'samples')

    """ Get this target function """
    def get_train_target(self, c):
        return self.Y[:,c]

    def get_test_target(self, c):
        return self.y[:,c]
