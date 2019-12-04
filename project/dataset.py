import numpy as np
import pandas as pd
import os, math

# fixme: change to dataset location
d = "D:\\Google Drive\\skool\\CS 5361\\datasets\\project\\"
f = "mbti.csv"

class Dataset(object):
    def __init__(self, first_time=False, lo=0.71, hi=0.77, to_load='', chop=0.0005):
        self.X = None; self.Y = None
        self.x = None; self.y = None
        self.lo = lo; self.hi = hi
        self.chop = chop
        self.rand = np.random.randint(1,100000)
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
        print('Using',math.ceil(self.chop*self.X.shape[0]),'training samples.')
        self.Y = self._representation(self.Y)
        self.X = self.X[self.rand+math.ceil(self.chop*self.X.shape[0]):]
        self.Y = self.Y[self.rand+math.ceil(self.chop*self.Y.shape[0]):]
        return self.X, self.Y

    def test(self):
        print('Using',math.floor(0.25*self.chop*self.X.shape[0]),'test samples.')
        self.y = self._representation(self.y)
        self.x = self.x[self.rand:self.rand+math.floor(0.25*self.chop*self.x.shape[0])]
        self.y = self.y[self.rand:self.rand+math.floor(0.25*self.chop*self.y.shape[0])]
        return self.x, self.y

    """ Change the representation of target values. """
    def _representation(self, y, one_hot=True):
        if one_hot:
            y = list(s.replace('E', '0') for s in y)
            y = list(s.replace('I', '1') for s in y)
            y = list(s.replace('S', '0') for s in y)
            y = list(s.replace('N', '1') for s in y)
            y = list(s.replace('F', '0') for s in y)
            y = list(s.replace('T', '1') for s in y)
            y = list(s.replace('P', '0') for s in y)
            y = list(s.replace('J', '1') for s in y)
        return np.array(y)

    """ Setter method: testing and training data after pre-processing """
    def set(self, X, Y, x, y, name=''):
        self.X = X
        self.Y = Y
        self.x = x
        self.y = y
        self._save(name)

    """ Get sample count of each personality/class type """
    def of_each(self, in_strings=False):
        if in_strings:
            classes = ['INTJ', 'INTP', 'INFJ', 'INFP', 'ISTJ', 'ISTP', 'ISFJ',
                       'ISFP', 'ENTJ', 'ENTP', 'ENFJ', 'ENFP', 'ESTJ', 'ESTP',
                       'ESFJ', 'ESFP']
        else:
            classes = ['1111', '1110', '1101', '1100', '1011', '1010', '1001',
                       '1000', '0111', '0110', '0101', '0100', '0011', '0010',
                       '0001', '0000']
        for c in classes:
            print(str(np.sum(self.Y == c)), c, 'samples')

    """ Get the target value for c """
    def get_train_target(self, c):
        return np.array([x[c] for x in self.Y])

    def get_test_target(self, c):
        return np.array([x[c] for x in self.y])
