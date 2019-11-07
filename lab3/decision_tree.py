# Program to build a one-node decision tree
# Programmed by Olac Fuentes
# Last modified September 16, 2019
import sys

import numpy as np
import time


class DecisionTreeNode(object):
    # Constructor
    def __init__(self, att, thr, left, right):
        self.attribute = att
        self.threshold = thr
        # left and right are either binary classifications or references to
        # decision tree nodes
        self.left = left
        self.right = right

    # @author mahdafr: part1.1
    def prune(self):
        l = False; r = False
        if self.left in [0,1] and self.right in [0,1]:
            if self.left==self.right:
                return self.left
            else:
                return False
        if self.left not in [0,1]:
            l = self.left.prune()
        if self.right not in [0,1]:
            r = self.right.prune()
        if l is not False:
            self.left = l
        if r is not False:
            self.right = r
        if l==r and l is not False:
            return l
        return False

    def print_tree(self, indent=''):
        # If prints the right subtree, corresponding to the condition x[attribute] > threshold
        # above the condition stored in the node
        if self.right in [0, 1]:
            print(indent + '       ', 'class=', self.right)
        else:
            self.right.print_tree(indent + '    ')

        print(indent, 'if x[' + str(self.attribute) + '] <=', self.threshold)

        if self.left in [0, 1]:
            print(indent + '       ', 'class=', self.left)
        else:
            self.left.print_tree(indent + '    ')

    # @author mahdafr: part1.2
    def count(self):
        if self.left in [0, 1]:
            count = 1
        else:
            count = self.left.count()
        if self.right in [0, 1]:
            return count + 1
        else:
            count += self.right.count()
        return count

    # @author mahdafr: part1.3
    def height(self):
        if self.right in [0, 1]:
            return 0
        if self.left in [0, 1]:
            return 0
        return max(self.left.height(),self.right.height())+1

    # @author mahdafr: part1.4
    def attr_imp(self, l):
        l[self.attribute] += 1
        if self.right not in [0,1]:  # left is child
            l = self.right.attr_imp(l)
        if self.left not in [0,1]:
            l = self.left.attr_imp(l)
        return l


class DecisionTreeClassifier(object):
    # Constructor
    def __init__(self, max_depth=8, min_samples_split=10, min_accuracy=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_accuracy = min_accuracy

    def fit(self, x, y):
        self.root = self._id3(x, y, depth=0)

    def predict(self, x_test):
        pred = np.zeros(len(x_test), dtype=int)
        for i in range(len(x_test)):
            pred[i] = self._classify(self.root, x_test[i])
        return pred

    def _id3(self, x, y, depth):
        orig_entropy = self._entropy(y, [])
        mean_val = np.mean(y)

        # if accuracy not attained and cannot go further in tree
        if depth >= self.max_depth or len(y) <= self.min_samples_split or max(
                [mean_val, 1 - mean_val]) >= self.min_accuracy:
            return int(round(mean_val))

        # @author mahdafr for part2-part3
        thr, best_att = self._threshold3(x, y, orig_entropy)

        less = x[:, best_att] <= thr[best_att]
        more = ~ less
        # @author mahdafr for part1
        lx = x[less]; ly = y[less]  # int(round(np.mean(y[less])))
        rx = x[more]; ry = y[more]  # int(round(np.mean(y[more])))
        return DecisionTreeNode(best_att, thr[best_att], self._id3(lx,ly,depth+1), self._id3(rx,ry,depth+1))

    # original code
    def _threshold(self, x, y, orig_entropy):
        thr = np.mean(x, axis=0)
        entropy_attribute = np.zeros(len(thr))

        # foreach training example, find entropy for each attribute
        for i in range(x.shape[1]):
            less = x[:, i] <= thr[i]
            more = ~ less
            entropy_attribute[i] = self._entropy(y[less], y[more])
        gain = orig_entropy - entropy_attribute
        # print('Gain:',gain)
        return thr, np.argmax(gain)

    # @author mahdafr for part3
    def _threshold3(self, x, y, orig_entropy):
        thr = []    # np.mean(x, axis=1)

        # @author mahdafr for part2
        # generate random values for each attribute
        VALS = 10
        for i in range(x.shape[1]):
            thr.append(np.random.uniform(min(x[:,i]),max(x[:,i]),size=(VALS)))
        thr = np.asarray(thr)
        entropy_attribute = np.zeros(len(thr))

        thresh = []
        # find entropy
        for i in range(x.shape[1]):
            m = sys.maxsize; ind = 0
            for j in range(len(thr[i])):
                less = x[:, i] <= thr[i][j]
                more = ~ less
                new = self._entropy(y[less], y[more])
                if new<m:
                    m = new
                    ind = j
            thresh.append(thr[i][ind])
            entropy_attribute[i] = m
        gain = orig_entropy - entropy_attribute
        # print('Gain:',gain)
        return np.asarray(thresh), np.argmax(gain)

    def _entropy(self, l, m):
        ent = 0
        for p in [l, m]:
            if len(p) > 0:
                pp = sum(p) / len(p)
                pn = 1 - pp
                if pp < 1 and pp > 0:
                    ent -= len(p) * (pp * np.log2(pp) + pn * np.log2(pn))
        ent = ent / (len(l) + len(m))
        return ent

    def _classify(self, dt_node, x):
        if dt_node in [0, 1]:
            return dt_node
        if x[dt_node.attribute] <= dt_node.threshold:
            return self._classify(dt_node.left, x)
        else:
            return self._classify(dt_node.right, x)

    # @author mahdafr: part1.1
    def _prune(self):
        self.root.prune()

    def display(self):
        print('Model:')
        self.root.print_tree()

    # @author mahdafr: part 1.2
    def count_nodes(self):
        n = self.root.count()
        print('Nodes in tree: ' + str(n))
        return n

    # @author mahdafr: part 1.3
    def height(self):
        print('Height of tree: ' + str(self.root.height()))

    # @author mahdafr: part1.4
    def attr_importance(self, s, n):
        # average number of times attr used to predict
        # target function value of each training example
        a = np.zeros(s)
        a = self.root.attr_imp(a)
        for i in range(len(a)):
            print('Attribute ' +str(i)+ ' Importance =' + str(a[i]/n))
        print('Best Attribute: ' + str(np.argmax(a)))

# methods for reading data and splitting data
def read():
    x = []; y = []
    infile = open("D:\Projects\cs5361-labs\lab2\magic04.txt", "r")
    for line in infile:
        y.append(int(line[-2:-1] == 'g'))
        x.append(np.fromstring(line[:-2], dtype=float, sep=','))
    infile.close()
    xa = np.zeros((len(x), len(x[0])))
    for i in range(len(xa)):
        xa[i] = x[i]
    x = xa
    return np.array(x).astype(np.float32), np.array(y)

def split(x,y):
    # Split data into training and testing
    ind = np.random.permutation(len(y))
    split_ind = int(len(y) * 0.8)
    x_train = x[ind[:split_ind]]
    x_test = x[ind[split_ind:]]
    y_train = y[ind[:split_ind]]
    y_test = y[ind[split_ind:]]
    return x_train, y_train, x_test, y_test


if __name__== "__main__":
    TREES = 5
    test_acc_avg = 0; train_acc_avg = 0
    test_time_avg = 0; train_time_avg = 0
    test_acc_avg_p = 0; train_acc_avg_p = 0
    test_time_avg_p = 0; train_time_avg_p = 0
    print('Forest Size:  ' + str(TREES))
    x, y = read()
    forest = []
    for i in range(TREES):
        x_train, y_train, x_test, y_test = split(x, y)

        model = DecisionTreeClassifier()
        start = time.time()
        model.fit(x_train, y_train)
        train_time_avg += time.time() - start
        train_pred = model.predict(x_train)

        start = time.time()
        test_pred = model.predict(x_test)
        test_time_avg += time.time() - start
        train_acc_avg += np.sum(train_pred == y_train) / len(train_pred)
        test_acc_avg += np.sum(test_pred == y_test) / len(test_pred)

        # model.display()
        n = model.count_nodes()
        model.height()
        forest.append(model)    # add the un-pruned tree

        print('\nPruned Tree: ')
        model = DecisionTreeClassifier()
        start = time.time()
        model.fit(x_train, y_train)
        model._prune()  # for testing purposes
        train_time_avg_p += time.time() - start
        train_pred = model.predict(x_train)

        start = time.time()
        test_pred = model.predict(x_test)
        test_time_avg_p += time.time() - start
        train_acc_avg_p += np.sum(train_pred == y_train) / len(train_pred)
        test_acc_avg_p += np.sum(test_pred == y_test) / len(test_pred)

        # model.display()
        n = model.count_nodes()
        model.height()
        model.attr_importance(x_train.shape[1],n)
        forest.append(model)    # add the pruned tree
        print('\n')

    print('Average training time  {0:.6f} '.format(train_time_avg/TREES))
    print('Average testing time  {0:.6f} '.format(test_time_avg/TREES))
    print('average train accuracy:', train_acc_avg/TREES)
    print('average test accuracy:', test_acc_avg/TREES)

    print('\nPruned:')
    print('Average training time {0:.6f} '.format(train_time_avg_p/TREES))
    print('Average testing time  {0:.6f} '.format(test_time_avg_p/TREES))
    print('average train accuracy:', train_acc_avg_p/TREES)
    print('average test accuracy:', test_acc_avg_p/TREES)
