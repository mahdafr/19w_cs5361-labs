# Program to build a one-node regression tree
# Programmed by Olac Fuentes
# Last modified September 16, 2019
import sys

import numpy as np
import time


class RegressionTreeNode(object):
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
        if isinstance(self.right, np.float32) and isinstance(self.left, np.float32):
            if self.left==self.right:
                return self.left
            else:
                return False
        if isinstance(self.left, np.float32):
            l = self.left.prune()
        if isinstance(self.right, np.float32):
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
        if isinstance(self.right, np.float32):
            print(indent + '       ', 'pred=', self.right)
        else:
            self.right.print_tree(indent + '    ')

        print(indent, 'if x[' + str(self.attribute) + '] <=', self.threshold)

        if isinstance(self.left, np.float32):
            print(indent + '       ', 'pred=', self.left)
        else:
            self.left.print_tree(indent + '    ')

    # @author mahdafr: part1.2
    def count(self):
        if isinstance(self.left, np.float32):
            count = 1
        else:
            count = self.left.count()
        if isinstance(self.right, np.float32):
            return count + 1
        else:
            count += self.right.count()
        return count

    # @author mahdafr: part1.3
    def height(self):
        if isinstance(self.right, np.float32):
            return 0
        if isinstance(self.left, np.float32):
            return 0
        return max(self.left.height(),self.right.height())+1

    # @author mahdafr: part1.4
    def attr_imp(self, l):
        l[self.attribute] += 1
        if not isinstance(self.right, np.float32):  # left is child
            l = self.right.attr_imp(l)
        if not isinstance(self.left, np.float32):
            l = self.left.attr_imp(l)
        return l


class DecisionTreeRegressor(object):
    # Constructor
    def __init__(self, max_depth=10, min_samples_split=5, max_mse=0.001):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_mse = max_mse

    def fit(self, x, y):
        self.root = self._id3(x, y, depth=0)

    def predict(self, x_test):
        pred = np.zeros(len(x_test), dtype=np.float32)
        for i in range(len(x_test)):
            pred[i] = self._predict(self.root, x_test[i])
        return pred

    def _id3(self, x, y, depth):
        orig_mse = np.var(y)
        # print('original mse:',orig_mse)
        mean_val = np.mean(y)
        if depth >= self.max_depth or len(y) <= self.min_samples_split or orig_mse <= self.max_mse:
            return mean_val

        # @author mahdafr part5
        thr, best_att = self._threshold5(x,y,orig_mse)

        # print('mse best attribute:',mse_attribute[best_att])
        less = x[:, best_att] <= thr[best_att]
        more = ~ less
        # print('subtree mse:',np.var(y[less]),np.var(y[more]))
        # @author mahdafr for part4
        lx = x[less]; ly = y[less]
        rx = x[more]; ry = y[more]
        return RegressionTreeNode(best_att, thr[best_att], self._id3(lx, ly, depth + 1), self._id3(rx, ry, depth + 1))
        # return RegressionTreeNode(best_att, thr[best_att], np.mean(y[less]), np.mean(y[more]))

    # original code
    def _threshold(self, x, y, orig_mse):
        thr = np.mean(x, axis=0)
        mse_attribute = np.zeros(len(thr))
        for i in range(x.shape[1]):
            less = x[:, i] <= thr[i]
            more = ~ less
            mse_attribute[i] = self._mse(y[less], y[more])
        gain = orig_mse - mse_attribute
        # print('Gain:',gain)
        return thr, np.argmax(gain)

    # @author mahdafr for part5
    def _threshold5(self,x,y,orig_mse):
        thr = []    # np.mean(x, axis=1)

        # @author mahdafr for part2
        # generate random values for each attribute
        VALS = 10
        for i in range(x.shape[1]):
            thr.append(np.random.uniform(min(x[:,i]),max(x[:,i]),size=(VALS)))
        thr = np.asarray(thr)
        mse_attribute = np.zeros(len(thr))

        thresh = []
        for i in range(x.shape[1]):
            m = sys.maxsize; ind = 0
            for j in range(len(thr[i])):
                less = x[:, i] <= thr[i][j]
                more = ~ less
                new = self._mse(y[less], y[more])
                if new<m:
                    m = new; ind = j
            mse_attribute[i] = m
            thresh.append(thr[i][ind])
        gain = orig_mse - mse_attribute
        # print('Gain:',gain)
        return np.asarray(thresh), np.argmax(gain)

    def _mse(self, l, m):
        err = np.append(l - np.mean(l), m - np.mean(m))  # It will issue a warning if either l or m is empty
        return np.mean(err * err)

    def _predict(self, dt_node, x):
        if isinstance(dt_node, np.float32):
            return dt_node
        if x[dt_node.attribute] <= dt_node.threshold:
            return self._predict(dt_node.left, x)
        else:
            return self._predict(dt_node.right, x)

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


if __name__=="__main__":
    TESTS = 5
    test_acc_avg = 0; train_acc_avg = 0
    test_time_avg = 0; train_time_avg = 0
    test_acc_avg_p = 0; train_acc_avg_p = 0
    test_time_avg_p = 0; train_time_avg_p = 0
    print('Tests:  ' + str(TESTS))
    dir = 'D:\Google Drive\skool\CS 5361\datasets\\'
    for i in range(TESTS):
        skip = np.random.randint(40,50)
        x_train = np.load(dir + 'x_ray_data_train.npy')[::skip]
        y_train = np.load(dir + 'x_ray_target_train.npy')[::skip]
        x_test = np.load(dir + 'x_ray_data_test.npy')[::skip]
        y_test = np.load(dir + 'x_ray_target_test.npy')[::skip]

        model = DecisionTreeRegressor()
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


        print('\nPruned Tree: ')
        model = DecisionTreeRegressor()
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
        print('\n')

    print('Average training time  {0:.6f} '.format(train_time_avg/TESTS))
    print('Average testing time  {0:.6f} '.format(test_time_avg/TESTS))
    print('average train accuracy:', train_acc_avg/TESTS)
    print('average test accuracy:', test_acc_avg/TESTS)

    print('\nPruned:')
    print('Average training time {0:.6f} '.format(train_time_avg_p/TESTS))
    print('Average testing time  {0:.6f} '.format(test_time_avg_p/TESTS))
    print('average train accuracy:', train_acc_avg_p/TESTS)
    print('average test accuracy:', test_acc_avg_p/TESTS)
