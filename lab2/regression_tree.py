# Program to build a one-node regression tree
# Programmed by Olac Fuentes
# Last modified September 16, 2019

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
        thr = np.mean(x, axis=0)
        mse_attribute = np.zeros(len(thr))
        for i in range(x.shape[1]):
            less = x[:, i] <= thr[i]
            more = ~ less
            mse_attribute[i] = self._mse(y[less], y[more])
        gain = orig_mse - mse_attribute
        # print('Gain:',gain)
        best_att = np.argmax(gain)
        # print('mse best attribute:',mse_attribute[best_att])
        less = x[:, best_att] <= thr[best_att]
        more = ~ less
        # print('subtree mse:',np.var(y[less]),np.var(y[more]))
        return RegressionTreeNode(best_att, thr[best_att], np.mean(y[less]), np.mean(y[more]))

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

    def display(self):
        print('Model:')
        self.root.print_tree()


print('\nSolar particle dataset')
skip = 10
x_train = np.load('x_ray_data_train.npy')[::skip]
y_train = np.load('x_ray_target_train.npy')[::skip]
x_test = np.load('x_ray_data_test.npy')[::skip]
y_test = np.load('x_ray_target_test.npy')[::skip]

model = DecisionTreeRegressor()
start = time.time()
model.fit(x_train, y_train)
elapsed_time = time.time() - start
print('Elapsed_time training  {0:.6f} '.format(elapsed_time))

pred = model.predict(x_train)
print('Mean square error traning set:', np.mean(np.square(pred - y_train)))
start = time.time()
pred = model.predict(x_test)
elapsed_time = time.time() - start
print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))
print('Mean square error test set:', np.mean(np.square(pred - y_test)))

model.display()