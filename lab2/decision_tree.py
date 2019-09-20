# Program to build a one-node decision tree
# Programmed by Olac Fuentes
# Last modified September 16, 2019

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


class DecisionTreeClassifier(object):
    # Constructor
    def __init__(self, max_depth=10, min_samples_split=10, min_accuracy=1):
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
        thr = np.mean(x, axis=0)
        entropy_attribute = np.zeros(len(thr))

        # foreach training example, find the attr w/best entropy
        for i in range(x.shape[1]):
            less = x[:, i] <= thr[i]
            more = ~ less
            entropy_attribute[i] = self._entropy(y[less], y[more])
        gain = orig_entropy - entropy_attribute

        print('Gain:',gain)
        best_att = np.argmax(gain)
        less = x[:, best_att] <= thr[best_att]
        more = ~ less
        lx = x[less]; ly = y[less]  # int(round(np.mean(y[less])))
        rx = x[more]; ry = y[more]  # int(round(np.mean(y[more])))
        return DecisionTreeNode(best_att, thr[best_att], self._id3(lx,ly,depth+1), self._id3(rx,ry,depth+1))

    def _build_tree(self):
        pass

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

    def display(self):
        print('Model:')
        self.root.print_tree()


x = []
y = []
infile = open("magic04.txt", "r")
for line in infile:
    y.append(int(line[-2:-1] == 'g'))
    x.append(np.fromstring(line[:-2], dtype=float, sep=','))
infile.close()

xa = np.zeros((len(x), len(x[0])))
for i in range(len(xa)):
    xa[i] = x[i]
x = xa

x = np.array(x).astype(np.float32)
y = np.array(y)

# Split data into training and testing
ind = np.random.permutation(len(y))
split_ind = int(len(y) * 0.8)
x_train = x[ind[:split_ind]]
x_test = x[ind[split_ind:]]
y_train = y[ind[:split_ind]]
y_test = y[ind[split_ind:]]


model = DecisionTreeClassifier()
start = time.time()
model.fit(x_train, y_train)
elapsed_time = time.time() - start
print('Elapsed_time training  {0:.6f} '.format(elapsed_time))

train_pred = model.predict(x_train)
start = time.time()
test_pred = model.predict(x_test)
elapsed_time = time.time() - start
print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))

train_acc = np.sum(train_pred == y_train) / len(train_pred)
print('train accuracy:', train_acc)
test_acc = np.sum(test_pred == y_test) / len(test_pred)
print('test accuracy:', test_acc)

model.display()
