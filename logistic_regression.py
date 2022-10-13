'''
A regression algorithm which does classification and calculates
the probility of belonging to a particular class
it takes your favourite features and labels and fits
a linear model (weights and biases)

And instead of giving you the result, it gives you the
logistic of the result
'''
# Train a logistic regression classifier to predict whether a flower
# is iris virginica or not

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np

# print(list(iris.keys()))
# print(iris['data'].shape)
# print(iris['target'])
# print(iris['DESCR'])
# print(Y)

iris = datasets.load_iris()
X = iris["data"][:, 3:]
Y = (iris["target"] == 2).astype(np.int)

# Train a logistic regression classifier
clf = LogisticRegression()
clf.fit(X, Y)
example = clf.predict(([[2.6]]))
print(example)

# Using matplotlib to plot the visualization
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
Y_prob = clf.predict_proba(X_new)
print(Y_prob)
plt.plot(X_new, Y_prob[:, 1], "g-", label="virginica")
plt.show()





















