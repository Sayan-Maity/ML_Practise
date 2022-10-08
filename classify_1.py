from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading the datasets
iris = datasets.load_iris()

# printing description and features
# print(iris.DESCR)
features = iris.data
labels = iris.target
# print(features[0], labels[0])

# Training the classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)

preds = clf.predict([[9.1, 9.5, 6.4, 0.2]])
print(preds)







