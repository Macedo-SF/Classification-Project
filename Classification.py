import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter

data = pd.read_csv("clean_loan.csv")
target = data.MIS_Status
features = data.drop(columns=['MIS_Status']) #go back and fix city
y = target
X = features
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=0)

#fit
clf = MLPClassifier(alpha=0.01,max_iter=2000)
print('it has begun')
print(datetime.datetime.now())

clf.fit(train_X,train_y)

print('fitted')
print(datetime.datetime.now())

yp=clf.predict(test_X)

print('predicted')
print(datetime.datetime.now())

print('Predicted == Real: ', clf.score(test_X,test_y))

#confusion matrix
titles_options = [("Confusion matrix", None), ("Confusion matrix, normalized", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, test_X, test_y, cmap=plt.cm.Blues, normalize=normalize)
    disp.ax_.set_title(title)
    plt.show()
#classification report
print('Classification Report: \n\n',classification_report(test_y, yp),
        '\n____________________________________________________________\n')

"""
size=len(target)
trSize=int(size*0.8)
teSize=size-trSize

#fits
clf = MLPClassifier(alpha=0.01,max_iter=2000)
y_train=target[:-trSize]
x_train=features[:-trSize]
y_test=target[-trSize:]
x_test=features[-trSize:]

print('it has begun')
print(datetime.datetime.now())

clf.fit(x_train,y_train)

print('fitted')
print(datetime.datetime.now())

yp=clf.predict(x_test)

print('predicted')
print(datetime.datetime.now())

comp=yp==y_test
c=Counter(comp)
print('Predicted == Real: ', c[1]/(c[0]+c[1]))

#confusion matrix
titles_options = [("Confusion matrix", None), ("Confusion matrix, normalized", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues, normalize=normalize)
    disp.ax_.set_title(title)
    plt.show()
#classification report
print('Classification Report: \n\n',classification_report(y_test, yp),
        '\n____________________________________________________________\n')
"""