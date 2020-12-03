import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter

#knei test
def kn(target, features, num):

    y=target
    X = features
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=0)
    clf = KNeighborsClassifier()

    print('knei has begun')
    print(datetime.datetime.now(),'\n')

    clf.fit(train_X,train_y)

    print('fitted')
    print(datetime.datetime.now(),'\n')

    yp=clf.predict(test_X)

    print('predicted')
    print(datetime.datetime.now(),'\n')

    print('Predicted == Real: ', clf.score(test_X,test_y),'\n')

    #confusion matrix
    titles_options = [("Confusion matrix", None), ("Confusion matrix, normalized", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, test_X, test_y, cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        plt.savefig('C:/Users/Saulo/source/repos/Classification-Project/Figures/Classification Tests/'+title+'_KN'+num+'.png')
        plt.close()
    #classification report
    print('Classification Report: \n\n',classification_report(test_y, yp),
            '\n____________________________________________________________\n')

#knei test end

#svm test
def svm_(target, features, num):

    y=target
    X = features
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=0)
    clf = svm.SVC()

    print('svm has begun')
    print(datetime.datetime.now(),'\n')

    clf.fit(train_X,train_y)

    print('fitted')
    print(datetime.datetime.now(),'\n')

    yp=clf.predict(test_X)

    print('predicted')
    print(datetime.datetime.now(),'\n')

    print('Predicted == Real: ', clf.score(test_X,test_y),'\n')

    #confusion matrix
    titles_options = [("Confusion matrix", None), ("Confusion matrix, normalized", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, test_X, test_y, cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        plt.savefig('C:/Users/Saulo/source/repos/Classification-Project/Figures/Classification Tests/'+title+'_SVM'+num+'.png')
        plt.close()
    #classification report
    print('Classification Report: \n\n',classification_report(test_y, yp),
            '\n____________________________________________________________\n')

#svm test end

#mlp test
def mlp(target, features, num):

    y=target
    X = features
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=0)
    clf = MLPClassifier()

    print('mlp has begun')
    print(datetime.datetime.now(),'\n')

    clf.fit(train_X,train_y)

    print('fitted')
    print(datetime.datetime.now(),'\n')

    yp=clf.predict(test_X)

    print('predicted')
    print(datetime.datetime.now(),'\n')

    print('Predicted == Real: ', clf.score(test_X,test_y),'\n')

    #confusion matrix
    titles_options = [("Confusion matrix", None), ("Confusion matrix, normalized", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, test_X, test_y, cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        plt.savefig('C:/Users/Saulo/source/repos/Classification-Project/Figures/Classification Tests/'+title+'_MLP'+num+'.png')
        plt.close()
    #classification report
    print('Classification Report: \n\n',classification_report(test_y, yp),
            '\n____________________________________________________________\n')

#mlp test end

#open txt
sys.stdout=open('C:/Users/Saulo/source/repos/Classification-Project/classification_tests_output.txt','w')

data = pd.read_csv("encoded_loan.csv")
target = data.MIS_Status
#1
features = data.drop(columns=['MIS_Status'])
mlp(target,features,'1')
#2
features = features.drop(columns=['CreateJob','RetainedJob'])
mlp(target,features,'2')
#3
features = features.drop(columns=['NewExist'])
mlp(target,features,'3')






#close txt
sys.stdout.close()
#State, NAICS, Term, NoEmp, NewExist, CreateJob, RetainedJob, isFranchise, UrbanRural, LowDoc, GrAppv
#CreateJob and RetainedJob have very little impact according to logit, NewExist looks low as well