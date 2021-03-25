#.Welcome to this Machine Learning lesson.
#.This exercise aims to demonstrate how to apply a logistic regression for a binary classification problem.
#.The exercise uses a logit  classifier to address the classification of a fruit regarding on its principal characteristics.

#1.Import libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer

#2.Import the dataset of fruits.
fruits = pd.read_csv('/home/juan-david/Documents/data_science/travail_personnel/machine_learning_michigan_university/fruit_data_with_colors.txt',delimiter = "\t")
print(fruits.head(5))

feature_names_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']
X_fruits_2d = fruits[['height', 'width']]
print(X_fruits_2d)
y_fruits_2d = fruits['fruit_label']
print(y_fruits_2d)

#3.Create the new target variable to check wether the target is apple or not.
y_fruits_apple = y_fruits_2d == 1
print(y_fruits_apple)

#4.Divide data into train and test set.
X_train, X_test, y_train, y_test = (train_test_split(X_fruits_2d.values,y_fruits_apple.values, random_state = 0))

#5.Scaling the data.
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#6.Training a generic logistic regression classifier.
clf = LogisticRegression(C=100).fit(X_train, y_train)
print('train score of the model :'+str(clf.score(X_train,y_train)))
print('test score of the model :'+str(clf.score(X_test,y_test)))

#7.Tuning the model.
results_train = []
results_test = []
classifiers = []

for c in [0.001,0.01,0.1, 0.5, 1,50, 100]:
    clf = LogisticRegression(C=c).fit(X_train, y_train)
    val_train = clf.score(X_train, y_train)
    val_test = clf.score(X_test, y_test)
    results_train.append(val_train)
    results_test.append(val_test)
    classifiers.append(clf)
        
best_test_result = max(results_test)
best_train_result = results_train[results_test.index(best_test_result)]
best_classifier = classifiers[results_test.index(best_test_result)]

print("Training set accuracy :"+ str(best_train_result))
print("Test set accuracy :"+ str(best_test_result))
print(best_classifier.get_params(deep=True))


#8.Applying logistic regression over the brest cancer data set.
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
print(X_cancer)
print(X_cancer.shape)
print(y_cancer)
print(y_cancer.shape)

#Split data into train and test sets.
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
print(X_train)
print(y_train)

#Scaling the data.
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Training a generic logit classifier.
clf = LogisticRegression().fit(X_train, y_train)
print('Breast cancer dataset')
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

#9.Tuning the model.
results_train = []
results_test = []
classifiers = []

for c in [0.001,0.01,0.1,0.5, 1,50, 100]:
    clf = LogisticRegression(C=c).fit(X_train, y_train)
    val_train = clf.score(X_train, y_train)
    val_test = clf.score(X_test, y_test)
    results_train.append(val_train)
    results_test.append(val_test)
    classifiers.append(clf)
        
best_test_result = max(results_test)
best_train_result = results_train[results_test.index(best_test_result)]
best_classifier = classifiers[results_test.index(best_test_result)]

print("Training set accuracy :"+ str(best_train_result))
print("Test set accuracy :"+ str(best_test_result))
print(best_classifier.get_params(deep=True))
