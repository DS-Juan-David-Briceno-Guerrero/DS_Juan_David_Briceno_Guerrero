#.Welcome to this Machine Learning lesson.
#.This exercise aims to demonstrate how to apply a decision tree for a binary classification problem.
#.The exercise uses a decision tree classifier to address the classification of a person having cancer or not.

#1.Import libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree
from adspy_shared_utilities import plot_feature_importances
from sklearn.model_selection import RandomizedSearchCV

#2.Import the dataset.
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
print(X_cancer)
print(X_cancer.shape)
print(y_cancer)
print(y_cancer.shape)

#3.Divide data into train and test set.
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)


#4.Scaling the data.
#Scaling the features.
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#4,Define a desicion tree classifier and training the model with the data. 
clf = DecisionTreeClassifier(max_depth = 1, min_samples_leaf = 8,
                            random_state = 0).fit(X_train, y_train)


print('Breast cancer dataset: decision tree')
print('Accuracy of DT classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of DT classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

print('test score of the model :'+str(clf.score(X_test,y_test)))


#5.Tuning the model.
clf = DecisionTreeClassifier(random_state = 0)
parameters = {
    'max_depth': range(1,12),
    'criterion' : ('gini', 'entropy'),
    'max_features' : ('auto', 'sqrt', 'log2'),
    'min_samples_leaf' : (2,4,6,8,10,12),
    'min_samples_split' : (2,4,6,8,10,12)
}

DT_grid = RandomizedSearchCV(clf, param_distributions = parameters, cv =5, verbose = True)
DT_grid.fit(X_train,y_train)


#5.Evaluating the score of thw model.
print('train score of the model :'+str(DT_grid.score(X_train,y_train)))
print('test score of the model :'+str(DT_grid.score(X_test,y_test)))
print(DT_grid.best_estimator_)

plot_decision_tree(DT_grid.best_estimator_, cancer.feature_names, cancer.target_names)
