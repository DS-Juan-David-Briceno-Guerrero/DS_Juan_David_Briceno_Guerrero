#.Welcome to this Machine Learning lesson.
#.This exercise aims to demonstrate how to apply a ridge regression for a regression task problem.
#.The exercise uses a linear regression model under L2 penalty of the weights to address the prediction problem.

#1.Import libraries.
import numpy as np
import pandas as pd
import inspect
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from adspy_shared_utilities import load_crime_dataset
from sklearn.preprocessing import MinMaxScaler

#2.Import the dataframe.
(X_crime, y_crime) = load_crime_dataset()
#Check the dataframe shapes.
print(X_crime.shape)
print(y_crime.shape)
#Check the dataframe types.
print(X_crime.dtypes)

#3.Divide the data into train and test sets.
from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)

linridge = Ridge(alpha=5).fit(X_train, y_train)

print('Crime dataset')
print('ridge regression linear model intercept: {}'
     .format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'
     .format(linridge.coef_))
print('Non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))
print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linridge.score(X_test, y_test)))
print('Number of non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))

#4.Scaling the data to see the improvent on the R squared.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linridge = Ridge(alpha=5).fit(X_train_scaled, y_train)

print('Crime dataset')
print('ridge regression linear model intercept: {}'
     .format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'
     .format(linridge.coef_))
print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linridge.score(X_test_scaled, y_test)))
print('Number of non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))

#5.Tuning of the model.
results_train = []
results_test = []
classifiers = []

for a in [0.01,0.1,1,10,20,23,25,50,100]:
    linridge = Ridge(alpha=a)
    val_train = linridge.score(X_train_scaled, y_train)
    val_test = linridge.score(X_test_scaled, y_test)
    results_train.append(val_train)
    results_test.append(val_test)
    classifiers.append(linridge)
        
best_test_result = max(results_test)
best_train_result = results_train[results_test.index(best_test_result)]
best_classifier = classifiers[results_test.index(best_test_result)]

print("Training set accuracy :"+ str(best_train_result))
print("Test set accuracy :"+ str(best_test_result))
print(best_classifier.get_params(deep=True))
