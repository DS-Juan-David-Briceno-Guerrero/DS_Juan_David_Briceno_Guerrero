#.Welcome to this Machine Learning lesson.
#.This exercise aims to demonstrate how to apply a Lasso regression for a regression problem.
#.The exercise uses a linear regression classifier with L1 penalty to in the weights to address the prediction problem.

#1.Import libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from adspy_shared_utilities import load_crime_dataset

#2.Import the dataframe.
(X_crime, y_crime) = load_crime_dataset()
#Check the dataframe shapes.
print(X_crime.shape)
print(y_crime.shape)
#Check the dataframe types.
print(X_crime.dtypes)

#3.Divide the data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)

#4.Scale the data before the training.
scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#5.Defining the linear regression classifier with L1 penalty, as well as training the model with a generic model (without tuning).
linlasso = Lasso(alpha=2.0, max_iter = 10000).fit(X_train_scaled, y_train)

#Showing the intercept, and the number of coefficients having not zero values for the weights in the regression.
print('Crime dataset')
print('lasso regression linear model intercept: {}'
     .format(linlasso.intercept_))
print('lasso regression linear model coeff:\n{}'
     .format(linlasso.coef_))
print('Non-zero features: {}'
     .format(np.sum(linlasso.coef_ != 0)))
print('R-squared score (training): {:.3f}'
     .format(linlasso.score(X_train_scaled, y_train)))

#Regression model score.
print('R-squared score (test): {:.3f}\n'
     .format(linlasso.score(X_test_scaled, y_test)))
print('Features with non-zero weight (sorted by absolute magnitude):')


#Showing the feature coefficients that are not equal to zero after the model's training.
for e in sorted (list(zip(list(X_crime), linlasso.coef_)),
    key = lambda e: -abs(e[1])):
    if e[1] != 0:
        print('\t{}, {:.3f}'.format(e[0], e[1]))
        
#6.Tuning the parameter alpha in the lasso regression model.
print('Lasso regression: effect of alpha regularization\n\
parameter on number of features kept in final model\n')

results_train = []
results_test = []
classifiers = []

for alpha in [0.5, 1, 2, 3, 5, 10, 20, 50]:
    linlasso = Lasso(alpha, max_iter = 10000).fit(X_train_scaled, y_train)
    r2_train = linlasso.score(X_train_scaled, y_train)
    r2_test = linlasso.score(X_test_scaled, y_test)
    classifiers.append(linlasso)
    results_train.append(r2_train)
    results_test.append(r2_test)
    
    print('Alpha = {:.2f}\nFeatures kept: {}, r-squared training: {:.2f}, \r-squared test: {:.2f}\n'.format(alpha, np.sum(linlasso.coef_ != 0), r2_train, r2_test))
    
    
best_test_result = max(results_test)
best_train_result = results_train[results_test.index(best_test_result)]
best_classifier = classifiers[results_test.index(best_test_result)]

print("Training set accuracy for tuned lasso regression model: "+ str(best_train_result))
print("Test set accuracy for tuned lasso regression model:"+ str(best_test_result))
print(best_classifier.get_params(deep=True))
