#.Welcome to this Machine Learning lesson.
#.This exercise aims to demonstrate how to apply a ridge regression for a regression task problem.
#.The exercise uses a linear regression model under L2 penalty of the weights to address the prediction.

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
from sklearn.linear_model import Ridge

#2.Import the dataframe.
(X_crime, y_crime) = load_crime_dataset()
#Check the dataframe shapes.
print(X_crime.shape)
print(y_crime.shape)
#Check the dataframe types.
print(X_crime.dtypes)

#3.Divide the data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)

#4.Scaling the data to see the improvent on the R squared.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Training a generic not tuned ridge regression classifier.
linridge = Ridge(alpha=1).fit(X_train_scaled, y_train)

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
#See the influence in the r_squared metric in ridge regression by varing the parameter alpha.
print('Ridge regression: effect of alpha regularization parameter\n')

results_train = []
results_test = []
classifiers = []

for this_alpha in [0, 1, 10, 20, 50, 100, 1000]:
    linridge = Ridge(alpha = this_alpha).fit(X_train_scaled, y_train)
    r2_train = linridge.score(X_train_scaled, y_train)
    r2_test = linridge.score(X_test_scaled, y_test)
    classifiers.append(linridge)
    results_train.append(r2_train)
    results_test.append(r2_test)
    
    num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
    print('Alpha = {:.2f}\nnum abs(coeff) > 1.0: {}, \
r-squared training: {:.2f}, r-squared test: {:.2f}\n'
         .format(this_alpha, num_coeff_bigger, r2_train, r2_test))
    
        
best_test_result = max(results_test)
best_train_result = results_train[results_test.index(best_test_result)]
best_classifier = classifiers[results_test.index(best_test_result)]

print("Training set accuracy for tuned ridge regression model: "+ str(best_train_result))
print("Test set accuracy for tuned ridge regression model:"+ str(best_test_result))
print(best_classifier.get_params(deep=True))
