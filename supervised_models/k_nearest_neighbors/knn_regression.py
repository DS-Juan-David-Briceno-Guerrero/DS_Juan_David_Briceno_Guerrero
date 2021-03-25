#.Welcome to this Machine Learning lesson.
#.This exercise aims to demonstrate how to apply k-nearest neighbors to resolve a regression task.
#.The exercise uses knn to address a regression problem.
#1.Import libraries.
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
np.set_printoptions(precision=2)


#2.Create the dataframe to perform the regression task.
plt.figure()
plt.title('Sample regression problem with one input variable')
X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)
plt.scatter(X_R1, y_R1, marker= 'o', s=50)
plt.show()



#3.Divide data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state = 0)

#4.Training a generic model.
#There is only one feature and for that reason there is not scaling part in this example.
knnreg = KNeighborsRegressor(n_neighbors = 2).fit(X_train, y_train)
print('R-squared training score: {:.3f}'
     .format(knnreg.score(X_train, y_train)))
print('R-squared test score: {:.3f}'
     .format(knnreg.score(X_test, y_test)))

#5.Tunning the model.
results_train = []
results_test = []
classifiers = []

for i in range(1,20):
    knnreg = KNeighborsRegressor(n_neighbors = i)
    knnreg.fit(X_train, y_train)
    val_train = knnreg.score(X_train, y_train)
    val_test = knnreg.score(X_test, y_test)
    results_train.append(val_train)
    results_test.append(val_test)
    classifiers.append(knnreg)
    
best_test_result = max(results_test)
best_train_result = results_train[results_test.index(best_test_result)]
best_classifier = classifiers[results_test.index(best_test_result)]

print("Training set accuracy :"+ str(best_train_result))
print("Test set accuracy :"+ str(best_test_result))
print(best_classifier.get_params(deep=True))
