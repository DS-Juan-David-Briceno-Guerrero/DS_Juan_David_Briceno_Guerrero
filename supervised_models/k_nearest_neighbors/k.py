#.Welcome to this Machine Learning lesson.
#.This exercise aims to demonstrate how to apply k-nearest neighbors to resolve a classification task.
#.The exercise uses knn to address a multiclass classification problem.
#1.Import libraries.
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from matplotlib.colors import ListedColormap
from adspy_shared_utilities import load_crime_dataset
from sklearn.preprocessing import MinMaxScaler
np.set_printoptions(precision=2)

#2.Import the dataframe.
#The dataframe contains labels of fruits and their characteristics such as weight, width, mass, color, and others.
fruits = pd.read_csv('fruit_data_with_colors.txt',delimiter = "\t")
print(fruits.head(5))

#3.Select features to train the model.
feature_names_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

#4.Split data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0)

#5.standardiz features with the min-max scaler.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)

#6.Creating the knn classifier.
knn = KNeighborsClassifier(n_neighbors = 5)

#7.training athe generic classifier.
knn.fit(X_train_scaled, y_train)
print("The test score of the model is:"+ str(knn.score(X_test_scaled, y_test)))

#8.Improving the model results with the tunning of the parameters K (if possible).
results_train = []
results_test = []
classifiers = []

for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train_scaled, y_train)
    val_train = knn.score(X_train_scaled, y_train)
    val_test = knn.score(X_test_scaled, y_test)
    results_train.append(val_train)
    results_test.append(val_test)
    classifiers.append(knn)
    
best_test_result = max(results_test)
best_train_result = results_train[results_test.index(best_test_result)]
best_classifier = classifiers[results_test.index(best_test_result)]

print("Training set accuracy :"+ str(best_train_result))
print("Test set accuracy :"+ str(best_test_result))
print(best_classifier.get_params(deep=True))

