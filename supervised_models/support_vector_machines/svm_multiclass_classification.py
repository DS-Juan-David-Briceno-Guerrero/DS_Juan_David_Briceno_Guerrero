#.Welcome to this Machine Learning lesson.
#.This exercise aims to demonstrate how to apply svm to to resolve a multiclass classification problem.
#.The exercise uses a linear support vector machine classifier to address the classification of a fruit regarding on its principal characteristics.

#1.Import libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.svm import LinearSVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

#2.Import the dataset of fruits.
#The exercise works with a linear classifier for this reason two features are selected in order to perform the training.
fruits = pd.read_csv('fruit_data_with_colors.txt',delimiter = "\t")
print(fruits.head(5))

feature_names_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_names_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

#Take two variables two explain the target variable.
X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']

#3.Divide the data into train and test sets.
#For each class in the target variable, a LSVM model is run, and then, for a set of features values (input), the model with higher score on positive classification is used to make the classification of the item.
X_train, X_test, y_train, y_test = train_test_split(X_fruits_2d, y_fruits_2d, random_state = 0)

#4.Scaling the data.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#4.Training the model while making the tuning.
results_train = []
results_test = []
classifiers = []

c_range = [0.001, 0.01, 0.1, 1,5,10, 15,30,50,75,100, 250,500]

for i in c_range:
    clf = LinearSVC(C=i, random_state = 67).fit(X_train_scaled, y_train)
    val_train = clf.score(X_train_scaled, y_train)
    val_test = clf.score(X_test_scaled, y_test)
    results_train.append(val_train)
    results_test.append(val_test)
    classifiers.append(clf)

best_test_result = max(results_test)
best_train_result = results_train[results_test.index(best_test_result)]
clf = classifiers[results_test.index(best_test_result)]

print("Training set accuracy :"+ str(best_train_result))
print("Test set accuracy :"+ str(best_test_result))
print(clf.get_params(deep=True))


#5.predicting the value for a fruit with height =2, and width 6.
#After evaluating the dot product between the model parameters(coeficients and weights), the biggest result gives us the answer about the classification for the evaluated inputs.
results_coe = clf.coef_
results_intercepts = clf.intercept_

print(results_coe.shape)
print(results_intercepts.shape)

x_test_fruit = [2,6]
x_test_fruit = np.array(x_test_fruit)
print(x_test_fruit.shape)


result_all_classes = np.dot(results_coe,x_test_fruit) + results_intercepts

#We can check that the biggest result wiithin the positives outputs is for the second class.
print("all_classes results")
print(result_all_classes)

prediction_result = clf.predict([x_test_fruit])
print("The classification throught the classifier is: "+str(prediction_result[0]))

#6.Plot the boundary of the linear classifier with SVM.
#Plotting the boundary desicions for each SVM model run on the multiclass classification task.
plt.figure(figsize=(6,6))
colors = ['r', 'g', 'b', 'y']
cmap_fruits = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#FFFF00'])

plt.scatter(X_fruits_2d[['height']], X_fruits_2d[['width']],
           c=y_fruits_2d, cmap=cmap_fruits, edgecolor = 'black', alpha=.7)

x_0_range = np.linspace(-10, 15)

for w, b, color in zip(clf.coef_, clf.intercept_, ['r', 'g', 'b', 'y']):
    # Since class prediction with a linear model uses the formula y = w_0 x_0 + w_1 x_1 + b, 
    # and the decision boundary is defined as being all points with y = 0, to plot x_1 as a 
    # function of x_0 we just solve w_0 x_0 + w_1 x_1 + b = 0 for x_1:
    plt.plot(x_0_range, -(x_0_range * w[0] + b) / w[1], c=color, alpha=.8)

plt.legend(target_names_fruits)
plt.xlabel('height')
plt.ylabel('width')
plt.xlim(-2, 12)
plt.ylim(-2, 15)
plt.show()

print("The overall accuracy of the classification done with LSVM is: "+str(clf.score(X_test, y_test)) )

