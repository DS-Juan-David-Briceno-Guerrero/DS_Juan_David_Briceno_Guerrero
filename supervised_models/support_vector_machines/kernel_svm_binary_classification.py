#.Welcome to this Machine Learning lesson.
#.This exercise aims to demonstrate how to apply support-vector machines to resolve a classification task.
#.The problem assists into a classification task where the space is not linearly separable. Thus, Kernels must be used.

#1.Import the libraries.
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_classification, make_blobs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.preprocessing import MinMaxScaler
from adspy_shared_utilities import plot_class_regions_for_classifierX_train_scaled

#2.Creating a synthetic dataset to do binary classification using KSVM.
#A two dimensional plot is shown with the results of the target value as color points, 
#Scaling is not done over the features because the dimensions of points is pretty similar.
cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])
X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers = 8,cluster_std = 1.3, random_state = 4)
y_D2 = y_D2 % 2
plt.figure()
plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2,marker= 'o', s=50, cmap=cmap_bold)
plt.show()

#Checking the dimensions of the data.
print(X_D2.shape)
print(y_D2.shape)

#3.Training, and showing the boundary regions of classification for by using KSVM with RBF.

#Spliting data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)

#4.Training the data for different values of parameters Gamma, and C in the KSVM.
#Gamma controls how much distance between points of the same class matters the depict the regions of classification. Higher values of Gamma involves the regions to be more precise with observation points.
#C controls the amount of regularization of the SVM model. This parameter is inversely proportional to the classifier margin. Thus, as long as the C increases the model regularizes less and try to fit better the data points.    

#Creating the plots for each iteration.
fig, subaxes = plt.subplots(3, 4, figsize=(15, 10), dpi=50)

#Training iteratively over sets of values for parameters Gamma, and C.
#The test scores (accuracy) for each combination is calculated as well as the classification regions of each model.
for this_gamma, this_axis in zip([0.01, 1, 5], subaxes):
    
    for this_C, subplot in zip([0.1, 1, 15, 250], this_axis):
        title = 'gamma = {:.2f}, C = {:.2f}'.format(this_gamma, this_C)
        clf = SVC(kernel = 'rbf', gamma = this_gamma,
                 C = this_C).fit(X_train, y_train)
        plot_class_regions_for_classifier_subplot(clf, X_train, y_train, X_test, y_test, title, subplot)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        
#5.Working over a real dataframe.
#Applying KSVM to a real life data set.
#df of the brest and cancer dataframe.
df = pd.read_csv('breast_cancer_dataset.csv')

print (df.head(3))
print("df shape :" +str(df.shape))

#5.1. The dataset is downloaded from scikit learn repositories. Categorical columns have been removed from the dataframe for exemplification proposals.
#The dataframe is converted then into a numpy array, and into scientific notation.
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
print("X_cancer shape :" +str(X_cancer.shape))
print("y_cancer shape :" +str(y_cancer.shape))

#Scaling the data points.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Variables' values in the training and test sets are now lying in the range between zero to one.
print(X_train_scaled)
print(X_test_scaled)

#Giving the results of a generic trained model.
clf = SVC().fit(X_train_scaled, y_train)
print('Breast cancer dataset (normalized with MinMax scaling)')
print('RBF-kernel SVC (with MinMax scaling) training set accuracy: {:.2f}'
     .format(clf.score(X_train_scaled, y_train)))
print('RBF-kernel SVC (with MinMax scaling) test set accuracy: {:.2f}'
     .format(clf.score(X_test_scaled, y_test)))

#6.Improving the model results with the tunning of the parameters C and Gamma.
results_train = []
results_test = []
classifiers = []

c_range = [0.001, 0.01, 0.1, 1,5,10, 15,30,50,75,100, 250,500]
g_range = [0.001,0.01,0.05,0.1,0.5, 1,2,5,10]

for this_gamma in g_range:
    for this_C in c_range:
        clf = SVC(kernel = 'rbf', gamma = this_gamma, C = this_C).fit(X_train_scaled, y_train)
        val_train = clf.score(X_train_scaled, y_train)
        val_test = clf.score(X_test_scaled, y_test)
        results_train.append(val_train)
        results_test.append(val_test)
        classifiers.append(clf)

best__test_result = max(results_test)
best_train_result = results_train[results_test.index(best__test_result)]
best_classifier = classifiers[results_test.index(best__test_result)]

print("Training set accuracy :"+ str(best_train_result))
print("Test set accuracy :"+ str(best__test_result))
print(best_classifier.get_params(deep=True))
        


