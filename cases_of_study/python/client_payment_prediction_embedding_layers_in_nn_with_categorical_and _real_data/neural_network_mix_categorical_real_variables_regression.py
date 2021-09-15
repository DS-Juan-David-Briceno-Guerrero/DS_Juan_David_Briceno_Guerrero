#The objective of this exercise is to apply neural networks within categorical, and real variables for the prediction  of a client payments amount wihtin a bank of Argentina. 
#Tensorflow must be at least the version 2.o
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
import kerastuner as kt
#import keras and the keras tuner to do hyperparametertuning
#pip install -q -U keras-tuner

#1.import libraries.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler

#Path to helper functions of neural_network_mix_categorical_real_variables_regression. 
import sys
sys.path.insert(0, 'path/to/your/py_file')
import helper_functions_neural_network_mix_categorical_real_variables_regression


# Load the TensorBoard notebook extension
%load_ext tensorboard
# Load the TensorBoard notebook extension
#rm -rf ./logs/
from tensorboard.plugins.hparams import api as hp

#2.Load data.
df = pd.read_csv (r'/home/juan-david/Documents/data_science/globant_work/pre_master_exercise_docs/pre_master.csv',sep=",")
print(df)
#print(df.shape)
#print(df.iloc[:,27:28].empty)

#3.Adapting and structuring the data.
#Count nan_percents in dataframe.
na_percent_values_dataframe(df)
x = df

#Create feature variables.
x= create_features_variables(df,"PaymentAmt")
print(x.shape)

#Capturing the categorical columns from the x dataframe.
#sellerId = x['SellerId']
currencyCode = x['CurrencyCode']
day_number = df['DayNumber']
visingerScoreBins_1 = x['VisingerScoreBins-1']
visingerScoreBins_2 = x['VisingerScoreBins-2']
visingerScoreBins_3 =  x['VisingerScoreBins-3']
visingerScoreBins_4 =  x['VisingerScoreBins-4']
visingerScoreBins_5 = x['VisingerScoreBins-5']
visingerScoreBins_6 = x['VisingerScoreBins-6']
visingerScoreBins_7 = x['VisingerScoreBins-7']
visingerScoreBins_8 = x['VisingerScoreBins-8']
visingerScoreBins_9 = x['VisingerScoreBins-9']
visingerScoreBins_10 = x['VisingerScoreBins-10']
visingerScoreBins_11 = x['VisingerScoreBins-11']
feedBackGroup_1 = x['FeedBackGroup-1']
feedBackGroup_2 = x['FeedBackGroup-2']
feedBackGroup_3 = x['FeedBackGroup-3']
feedBackGroup_4 = x['FeedBackGroup-4']
feedBackGroup_5 = x['FeedBackGroup-5']
feedBackGroup_6 = x['FeedBackGroup-6']
feedBackGroup_7 = x['FeedBackGroup-7']
feedBackGroup_8 = x['FeedBackGroup-8']
feedBackGroup_9 = x['FeedBackGroup-9']
feedBackGroup_10 = x['FeedBackGroup-10']
feedBackGroup_11 = x['FeedBackGroup-11']

month = df['MonthDate']
month = pd.to_datetime(month, dayfirst = True)
month = month.dt.month 

#Defining a dataframe containing the categorical variables after modifications.
x_categoielle = pd.concat([currencyCode, month, day_number,visingerScoreBins_1,visingerScoreBins_2,visingerScoreBins_3,visingerScoreBins_4,visingerScoreBins_5,visingerScoreBins_6,visingerScoreBins_7,visingerScoreBins_8,visingerScoreBins_9,visingerScoreBins_10,visingerScoreBins_11,feedBackGroup_1,feedBackGroup_2,feedBackGroup_3,feedBackGroup_4,feedBackGroup_5,feedBackGroup_6,feedBackGroup_7,feedBackGroup_8,feedBackGroup_9,feedBackGroup_10,feedBackGroup_11], axis=1)
print(x_categoielle.shape)

x_categoielle = erase_na_rows_dataframe(x_categoielle)
dataframe = x_categoielle[0]
list_removed_rows = x_categoielle[1]

x_categoielle = dataframe 
print(x_categoielle.shape)

#Define the dependent variables of the model
y = y.drop(y.index[list_removed_rows]).reset_index(drop=True)
print(y.shape)


#Creating a dataframe containing real variables from the set of features. 
#Deleting all variable columns contained in the dataframe of categorical variables.
x_reelles = x.drop(['SellerId','CurrencyCode','MonthDate','DayNumber','CycleNumber','DayDate','VisingerScoreBins-1','VisingerScoreBins-2','VisingerScoreBins-3','VisingerScoreBins-4','VisingerScoreBins-5','VisingerScoreBins-6','VisingerScoreBins-7','VisingerScoreBins-8','VisingerScoreBins-9','VisingerScoreBins-10','VisingerScoreBins-11','FeedBackGroup-1','FeedBackGroup-2','FeedBackGroup-3','FeedBackGroup-4','FeedBackGroup-5','FeedBackGroup-6','FeedBackGroup-7','FeedBackGroup-8','FeedBackGroup-9','FeedBackGroup-10','FeedBackGroup-11'], axis=1)
x_reelles = x_reelles.drop(x_reelles.index[list_removed_rows]).reset_index(drop=True)
print(x_reelles.shape)

real_features = columns_names_dataframe(x_reelles)
categorical_features = columns_names_dataframe(x_categoielle)
print("Does the dataframe contains infs values? : "+str(contain_infs_values(x_reelles)))

x_reelles = replace_infs_by_zero_dataframe(x_reelles)
print("Does the dataframe contains infs values? : "+str(contain_infs_values(x_reelles)))

x_reelles = replace_na_values_zeros_dataframe(x_reelles)
print(x_reelles.shape)

print(x_categoielle.shape)
print(x_reelles.shape)

real_names = columns_names_dataframe(x_reelles)
categorical_names = columns_names_dataframe(x_categoielle)

features = append_columns_names(real_names,categorical_names)
print(len(features))


#Create a train set x concatenating the two set of features variables.
x = concatenate_real_categorical_variables_dataframes(x_reelles,x_categoielle,features)
print(x.shape)
print(x)

X_train, X_test, y_train, y_test = train_test_split_dataframe(x, y, 0.3,42)

x_reelles_1, x_categoielle_1 = deconcatenate_real_categorical_variables_dataframe(X_train, real_features, categorical_features)

x_reelles_2, x_categoielle_2 = deconcatenate_real_categorical_variables_dataframe(X_test, real_features, categorical_features)

print("train")
print(x_reelles_1.shape)
print(x_categoielle_1.shape)
print("test")
print(x_reelles_2.shape)
print(x_categoielle_2.shape)
print(x_reelles_1)
print(x_categoielle_1)

x_reelles_1 , std_training = standardize_training_set(x_reelles_1, real_features)
print(x_reelles_1)

x_reelles_2 = standardize_test_set(x_reelles_2, real_features,std_training )
print(x_reelles_2)

print(x_categoielle_2)
x_categoielle_2  =  label_encode_dataframe_test(x_categoielle_2, metadata)

print("x_categoielle_2_encoded")
print(x_categoielle_2)

X_train = concatenate_real_categorical_variables_dataframes(x_reelles_1,x_categoielle_1, features)
print(X_train.shape)

X_test = concatenate_real_categorical_variables_dataframes(x_reelles_2,x_categoielle_2, features)
print(X_test.shape)


#3.Create the model with the embedding layers.
#Creer un model avec tous les variables (apres faire le encoder avec embeddings) pour developper le reseau de neuron.
#layers_nn is a list containing the hiden layers and their nodes to built the neural_network model.

import kerastuner as kt
hp = kt.HyperParameters()

#--------------------------------------------------
#Tunning with hyperband.
tuner = kt.Hyperband(model_builder,
                     objective='val_mean_absolute_error',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

#Create a callback to stop training early after reaching a certain value for the validation loss.
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search([x_train.loc[:,[f]].values for f in features], y_train.values, epochs=50, validation_split=0.2, callbacks=[stop_early])


# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(best_hps.get("units_1"))
print(best_hps.get("units_2"))
print(best_hps.get("units_3"))
print(best_hps.get("dropout_1"))
print(best_hps.get("dropout_2"))
print(best_hps.get("dropout_3"))
print(best_hps.get("learning_rate"))

checkpoint_path = "training_hypertuner_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1,period = 1)

model = tuner.hypermodel.build(best_hps)
history = model.fit([x_train.loc[:,[f]].values for f in features], y_train.values, epochs=2, validation_split=0.2, callbacks =[cp_callback])

val_mean_absolute_error_per_epoch = history.history['val_mean_absolute_error']
best_epoch = val_mean_absolute_error_per_epoch.index(min(val_mean_absolute_error_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


hypermodel = tuner.hypermodel.build(best_hps)

#4.Retrain the model
hypermodel.fit([x_train.loc[:,[f]].values for f in features], y_train.values, epochs=best_epoch,callbacks =[cp_callback])



eval_result = hypermodel.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)

#4.Obatain the best model find by the hypertunner.
#Parameters to include in the model.

#256
#320
#256
#0.2
#0.2
#0.1
#0.001

layers_units = [256,320,256]
#print(layers_units[0])
#print(len(layers_units))

layers_dropouts = [0.2,0.2,0.1]
#print(layers_dropouts[0])
#print(len(layers_dropouts))
learning_rate_best = 0.001


#5.Tune the best model found with the hypertunner.
checkpoint_path = "training_hypertuner_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1,period = 1)

model = get_model_for_hypertuner_callback(real_features, x_categoielle, categorical_features,len(layers_units),layers_units,layers_dropouts,learning_rate_best)

model.load_weights("training_hypertuner_1/cp-0044.ckpt")
#loss , mae = model.evaluate([x_test.loc[:,[f]].values for f in features],y_test.values,verbose = 2)
#print("restored model, accuracy:"+str(mae))

model.fit([x_train.loc[:,[f]].values for f in features], y_train.values, epochs=75, validation_data=([x_test.loc[:,[f]].values for f in features],y_test.values),verbose=2, callbacks =[cp_callback])


