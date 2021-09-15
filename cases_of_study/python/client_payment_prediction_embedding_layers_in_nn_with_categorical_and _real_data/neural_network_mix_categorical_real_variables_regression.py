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

#3.helper functions.
#Compter le pourcentage des donnees NA de chaque column d'un Dataframe specifie par argument.
#Summarize the percentage of NA values in each column of a dataframe passed by parameter.
def na_percent_values_dataframe(df):
    print("The number of rows in the dataframe is:"+str(df.shape[0]))
    print("")
    print("-------------------counting Nan Values------")

    for i in range(df.shape[1]):
	    # count number of rows with missing values
        n_miss = df.iloc[:,[i]].isna().sum()
        n_miss = n_miss.to_numpy(n_miss)
        n_miss = n_miss[0]
        perc = n_miss/df.shape[0]
	    #perc = (n_miss)/(df.shape[0])
        print("Column index:"+str(i)+","+" Number of missings :"+str(n_miss)+","+" Percent :"+str(perc))

na_percent_values_dataframe(df)

x = df
#x = x.drop(["PaymentAmt"], axis =1)
#print(x.shape)
def create_features_variables(df, target_column_name):
    x= df
    x= x.drop([target_column_name], axis=1)
    return x
x= create_features_variables(df,"PaymentAmt")
print(x.shape)

def create_target_variable(df, target_column_name):
    y = df[target_column_name]
    return y
y = create_target_variable(df,"PaymentAmt")
print(y.shape)

#Method that removes from a dataframe the rows in which there are NaN values.
def erase_na_rows_dataframe(df):
    
    rows_with_nan = []
    metadata = []

    for index, row in df.iterrows():

        is_nan_series = row.isnull()
        if is_nan_series.any():
            rows_with_nan.append(index)
    
    df = df.drop(df.index[rows_with_nan]).reset_index(drop=True)

    metadata.append(df)
    metadata.append(rows_with_nan)
    return(metadata)

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
#y = df["PaymentAmt"]
y = y.drop(y.index[list_removed_rows]).reset_index(drop=True)
print(y.shape)


#Creating a dataframe containing real variables from the set of features. 
#Deleting all variable columns contained in the dataframe of categorical variables.
x_reelles = x.drop(['SellerId','CurrencyCode','MonthDate','DayNumber','CycleNumber','DayDate','VisingerScoreBins-1','VisingerScoreBins-2','VisingerScoreBins-3','VisingerScoreBins-4','VisingerScoreBins-5','VisingerScoreBins-6','VisingerScoreBins-7','VisingerScoreBins-8','VisingerScoreBins-9','VisingerScoreBins-10','VisingerScoreBins-11','FeedBackGroup-1','FeedBackGroup-2','FeedBackGroup-3','FeedBackGroup-4','FeedBackGroup-5','FeedBackGroup-6','FeedBackGroup-7','FeedBackGroup-8','FeedBackGroup-9','FeedBackGroup-10','FeedBackGroup-11'], axis=1)
x_reelles = x_reelles.drop(x_reelles.index[list_removed_rows]).reset_index(drop=True)
print(x_reelles.shape)

#Method aiming to capture the columns in a dataframe passed by argument.
def columns_names_dataframe(df):
    names = [f for f in df.columns]
    list_of_real_values = []

    if len(names[1][0])==1:
        for i in range(len(names)):
            name = names[i]
            list_of_real_values.append(name)
        names = list_of_real_values
    if len(names[1][0])!=1:
        for i in range(len(names)):
            name = names[i][0]
            list_of_real_values.append(name)
        names = list_of_real_values    
    return names

real_features = columns_names_dataframe(x_reelles)
categorical_features = columns_names_dataframe(x_categoielle)


def contain_infs_values(df):
    df_contient_inf = df[(df == np.inf).any(axis=1)]
    a = True
    if df_contient_inf.empty:
        a=False
    return(a)

print("Does the dataframe contains infs values? : "+str(contain_infs_values(x_reelles)))

#Method that replaces the infinite values of dataframe by zero
def replace_infs_by_zero_dataframe(df):
    df = pd.DataFrame(x_reelles)
    df = df.replace([np.inf, -np.inf], 0)

    
    return df

x_reelles = replace_infs_by_zero_dataframe(x_reelles)
print("Does the dataframe contains infs values? : "+str(contain_infs_values(x_reelles)))

#x_reelles = erase_inf_dataframe(x_reelles,0)
#print(x_reelles)

Method that change NA values in rows by the average of its corresponding column,
#The method applies for a dataframe containing only real values,
def replace_na_values_column_average_dataframe(df, df_columns):
    df_modified = df
    for i in range(len(df_columns)):
        df_modified.iloc[:, i].fillna(value=df_modified.iloc[:, i].mean(), inplace=True)
    return(df_modified)

#x_reelles = replace_na_values_column_average_dataframe(x_reelles, real_features)

#Method that change NA values in rows by the average of its corresponding column,
#The method applies for a dataframe containing only real values
def replace_na_values_zeros_dataframe(df):
    df_modified = df 
    df_modified = df_modified.fillna(0)
    return(df_modified)

x_reelles = replace_na_values_zeros_dataframe(x_reelles)
print(x_reelles.shape)

print(x_categoielle.shape)
print(x_reelles.shape)

real_names = columns_names_dataframe(x_reelles)
categorical_names = columns_names_dataframe(x_categoielle)

#Method that receives by argument two lists of names, and concatenate them.
def append_columns_names(list_1, list_2):
    features = []

    for i in real_names:
        features.append(i)
    for j in categorical_names:
        features.append(j)

    return(features)

features = append_columns_names(real_names,categorical_names)
print(len(features))

#Creations des fonctions pour afficher les labels_encoders, le dimension des vecteurs d'encoding et le numbero d'unique classes.
def label_encode_column(df, column): 
   
    lbl_enc = LabelEncoder()
    lbl_enc.fit(df[column].astype(str).values)
    #lbl_enc_classes = lbl_enc.classes_
    df.loc[:,column] = lbl_enc.transform(df[column].astype(str).values)

    return df, lbl_enc#, #lbl_enc_classes

#Ca fait la transformation des donnees pour les passes avec le format label-encodng.
def label_encode_dataframe(df, df_columns):
    
    list_columns_names = []
    list_lbl_encoders = []
    #list_lbl_encoders_classes = []

    for i in df_columns:
        df_enc, lbl_enc_column = label_encode_column(df,i)
        list_columns_names.append(i)
        list_lbl_encoders.append(lbl_enc_column)
        #list_lbl_encoders_classes.append(classes_column)

    metadata = np.column_stack((list_columns_names,list_lbl_encoders))#,list_lbl_encoders_classes))
    
    return df , metadata

def encode_test_column(df,column,lbl_enc):
    lbl_enc_test = lbl_enc
    df.loc[:,column] = lbl_enc_test.transform(df[column].astype(str).values)
    return df

def obtain_lists_metadata(metadata):
    column_names = metadata[:, 0]
    column_names = column_names.tolist()
    column_encoders = metadata[:, 1]
    column_encoders = column_encoders.tolist()
    return column_names , column_encoders

def label_encode_dataframe_test(df,metadata):

    a , b = obtain_lists_metadata(metadata)

    for i in range(0,len(a)):
        column = a[i]
        lbl_column = b[i]
        encode_test_column(df,column,lbl_column)
    return df

def concatenate_real_categorical_variables_dataframes(df_real_variables, df_categorical_variables, features_names):

    unified_df = pd.concat([df_real_variables, df_categorical_variables], axis=1)

    unified_df.columns = features_names

    return(unified_df)

#Create a train set x concatenating the two set of features variables.
x = concatenate_real_categorical_variables_dataframes(x_reelles,x_categoielle,features)
print(x.shape)
print(x)

def deconcatenate_real_categorical_variables_dataframe(df, real_variables_names, categorical_variables_names):
    real_variables = df[real_variables_names]
    categorical_variables = df[categorical_variables_names]
    return real_variables, categorical_variables

#Perform the division into train and split sets for a dataframe passed by parameter. 
def train_test_split_dataframe(x,y, test_sample_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_sample_size, random_state=random_state)
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    return X_train, X_test, y_train, y_test

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


#Create a method that standardizes a dataframe passed by argument and returns it afterwards.
def standardize_training_set(df_to_standardize, columns):
    std = StandardScaler()
    df = std.fit_transform(df_to_standardize)
    df = std.transform(df_to_standardize)
    df = pd.DataFrame(df)
    df.columns = [columns] 
    return df, std

x_reelles_1 , std_training = standardize_training_set(x_reelles_1, real_features)

print(x_reelles_1)

def standardize_test_set(df_test, columns, std_trai):
    std = std_trai
    df_test = std.transform(df_test)
    df_test = pd.DataFrame(df_test)
    df_test.columns = [columns]  
    return df_test

x_reelles_2 = standardize_test_set(x_reelles_2, real_features,std_training )

print(x_reelles_2)

print(x_categoielle_2)
x_categoielle_2  =  label_encode_dataframe_test(x_categoielle_2, metadata)

print("x_categoielle_2_encoded")
print(x_categoielle_2)

def variable_num_categories(df):
    df = pd.DataFrame.to_numpy(df)
    df_categories = np.unique(df)
    num_categories = df_categories.shape[0]
    return num_categories

def emb_rule_size(n_categories):
    return int(min(np.ceil((n_categories)/2),50))

X_train = concatenate_real_categorical_variables_dataframes(x_reelles_1,x_categoielle_1, features)
print(X_train.shape)

X_test = concatenate_real_categorical_variables_dataframes(x_reelles_2,x_categoielle_2, features)
print(X_test.shape)


#Create the model with the embedding layers.
#Creer un model avec tous les variables (apres faire le encoder avec embeddings) pour developper le reseau de neuron.
#layers_nn is a list containing the hiden layers and their nodes to built the neural_network model.
def get_model_for_hypertuner(real_variables_columns, df_categorical_variables, categorical_variables_columns, num_layers,hp):
    inputs = []
    outputs = []

    for j in real_variables_columns:
        inp_variables_reelles = tf.keras.layers.Input(shape=(1,))
        out_variables_reelles = tf.keras.layers.Reshape(target_shape=(1,))(inp_variables_reelles)
        inputs.append(inp_variables_reelles)
        outputs.append(out_variables_reelles)
    
    for c in categorical_variables_columns:
        num_unique_vals = variable_num_categories(df_categorical_variables[c])
        embed_dim = emb_rule_size(num_unique_vals)
        inp = tf.keras.layers.Input(shape=(1,))
        out = tf.keras.layers.Embedding(num_unique_vals + 1, embed_dim , name =c)(inp)
        out = tf.keras.layers.Reshape(target_shape=(embed_dim,))(out)
        inputs.append(inp)
        outputs.append(out)
   
    x = tf.keras.layers.Concatenate()(outputs)
    
    if num_layers ==1:
        hp_units = hp.Int('units_1' , min_value=32, max_value=350, step=32)
        x = tf.keras.layers.Dense(hp_units, activation='relu',kernel_initializer ='he_normal')(x)

        dropout_rate = hp.Float('dropout_1', 0.1, 0.3, step=0.1)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    else:
        for i in range(1,num_layers+1):
            hp_units = hp.Int('units_' + str(i), min_value=32, max_value=350, step=32)
            x = tf.keras.layers.Dense(hp_units, activation='relu',kernel_initializer ='he_normal')(x)

            dropout_rate = hp.Float('dropout_'+ str(i), 0.1, 0.3, step=0.1)
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    #for i, nodes in enumerate(layers_nn):
            #x = tf.keras.layers.Dense(nodes,activation='relu',kernel_initializer='he_normal')(x)
            #x = tf.keras.layers.Dropout(0.3)(x)

    #x = tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation='relu',kernel_initializer ='he_normal')(x)
    #x = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(x)

    y = tf.keras.layers.Dense(1, activation='relu',)(x)
    model = tf.keras.Model(inputs = inputs, outputs = y)
    #   model.summary()
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])

    model.compile(
    loss=tf.keras.losses.MeanAbsoluteError(name='mean_absolute_error'),
    optimizer= tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
    metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    return model

import kerastuner as kt
hp = kt.HyperParameters()
#Running the model with one layer
def model_builder(hp):
  model = get_model_for_hypertuner(real_features, x_categoielle, categorical_features, 3,hp)

  return model

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

# Retrain the model
hypermodel.fit([x_train.loc[:,[f]].values for f in features], y_train.values, epochs=best_epoch,callbacks =[cp_callback])



eval_result = hypermodel.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)




#4.Obatain the best model find by the hypertunner.
#Defining a method to obtain a generic model of the already trained NN.
#Charge the training state of the hypermodel.
def get_model_for_hypertuner_callback(real_variables_columns, df_categorical_variables, categorical_variables_columns, num_layers,layers_units,layer_dropouts, learning_rate):
    inputs = []
    outputs = []

    for j in real_variables_columns:
        inp_variables_reelles = tf.keras.layers.Input(shape=(1,))
        out_variables_reelles = tf.keras.layers.Reshape(target_shape=(1,))(inp_variables_reelles)
        inputs.append(inp_variables_reelles)
        outputs.append(out_variables_reelles)
    
    for c in categorical_variables_columns:
        num_unique_vals = variable_num_categories(df_categorical_variables[c])
        embed_dim = emb_rule_size(num_unique_vals)
        inp = tf.keras.layers.Input(shape=(1,))
        out = tf.keras.layers.Embedding(num_unique_vals + 1, embed_dim , name =c)(inp)
        out = tf.keras.layers.Reshape(target_shape=(embed_dim,))(out)
        inputs.append(inp)
        outputs.append(out)
   
    x = tf.keras.layers.Concatenate()(outputs)
    
    if num_layers ==1:
        #hp_units = hp.Int('units_1' , min_value=32, max_value=512, step=32)
        x = tf.keras.layers.Dense(layers_units[0], activation='relu',kernel_initializer ='he_normal')(x)

        #dropout_rate = hp.Float('dropout_1', 0.1, 0.3, step=0.1)
        x = tf.keras.layers.Dropout(layer_dropouts[0])(x)

    else:
        for i in range(1,num_layers+1):
            #hp_units = hp.Int('units_' + str(i), min_value=32, max_value=512, step=32)
            x = tf.keras.layers.Dense(layers_units[i-1], activation='relu',kernel_initializer ='he_normal')(x)

            #dropout_rate = hp.Float('dropout_'+ str(i), 0.1, 0.3, step=0.1)
            x = tf.keras.layers.Dropout(layer_dropouts[i-1])(x)

    #for i, nodes in enumerate(layers_nn):
            #x = tf.keras.layers.Dense(nodes,activation='relu',kernel_initializer='he_normal')(x)
            #x = tf.keras.layers.Dropout(0.3)(x)

    #x = tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation='relu',kernel_initializer ='he_normal')(x)
    #x = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(x)

    y = tf.keras.layers.Dense(1, activation='relu',)(x)
    model = tf.keras.Model(inputs = inputs, outputs = y)
    #   model.summary()
    
    #hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])

    model.compile(
    loss=tf.keras.losses.MeanAbsoluteError(name='mean_absolute_error'),
    optimizer= tf.keras.optimizers.Adam(learning_rate=learning_rate),
    metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    return model

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


