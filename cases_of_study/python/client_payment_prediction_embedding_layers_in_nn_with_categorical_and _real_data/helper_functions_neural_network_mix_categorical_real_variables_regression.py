#Helper functions for the client payment prediction exercise.
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

def create_features_variables(df, target_column_name):
    x= df
    x= x.drop([target_column_name], axis=1)
    return x

def create_target_variable(df, target_column_name):
    y = df[target_column_name]
    return y

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

#Defining a method aiming to capture the columns in a dataframe passed by argument.
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


def contain_infs_values(df):
    df_contient_inf = df[(df == np.inf).any(axis=1)]
    a = True
    if df_contient_inf.empty:
        a=False
    return(a)

def replace_infs_by_zero_dataframe(df):
    df = pd.DataFrame(x_reelles)
    df = df.replace([np.inf, -np.inf], 0)

    
    return df

#The method applies for a dataframe containing only real values,
def replace_na_values_column_average_dataframe(df, df_columns):
    df_modified = df
    for i in range(len(df_columns)):
        df_modified.iloc[:, i].fillna(value=df_modified.iloc[:, i].mean(), inplace=True)
    return(df_modified)

#Method that change NA values in rows by the average of its corresponding column,
#The method applies for a dataframe containing only real values
def replace_na_values_zeros_dataframe(df):
    df_modified = df 
    df_modified = df_modified.fillna(0)
    return(df_modified)

#Method that receives by argument two lists of names, and concatenate them.
def append_columns_names(list_1, list_2):
    features = []

    for i in real_names:
        features.append(i)
    for j in categorical_names:
        features.append(j)

    return(features)

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

#Create a method that standardizes a dataframe passed by argument and returns it afterwards.
def standardize_training_set(df_to_standardize, columns):
    std = StandardScaler()
    df = std.fit_transform(df_to_standardize)
    df = std.transform(df_to_standardize)
    df = pd.DataFrame(df)
    df.columns = [columns] 
    return df, std

def standardize_test_set(df_test, columns, std_trai):
    std = std_trai
    df_test = std.transform(df_test)
    df_test = pd.DataFrame(df_test)
    df_test.columns = [columns]  
    return df_test

def variable_num_categories(df):
    df = pd.DataFrame.to_numpy(df)
    df_categories = np.unique(df)
    num_categories = df_categories.shape[0]
    return num_categories

def emb_rule_size(n_categories):
    return int(min(np.ceil((n_categories)/2),50))


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


def model_builder(hp):
    model = get_model_for_hypertuner(real_features, x_categoielle, categorical_features, 3,hp)
    return model

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


