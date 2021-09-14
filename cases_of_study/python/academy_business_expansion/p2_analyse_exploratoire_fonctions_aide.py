#Implementing helper functions for the different project parts. 
def give_df_info(df):
    types = pd.Series(df.dtypes).to_frame()
    types.columns = ["Data_types"] 
     
    percent_missing = df.isnull().sum() / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,'percent_missing': percent_missing})
    percent_missing = pd.Series(percent_missing).to_frame()
    percent_missing.columns = ["Nan_percent"]
    
    uniqueValues = pd.Series(df.nunique()).to_frame()
    uniqueValues.columns = ["Unique_values"]
    
    #Extracts the variables names to add at the final dataframe.
    variables_names = list(types.index)
    variables_names = pd.DataFrame(variables_names)
    variables_names.columns = ["Variable_names"]
    
    #Aggregate dataframe with information values.
    df_info = pd.concat([types, percent_missing, uniqueValues], axis=1)

    #Creating the dataframe containing all the information of the edStatsCountry dataframe.
    Unique_values = []

    for index, row in df_info.iterrows():
        data_type = row['Data_types']
        Unique_value = row['Unique_values']
    
        if(data_type=="float64"):
            Unique_value = "real_variable"
        Unique_values.append(Unique_value)
    
    Unique_values = pd.DataFrame({'Unique_values':Unique_values})
    Unique_values = Unique_values.reset_index()
    Unique_values = Unique_values.drop(['index'], axis=1)

    df_info = df_info.drop(['Unique_values'], axis=1)
    df_info.reset_index(inplace=True)
    df_info = df_info.drop(['index'], axis=1)
    df_info = pd.concat([variables_names, df_info, Unique_values], axis=1)
    
    return df_info

#Defining a method that delete from the database the series that are not between the time periods (yearAnc, and yearFut).
def delete_ancient_and_future_series(df, yearAnc, yearFut, verbose=False):
    df_mod = df
    for (columnName, columnData) in df_mod.iteritems():
        try:
            columnNameInt = int(columnName)
            if(columnNameInt<yearAnc):
                df_mod = df_mod.drop([columnName], axis=1)
            elif(columnNameInt>yearFut):
                df_mod = df_mod.drop([columnName], axis=1)
        except Exception as e:
            if(verbose):
                print('The column '+columnName+ 'name is not a year')
    return df_mod

#Defining a method to avoid row values for unfilled indicator in year series.
def indicator_is_null_in_df(df,ind_name,year_start_series):
    year = str(year_start_series)
    db = df[df["Indicator Name"].isin([ind_name])]
    columns = list(db.columns)
    index = columns.index(year)

    #Obtaining only series values in the dataframe.
    db = db.iloc[:,index:-1]

    #checking if rows are null in the df. Assign true or false if the row contains only NaN values.
    array = db.isnull().values.all(axis=0)

    #Replacing booleans for numbers. True = 0, False = 1. 
    array = np.where(array, 0, 1)

    #If the row contains only NaN values, Then the sum of the associated number should be zero.
    result = np.sum(array)
    
    if(result==0):
        return True
    else:
        return False
    
#Defining a method that returns true if an indicator contains several nan rows.
#The percent bound controls the boundary that defines if the indicator has too many NaN variables.
def big_percent_nan_values_indicator_in_df(df,ind_name,year_start_series, percent_boud):
    year = str(year_start_series)
    db = df[df["Indicator Name"].isin([ind_name])]
    columns = list(db.columns)
    index = columns.index(year)

    #Obtaining only series values in the dataframe.
    db = db.iloc[:,index:-1]
    
    total_vals = db.shape[0]*db.shape[1]
    num_nan = db.isnull().sum().sum()
    
    percent = num_nan/total_vals
    
    if(percent >= percent_boud):
        #print(ind_name+' has a lot of NaN values: '+str(percent))
        return True
    else:
        return False

    
#Defining a method that return a boolean indicating if all values in an array ar null.
def all_values_are_nan_in_array(array):
    x = array.astype(float)
    all_are_null = np.sum(np.isnan(x)*1)==len(x)
    return all_are_null

#Defining a method that replaces zero or nan values in an array with the average of the corresponding non null enttrie in their rows.
def replace_zeros_with_row_average_in_array(array):
    arr = []
    for l in array:
        l = l.astype(float)
        l[l==0] = np.nan
        condition = all_values_are_nan_in_array(l)
        if(condition):
            l = np.nan_to_num(l, nan=0.0)
            row_mean = np.mean(l)
            l = np.where(l==0,row_mean,l)
        else:
            #convertir todos a nan para calcular le buen promedio.
            row_mean = np.nanmean(l)
            l = np.nan_to_num(l, nan=row_mean)
        arr.append(l)
    
    arr = np.array(arr)
    return arr

#Creates a method that normalize the values of a series matrix by calculating the min-max values in its rows. 
def normalize_df_series(df):
    df_T = df.T
    df_T = df_T.to_numpy()
    scaler = MinMaxScaler()
    df_T = scaler.fit_transform(df_T)
    df_to_return = df_T.T
    #df_to_return = pd.DataFrame(df_to_return, index = df.index, columns = df.columns)
    return df_to_return

#Defining a method that returns indicators having so many NaN values.
def indicators_almost_null_in_df(df,list_inds,year_start_series,percent_bound):
    almost_null_indicators = []
    for i in list_inds:
        if(big_percent_nan_values_indicator_in_df(df,i,year_start_series,percent_bound)):
            almost_null_indicators.append(i)
    
    return almost_null_indicators

#Defining a method that allows to check if an indicator has only NaN values.
def indicators_null_in_df(df,list_inds,year_start_series):
    null_indicators = []
    for i in list_inds:
        if(indicator_is_null_in_df(df,i,year_start_series)):
            print('The indicator '+i+' is null')
            null_indicators.append(i)
    
    return null_indicators

#Creating a function that allows to return a list of indicators matching some specific words.
#The lists are used then to filter the database with the found indicators.
def indicators_that_match_pattern(ind_list, list_words):
    inds_matched_pattern = []
    for i in ind_list:
        indicator = i.lower()
        all_words_match = True
        
        for j in list_words:
            word = j.lower()
            if(word not in indicator):
                all_words_match = False
        
        if(all_words_match):
            inds_matched_pattern.append(i)
            
    return inds_matched_pattern


#Create a method that calculates the percent increment of an indicator regarding on its initial value.
def indicator_increment_from_initial_value(init_col_index ,matx):
    col = matx[:,init_col_index]
    col = col.reshape(col.shape[0],1)
    inc_matx = matx-col
    return inc_matx

#define a function that creates the country's continent column in a dataframe.
#A map equiv must be defined before.
def df_country_continent_mapping(df, equiv):
    df["Country Continent"] = df["Country Name"].map(equiv)
    return df

#Define a funcion that creates a series line plot for a df series passing by argument.
def series_lines_plot(serie, serie_name):
    plt.figure(figsize=(20,14))
    series_plot = serie 
    series_plot.index = series_plot["TimeStamp"]
    lines = series_plot.plot.line()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={"size":10})
    #plt.title(serie_name+" values along years",fontweight="bold")
    plt.xlabel("Years")
    plt.ylabel("Serie values") 

#Filling missing values in a dataframe with previous series values having no null entries.
#If after filling with previous values, there are still values to fill, it complete them with zeros.
def filling_mising_values_with_previous_values(df_series):
    df = df_series.ffill(axis=1).bfill(axis=1)
    df = df.fillna(0)
    return df

#Normalize the numeric values of a dataframe.
def normalize_df_series(df):
    df_T = df.T
    df_T = df_T.to_numpy()
    scaler = MinMaxScaler()
    df_T = scaler.fit_transform(df_T)
    df_to_return = df_T.T
    #df_to_return = pd.DataFrame(df_to_return, index = df.index, columns = df.columns)
    return df_to_return

#Funcion that plost the indicators growth within the series' years.
def plotting_indicators_growth_country(db_indicators, country, list_indicators):
    #Seleeting the country.
    selected_country = [country]
    db_indicators_algorithm = db_indicators[db_indicators["Country Name"].isin(selected_country)]

    #plotting the country's indicator.
    plt.figure(figsize=(8, 6))
    
    for i in list_indicators:
        db = db_indicators_algorithm
        db = db[db["Indicator Name"].isin([i])]
        #Create the plot.
        x = db['TimeStamp']
        y = db['Value']
        plt.plot(x, y, label = i)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={"size":10})
        
    plt.title("Evolution of "+country+"'s "+"principal educational indicators", fontsize=16)
    plt.xlabel("Year")
    plt.ylabel("Value (ndicator increment)")  
    plt.xticks(rotation=90)
    plt.show()
    
#Defining a function that selects from the datavase the top-n countries in an indicator aspect in an specified year.
def top_n_countries_educational_investment(df_indicators_summary_unnormalized, year, top_n, plotting=True):
    n = [str(year)]
    #1.
    top_countries_data_1 = df_indicators_summary_unnormalized
    #Filtering by indicator.
    indicator_1 = ['GovernmentExpenditureSecondary_as_%_GDP']
    top_countries_data_1 = top_countries_data_1[top_countries_data_1["Indicator Name"].isin(indicator_1)]

    #Filtering by year
    try:
        top_countries_data_1 = top_countries_data_1[top_countries_data_1["TimeStamp"].isin(n)].sort_values(by=['Country Name'])
    except Exception as e: 
        print('The writtten year does not exist in the provided database: '+ str(e))
        sys.exit(1)
            
    top_countries_data_1 = top_countries_data_1.reset_index()
                     
    #2.
    top_countries_data_2 = df_indicators_summary_unnormalized
    #Filtering by indicator.
    indicator_2 = ['GovernmentExpenditureTertiary_as_%_GDP']
    top_countries_data_2 = top_countries_data_2[top_countries_data_2["Indicator Name"].isin(indicator_2)]

    #Filtering by year.
    try:
        top_countries_data_2 = top_countries_data_2[top_countries_data_2["TimeStamp"].isin(n)].sort_values(by=['Country Name'])
    except Exception as e:
        print('The writtten year does not exist in the provided database: '+ str(e))
        sys.exit(1)
                     
    top_countries_data_2 = top_countries_data_2.reset_index()
                     
    #3.
    top_countries_data_3 = df_indicators_summary_unnormalized
    #Filtering by indicator.
    indicator_3 = ['GDP per capita (current US$)']
    top_countries_data_3 = top_countries_data_3[top_countries_data_3["Indicator Name"].isin(indicator_3)]

    #Filtering by year.
    try:
        top_countries_data_3 = top_countries_data_3[top_countries_data_3["TimeStamp"].isin(n)].sort_values(by=['Country Name'])
    except Exception as e:
        print('The writtten year does not exist in the provided database: '+ str(e))
        sys.exit(1)
    
    top_countries_data_3 =top_countries_data_3.reset_index()
    
    #Creating the new dataframe containing the expenditure on education.
    top_countries_data = pd.concat([top_countries_data_1["Country Name"], top_countries_data_1["Country Continent"], top_countries_data_1["Value"], top_countries_data_2["Value"],top_countries_data_3["Value"]], axis =1)
    top_countries_data.columns = ['Country Name','Country Continent','%_expenditure_secondary_education','%_expenditure_tertiary_education','GDP_per_capita']
    top_countries_data["Invested_money_secondary_education"] = top_countries_data["%_expenditure_secondary_education"] * top_countries_data["GDP_per_capita"]
    top_countries_data["Invested_money_tertiary_education"] = top_countries_data["%_expenditure_tertiary_education"] * top_countries_data["GDP_per_capita"]
    top_countries_data["Total_invested_money_secondary_and_tertiary_education"] = top_countries_data["Invested_money_secondary_education"] + top_countries_data["Invested_money_tertiary_education"]
    top_countries_data = top_countries_data.sort_values(by=['Total_invested_money_secondary_and_tertiary_education'],  ascending=False)

    top_n_investing_in_education = top_countries_data.iloc[0:top_n,:]
    year_analysis = str(year)

    #Plotting
    if(plotting==True):
        try:
            plt.figure(figsize=(12,8))
            barp_top_n_investing_in_education = top_n_investing_in_education.plot.bar(x='Country Name', y='Total_invested_money_secondary_and_tertiary_education')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={"size":10})
            plt.title("Top "+str(top_n)+" of countries investing in secondary and tertiary education during the "+year_analysis+" year", fontsize=14,fontweight="bold")
            plt.xlabel("Country")
            plt.ylabel("Money (GDP/Capita Units)")  
            plt.xticks(rotation=90)
            plt.show()
        except Exception as e:
            print('The writtten year does not exist in the provided database: '+ str(e)) 
    
    names_countries = top_n_investing_in_education["Country Name"].to_frame()
    names_countries.columns = ['Country Name']
    
    inv_education = top_n_investing_in_education["Total_invested_money_secondary_and_tertiary_education"].to_frame()
    inv_education.columns = ['investment_education_'+str(year)]
    return names_countries, inv_education


#Create a method that obtain the series of investment in education within two different periods.
#Filtering by indicator and countries.
def initialize_data_countries_investment_education(db_countries_edu, list_countries):
    indicator_1 = ['GovernmentExpenditureSecondary_as_%_GDP']
    countries_data_1 = db_countries_edu[db_countries_edu["Indicator Name"].isin(indicator_1)]
    countries_data_1 = countries_data_1[countries_data_1["Country Name"].isin(list_countries)]
    countries_data_1 = countries_data_1.sort_values(by=['Country Name'],  ascending=True)

    indicator_2 = ['GovernmentExpenditureTertiary_as_%_GDP']
    countries_data_2 = db_countries_edu[db_countries_edu["Indicator Name"].isin(indicator_2)]
    countries_data_2 = countries_data_2[countries_data_2["Country Name"].isin(list_countries)]
    countries_data_2 = countries_data_2.sort_values(by=['Country Name'],  ascending=True)

    indicator_3 = ['GDP per capita (current US$)']
    countries_data_3 = db_countries_edu[db_countries_edu["Indicator Name"].isin(indicator_3)]
    countries_data_3 = countries_data_3[countries_data_3["Country Name"].isin(list_countries)]
    countries_data_3 = countries_data_3.sort_values(by=['Country Name'],  ascending=True)
    
    return countries_data_1, countries_data_2, countries_data_3

#Creating series by years.
def calculate_investment_education(db_countries_edu, list_countries, list_years):
    
    countries_data_1, countries_data_2, countries_data_3 = initialize_data_countries_investment_education(db_countries_edu,list_countries)
    idx = 0
    
    for i in list_years:
        n = [i]  
        if(idx==0):
            countries_data_1 = countries_data_1[countries_data_1["TimeStamp"].isin(n)]
            df_series_education_investment = countries_data_1["Country Name"]
            df_series_education_investment = df_series_education_investment.reset_index(drop=True)

        try:
            countries_data_1 = countries_data_1[countries_data_1["TimeStamp"].isin(n)]
            countries_data_2 = countries_data_2[countries_data_2["TimeStamp"].isin(n)]
            countries_data_3 = countries_data_3[countries_data_3["TimeStamp"].isin(n)]

        except Exception as e: 
            print('The writtten year does not exist in the provided database: '+ str(e))
            sys.exit(1)
        
        countries_data_1 = countries_data_1.reset_index(drop=True)
        countries_data_2 = countries_data_2.reset_index(drop=True)
        countries_data_3 = countries_data_3.reset_index(drop=True)
      
        df_series_education_investment_info = pd.concat([countries_data_1["Value"], countries_data_2["Value"], countries_data_3["Value"]], axis =1)
        df_series_education_investment_info.columns = ['%_expenditure_secondary_education_'+i,'%_expenditure_tertiary_education_'+i,'GDP_per_capita_'+i]
        df_series_education_investment = pd.concat([df_series_education_investment,df_series_education_investment_info],axis =1)
        df_series_education_investment["Invested_money_secondary_education_"+i] = df_series_education_investment["%_expenditure_secondary_education_"+i] * df_series_education_investment["GDP_per_capita_"+i]
        df_series_education_investment["Invested_money_tertiary_education_"+i] = df_series_education_investment["%_expenditure_tertiary_education_"+i] * df_series_education_investment["GDP_per_capita_"+i]
        df_series_education_investment["Total_invested_money_secondary_and_tertiary_education_"+i] = df_series_education_investment["Invested_money_secondary_education_"+i] + df_series_education_investment["Invested_money_tertiary_education_"+i]
        
        countries_data_1, countries_data_2, countries_data_3 = initialize_data_countries_investment_education(countries_education_investment_series,list_countries)
    
        idx = idx + 1

    return df_series_education_investment

#From a df with series of education investment in columns, it obtain a df in an specific year.
def obtain_indicator_column_year(db, indicator_name, year):
    y = str(year)
    df = db[['Country Name',indicator_name+'_'+y, "Country Continent"]]
    return df


#The method transform a dataframe to exhibit its information in the way that countries are in columns , and the series values (years) are in rows.
#The information is aggregated by an indicator that should be passed by parameter.
def transform_row_series_into_columns_series_per_country(df_to_transform, indicator_name):
    filter_col = [col for col in df_to_transform if col.startswith(indicator_name)]
    df_to_return = df_to_transform[filter_col]
    
    columns_names = list(df_to_return.columns)
    first_numeric_digit = int(columns_names[0].find('1'))
    new_columns_names = [x[first_numeric_digit:] for x in columns_names]
    df_to_return.columns = new_columns_names
    df_to_return = pd.concat([df_to_transform["Country Name"],df_to_return], axis =1)
    
    df_to_return = pd.melt(df_to_return, id_vars=["Country Name"], 
                           var_name="TimeStamp", value_name="Value")
    
    df_to_return = pd.pivot_table(df_to_return, values='Value', index=['TimeStamp'], columns=['Country Name'])
    df_to_return.columns.name = None               
    df_to_return = df_to_return.reset_index()
    
    return df_to_return


#Filtering by indicator and countries.
def initialize_data_indicator(db, indicator_name):
    indicator = [indicator_name]
    data_1 = db[db["Indicator Name"].isin(indicator)]
    data_1 = data_1.sort_values(by=['Country Name','TimeStamp'],  ascending=True)
    data_1 = data_1.reset_index(drop=True)
    
    return data_1

#Obtains a dataframe with info for indicator in the way (years,indicator-year)
def calculate_indicator_series_columns(db, indicator_name, list_years, list_countries):
    data_1 = initialize_data_indicator(db[db["Country Name"].isin(list_countries)],indicator_name)
    idx = 0
    
    for i in list_years:
        n = [i]  
        if(idx==0):
            data_1 = data_1[data_1["TimeStamp"].isin(n)]
            df_series_info = data_1["Country Name"]
            df_series_info = df_series_info.reset_index(drop=True)
            
        try:
            data_1 = data_1[data_1["TimeStamp"].isin(n)]

        except Exception as e: 
            print('The writtten year does not exist in the provided database: '+ str(e))
            sys.exit(1)
        
        data_1 = data_1.reset_index(drop=True)
        df_series = pd.concat([data_1["Value"]], axis =1)
        df_series.columns = [indicator_name+'_'+i]
        df_series_info = pd.concat([df_series_info, df_series],axis =1)
        
        data_1 = initialize_data_indicator(db[db["Country Name"].isin(list_countries)], indicator_name)
    
        idx = idx + 1

    return df_series_info

      
#Obtains top_n countries having hight values for an indicator.
def obtain_list_countries_top_n_indicator(db, year, indicator_name, top_n , smallest_to_biggest = False):
    y = str(year)
    df = db[db["Indicator Name"].isin([indicator_name])]
    df = df[df["TimeStamp"].isin([y])]
    df = df.sort_values(by=['Value'],  ascending= smallest_to_biggest)
    df = df.iloc[0:15,:]
    list_countries = list(df["Country Name"].unique())
    return list_countries

#Define a function that normalizes all dataframe values within an scale 0 to 1 by scaliing over the maximun dataframe value.
def df_normalized_with_max_value_years_countries(df):
    df_mod = df
    df_years = df_mod["TimeStamp"]
    df_mod.drop("TimeStamp", axis=1, inplace=True)
    df_numbers = df_mod
    column_maxes = df_numbers.max()
    df_max = column_maxes.max()
    normalized_df = df_numbers / df_max
    df_to_return = pd.concat([df_years,normalized_df],axis =1)
    return df_to_return

#Define a function that plots an histogram given an indicator series.
def plot_indicator_histogram(ind_serie, ind_name, i_fig, n_bins =100):
    serie = ind_serie
    serie = serie.drop("TimeStamp", axis=1)
    serie = serie.to_numpy()
    serie = serie.reshape(serie.shape[0]*serie.shape[1],)
    serie = serie[serie != 0]
    
    print(ind_name)
    plt.figure(i_fig,figsize=(3,3))
    plt.hist(serie, density=True, bins=n_bins)
    #plt.suptitle('Histogram : '+ind_name, fontsize=16, fontweight="bold")
    plt.ylabel('Density')
    plt.xlabel('Indicator_values')
    

#Define a function that calculates the variance of each indicator in the database.
def create_indicator_serie(df,ind_name,list_years):
    db_normalized = df
    ind_series_columns = calculate_indicator_series_columns(db_normalized,ind_name,list_years,list(df["Country Name"].unique()))
    ind_series_countries = transform_row_series_into_columns_series_per_country(ind_series_columns, ind_name)
    ind_series_countries = df_normalized_with_max_value_years_countries(ind_series_countries)
    return ind_series_countries

def create_indicator_serie_unnormalized(df,ind_name,list_years):
    db_normalized = df
    ind_series_columns = calculate_indicator_series_columns(db_normalized,ind_name,list_years,list(df["Country Name"].unique()))
    ind_series_countries = transform_row_series_into_columns_series_per_country(ind_series_columns, ind_name)
    return ind_series_countries

#Define specifies if the zeros within a dataframe are significat or not (threshold is equal to 0.2 to evaluate).
def non_significant_zeros(vector):
    cond = False                                                
    ratio = sum(vector==0)/vector.shape[0]                                                        
    
    if(ratio<=0.2):
        cond =True
    return cond

#Define specifies if the zeros within a dataframe are significat or not (threshold is equal to 0.2 to evaluate).
def erase_too_many_zeros_vector(df_series):
    serie = df_series
    serie = serie.drop("TimeStamp", axis=1).to_numpy()
    serie = serie.reshape(serie.shape[0]*serie.shape[1],)
    if(non_significant_zeros(serie)):
        serie = serie[serie != 0]
    return serie
                                                            
#Define a function that returns a dataframe with the standard deviation of each indicator.
#A dataframe having the information of all countries and indicators should be passed by argument.
def db_indicators_dev_standard(df, list_years):
    indicators = list(df['Indicator Name'].unique())
    std_names = []
    std_inds = []
    for i in indicators:
        serie = erase_too_many_zeros_vector(create_indicator_serie(df, i, list_years))
        std_names.append(i)
        std_inds.append(np.std(serie))
                               
    df_to_return = pd.DataFrame(list(zip(std_names, std_inds)))
    return df_to_return

#Define a function that calculates the averages through years for an indicator serie.
def df_obtain_averages_indicator_serie(serie, indicator_name, country_names_column=False):
    serie_avg_timestamp = serie['TimeStamp']
    serie_avg = serie.drop("TimeStamp", axis=1)
    serie_avg_columns = serie_avg.columns
    serie_avg = serie_avg.to_numpy()
    serie_avg = serie_avg.mean(axis=0)
    serie_avg = serie_avg.reshape(serie_avg.shape[0],1)
    names = np.array(serie_avg_columns).reshape(len(serie_avg_columns),1)
    array_complete = np.concatenate((names, serie_avg), axis=1)
    df_avg = pd.DataFrame(array_complete)
    df_avg.columns = ['Country Name', indicator_name]
    
    df_avg_2 = df_avg.iloc[:,1:2]
    df_avg_2.columns = [indicator_name]
    
    if(country_names_column==False):
        return df_avg_2
    else:
        return df_avg
    

def obtain_averages_indicators_per_country(df, list_years):
    indicators = list(df['Indicator Name'].unique())
    n = len(list(df['Country Name'].unique()))
    array = np.zeros((n, 1))
    cont = 0
    db = 0
    for i in indicators:
        serie = create_indicator_serie(df, i,list_years)
        
        if (cont == 0):
            df_averages_ind = df_obtain_averages_indicator_serie(serie,i,country_names_column=True)
            db = df_averages_ind
        else:
            df_averages_ind = df_obtain_averages_indicator_serie(serie,i,country_names_column=False)
            db = pd.concat([db,df_averages_ind], axis = 1)
            
        cont = cont + 1

    return db
