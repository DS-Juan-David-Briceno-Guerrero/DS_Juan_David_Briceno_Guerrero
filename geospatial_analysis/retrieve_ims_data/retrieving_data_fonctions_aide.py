#Define a method that closes a session of URI requests.
def close(s):
    """Disposes of any internal state.

    Currently, this closes the PoolManager and any active ProxyManager,
    which closes any pooled connections.
    """
    s.poolmanager.clear()
    for proxy in s.proxy_manager.values():
        proxy.clear()

#Defining a function that prints a json object.
def jprint(obj):
    # create a formatted string of the Python JSON object
    text = json.dumps(obj, sort_keys=False, indent=4)
    print(text)

#Defining a function that prints n_lines of a json object.
def print_n_lines_json(json_obj, n_limit):
    c = 0
    for line in json.dumps(json_obj, sort_keys=False, indent=4).splitlines():
        if(c <= n_limit):
            print(line)
            c = c+1

#Methods that obtain information from meteorological requests.
#Defining a method that obtains from a request of data, lists of attributes for the data
def obtain_from_request_data_lists_attributes(dic_from_json):
    #Obtain dictionary from the data key in initial dictionary obtained from request.
    dic = dic_from_json['data']
    stationId = []
    dates = []
    channels = []
    for key in dic:
        stationId.append(dic_from_json["stationId"])
        dates.append(key["datetime"])
        channels.append(key["channels"])
    return stationId, dates, channels

#Defining a method that creates a dataframe with attributes from a channel dictionary.
def obtain_from_channel_df_attributes(channel_dic, str_station_id, str_datatime):
    channel_id = []
    name_ = []
    alias_ = []
    value_ = []
    status_ = []
    valid_ = []
    description_ = []

    #Values received by arguemnt.
    stationId = str_station_id
    datetime = str_datatime
    stationId_ = []
    datetime_ = []

    for i in channel_dic:
        channel_id.append(i["id"])
        name_.append(i["name"])
        alias_.append(i["alias"])
        value_.append(i["value"])
        status_.append(i['status'])
        valid_.append(i["valid"])
        description_.append(i["description"])
        stationId_.append(stationId)
        datetime_ .append(datetime)
    
    ls_1 = [channel_id, name_, alias_, value_, status_, valid_, description_,stationId_,datetime_]
    df_channel_1 = pd.DataFrame(ls_1)
    df_channel_1 = df_channel_1.T
    df_channel_1.columns = ["channelId","name","alias","value","status","valid","description","stationId","datetime"]
    return df_channel_1

#Defining a function that creates a consolidates dataframe for meteorological data from lists.
def obtain_from_data_lists_consolidated_df(list_stations, list_dates, list_channels):
    #Ontain initial lists objects.
    init_channel = list_channels[0]
    init_station = list_stations[0]
    init_date = list_dates[0]
    
    #Initial dataframe.
    df_init = obtain_from_channel_df_attributes(init_channel,init_station, init_date)
    
    #Concatenate all monitors dataframes.
    df_mod = df_init.copy()
    for i in range(1,len(list_stations)):
        df_update = obtain_from_channel_df_attributes(list_channels[i],list_stations[i],list_dates[i])
        df_mod = pd.concat([df_mod, df_update], ignore_index=True, axis=0)
    
    df_mod = df_mod[["stationId","datetime","channelId","name","alias","value","status","valid","description",]]
    
    return df_mod

#---------------------------------------------------------------------------------------------------------------------
#Method that creates a consolidated dataframe for data coming from multiple station of mis.
def obtain_df_data_from_multiple_stations_requests(session, init_date, final_date ,ls_stations, ApiToken):
    #Create copy of list.
    ls = ls_stations.copy()
    
    #Obtain request dataframe for first station.
    j = str(ls[0])
    my_uri = 'https://api.ims.gov.il/v1/envista/stations/'+j+'/data?from='+init_date+'&to='+final_date
    r= session.get(my_uri, headers={'Authorization': 'ApiToken '+ApiToken})
    
    #Obtain consolidated dataframe from request.
    ls_1, ls_2, ls_3 = obtain_from_request_data_lists_attributes(r.json())
    df_init = obtain_from_data_lists_consolidated_df(ls_1, ls_2, ls_3)
    
    if(len(ls)==1 and (r.status_code!=204)):
        return df_init
    else:
        new_list = ls.copy()
        new_list.pop(0)
        df_mod = df_init.copy()
        
        for i in new_list:
            my_uri_new = 'https://api.ims.gov.il/v1/envista/stations/'+str(i)+'/data?from='+init_date+'&to='+final_date
            r_new = session.get(my_uri_new, headers={'Authorization': 'ApiToken '+ApiToken})
            #Obtain consolidated dataframe from request.
            if(r_new.status_code!=204):
                ls_1_new, ls_2_new, ls_3_new = obtain_from_request_data_lists_attributes(r_new.json())
                df_update = obtain_from_data_lists_consolidated_df(ls_1_new, ls_2_new, ls_3_new)
                df_mod = pd.concat([df_mod, df_update], ignore_index=True, axis=0)
                #print(i)
        return df_mod

#Method that creates a consolidated dataframe for data coming from multiple station of mis(part2).
async def obtain_df_data_from_multiple_stations_requests_daily_improved(init_date, final_date ,ls_stations, ApiToken):
    headers={'Authorization': 'ApiToken '+ApiToken, 'Connection': 'keep-alive'}
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        for station in ls_stations:
            task = asyncio.ensure_future(get_station_data_daily(session, init_date, final_date,station))
            tasks.append(task)
            
        list_dfs = await asyncio.gather(*tasks) 
        print('finished')
        df_mod = pd.concat(list_dfs, ignore_index=True, axis=0)
        return  df_mod
    
#Method that creates a consolidated dataframe for data coming from multiple station of mis(part1).
async def get_station_data_daily(session, init_date, final_date, station):
    my_uri = 'https://api.ims.gov.il/v1/envista/stations/'+str(station)+'/data?from='+init_date+'&to='+final_date
    async with session.get(my_uri) as r_new:
        if(r_new.status!=204):
            results_data = await r_new.json()
            ls_1_new, ls_2_new, ls_3_new = obtain_from_request_data_lists_attributes(results_data)
            df_update = obtain_from_data_lists_consolidated_df(ls_1_new, ls_2_new, ls_3_new)
            return df_update
        else:
            keys = ['stationId','datetime','channelId','name','alias','value','status','valid','description','period']
            values = [[],[],[],[],[],[],[],[],[],[]]
            d = dict(zip(keys,values))
            df_empty = pd.DataFrame(d)
            return df_empty
#------------------------------------------------------------------------------------------------------------ #Define a function that keeps alive a connection using aiohttp.   
class KeepAliveClientRequest(ClientRequest):
    async def send(self, conn: "Connection") -> "ClientResponse":
        sock = conn.protocol.transport.get_extra_info("socket")
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 2)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
        return (await super().send(conn))

#Defines a function that helps obtain data montly data from the IMS server.
async def get_station_data_montly(session, year_month_date, station, asyncio_semaphore):
    async with asyncio_semaphore:
        print(station)
        my_uri = 'https://api.ims.gov.il/v1/envista/stations/'+str(station)+'/data/monthly/'+year_month_date
        async with session.get(my_uri) as resp:
            if(resp.status!=204):
                async with aiofile.async_open('IMS_data_'+year_month_date.replace("/", "_")+'/station_'+str(station),'wb+') as afp:
                    async for chunk in resp.content.iter_chunked(400):
                        await afp.write(chunk)
                        
#Defines a function that helps obtain data montly data from the IMS server.
async def get_stations_data_montly(session,year_month_date, ls_stations):
    #asyncio_semaphore = asyncio.Semaphore(20)
    asyncio_semaphore = asyncio.BoundedSemaphore(4)
    coros = [ asyncio.ensure_future(get_station_data_montly(session, year_month_date, i, asyncio_semaphore)) for i in ls_stations]
    await asyncio.gather(*coros)
    
#Defines a function that obtain data montly data from the IMS server.
async def obtain_df_data_from_multiple_stations_requests_montly(year_month_date,ls_stations, ApiToken): 
    #Create folder of storage.
    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, 'IMS_data_'+year_month_date.replace("/", "_"))
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path) 
    
    #Obtain data.
    headers={'Authorization': 'ApiToken '+ApiToken, 'Connection': 'keep-alive'}
    async with aiohttp.ClientSession(request_class=KeepAliveClientRequest, headers=headers) as session:
        await get_stations_data_montly(session, year_month_date, ls_stations)
        return "finished"

#Defines a function that returns lists with attributes from a request metadata.         
def obtain_from_request_metadata_lists_attributes(dic_from_json):
    stations = []
    names = []
    locations = []
    monitors = []
    regions = []
    for i in dic_from_json:
        for keys in i:
            if(keys =="stationId"):
                stations.append(i["stationId"])
            if(keys =="name"):
                names.append(i["name"])
            if(keys =="location"):
                locations.append(i["location"])
            if(keys =="regionId"):
                regions.append(i["regionId"])
            if(keys =="monitors"):
                monitors.append(i["monitors"])
    return stations, names, locations, regions, monitors

#Definie a function to obtain from location dictionary the values of latitude and longitude.
def obtain_from_location_tuple_lat_long(loc_dict):
    lat, long = loc_dict["latitude"], loc_dict["longitude"]
    return lat, long

#Defining for a monitor the dataframe containning its attributes information.           
def obtain_from_monitor_df_attributes(monitor_list, str_station_id, str_name, str_latitude, str_longitude, str_region):
    #Attributes obtained by parameter
    stations = []
    names = []
    latitude = []
    longitude = []
    regions = []
    stationId = str_station_id
    stationName = str_name
    latitude_ = str_latitude
    longitude_ = str_longitude
    stationRegion = str_region

    #Attributes Calculated.
    channels = []
    units = []
    for keys in monitor_list:
        channels.append(keys["channelId"])
        units.append(keys["units"])
        
        stations.append(stationId)
        names.append(stationName)
        latitude.append(latitude_)
        longitude.append(longitude_)
        regions.append(stationRegion)
    
    ls_1 = [channels, units, stations, names, latitude, longitude, regions ]
    df_monitor_1 = pd.DataFrame(ls_1)
    df_monitor_1 = df_monitor_1.T
    df_monitor_1.columns = ["channelId","units","stationId","name","latitude","longitude", "regionId"]
    return df_monitor_1

#Defines a function that creates from a metadata request, a dataframe summarizing all the information.
def obtain_from_metadata_lists_consolidated_df(list_stations, list_names, list_locations, list_regions, list_monitors):
    #Ontain initial lists objects.
    init_monitor = list_monitors[0]
    init_station = list_stations[0]
    init_name = list_names[0]
    init_lat , init_long = obtain_from_location_tuple_lat_long(list_locations[0])
    init_region = list_regions[0]
    
    #Initial dataframe.
    df_init = obtain_from_monitor_df_attributes(init_monitor,init_station, init_name,init_lat, init_long, init_region)
    
    #Concatenate all monitors dataframes.
    df_mod = df_init.copy()
    for i in range(1,len(list_monitors)):
        lat , long = obtain_from_location_tuple_lat_long(list_locations[i])
        df_update = obtain_from_monitor_df_attributes(list_monitors[i],list_stations[i],list_names[i],lat,long,list_regions[i])
        df_mod = pd.concat([df_mod, df_update], ignore_index=True, axis=0)
    
    df_mod = df_mod[["stationId","name","latitude","longitude","channelId","units"]]
    
    return df_mod

#Obtain periods from datetime from a data dataframe.
def df_obtain_period(df, col_name):
    df_mod = df.copy()
    df_mod['period'] =  df_mod.apply(lambda x: x[col_name][0:10] ,axis=1)
    #df_mod['period'] = df_mod.apply(lambda x: datetime.fromisoformat(x['period']),axis=1)
    return df_mod


#Defines a function that creates a mapping to map the column for one dataframe to another.
def create_column_mapping(df, col1, col2):
    df_mod = df.copy()
    mp = df_mod[[col1,col2]]
    mp = dict(mp[[col1, col2]].values)
    return mp

#Defines a function that creates a column from a mapping.
def df_new_column_from_mapping(df, mp, new_col_name, mp_col):
    df_mod = df.copy()
    df_mod[new_col_name] = df_mod[mp_col].map(mp)
    return df_mod

#Defining a function that Gives general information from a dataframe. 
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


#Define a function that allows to calculate the hour of a string date passed by argument.
def calculate_hour_from_date(str_date):
    string = str_date
    position = 14
    new_character = '0'
    string = string[:position] + new_character + string[position+1:]
    return string

