#Create a list of hours to faciliate the process of retrieval.
def get_hours_era5_data_retrieval():
    numbers = [i for i in range(0,23+1)]
    hours = []
    for i in numbers:
        if i <10:
            hours.append('0'+str(i)+':00')
        else:
            hours.append(str(i)+':00')
    return hours

#Create a list of days to faciliate the process of retrieval.
def get_days_era5_data_retrieval():
    numbers = [i for i in range(1,31+1)]
    days = []
    for i in numbers:
        if i <10:
            days.append('0'+str(i))
        else:
            days.append(str(i))
    return days

#Defines a function to get dictionaries of retrieval.
def get_dictionaries_era5_data_retrieval():
    #Create dictionary for retrieving data in regads to reanalysis-era5-pressure-levels.
    variables = ["variable","pressure_level","product_type","year","month", "day","time","format"]
    values = ["temperature","1000","reanalysis","2008","01","01","11:00","netcdf"]
    d_reanalysis_era5_pressure_levels = dict(zip(variables,values))

    #Create dictionary of dictionaries for data retrieval of different datasets.
    dict_names = ["reanalysis-era5-pressure-levels"]
    dict_objects = [d_reanalysis_era5_pressure_levels]
    d = dict(zip(dict_names,dict_objects))
    return d

#Defines a function to retrieve hourly data from the API of era5.
def retrieve_ERA5_data_in_hour(connection,dictionary,dataset_name, year, month, day, hour, path):
    #Obtain path.
    path = path
        
    try:     
        #Create file name.
        desired_file_name = 'r_era5_temp_'+year+'_'+month+'_'+day+'_'+hour[0:2]+'h.nc'

        #Adjust paramenters.
        d = dictionary.copy()
        d = d[dataset_name].copy()
        d["year"] = year
        d["month"] = month
        try:
            d["day"] = day
        except :
            print("day is not a parameter of choice for this retrieval")

        try:
            d["time"] = hour
        except :
            print("time is not a parameter of choice for this retrieval")

        #Obtain data.
        c.retrieve('reanalysis-era5-pressure-levels',
        d, path+'/'+desired_file_name)
        
    except :
            print("Request is not valid")
            
#Defines a function to retrieve daily data from the API of era5.
def retrieve_ERA5_data_in_day(connection,dictionary,dataset_name, year, month, day, hours, path):
    print(day)
    #Obtain path.
    path = path
    
    #request data.
    try:
        #Create file name.
        desired_file_name = 'r_era5_temp_'+year+'_'+month+'_'+day+'.nc'

        #Adjust paramenters.
        d = dictionary.copy()
        d = d[dataset_name].copy()
        d["year"] = year
        d["month"] = month
        try:
            d["day"] = day
        except :
            print("day is not a parameter of choice for this retrieval")
        
        #Request data for all hours.
        
        try:
            d["time"] = hours
        except :
            print("time is not a parameter of choice for this retrieval")

        #Obtain data.
        c.retrieve('reanalysis-era5-pressure-levels',
        d, path+'/'+desired_file_name)
        
    except :
            print("Request is not valid")
            
#Defines a function to retrieve daily data from the API of era5.           
def retrieve_ERA5_data_in_day_semaphore(connection,dictionary,dataset_name, year, month, day, hours, path, semaphore):
    semaphore.acquire()
    print(day)
    #Obtain path.
    path = path

    #request data.
    try:
        #Create file name.
        desired_file_name = 'r_era5_temp_'+year+'_'+month+'_'+day+'.nc'

        #Adjust paramenters.
        d = dictionary.copy()
        d = d[dataset_name].copy()
        d["year"] = year
        d["month"] = month
        
        try:
            d["day"] = day
        except :
            print("day is not a parameter of choice for this retrieval")

        try:
            d["time"] = hours
        except :
            print("time is not a parameter of choice for this retrieval")

        #Obtain data.
        c.retrieve('reanalysis-era5-pressure-levels',
        d, path+'/'+desired_file_name)

    except :
        print("Request is not valid")
    
    semaphore.release()

#Defines a function to retrieve daily data from the API of era5.
def retrieve_ERA5_data_in_month(connection,dictionary,dataset_name, year, month, days, hours):
    #Create folder of storage.
    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, 'ERA5_data_'+year+'_'+month)
    if os.path.exists(path)==False:
        os.mkdir(path)
        
    #Create semaphore.
    semaphore = Semaphore(31)
    
    #Initiate processes.
    processes = []
    for i in days:
        p = multiprocessing.Process(target=retrieve_ERA5_data_in_day_semaphore,args=(connection,dictionary,dataset_name, year, month, i, hours, path, semaphore))
        if __name__ =="__main__":
            p.start()
            processes.append(p)
    for p in processes:
        p.join()
    
#----------------------------------------------------------------------------------------------------------
#Defines a function to retrieve daily data from the API of era5.           
def retrieve_in_day(i):
    #async with asyncio_semaphore:
    print(i)
        #request data.
    try:
        time.sleep(2)
        #    print("finish"+str(i))
    except :
        print("Request is not valid")
    print("finish"+str(i))
      
#
#Defines a function to retrieve daily data from the API of era5.           
def retrieve_in_day_semaphore(i, semaphore):
    semaphore.acquire()
    #async with asyncio_semaphore:
    print(i)
        #request data.
    try:
        time.sleep(2)
        #    print("finish"+str(i))
    except :
        print("Request is not valid")
    print("finish"+str(i))
    semaphore.release()

#Retrieving data in parallel.
def retrieve_in_month(n):
    for i in range(n):
        retrieve_in_day(i)
        
#.
def retrieve_in_month_imp(n):
    #Create semaphore.
    sema = Semaphore(10)
    
    #Initiate processes.
    processes = []
    for i in range(n):
        p = multiprocessing.Process(target=retrieve_in_day_semaphore,args=(i,sema))
        if __name__ =="__main__":
            p.start()
            processes.append(p)
    for p in processes:
        p.join()