#Define a function that gives information about a vecotr of modis files.
era5_data_give_info <- function(files_ERA5) {
  files_split_ = strsplit(files_ERA5, "\\.")
  split_1 = stri_sub(sapply(files_split_, "[", 1),-10)
  variable = str_remove(sapply(files_split_, "[", 1), split_1)
  variable = substr(variable,1,nchar(variable)-1)
  split_2 = strsplit(split_1, "_")
  year = sapply(split_2, "[", 1)
  month = sapply(split_2, "[", 2)
  day = sapply(split_2, "[", 3)
  dates = as.Date(paste(year,month,day,sep="-"))
  info_list = list(variable,year,month,day,dates)
  return(info_list)
}

#Defines a function that gets unique calendar dates in era5 data.
era5_data_get_unique_calendar_dates <- function(files_ERA5){
  info_list_era5 = era5_data_give_info(files_ERA5)
  dates_calendar = unique(info_list_era5[[5]])
  return(dates_calendar)
}

#Defines a function that filters a vector of modis files based on a time range.
era5_data_subset_by_dates <- function(files_ERA5, startime, endtime){
  #First processing.
  info_list_era5 = era5_data_give_info(files_ERA5)
  calendar_dates = info_list_era5[[5]]
  
  #Seconf processing.
  df <- data.frame(date = calendar_dates ,
                   files = files_ERA5)
  df = df[df$date >= startime & df$date <= endtime, ]
  files_ <- df$files
  return(files_)
}


#Defines a function that obtain a raster from a era5 netcdf file.
era5_obtain_raster_from_file = function(file_path, variable){
  #Subset files by variable.
  file_path_new = file_path[grepl(variable,file_path)]
  #Create raster.
  raster = read_ncdf(file_path_new, make_time=TRUE, make_units=TRUE)
  
  #Assign values of time layer.
  times = st_get_dimension_values(raster, "time")
  st_dimensions(raster)$time$values = times
  
  #Return raster.
  return(raster) 
}

#Define a function that from multiple files or a file, obtain a list of era5 rasters.
era5_obtain_list_of_rasters_from_files_shapefile_modis_template = function(files,variable, israel_32636, r_modis_template){
  #Obtain shape file to crop rasters.
  israel_WSG84 = st_transform(israel_32636, 4326)
  
  #create list of rasters.
  list_of_rasters <- foreach(i = 1:length(files)) %dopar%{
    #r = st_warp(r,r_multiband_template, crs= st_crs(r_multiband_template))
    st_warp(era5_obtain_raster_from_file(files[i],variable)[israel_WSG84],r_modis_template, crs = st_crs(r_modis_template))
  }
  
  #return.
  return(list_of_rasters)
}


#Create a era5 raster from a selected set of paths and a single variable.
era5_data_create_raster <- function(files_, path, shapefile, variable,r_modis_template){
  #Obtain complete path files.
  files_paths = modis_data_concatenate_path_and_files(files_,path)
  
  #Obtain list of rasters
  ls_r = era5_obtain_list_of_rasters_from_files_shapefile_modis_template(files_paths,variable,shapefile, r_modis_template)
  
  #Create raster.
  r = modis_data_concatenate_list_of_rasters_imp(ls_r)
  
  #return.
  return(r)
}


#Create a multiband raster of era5 data of one single variable, according to the desired dates.
era5_data_create_multiband_raster <- function(files_,path,shapefile, unique_dates_calendar, variable, r_modis_template){
  
  #Create list of rasters.
  list_of_rasters <- foreach(i = 1:length(unique_dates_calendar)) %dopar%{
    f = era5_data_subset_by_dates(files_, unique_dates_calendar[i], unique_dates_calendar[i])
    t = era5_data_create_raster(f, path, shapefile, variable,r_modis_template)
    if(is.null(t) == TRUE){
    }else{
      t
    }
  }
  
  #Delte null values.
  list_of_rasters = compact(list_of_rasters)
  
  #Second stage processing.
  r = modis_data_concatenate_list_of_rasters_imp(list_of_rasters)
  times = st_get_dimension_values(r, "time")
  st_dimensions(r)$time$values = times
  
  #Assigning name to raster variable.
  names(r) = variable
  
  return(r)
}

#Define a function that allows to filter a multiband raster by a defined interval of times.
stars_multiband_raster_object_subset_by_dates <- function(r, startime, endtime){
  
  #Obtain times and their indexes from raster's time band.
  times = st_dimensions(r)$time$values
  times_idx = seq(1,length(times),1)
  
  #Create df.
  df <- data.frame(times = times ,
                   times_idx = times_idx)
  
  #Filtering df.
  df = df[df$times >= startime & df$times <= endtime, ]
  
  #Get indexes.
  idx = df$times_idx
  
  #Filter raster.
  r_mod = r[,,,idx]
  
  #return
  return(r_mod)
}