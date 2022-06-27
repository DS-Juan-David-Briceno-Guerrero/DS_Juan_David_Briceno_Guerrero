#Define a function that gives for list of files, the variables on them.
modis_data_give_variables <-function(files_, path_to_data){
  
  #Obtain path to files.
  path_to_files = path_to_data

  #Get one file name.
  file = files_[1]
  
  #Complete path.
  path_to_file = paste0(path_to_files,'/',file)
  
  #Get subdatasets.
  sd = gdal_subdatasets(path_to_file)
  
  #get variables names.
  list_of_variables = c()
  for(i in 1:length(sd)){
    v_i = strsplit(sd[[i]], split = ":")
    v_i = v_i[[1]][length(v_i[[1]])]
    list_of_variables = append(list_of_variables, list(v_i))
  } 
  
  #Return list of variables names.
  list_of_variables = unlist(list_of_variables)
  return(list_of_variables)
}

#Define a function that gives information about a vecotr of modis files.
modis_data_give_info <- function(files_) {
  files_split_ = strsplit(files_, "\\.")
  tiles_ = sapply(files_split_, "[", 3)
  dates = sapply(files_split_, "[", 2)
  years = substr(dates, 2, 5)
  info_list = list(tiles_, dates, years)
  return(info_list)
}

#Defines a function that takes a vector of day_of_year, and vector of string years, and creates a date.
get_calendar_date <- function(day_year,str_year) {
  o = paste(str_year,"-01-01",sep="")
  date = as.Date((day_year-1), origin = o)
  return(date)
}

#Defines a function that takes a vector with modis files, and transform into calendar dates. 
#convert_modis_date_to_calendar_date
modis_get_calendar_dates <- function(files_){
  #First processing.
  info_list = modis_data_give_info(files_)
  dates = info_list[[2]]
  years = info_list[[3]]
  
  #Second processing.
  day_year = substr(dates,6,8)
  day_year = as.double(day_year)
  x = mapply(get_calendar_date, day_year, years)
  x = as.Date(x, origin='1970-01-01')
  return(x)
}

#Defines a function that gets unique calendar dates in data.
modis_data_give_unique_calendar_dates <- function(files_){
  dates_calendar = unique(modis_get_calendar_dates(files_))
  return(dates_calendar)
}

#Defines a function that filters a vector of modis files based on a time range.
modis_data_subset_by_dates <- function(files_, startime, endtime){
  #First processing.
  calendar_dates = modis_get_calendar_dates(files_)
  
  #Seconf processing.
  df <- data.frame(date = calendar_dates ,
                   files = files_)
  df = df[df$date >= startime & df$date <= endtime, ]
  files_ <- df$files
  return(files_)
}

#Defines a function that filters modis files by the desired tiles.
modis_data_subset_by_tiles <- function(files_, vector_tiles, path_to_files){
  #Obtain desired tiles for filtering.
  list__desired_tiles = vector_tiles
  
  #First stage processing.
  info_list = modis_data_give_info(files_)
  tiles_ = info_list[[1]]
  dates = info_list[[2]]
  years = info_list[[3]]
  copy_files = files_
  
  #Second stage processing.
  new_files_ = vector()
  for (t in list__desired_tiles){
    f <- copy_files[tiles_ == t]
    new_files_ = c(new_files_,f)
  }
  files_ <- new_files_
  theNAs <- is.na(files_)
  files_ = files_[!theNAs]
  return(files_)
}

#Defines a function that obtain a raster from a file.
modis_obtain_raster_from_file = function(file_path, variable){
  raster = raster(get_subdatasets(file_path)[grepl(variable, get_subdatasets(file_path))])
  return(raster) 
}

#Define a function that from multiple files or a file, obtain rasters.
modis_obtain_rasters_from_files = function(files,variable){
  rasters = mapply(modis_obtain_raster_from_file,files,variable)
  rasters = unname(rasters)
  return(rasters)
}

#Defines a function that creates total file path.
modis_data_concatenate_path_and_file = function(file,path){
  new_path = paste0(path,'/',file)
  return(new_path)
}

#Defines a function that creates for each file its total path.
modis_data_concatenate_path_and_files = function(files, path){
  new_paths = mapply(modis_data_concatenate_path_and_file,files,path)
  new_paths = unname(new_paths)
  return(new_paths)
}

#Defines a function that create a raster from modis data files, a shapefile, and a selected variable.
#Returns true if a problem arises when trying to create the raster from the filtered files.
modis_data_create_raster <- function(files_, path, shapefile, variable){
  files_ = modis_data_concatenate_path_and_files(files_, path)
  tryCatch(
    {
      #print("1")
      rasters <- modis_obtain_rasters_from_files(files_,variable)
      proj = projection(rasters[[1]])
      #Mosaick rasters.
      if(length(rasters)==1){
        mos = rasters[[1]]
      }else{
        mos <- do.call(merge, rasters)
      }
      mos = st_as_stars(mos)
      #print("2")
      #Crop raster with shapefile.
      proj.sinu <- proj
      shapefile_sinu <- shapefile %>% st_transform(proj.sinu)
      mos = st_as_stars(mos)
      #print("3")
      #mos <-  mos %>% mask(shapefile_sinu)  # crop to Israel
      mos = mos[shapefile_sinu]
      #print("4")
      #print('ok_3')
      return(mos) 
    },
    error = function(e){
      return(NULL)
    }
  )
}

#Defines a function that concatenates two rasters.
modis_data_concatenate_rasters <- function(r1_band, r2_band) {
  r_mod = c(r1_band, r2_band, along = 3) 
  return(r_mod)
}

#Define a function that concatenates a list of rasters.
modis_data_concatenate_list_of_rasters <- function(list_r){
  #Creating copy.
  l <- list_r
  if(length(l)==1){
    l_mod = l[[1]]
    return(l_mod)
  }else if(length(l)==2){
    l_mod = modis_data_concatenate_rasters(l[[1]], l[[2]])
    return(l_mod)
  }else{
    l_mod = modis_data_concatenate_rasters(l[[1]], l[[2]])
    for(i in 3:length(l)){
      l_mod = modis_data_concatenate_rasters(l_mod, l[[i]])
    }
    return(l_mod)
  }
}

#Newone
modis_data_concatenate_list_of_rasters_imp <- function(list_r){
  #Create list of rasters.
  list_of_rasters = foreach(i = 1:length(list_r), .combine = 'c') %dopar%{
    list_r[[i]]
  }
  
  return(list_of_rasters)
}

#Multiband raster.
modis_data_create_multiband_raster <- function(files_,path,shapefile, unique_dates_calendar, variable){
  #Initialize empty list.
  list_of_rasters = c()
  #print('1')
  
  #First stage processing.
  for (i in 1:length(unique_dates_calendar)){
    f = modis_data_subset_by_dates(files_, unique_dates_calendar[i], unique_dates_calendar[i])
    t = modis_data_create_raster(f, path, shapefile, variable)
    print(t)
    if(is.null(t) == TRUE){
    }else{
      r = create_date_t_band_in_stars(t,unique_dates_calendar[i], 'time')
      list_of_rasters = append(list_of_rasters, list(r))
    }
  print(i)
  }
  
  #Second stage processing.
  list_of_rasters = modis_data_concatenate_list_of_rasters(list_of_rasters)
  
  #Assigning name to raster variable.
  names(list_of_rasters) = variable
  
  return(list_of_rasters)
}


#Multiband raster.
modis_data_create_multiband_raster_imp <- function(files_,path,shapefile, unique_dates_calendar, variable){
  
  #Create list of rasters.
  list_of_rasters <- foreach(i = 1:length(unique_dates_calendar)) %dopar%{
    f = modis_data_subset_by_dates(files_, unique_dates_calendar[i], unique_dates_calendar[i])
    t = modis_data_create_raster(f, path, shapefile, variable)
    if(is.null(t) == TRUE){
    }else{
      create_date_t_band_in_stars(t,unique_dates_calendar[i], 'time')
    }
  }
  
  #Delte null values.
  list_of_rasters = compact(list_of_rasters)
  
  #Second stage processing.
  list_of_rasters = modis_data_concatenate_list_of_rasters_imp(list_of_rasters)
  
  #Assigning name to raster variable.
  names(list_of_rasters) = variable
  
  return(list_of_rasters)
}