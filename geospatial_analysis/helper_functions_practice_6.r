#Define a function that allows to work with dataframe columns positions.
moveme <- function (invec, movecommand) {
  movecommand <- lapply(strsplit(strsplit(movecommand, ";")[[1]], 
                                 ",|\\s+"), function(x) x[x != ""])
  movelist <- lapply(movecommand, function(x) {
    Where <- x[which(x %in% c("before", "after", "first", 
                              "last")):length(x)]
    ToMove <- setdiff(x, Where)
    list(ToMove, Where)
  })
  myVec <- invec
  for (i in seq_along(movelist)) {
    temp <- setdiff(myVec, movelist[[i]][[1]])
    A <- movelist[[i]][[2]][1]
    if (A %in% c("before", "after")) {
      ba <- movelist[[i]][[2]][2]
      if (A == "before") {
        after <- match(ba, temp) - 1
      }
      else if (A == "after") {
        after <- match(ba, temp)
      }
    }
    else if (A == "first") {
      after <- 0
    }
    else if (A == "last") {
      after <- length(myVec)
    }
    myVec <- append(temp, values = movelist[[i]][[1]], after = after)
  }
  myVec
}

#Define a function that regarding ona dataframe with IMS data (israel), gives a vector layer representing the stations on israle territory.
ims_data_give_template <- function(df_IMS_data, crs){
  
  #Obtain data.
  df_IMS_data_mod = df_IMS_data
  
  #Rename columns,
  colnames(df_IMS_data_mod)[which(names(df_IMS_data_mod) == "longitude")] <- "x"
  colnames(df_IMS_data_mod)[which(names(df_IMS_data_mod) == "latitude")] <- "y"
  
  #Obtaining unique stations.
  df_IMS_data_mod = unique(df_IMS_data_mod[,c("x","y","stationId")])
  
  #Obtain vector layer.
  vector_israel_stations = st_as_sf(df_IMS_data_mod,  coords = c("x", "y"))
  
  #Assign crs (long,lat).
  vector_israel_stations = st_set_crs(vector_israel_stations, 4326)
  
  #Reproject to sinusoidal.
  vector_israel_stations_sinusoidal = st_transform(vector_israel_stations, crs)
  
  #Return.
  return(vector_israel_stations_sinusoidal)
}

#Define a function that gives a dataframe representing a raster.
stars_raster_object_give_dataframe_singleband_time <- function(r){
  #Obtain df.
  df_r = as.data.frame(r,xy=T,na.rm=T)
  
  #Obtain variable name.
  name = names(r)
  
  #Adding id for cell in raster grid.
  rownames(df_r) <- 1:nrow(df_r)
  colnames(df_r)[1] <- "x"
  colnames(df_r)[2] <- "y"
  
  #Obtain id for each cell.
  v <-  st_as_sf(df_r, coords = c("x", "y"))
  v$id <- 1:nrow(v)
  df_v_coords = st_coordinates(v)
  colnames(df_v_coords)[1] <- "x"
  colnames(df_v_coords)[2] <- "y"
  df_v = st_drop_geometry(v)
  df_v <- cbind(df_v,df_v_coords)
  
  #Reorder df columns.
  df_v = df_v[moveme(names(df_v), "y first")]
  df_v = df_v[moveme(names(df_v), "x first")]
  df_v = df_v[moveme(names(df_v), "id first")]
  
  if("Day_view_time_overpass" %in% names(df_v)){
    df_v[,c("Day_view_time_overpass")] = as.POSIXct(df_v[,c("Day_view_time_overpass")],tz="UTC")
  }
  
  if("Night_view_time_overpass" %in% names(df_v)){
    df_v[,c("Night_view_time_overpass")] = as.POSIXct(df_v[,c("Night_view_time_overpass")],tz="UTC")
  }
  
  
  #Return the dataframe.
  return(df_v)
}

#Define a function that creates a dataframe for a multiband time raster r. 
stars_raster_object_give_dataframe_multiband_time <- function(r){
  
  #Obtain time values.
  n_t_values = st_dimensions(r)$time$to-st_dimensions(r)$time$from +1
  
  ##Create list of dfs.
  list_df = list()
  
  #Append dataframes in list.
  for(i in 1:n_t_values){
    r_i = r[,,,i]
    df_i =  stars_raster_object_give_dataframe_singleband_time(r_i)
    list_df[[i]] = df_i
  }
  
  #Merge dataframes in list.
  df = data.table::rbindlist(list_df)
  
  #Convert data.table into dataframe.
  class(df) <- class(as.data.frame(df))
  
  #Return.
  return(df)
}

#Define a function that creates a dataframe for a multiband time raster r. 
stars_raster_object_give_dataframe_multiband_time_imp <- function(r){
  
  #Obtain time values.
  n_t_values = st_dimensions(r)$time$to-st_dimensions(r)$time$from +1
  
  #Append dataframes in a list.
  #list_df <- foreach(i = 1:n_t_values,.export = ls(globalenv()),.packages = "stars") %dopar%{
  list_df <- foreach(i = 1:n_t_values) %dopar%{
    stars_raster_object_give_dataframe_singleband_time(r[,,,i])
  }
  
  #Merge dataframes in list.
  df = data.table::rbindlist(list_df)
  
  #Convert data.table into dataframe.
  class(df) <- class(as.data.frame(df))
  
  #Return.
  return(df)
}

#Define a function that returns a dataframe for a raster r.
#stars_raster_object_give_dataframe <-function(r){
#  #Obtain band size.
#  band_size = st_dimensions(r)$time$to-st_dimensions(r)$time$from +1
#  if(band_size<=1){
#    df = stars_raster_object_give_dataframe_singleband_time(r)
#  }
#  else{
#    #df = stars_raster_object_give_dataframe_multiband_time(r)
#    df = stars_raster_object_give_dataframe_multiband_time_imp(r)
#  }
  
#  return(df)
#}

#Define a function that returns a dataframe for a raster r.
stars_raster_object_give_dataframe <- function(r){
  
  #Check wether it contains time band or not.
  band = st_dimensions(r)$time
  
  if(is.null(band)){
    #Obtain df for single band raster.
    df = stars_raster_object_give_dataframe_singleband_time(r)
    
  }else{
    
    #Obtain band size.
    band_size = st_dimensions(r)$time$to-st_dimensions(r)$time$from +1
    if(band_size<=1){
      df = stars_raster_object_give_dataframe_singleband_time(r)
    }
    else{
      #df = stars_raster_object_give_dataframe_multiband_time(r)
      df = stars_raster_object_give_dataframe_multiband_time_imp(r)
    }
  }
  #Return df.
  return(df)
}

#Define a function that gives a dataframe representing a vector layer.
st_vector_object_give_dataframe <- function(v){
  #names(v)[1]
  
  #Obtain df.
  df_stations_coords = st_coordinates(v)
  colnames(df_stations_coords)[1] <- "x"
  colnames(df_stations_coords)[2] <- "y"
  df_stations = st_drop_geometry(v)
  df_stations <- cbind(df_stations,df_stations_coords)
  
  #Reorder df columns.
  df_stations = df_stations[moveme(names(df_stations), "y first")]
  df_stations = df_stations[moveme(names(df_stations), "x first")]
  
  return(df_stations)
}


#Defines a function that gives a raster template representing for a raster passed by argument.
stars_raster_object_give_stars_raster_template <- function(r){
  dx = st_dimensions(r)$x$delta 
  dy = st_dimensions(r)$y$delta
  templete = st_as_stars(st_bbox(r), dx = dx, dy = dy)
  return(templete)
}

#Defines a function that gives a template (vector layers) representing the grid-cells of a raster.
#Returns vector of points or spatial points if indicated.
stars_raster_object_give_sf_vector_layer_template <- function(r, return_sp=TRUE){
  
  r <- tryCatch({
    r = r[,,,1] 
  },
  error = function(cond){
    r = r 
  },
  finally = {
    #pass
  })
  
  #Obtain crs.
  r_crs = st_crs(r)
  
  #Obtain df of vector layer associated with the raster.
  df_pts = stars_raster_object_give_dataframe(r)
  df_pts = df_pts[,c("id","x","y")]

  #Convert as vector layer and assign crs.
  pts = st_as_sf(df_pts, coords = c("x", "y"))
  pts = st_set_crs(pts, r_crs)
  
  #Obtain templete as sp.
  pixels <- as(pts, 'Spatial')
  pixels@data <- data.frame(id=as.integer(rownames(pts)))
  gridded(pixels) <- TRUE
  pixels
  
  #return.
  if (return_sp==TRUE)
    return(pixels)
  else
    return(pts)
}

#Defines a function that gives a template of sp pixels regarding a vector layer.
st_vector_object_give_sp_template <- function(v){
  
  #Obtain column names in vector layer.
  #Obtain templete as sp.
  pixels <- as(v, 'Spatial')
  pixels@data <- data.frame(id=as.integer(rownames(v)))
  gridded(pixels) <- TRUE
  pixels
  
  #return.
  return(pixels)
}

#Define a function that gives a dataframe representing a raster.
modis_data_give_sf_vector_layer_israel_stations <- function(template_raster_vector_points_grid, vector_stations){
  
  #Obtain sp pixels/cells from vector of points grid.
  pixels = st_vector_object_give_sp_template(template_raster_vector_points_grid)
  
  #Obtain sp points from vector of stations points.
  v_points <- as(vector_stations, 'Spatial')
  
  #Obtain column giving for eact station its grid within a raster tempalte.
  id_grid = over(v_points, pixels)
  
  #Obtain df from vector representing the stations. 
  df_vector_stations_1 = st_vector_object_give_dataframe(vector_stations)
  df_vector_stations_2 <- cbind(df_vector_stations_1,id_grid)
  
  #Reorder df columns.
  df_vector_stations_2 = df_vector_stations_2[moveme(names(df_vector_stations_2), "y first")]
  df_vector_stations_2 = df_vector_stations_2[moveme(names(df_vector_stations_2), "x first")]
  df_vector_stations_2 = df_vector_stations_2[moveme(names(df_vector_stations_2), "id first")]
  colnames(df_vector_stations_2)[1] <- "id_in_grid"
  
  #Perform modifications.
  colnames(df_vector_stations_2)[which(names(df_vector_stations_2) == "id_in_grid")] <- "id" 
  df_vector_stations_2 = left_join(df_vector_stations_2,template_raster_vector_points_grid, by = c("id"))
  df_vector_stations_2 = df_vector_stations_2[,c("x","y","id","stationId")]
  df_vector_stations_2 = st_as_sf(df_vector_stations_2,  coords = c("x", "y"))
  df_vector_stations_2 = st_set_crs(df_vector_stations_2, st_crs(template_raster_vector_points_grid))
  
  #return vector layer.
  return(df_vector_stations_2)
}


#Define a function that from a dataframe of an specific(single-band period) builds a raster.
#dx, and dy from a raster template should be passed by parameter.
stars_raster_object_from_dataframe_singleband_time <- function(df_r, var_name, crs, dx, dy){
  #Obtain dataframe.
  df_r_mod = df_r

  #Store id column.
  id_col = df_r_mod$id
  
  #Change to characters in the case of day overpass.
  if(var_name == "Day_view_time_utc"){
    df_r_mod[,c("Day_view_time_utc")] = as.character(df_r_mod[,c("Day_view_time_utc")])
  }
  
  if(var_name == "Night_view_time_utc"){
    df_r_mod[,c("Night_view_time_utc")] = as.character(df_r_mod[,c("Night_view_time_utc")])
  }
  
  #Create raster.
  r <- tryCatch({
    #Avoid id column.
    df_r_mod = df_r_mod[,c("x","y","time",var_name)]
    #process.
    r = st_as_stars(df_r_mod, dims = c("x","y","time"), pretty= TRUE, dx = dx, dy = dy)
    r = st_set_crs(r,st_crs(crs))
  },
  error = function(cond){
    #Avoid id column.
    df_r_mod = df_r_mod[,c("x","y",var_name)]
    #Process.
    r = st_as_stars(df_r_mod, dims = c("x","y"), pretty= TRUE, dx = dx, dy = dy)
    r = st_set_crs(r,st_crs(crs))
  },
  finally = {
    #pass
  })
  #return raster and the id of the cells if needed.
  return(r)
}

#Define a function that  builds a raster from a dataframe.
#dx, and dy from a raster template should be passed by parameter.
stars_raster_object_from_dataframe <- function(df_r, var_name, crs, dx, dy){
  
  #Create empty list of rasters.
  list_of_rasters = c()
  
  #Obtain df.
  df_r_mod = df_r
  
  #Obtain columns of df.
  df_cols = names(df_r_mod)
  
  #Obtain band periods.
  periods = unique(df_r_mod$time)
  
  #Validate condition.
  if(('time' %in% df_cols)==TRUE){
    #Filter iteratively.
    for(i in 1:length(periods)){
      print(i)
      print(periods[i])
      df_i = df_r_mod[df_r_mod$time == periods[i], ]
      print(nrow(df_i))
      r_i = stars_raster_object_from_dataframe_singleband_time(df_i,var_name,crs, dx, dy)
      list_of_rasters = append(list_of_rasters, list(r_i))
    } 
  }else{
    r_j = stars_raster_object_from_dataframe_singleband_time(df_r_mod,var_name,crs, dx, dy)
    list_of_rasters = append(list_of_rasters, list(r_j))
  }
  
  #Last stage processing.
  list_of_rasters = modis_data_concatenate_list_of_rasters(list_of_rasters)
  #Assigning name to raster variable.
  names(list_of_rasters) = var_name
  return(list_of_rasters)
}

#Define a function that create a list of rasters regarding on a list of variables, dates, and files. 
modis_data_create_list_stars_rasters <- function(files_from_selected_dates,path_modis_data, israel_32636, unique_dates_in_files_from_selected_dates, variables){
  #Obtain rasters.
  list_of_rasters = c()
  for(i in 1:length(variables)){
    r_i = modis_data_create_multiband_raster(files_from_selected_dates, path_modis_data, israel_32636,unique_dates_in_files_from_selected_dates, variables[i])
    list_of_rasters = append(list_of_rasters, list(r_i))
  }
  
  #return.
  return(list_of_rasters)  
}

#Define a function that create a list of rasters regarding on a list of variables, dates, and files. 
modis_data_create_list_stars_rasters_imp <- function(files_from_selected_dates,path_modis_data, israel_32636, unique_dates_in_files_from_selected_dates, variables){
  
  #create list of rasters.
  list_of_rasters <- foreach(i = 1:length(variables)) %dopar%{
    modis_data_create_multiband_raster_imp(files_from_selected_dates, path_modis_data, israel_32636,unique_dates_in_files_from_selected_dates, variables[i])
  }
  
  #return.
  return(list_of_rasters)  
}

#Define a function that builds a dataframe from a list of multiband rasters.
list_stars_raster_objects_give_dataframe <- function(list_stars_rasters){
  
  #Obtain first df.
  df_r = stars_raster_object_give_dataframe(list_stars_rasters[[1]])
  
  #Append dataframes.
  if(length(list_stars_rasters)>1){
    for(i in 2:length(list_stars_rasters)){
      df_r_i = stars_raster_object_give_dataframe(list_stars_rasters[[i]])
      df_r = left_join(df_r,df_r_i, by = c("id", "x", "y", "time")) 
    }  
  }
  
  #Return.
  return(df_r)
}

#Defines a function that create a list of raster from a df containing one or more variables.
list_stars_raster_objects_from_dataframe <-function(df,crs, dx, dy){
  #Get unique values for variables.
  var_names = names(df)
  y<-c("id", "x", "y", "time") # values to be removed
  idx = which(var_names %in% y) # Positions of the values of y in x   
  var_names = var_names[-idx]
  
  #Initialize list of rasters.
  list_of_rasters = c()
  
  #create list of rasters.
  for(i in 1:length(var_names)){
    r_i = stars_raster_object_from_dataframe(df, var_names[i], crs, dx, dy)
    list_of_rasters = append(list_of_rasters, list(r_i))
  }
  
  return(list_of_rasters)
}

#Defines a function that create a list of raster from a df containing one or more variables.
list_stars_raster_objects_from_dataframe_imp <-function(df,crs, dx, dy){
  #Get unique values for variables.
  var_names = names(df)
  y<-c("id", "x", "y", "time") # values to be removed
  idx = which(var_names %in% y) # Positions of the values of y in x   
  var_names = var_names[-idx]
  
  #create list of rasters.
  list_of_rasters <- foreach(i = 1:length(var_names)) %dopar%{
    stars_raster_object_from_dataframe(df, var_names[i], crs, dx, dy)
  }
  
  return(list_of_rasters)
}