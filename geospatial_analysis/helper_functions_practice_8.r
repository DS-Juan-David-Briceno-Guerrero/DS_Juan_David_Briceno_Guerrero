#Define a function that reprpjects a raster to an specified crs.
stars_raster_object_reproject <- function(r, r_temp = NULL, new_crs){
  #reproject raster to new crs.
  if(is.null(r_temp)){
    r = st_warp(r, crs = new_crs)
  }
  else{
    r = st_warp(r, r_temp, crs = new_crs)
  }
  return(r)
}

#Defines a function that reprojects a list of rasters.
#If template is passed by argument, it should have the same crs than the one passed by parameter.
list_stars_raster_objects_reproject <- function(ls_r, r_temp = NULL, new_crs){
  
  #create list of rasters.
  if(is.null(r_temp)){
    list_of_rasters <- foreach(i = 1:length(ls_r)) %dopar%{
      stars_raster_object_reproject(ls_r[[i]],new_crs = new_crs)
    }
  }else{
    list_of_rasters <- foreach(i = 1:length(ls_r)) %dopar%{
      stars_raster_object_reproject(ls_r[[i]], r_temp, new_crs = new_crs)
    }
  }
  
  #Return desired output.
  return(list_of_rasters)
}

#Defines a function that calculates utc time from solar time.
get_utc_time_from_solar_time <- function(locST_view_time,lon,date)
{
  #Calculate desired time.
  if(is.na(locST_view_time)){
    new_time_1 = NA 
  }else{
    new_time_1 = LocST_UTC(LocST=locST_view_time, lon=lon, utc=date)
    new_time_1 = new_time_1[[4]][1]
    new_time_1 = as.character(new_time_1)
  }
  
  return(new_time_1)
}

#Defines a function that for a dataframe representing a EPGS 4326 raster, creates overpass times columns.
dataframe_EPGS_4326_give_utc_overpass_time <- function(df, solar_view_time_col){
  
  #Obtain df.
  df_r_mod = df
  
  #Obtain utc overpass times from solar time columns. 
  if(solar_view_time_col %in% names(df_r_mod)){
    new_col_name = paste(solar_view_time_col,"_utc",sep = "")
    df_r_mod[,c(new_col_name)] = mapply(get_utc_time_from_solar_time, locST_view_time=df_r_mod[,c(solar_view_time_col)],lon=df_r_mod$x,date = df_r_mod$time)
    df_r_mod[,c(new_col_name)] = as.POSIXct(df_r_mod[,c(new_col_name)],tz="UTC")
  }
  
  #Return desired value.
  return(df_r_mod)
}

#Define a functon that for a list of rasters, return the names of each of them.
list_stars_raster_objects_give_variables <- function(ls_r){
  
  #Obtain names.
  list_of_variables <- foreach(i = 1:length(ls_r)) %dopar%{
    list_of_variables = names(ls_r[[i]])
  }
  
  #Return names.
  return(unlist(list_of_variables))
}

#Defines a function that gives for a dataframe, its template.
dataframe_give_dataframe_template <- function(df){
  
  #Give geospatial column identifiers for a raster.
  df_mod = df[,c("id","x","y")]
  
  #Remove duplicates.
  df_mod = distinct(df_mod)
  
  #Return desired dataframe.
  return(df_mod)
}

#Defines a function that gives from a dataframe representing a raster, its geospatial column identifiers.
stars_raster_object_give_dataframe_template <- function(r){
  
  #Obtain spatial raster.
  r = stars_raster_object_give_stars_raster_template(r)
  
  #Obtain dataframe template.
  df = stars_raster_object_give_dataframe(r)
  
  #Obtain template.
  df_mod = dataframe_give_dataframe_template(df)
  
  #Return desired dataframe.
  return(df_mod)
}

#Defines a function that from a list of stars rasters, calculate utc_overpass time raster if possible.
list_stars_raster_objects_calculate_utc_overpass_time_raster <- function(ls_r,i_d){
  
  #Subset list by index.
  ls_d = ls_r[][i_d]
  
  #Get template.
  ls_d_template = stars_raster_object_give_stars_raster_template(ls_d[[1]])
  
  #Obtain initial raster characteristics and reproject raster to EPGS:4326.
  sinu = st_crs(ls_d_template)
  ls_d_4326 = list_stars_raster_objects_reproject(ls_d, new_crs= 4326)
  
  #Obtain characteristics of raster in 4326.
  dx = st_dimensions(ls_d_4326[[1]])$x$delta
  dy = st_dimensions(ls_d_4326[[1]])$y$delta
  crs = st_crs(ls_d_4326[[1]])
  
  #Obtain df of 4326 raster.
  df_ls_d_4326 = list_stars_raster_objects_give_dataframe(ls_d_4326)
  
  #Calculate utc column.
  ls_d_name = names(ls_d[[1]])
  print(ls_d_name)
  df_ls_d_4326 = dataframe_EPGS_4326_give_utc_overpass_time(df_ls_d_4326,ls_d_name)
  print(head(df_ls_d_4326))
  
  #Obtain list of rasters from df.
  new_ls_d_4326 = list_stars_raster_objects_from_dataframe(df_ls_d_4326,crs,dx,dy)
  
  #Reproject list of rasters.
  new_ls_d = list_stars_raster_objects_reproject(new_ls_d_4326, ls_d_template, new_crs= sinu)
  
  return(new_ls_d)
}

#Defines a function that from a list of stars rasters, calculate utc_overpass time rasters if possible.
list_stars_raster_objects_calculate_utc_overpass_time_rasters <- function(ls_r){
  
  vars = vars = list_stars_raster_objects_give_variables(ls_r)
  a = which(vars == "Day_view_time")
  b = which(vars == "Night_view_time")
  d = c(a,b)
  
  if(is_empty(d)){
    
  }else{
    l <- foreach(i = 1:length(d)) %dopar%{
      l_i = list_stars_raster_objects_calculate_utc_overpass_time_raster(ls_r,d[i])
    }
    
  }
  
  #Working with lists.
  ls_1 = l[[1]][[2]]
  ls_2 = l[[2]][[2]]
  ls = list(ls_1,ls_2)
  
  #Append to original.
  ls_r_mod = c(ls_r,ls)
  
  #Return list of rasters.
  return(ls_r_mod)
}
