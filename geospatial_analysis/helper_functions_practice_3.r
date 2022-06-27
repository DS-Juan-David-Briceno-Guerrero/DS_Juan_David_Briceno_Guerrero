subset_ERAS5_nc_file_raster_data <- function(nc_file, var_name, dg_lon_max,dg_lon_min, dg_lat_max, dg_lat_min){
  
  #Obtain variables from raster.
  lon <- ncvar_get(nc_file, "longitude")
  lat <- ncvar_get(nc_file, "latitude", verbose = F)
  t <- ncvar_get(nc_file, "time")
  temp <- ncvar_get(nc_file, var_name)
  
  #Change col names in df.
  arr = temp
  df_arr = melt(arr)
  names(df_arr) <- c("id_lon", "id_lat", "value")
  
  #Capture long coordinates.
  df_lon <- data.frame(lon)
  df_lon["id_lon"] <- data.frame(as.integer(rownames(df_lon)))
  names(df_lon) <- c("lon","id_lon")
  df_lon = df_lon[,c(2,1)]
  
  #Capture long coordinates.
  df_lat <- data.frame(lat)
  df_lat["id_lat"] <- data.frame(as.integer(rownames(df_lat)))
  names(df_lat) <- c("lat","id_lat")
  df_lat = df_lat[,c(2,1)]
  
  #Assign coordiantes to dataframe.
  df_arr <- merge(df_arr, df_lon, by.x = "id_lon", by.y = "id_lon")
  df_arr <- merge(df_arr, df_lat, by.x = "id_lat", by.y = "id_lat")
  df_arr <- df_arr[,c(2,1,3,4,5)]
  
  #Subset the dataframe by desired longitudes and latitudes.
  df_arr <- df_arr %>% filter(lon >= dg_lon_min, lon<= dg_lon_max)
  df_arr <- df_arr %>% filter(lat <= dg_lat_max, lat>= dg_lat_min)
  
  #Getting lon nad lat arrays.
  lon <- df_arr["lon"] %>% as.matrix() %>%  array()
  lat <- df_arr["lat"] %>% as.matrix() %>%  array()
  
  #Sort dataframe.
  df_arr <- arrange(df_arr,id_lon,id_lat)
  df_arr <- df_arr[,c(1,2,3)]
  df_arr_1 <- dcast(df_arr, id_lon ~ id_lat, value.var = 'value')
  df_arr_1 <- df_arr_1[,c(-1)]
  
  #Convert df to array.
  arr_mod <- data.matrix(df_arr_1)
  
  #Bind all to a vector.
  result <- list("matrix_values" = arr_mod, "lon" =lon, "lat"=lat)
  return(result)
}