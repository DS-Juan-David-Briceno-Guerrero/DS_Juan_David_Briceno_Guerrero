#Creating a set of helping functions to help the work in the practice 1.

#1.Defining a function that add a band layer to a raster indicating a period of time.
#The raster values are constant along the time.
create_date_t_band_in_stars <- function(r, band_period, band_name){
  #Creating dataframe object from stars object.
  df_r = as.data.frame(r)
  date_n = rep(c(band_period), times=nrow(df_r))
  df_r[band_name] <- date_n
  
  #Transforming df object into stars object(coming back). 
  l <- st_as_stars(df_r, dims = c("x","y",band_name), pretty= TRUE, dx = st_dimensions(r)$x$delta, dy = st_dimensions(r)$y$delta)
  l = st_set_crs(l,st_crs(r))
  #l = st_set_dimensions(l, "x", offset=st_dimensions(r)$x$offset)
  #l = st_set_dimensions(l, "y", offset=st_dimensions(r)$y$offset)
  return(l) 
}

#2.Defining a function that creates n time bands replicating values of a raster.
modis_data_create_multiband_raster <- function(list_rasters, lists_periods, time_band_name){
  
  #Creating copy.
  l <- r
  if(n==1){
    new = create_date_t_band_in_stars(l, n, band_name)
    return(new)
  }
  else{
    l_mod = create_date_t_band_in_stars(l, 1, band_name)
    for (i in 2:n){
      new = create_date_t_band_in_stars(l, i, band_name)
      l_mod = c(l_mod, new, along = 3) 
    }
    return(l_mod)
  }
}


#3.Histogram for raster band.
#Deploying histogram distribution for first raster band.
hist_raster_band <- function(r, n_band, n_breaks){
  h <- hist(r[[1]][,, n_band], breaks = n_breaks, plot=FALSE)
  h$counts=h$counts/sum(h$counts)
  print(h$counts)
  barplot(h$counts,
          main= paste("Histogram: Pixel values band",n_band),
          xlab ="Ranges",
          ylab ="Density")
  return(h)
}


#Defining the min and max of vector when NA values are present.
min2 <- function(x)if (all(is.na(x))) NA else min(x,na.rm = TRUE)
max2 <- function(x)if (all(is.na(x))) NA else max(x,na.rm = TRUE)
