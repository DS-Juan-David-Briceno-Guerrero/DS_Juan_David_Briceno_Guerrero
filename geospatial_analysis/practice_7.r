#Import libraries.
#install.packages("ncmeta")
#install.packages("doParallel",dependencies=T)
library(stringi)
library(foreach)
library(ncmeta)
library(data.table)
library(dplyr)
library(tidyverse)
library(magrittr)  # %>% pipe-like operator
library(parallel)  # parallel computation
library(sp)        # classes and methods for spatial data
library(rgdal)     # wrapper for GDAL and proj.4 to manipulate spatial data
library(raster)    # methods to manipulate gridded spatial data
library(gdalUtils) # extends rgdal and raster to manipulate HDF4 files
library(sf)
library(data.table)
library(stars)
library(mapview)
library(MASS)
library(reshape) #
library(ncdf4) # package for netcdf manipulation
library(ggplot2)
library(tidyr)
source("/home/juan-david/Documents/data_science/UGA_phd/documents/Spatial_Data_programming_with_R_course_Michael_Dorman/helper_functions_practice_1.r")
source("/home/juan-david/Documents/data_science/UGA_phd/documents/Spatial_Data_programming_with_R_course_Michael_Dorman/section_5/helper_functions_practice_5.r")
source("/home/juan-david/Documents/data_science/UGA_phd/documents/Spatial_Data_programming_with_R_course_Michael_Dorman/section_6/helper_functions_practice_6.r")
source("/home/juan-david/Documents/data_science/UGA_phd/documents/Spatial_Data_programming_with_R_course_Michael_Dorman/section_7/helper_functions_practice_7.r")

#1.Create cluster for parallel computation(optional).
parallel::detectCores()
n.cores <- parallel::detectCores() - 1
my.cluster <- parallel::makeCluster(
  n.cores, 
  #type = "FORK"
  type = "PSOCK"
)
doParallel::registerDoParallel(cl = my.cluster)
clusterExport(my.cluster, varlist=ls(globalenv()), envir=environment())
parallel::clusterEvalQ(my.cluster,{library(stars)})
parallel::clusterEvalQ(my.cluster,{library(rgdal)})
parallel::clusterEvalQ(my.cluster,{library(gdalUtils)})
parallel::clusterEvalQ(my.cluster,{library(raster)})
parallel::clusterEvalQ(my.cluster,{library(sf)})
parallel::clusterEvalQ(my.cluster,{library(rgdal)})
parallel::clusterEvalQ(my.cluster,{library(doParallel)})
parallel::clusterEvalQ(my.cluster,{library(doParallel)})
parallel::clusterEvalQ(my.cluster,{library(tidyverse)})
parallel::clusterEvalQ(my.cluster,{library(stringi)})
#parallel::clusterEvalQ(my.cluster,{library(lubridate)})

#Check the function doParallel was registered correctly.
foreach::getDoParRegistered()
#Check the number of workers availabe.
foreach::getDoParWorkers()
#Recommendable to stop the cluster when we are done working with it.
#parallel::stopCluster(cl = my.cluster)

#1.Specifying paths and getting israel shape.
path_modis_data = '/home/juan-david/Documents/data_science/UGA_phd/documents/Instructions_for_Earthdatasearch_data_download/Downloaded_data/MYD11A1.006'
path_ERA5_data = "/home/juan-david/Documents/data_science/UGA_phd/documents/ERA5_data_download/ERA5_data_2019_02" #path to the downloaded file
path_israel_shape = '/home/juan-david/Documents/data_science/UGA_phd/documents/Instructions_for_Earthdatasearch_data_download/Shape_files/Israel/israel_borders.shp'
israel_32636 <-file.path(path_israel_shape) %>% st_read()

#2.Obtain a list of multiband rasters from modis data.
#Get info from data.
files_ = list.files(path_modis_data)
var_names = modis_data_give_variables(files_,path_modis_data)
info_list = modis_data_give_info(files_) #Get tiles, dates, years in list.
tiles_ = info_list[[1]]
dates = info_list[[2]]
years = info_list[[3]]
files_from_selected_tiles = modis_data_subset_by_tiles(files_, c("h20v05","h20v06","h21v05","h21v06"))
dates_in_files_from_selected_tiles = modis_data_get_unique_calendar_dates(files_from_selected_tiles)
files_from_selected_dates = modis_data_subset_by_dates(files_from_selected_tiles, "2022-01-01", "2022-01-8")
unique_dates_in_files_from_selected_tiles = modis_data_get_unique_calendar_dates(files_from_selected_dates)

start_time <- Sys.time()
ls_r_multiband = modis_data_create_list_stars_rasters(files_from_selected_dates,path_modis_data, israel_32636,c("2022-01-01","2022-01-02"), var_names[0:1])
end_time <- Sys.time()
print(end_time-start_time)

start_time <- Sys.time()
ls_r_multiband = modis_data_create_list_stars_rasters_imp(files_from_selected_dates,path_modis_data, israel_32636, c("2022-01-01","2022-01-02"), var_names[0:1])
end_time <- Sys.time()
print(end_time-start_time)

#Obtain one raster info.
r_multiband = ls_r_multiband[[1]]
r_multiband_template = stars_raster_object_give_stars_raster_template(r_multiband)
mapview(r_multiband_template)

#plot.
r_multiband_filtered = stars_multiband_raster_object_subset_by_dates(r_multiband, "2022-01-01","2022-01-07")
times = st_dimensions(r_multiband_filtered)$time$values
plot(r_multiband_filtered)

#2.Read ERA5 data.
files_ERA5 = list.files(path_ERA5_data)
info_list_era5 = era5_data_give_info(files_ERA5)
var_names = info_list_era5[[1]]
years = info_list_era5[[2]]
months = info_list_era5[[3]]
days = info_list_era5[[4]]
dates = info_list_era5[[5]]
files_from_selected_dates = era5_data_subset_by_dates(files_ERA5, "2019-02-01","2019-02-02")
unique_dates_in_files_from_selected_dates= era5_data_get_unique_calendar_dates(files_from_selected_dates)

#Obtain multiband raster and some characteristics.
r_multiband_era5 = era5_data_create_multiband_raster(files_from_selected_dates,path_ERA5_data,israel_32636, unique_dates_in_files_from_selected_dates, "r_era5_temp",r_multiband_template)
times = st_dimensions(r_multiband_era5)$time$values
r_multiband_era5_filtered = stars_multiband_raster_object_subset_by_dates(r_multiband_era5, "2019-02-01 00:00:00","2019-02-01 07:00:00")
times = st_dimensions(r_multiband_era5_filtered)$time$values

#3.Visualize data.
plot(r_multiband_era5_filtered)
mapview(r_multiband_era5_filtered[,,,2])

#4.Obtain dataframe from raster.
start_time <- Sys.time()
df_r = stars_raster_object_give_dataframe(r_multiband_era5)
end_time <- Sys.time()
print(end_time - start_time)

tail(df_r)
length(unique(df_r$id))
length(unique(df_r$time))
unique(df_r$time)
unique(df_r$r_era5_temp)
nrow(df_r)

#7.Ontain same raster from df.
#Alternative 1.
dx = st_dimensions(r_multiband_era5_filtered)$x$delta
dy = st_dimensions(r_multiband_era5_filtered)$y$delta
list_r_new = list_stars_raster_objects_from_dataframe_imp(df_r, st_crs(r_multiband_era5_filtered),dx,dy)
r_new = list_r_new[[1]]
plot(r_new)

#Alternative 2.
r_new_2 = stars_raster_object_from_dataframe(df_r, 'r_era5_temp', st_crs(r_multiband_era5_filtered),dx,dy)
plot(r_new_2)

#get associated dataframe.
df_r_new = list_stars_raster_objects_give_dataframe(list_r_new)
head(df_r_new)
length(unique(df_r_new$id))
length(unique(df_r_new$time))
nrow(df_r_new)