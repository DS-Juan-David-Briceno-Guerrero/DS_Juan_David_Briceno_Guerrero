#Import libraries.
#install.packages("ncmeta")
#install.packages("doParallel",dependencies=T)
#Install library to deal with solar time units.
#remotes::install_github('MLezamaValdes/LocST')
library(purrr)
library(LocST)
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
source("/home/juan-david/Documents/data_science/UGA_phd/documents/Spatial_Data_programming_with_R_course_Michael_Dorman/section_8/helper_functions_practice_8.r")

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
parallel::clusterEvalQ(my.cluster,{library(LocST)})
#parallel::clusterEvalQ(my.cluster,{library(lubridate)})

#Check the function doParallel was registered correctly.
foreach::getDoParRegistered()
#Check the number of workers availabe.
foreach::getDoParWorkers()
#Recommendable to stop the cluster when we are done working with it.
#parallel::stopCluster(cl = my.cluster)

#1.Specifying paths and getting israel shape.
path_modis_aqua_data = '/home/juan-david/Documents/data_science/UGA_phd/documents/Instructions_for_Earthdatasearch_data_download/Downloaded_data/MYD11A1.006'
path_modis_terra_data = '/home/juan-david/Documents/data_science/UGA_phd/documents/Instructions_for_Earthdatasearch_data_download/Downloaded_data/MOD11A1.006'
path_ERA5_data = "/home/juan-david/Documents/data_science/UGA_phd/documents/ERA5_data_download/ERA5_data_2022_01" #path to the downloaded file
path_israel_shape = '/home/juan-david/Documents/data_science/UGA_phd/documents/Instructions_for_Earthdatasearch_data_download/Shape_files/Israel/israel_borders.shp'
israel_32636 <-file.path(path_israel_shape) %>% st_read()

#2.Obtain Modis data.
#Get aqua data.
files_ = list.files(path_modis_aqua_data)
var_names = modis_data_give_variables(files_,path_modis_aqua_data)
info_list = modis_data_give_info(files_) #Get tiles, dates, years in list.
tiles_ = info_list[[1]]
dates = info_list[[2]]
years = info_list[[3]]
files_from_selected_tiles = modis_data_subset_by_tiles(files_, c("h20v05","h20v06","h21v05","h21v06"))
dates_in_files_from_selected_tiles = modis_data_give_unique_calendar_dates(files_from_selected_tiles)
files_from_selected_dates = modis_data_subset_by_dates(files_from_selected_tiles, "2022-01-01","2022-01-7")
unique_dates_in_files_from_selected_tiles = modis_data_give_unique_calendar_dates(files_from_selected_dates)
ls_r_multiband_aqua = modis_data_create_list_stars_rasters_imp(files_from_selected_dates,path_modis_aqua_data, israel_32636, unique_dates_in_files_from_selected_tiles, c("LST_Day_1km","Day_view_time","Night_view_time"))
plot(ls_r_multiband_aqua[[1]])

#Obtain one raster info.
r_multiband_aqua = ls_r_multiband_aqua[[1]]
r_multiband_aqua_template = stars_raster_object_give_stars_raster_template(r_multiband_aqua)
r_multiband_aqua_template_sf = stars_raster_object_give_sf_vector_layer_template(r_multiband_aqua,FALSE)

#Get terra data.
files_ = list.files(path_modis_terra_data)
var_names = modis_data_give_variables(files_,path_modis_terra_data)
info_list = modis_data_give_info(files_) #Get tiles, dates, years in list.
tiles_ = info_list[[1]]
dates = info_list[[2]]
years = info_list[[3]]
files_from_selected_tiles = modis_data_subset_by_tiles(files_, c("h20v05","h20v06","h21v05","h21v06"))
dates_in_files_from_selected_tiles = modis_data_give_unique_calendar_dates(files_from_selected_tiles)
files_from_selected_dates = modis_data_subset_by_dates(files_from_selected_tiles, "2022-01-01","2022-01-7")
unique_dates_in_files_from_selected_tiles = modis_data_give_unique_calendar_dates(files_from_selected_dates)
ls_r_multiband_terra = modis_data_create_list_stars_rasters_imp(files_from_selected_dates,path_modis_terra_data, israel_32636, unique_dates_in_files_from_selected_tiles, c("LST_Day_1km","Day_view_time","Night_view_time"))
plot(ls_r_multiband_terra[[1]])

#Check terra grid.
ls_r_multiband_aqua
ls_r_multiband_terra

#Gurantee the same template.
ls_r_multiband_terra = list_stars_raster_objects_reproject(ls_r_multiband_terra, r_multiband_aqua_template, st_crs(r_multiband_aqua_template))
h = c(ls_r_multiband_aqua, ls_r_multiband_terra)
h

modis_data_create_list_stars_aqua_rasters_

#Obtain utc overpass times from raster.
ls_r_multiband_terra = list_stars_raster_objects_calculate_utc_overpass_time_rasters(h)

vars = vars = list_stars_raster_objects_give_variables(h)
a = which(vars == "Day_view_time")
b = which(vars == "Night_view_time")
d = c(a,b)


#Give dataframe explaining Modis data.
df_r_multi = list_stars_raster_objects_give_dataframe(ls_r_multiband)
head(df_r_multi)
length(unique(df_r_multi$id))
nrow(df_r_multi)
unique(df_r_multi$time)
unique(df_r_multi$LST_Day_1km)
unique(df_r_multi$Day_view_time)
unique(df_r_multi$Night_view_time)
unique(df_r_multi$Day_view_time_utc)
unique(df_r_multi$Night_view_time_utc)

#2.Obtain IMS data.
df_israel_stations = read.csv("/home/juan-david/Documents/data_science/UGA_phd/documents/IMS_data_download/df_IMS_data_2022_01/df_stations_hourly.csv", header = TRUE, sep = ",")
head(df_israel_stations)
length(unique(df_israel_stations$period))
unique(df_israel_stations$stationId)
nrow(df_israel_stations)

#Obtain sf template of stations and their locations.
temp_israel_stations = ims_data_give_template(df_israel_stations, st_crs(r_multiband_templeate_sf))
mapview(temp_israel_stations)

#Obtain stations in raster grid.
vector_layer_israel_stations = modis_data_give_sf_vector_layer_israel_stations(r_multiband_templeate_sf,temp_israel_stations)
head(vector_layer_israel_stations)
mapview(vector_layer_israel_stations)
length(unique(vector_layer_israel_stations$id))
head(df_r_multi)
head(vector_layer_israel_stations)

#Filter df modis data by cell containing stations in IMS.
nrow(df_r_multi)
df_Modis_IMS_stations = filter(df_r_multi, id %in% unique(vector_layer_israel_stations$id))
head(df_Modis_IMS_stations)
nrow(df_Modis_IMS_stations)

#Adding the stationid to the dataframe.
df_Modis_IMS_stations = left_join(df_Modis_IMS_stations,vector_layer_israel_stations, by=c("id"))
df_Modis_IMS_stations = df_Modis_IMS_stations[moveme(names(df_Modis_IMS_stations), "stationId first")]

#Give dataframe explaining IMS data: Dropping geometry of sf to obtain dataframe.
df_Modis_IMS_stations$geometry <- NULL
head(df_Modis_IMS_stations)
dim(df_Modis_IMS_stations)
unique(df_Modis_IMS_stations$stationId)
unique(vector_layer_israel_stations$stationId)
unique(df_Modis_IMS_stations$time)

#Playing.
names(df_Modis_IMS_stations)
df = df_Modis_IMS_stations[,c("stationId","time","LST_Day_1km","Day_view_time","Day_view_time_utc")]
df

#3.Obtain ERA5 hourly data.
files_ERA5 = list.files(path_ERA5_data)
info_list_era5 = era5_data_give_info(files_ERA5)
var_names = info_list_era5[[1]]
years = info_list_era5[[2]]
months = info_list_era5[[3]]
days = info_list_era5[[4]]
dates = info_list_era5[[5]]
files_from_selected_dates = era5_data_subset_by_dates(files_ERA5, "2022-01-01","2022-01-03")
unique_dates_in_files_from_selected_dates= era5_data_get_unique_calendar_dates(files_from_selected_dates)

#Obtain multiband raster and some characteristics.
r_multiband_era5 = era5_data_create_multiband_raster(files_from_selected_dates,path_ERA5_data,israel_32636, unique_dates_in_files_from_selected_dates, "r_era5_temp",r_multiband_template)
times = st_dimensions(r_multiband_era5)$time$values

#Visualize data.
plot(r_multiband_era5)
mapview(r_multiband_era5[,,,2])

#Obtain dataframe from raster.
df_era5 = stars_raster_object_give_dataframe(r_multiband_era5)
head(df_era5)
length(unique(df_era5$id))
length(unique(df_era5$time))
unique(df_era5$time)
unique(df_era5$r_era5_temp)
nrow(df_era5)

#
#Filter df era5 data by cell containing stations in IMS.
df_era5_IMS_stations = filter(df_era5, id %in% unique(vector_layer_israel_stations$id))
head(df_era5_IMS_stations)
length(unique(df_era5_IMS_stations$id))
nrow(df_era5_IMS_stations)

#Adding the stationid to the dataframe.
df_era5_IMS_stations = left_join(df_era5_IMS_stations,vector_layer_israel_stations, by=c("id"))
head(df_era5_IMS_stations)
df_era5_IMS_stations = df_era5_IMS_stations[moveme(names(df_Modis_IMS_stations), "stationId first")]
unique(df_era5_IMS_stations$stationId)


unique(df_Modis_IMS_stations)
unique(df_era5_IMS_stations$time)



#Adding variables from Modis to IMS data.
#Convert to data.table.
df_Modis_IMS_stations = setDT(df_Modis_IMS_stations)
df_Modis_IMS_stations <- df_Modis_IMS_stations[, time:=as.character(time)]
setkey(df_Modis_IMS_stations,stationId,time)
head(df_Modis_IMS_stations)
dim(df_Modis_IMS_stations)
df_israel_stations = setDT(df_israel_stations)
setkey(df_israel_stations,stationId,period)
dim(df_israel_stations)
df_Modis_IMS_stations <- df_Modis_IMS_stations[df_israel_stations]
df_Modis_IMS_stations = df_Modis_IMS_stations[df_Modis_IMS_stations$id!=0]
head(df_Modis_IMS_stations)
dim(df_Modis_IMS_stations)

#Come back to df.
class(df_Modis_IMS_stations) <- class(as.data.frame(df_Modis_IMS_stations))
class(df_israel_stations) <- class(as.data.frame(df_israel_stations))
dim(df_Modis_IMS_stations)
head(df_Modis_IMS_stations)