#Import libraries.
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
library(dplyr)
library(MASS)
library(reshape) #
library(ncdf4) # package for netcdf manipulation
library(rgdal) # package for geospatial analysis
library(ggplot2)
library(tidyr)
source("/home/juan-david/Documents/data_science/UGA_phd/documents/Spatial_Data_programming_with_R_course_Michael_Dorman/helper_functions_practice_1.r")
source("/home/juan-david/Documents/data_science/UGA_phd/documents/Spatial_Data_programming_with_R_course_Michael_Dorman/section_5/helper_functions_practice_5.r")
source("/home/juan-david/Documents/data_science/UGA_phd/documents/Spatial_Data_programming_with_R_course_Michael_Dorman/section_6/helper_functions_practice_6.r")

#1.Prepare data.
#Load IMS data.
df_israel_stations = read.csv("/home/juan-david/Documents/data_science/UGA_phd/documents/IMS_data_download/df_IMS_data_2022_01/df_stations.csv", header = TRUE, sep = ",")
df_israel_station_20 = filter(df_israel_stations, stationId %in%  20)
nrow(df_israel_station_20)
head(df_israel_stations)
unique(df_israel_stations$period)
unique(df_israel_stations$stationId)

#Get data (Modis) and israel shape file.
path_modis_data = '/home/juan-david/Documents/data_science/UGA_phd/documents/Instructions_for_Earthdatasearch_data_download/Downloaded_data/MYD11A1.006'
path_israel_shape = '/home/juan-david/Documents/data_science/UGA_phd/documents/Instructions_for_Earthdatasearch_data_download/Shape_files/Israel/israel_borders.shp'
israel_32636 <-file.path(path_israel_shape) %>% st_read()
mapview(israel_32636)

#Get info from data.
files_ = list.files(path_modis_data)
var_names = modis_data_give_variables(files_,path_modis_data)
info_list = modis_data_give_info(files_) #Get tiles, dates, years in list.
tiles_ = info_list[[1]]
dates = info_list[[2]]
years = info_list[[3]]

#Obtain a list of multiband rasters from data.
files_from_selected_tiles = modis_data_subset_by_tiles(files_, c("h20v05","h20v06","h21v05","h21v06"))
dates_in_files_from_selected_tiles = modis_data_get_unique_calendar_dates(files_from_selected_tiles)
files_from_selected_dates = modis_data_subset_by_dates(files_from_selected_tiles, "2022-01-01", "2022-01-8")
unique_dates_in_files_from_selected_tiles = modis_data_get_unique_calendar_dates(files_from_selected_dates)
ls_r_multiband = modis_data_create_list_stars_rasters(files_from_selected_dates,path_modis_data, israel_32636, unique_dates_in_files_from_selected_tiles, c("LST_Day_1km","Emis_31","LST_Night_1km"))

#Subset one multiband raster from the list.
r_multiband = ls_r_multiband[[1]]
st_bbox(r_multiband)
plot(r_multiband)

#2.Obtain a dataframe with stations along the Israel territory.
#Obtain modis grid template(sf layer). 
temp_r_multiband = stars_raster_object_give_sf_template(r_multiband,FALSE)
mapview(temp_r_multiband)

#Obtain stations template(sf layer).
temp_israel_stations = ims_data_give_template(df_israel_stations, st_crs(temp_r_multiband))
mapview(temp_israel_stations)

#Obtain stations in grid.
vector_layer_israel_stations = modis_data_give_sf_vector_layer_israel_stations(temp_r_multiband,temp_israel_stations)
head(vector_layer_israel_stations)
mapview(vector_layer_israel_stations)
length(unique(vector_layer_israel_stations$id))
length(unique(temp_r_multiband$id))

#3.Give dataframe explaning raster.
df_r_multi = modis_data_get_dataframe_from_list_stars_rasters(ls_r_multiband)
head(df_r_multi)
length(unique(df_r_multi$id))
nrow(df_r_multi)
unique(df_r_multi$time)
unique(df_r_multi$LST_Day_1km)
head(df_r_multi)

#Select a time period from the band.
raster_1 = ls_r_multiband[[1]][,,,6]
mapview(raster_1)
df_raster_1 = df_r_multi[df_r_multi$time == '2022-01-06',]
dim(df_raster_1)
unique(df_raster_1$LST_Day_1km)
hist_raster_band(raster_1, 1,100)

#4.Create list of rasters from dataframe.
#dx = st_dimensions(r_multiband)$x$delta
#dy =st_dimensions(r_multiband)$x$delta
#ls_r_multiband_new = modis_data_get_list_stars_rasters_from_dataframe(df_r_multi, st_crs(temp_r_multiband),dx, dy)
#r_multiband_new = ls_r_multiband_new[[2]]

#Define a function to convert sf as df and viceversa.
#Define a function to represent each raster as a matrix.
#Define a function to transform a matrix into a raster.

#5.Subset raster values from stations data.
head(df_raster_1)
head(vector_layer_israel_stations)
head(df_israel_stations)
unique(df_israel_stations$period)
mapview(raster_1)
mapview(vector_layer_israel_stations)

#Disolve modis data in IMS data.
df_Modis_IMS_stations = filter(df_r_multi, id %in%  unique(vector_layer_israel_stations$id))
head(df_Modis_IMS_stations)
unique(df_Modis_IMS_stations$time)
unique(df_Modis_IMS_stations$stationId)
df_Modis_IMS_stations = left_join(df_Modis_IMS_stations,vector_layer_israel_stations, by=c("id"))
df_Modis_IMS_stations = df_Modis_IMS_stations[moveme(names(df_Modis_IMS_stations), "stationId first")]
df_Modis_IMS_stations$geometry <- NULL
dim(df_Modis_IMS_stations)

#Cnvert to data.table to speed performance.
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

#Convert temperature of stations to kelvin to compare.
df_Modis_IMS_stations$TD_kelvin = df_Modis_IMS_stations$TD + 273.15
df_Modis_IMS_stations$TD = NULL
df_Modis_IMS_stations = df_Modis_IMS_stations[moveme(names(df_Modis_IMS_stations), "stationId first")]
head(df_Modis_IMS_stations)

#Get layer.
sf_Modis_IMS_stations = st_as_sf(df_Modis_IMS_stations,  coords = c("x", "y"))
sf_Modis_IMS_stations = st_set_crs(sf_Modis_IMS_stations, st_crs(raster_1))
#mapview(sf_Modis_IMS_stations[sf_Modis_IMS_stations[c("TD_kelvin"),]])

#6,Create raster from vector layer.
dx = st_dimensions(raster_1)$x$delta
dy =st_dimensions(raster_1)$x$delta
raster_template = st_as_stars(st_bbox(raster_1), dx =dx, dy=dy)
sf_Modis_IMS_stations = filter(sf_Modis_IMS_stations, time %in%  "2022-01-01")
r_Modis_IMS_stations = st_rasterize(sf_Modis_IMS_stations,raster_template)
mapview(r_Modis_IMS_stations)
