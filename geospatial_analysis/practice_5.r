#Create a multiband raster for range period of time selected within data stored in a repository. 

#1.Loading required libraries.
#install.packages("MODIS", repos="http://R-Forge.R-project.org")
#library(MODIS)
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
source("/home/juan-david/Documents/data_science/UGA_phd/documents/Spatial_Data_programming_with_R_course_Michael_Dorman/helper_functions_practice_1.r")
source("/home/juan-david/Documents/data_science/UGA_phd/documents/Spatial_Data_programming_with_R_course_Michael_Dorman/section_5/helper_functions_practice_5.r")

#Oefine path to Modis data.
#path_modis_data = '/home/juan-david/Documents/data_science/UGA_phd/documents/Instructions_for_Earthdatasearch_data_download/Downloaded_data/LST_data'
#path_modis_data = '/home/juan-david/Documents/data_science/UGA_phd/documents/Instructions_for_Earthdatasearch_data_download/Downloaded_data/Israel_2020_1_1_2020_3_1_terra'
path_modis_data = '/home/juan-david/Documents/data_science/UGA_phd/documents/Instructions_for_Earthdatasearch_data_download/Downloaded_data/MYD'
path_israel_shape = '/home/juan-david/Documents/data_science/UGA_phd/documents/Instructions_for_Earthdatasearch_data_download/Shape_files/Israel/israel_borders.shp'

#Obtain israel_shape file.
israel_32636 <-file.path(path_israel_shape) %>% st_read()

#1.Get files.
#Update info from data.
files_ = list.files(path_modis_data)
info_list = modis_data_give_info(files_)
tiles_ = info_list[[1]]
dates = info_list[[2]]
years = info_list[[3]]
#check
length(files_)

#2.1 Subset by desired tiles.
files_filtered_tiles = modis_data_subset_by_tiles(files_, c("h20v05","h20v06","h21v05","h21v06"))
info_list = modis_data_give_info(files_filtered_tiles)
tiles_ = info_list[[1]]
dates = info_list[[2]]
years = info_list[[3]]
#check
length(files_filtered_tiles)

#2.2 Get available dates.
unique_dates_calendar = modis_get_unique_calendar_dates(files_filtered_tiles)
print(unique_dates_calendar)

#3.1 Subset by dates.
#files_t_1 = modis_data_subset_by_dates(files_filtered_tiles, "2020-01-02", "2020-01-03")
#files_t_1 = modis_data_subset_by_dates(files_filtered_tiles, "2022-03-01", "2022-03-20")
files_t_1 = modis_data_subset_by_dates(files_filtered_tiles, "2021-12-23", "2021-12-24")
info_list = modis_data_give_info(files_t_1)
tiles_ = info_list[[1]]
dates = info_list[[2]]
years = info_list[[3]]
#check
length(files_t_1)
unique_dates_calendar = modis_get_unique_calendar_dates(files_t_1)
print(unique_dates_calendar)

#4.Create multiband raster from selected dates and tiles.
#"LST_Day_1km$"
ls = modis_data_create_multiband_raster(files_t_1, path_modis_data ,israel_32636, unique_dates_calendar, 'LST_Day_1km$')
plot(ls)
print(ls)

#Suset raster band (select for example one calendar date).
r = ls[,,,which(unique_dates_calendar=="2021-12-24")]

#Reproject israel shape file with raster CRS.
israel_sin = st_transform(israel_32636,st_crs(r))
st_bbox(israel_sin)
st_bbox(r)

#Check results.
mapview(israel_sin)
mapview(r)

#5.Put in action all developed functions.
#Merging all function together.
#Available data.
files_filtered_tiles = modis_data_subset_by_tiles(files_, c("h20v05","h20v06","h21v05","h21v06"))
available_dates = modis_get_unique_calendar_dates(files_filtered_tiles)
#Selected filters.
files_from_selected_dates = modis_data_subset_by_dates(files_filtered_tiles, "2021-12-22", "2021-12-24")
unique_dates_in_files_from_selected_dates = modis_get_unique_calendar_dates(files_from_selected_dates)

#Create raster.
raster_from_selected_dates = modis_data_create_multiband_raster(files_from_selected_dates, path_modis_data, israel_32636,unique_dates_in_files_from_selected_dates, "LST_Day_1km$")
st_bbox(raster_from_selected_dates)
plot(raster_from_selected_dates)

#Subset one raster from the time band.
raster_1 = raster_from_selected_dates[,,,1]
mapview(raster_1)
names(raster_1)

#Validate results with one creation raster.
files_from_selected_dates_2 = modis_data_subset_by_dates(files_filtered_tiles, "2021-12-22", "2021-12-22")
unique_dates_in_files_from_selected_dates_2 = modis_get_unique_calendar_dates(files_from_selected_dates_2)
raster_1_validation = modis_data_create_raster(files_from_selected_dates_2, path_modis_data, israel_32636, "LST_Day_1km$")
mapview(raster_1_validation)
names(raster_1_validation)
