#Assign values from one raster to another one, by reading data from a vector layer

#1.Loading required libraries.
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
source("/home/juan-david/Documents/data_science/UGA_phd/documents/Spatial_Data_programming_with_R_course_Michael_Dorman/section_3/helper_functions_practice_3.r")

#Updating path way.
pre_path <-  Sys.info()["sysname"] %>%
  switch(Windows = 'P:/',
         Linux = '/home/juan-david/Documents/data_science/UGA_phd/documents/')
print(pre_path)


#2.Loading files.
#Importing modis vector layer data.
modis_data <-
  file.path(paste0(pre_path,'Spatial_Data_programming_with_R_course_Michael_Dorman/section_2/files/'),'MODIS_vector_layer.shp') %>%
  st_read()
st_crs(modis_data)

#Importing ERA5 vector layer data.
ERA5_data <-
  file.path(paste0(pre_path,'Spatial_Data_programming_with_R_course_Michael_Dorman/section_2/files/'),'ERA5_vector_layer.shp') %>%
  st_read()
ERA5_data <- st_transform(ERA5_data, st_crs(modis_data))

#Importing Modis raster for template.
modis_data_raster = read_stars(paste0(pre_path,'Spatial_Data_programming_with_R_course_Michael_Dorman/section_2/files/MODIS_raster.tif'))
modis_data_raster = modis_data_raster[,,,1]

#Importing israel shape file.
israel_32636 <-
  file.path(paste0(pre_path,'/Instructions_for_Earthdatasearch_data_download/Shape_files/Israel/'),'israel_borders.shp') %>%
  st_read()
israel_sinusoidal <- st_transform(israel_32636, st_crs(modis_data))

#Crop modis_data_raster.
modis_data_raster = modis_data_raster[israel_sinusoidal]
plot(modis_data_raster)
mapview(modis_data_raster)
st_crs(modis_data_raster)

#3.Chech obtained objects.
#Observe the raster.
dim(modis_data_raster)
#Get indexes of non NA values in raster values.
non_na_indexes = apply(modis_data_raster[[1]],1,function(x) which(!is.na(x)))
print(non_na_indexes[0:20])

#Exploring one pixel value.
modis_data_raster[[1]][2,243,1]

#4.Create raster template with empty values to transform vector layers into rasters.
template = modis_data_raster
template[[1]][] = 0

#5.Convert layers into rasters.
#Convert modis layer into raster by rasterzing..
#modis_val_rasterized = st_rasterize(modis_data[,"layer"], template)
modis_val_rasterized = st_rasterize(modis_data[,"layer"])
modis_val_rasterized = modis_val_rasterized[israel_sinusoidal]

#ERA5_val_rasterized = st_rasterize(ERA5_data[,"layer"],template)
ERA5_val_rasterized = st_rasterize(ERA5_data[,"layer"])
ERA5_val_rasterized = ERA5_val_rasterized[israel_sinusoidal]

plot(modis_val_rasterized)
plot(ERA5_val_rasterized)

#Check values after rasterization.
#modis
dim(modis_val_rasterized)
non_na_modis_val_rasterized = apply(modis_val_rasterized[[1]],1,function(x) which(!is.na(x)))
print(non_na_modis_val_rasterized[0:20])

#ERA5.
dim(ERA5_val_rasterized)
non_na_ERA5_val_rasterized = apply(ERA5_val_rasterized[[1]],1,function(x) which(!is.na(x)))
print(non_na_ERA5_val_rasterized[0:20])

#Exploring one pixel value.
modis_val_rasterized[[1]][2,243]
ERA5_val_rasterized[[1]][2,243]

#Dissolving values from ERA5 to Modis..
ERA5_val_rasterized_1 = st_warp(ERA5_val_rasterized, modis_val_rasterized, method = 'near')
dim(ERA5_val_rasterized_1)
non_na_ERA5_val_rasterized_1 = apply(ERA5_val_rasterized_1[[1]],1,function(x) which(!is.na(x)))
print(non_na_ERA5_val_rasterized_1[0:20])
plot(ERA5_val_rasterized_1)

#Check
st_bbox(modis_val_rasterized)
st_bbox(ERA5_val_rasterized_1)

#Visualize results.
mapview(modis_val_rasterized)
mapview(ERA5_val_rasterized_1)
