# Create a reference grid for Israel aligning data from MODIS LST data with ERAS5 data.
#Obtaining information from ERAS5 instruments.

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

#2.Adjusting path to get files.
pre_path <-  Sys.info()["sysname"] %>%
  switch(Windows = 'P:/',
         Linux = '/home/juan-david/Documents/data_science/UGA_phd/documents/')
print(pre_path)
#data path of ERAS5 data.
data_path <- paste0(pre_path, 'ERA5_data_download')
print(data_path)

#3.Importing Israel shape file.
#Israel shape file in utm degree units.
#Israel in UTM.
israel_shape_file_utm <-
  file.path(paste0(pre_path,'/Instructions_for_Earthdatasearch_data_download/Shape_files/Israel/'),'israel_borders.shp') %>%
  st_read()
st_crs(israel_shape_file_utm)
projection(israel_shape_file_utm)

#Israel shape file projected in WGS84 degrees by projecting the vector layer with sf.
wgs84_degrees_sf <- st_crs(4326)
israel_shape_file_wgs84_degrees <- israel_shape_file_utm %>% 
  st_transform(wgs84_degrees_sf)
mapview(israel_shape_file_wgs84_degrees)

#print shape file attributes.
st_crs(israel_shape_file_wgs84_degrees)
projection(israel_shape_file_wgs84_degrees)
print(israel_shape_file_wgs84_degrees)
plot(israel_shape_file_wgs84_degrees)

#Obtain bbox covering israel in WGS84 geographic.
coordiantes_israel_wgs84 <- st_bbox(israel_shape_file_wgs84_degrees)
x_min <- floor(coordiantes_israel_wgs84["xmin"] %>% as.vector())
x_max <- ceiling(coordiantes_israel_wgs84["xmax"] %>% as.vector())
y_min <- floor(coordiantes_israel_wgs84["ymin"] %>% as.vector())
y_max <- ceiling(coordiantes_israel_wgs84["ymax"] %>% as.vector())

#4.Importing data from ERAS5 ".nc" file (netCDF file type).
#Read data from file.
nc_data <- nc_open(paste0(data_path, '/my_downloaded_data_temperature.nc'))

#Obtain file variables.
list_cov_israel = subset_ERAS5_nc_file_raster_data(nc_data, "t", x_max, x_min, y_max, y_min)
temp_cov_israel <-list_cov_israel$matrix_values
lon_cov_israel <- list_cov_israel$lon
lat_cov_israel <- list_cov_israel$lat

#Close the nc file.
nc_close(nc_data) 

#5.Create raster from data.
r <- raster(t(temp_cov_israel), xmn=min(lon_cov_israel), xmx=max(lon_cov_israel), ymn=min(lat_cov_israel), ymx=max(lat_cov_israel),crs="+proj=lcc +lat_1=48 +lat_2=33 +lon_0=-100 +datum=WGS84")
#Printing propoerties.
print(r)
plot(r)
plot(st_as_stars(r))

#Cropping for israel territory the values obtained from the .nc file.
r <-  r %>%   
  mask(israel_shape_file_wgs84_degrees)  # crop to Israel
#print properties.
print(r)
plot(r)
plot(st_as_stars(r))

# Store the cell row and column numbers
r$row <- rowFromCell(r, 1:ncell(r))
r$col <- colFromCell(r, 1:ncell(r))
pts = as.data.frame(r,xy=T,na.rm=T) %>% as.data.table()
pts = st_as_sf(pts, coords = c("x", "y"), crs = 4326)

#Geographic points projected into Modis sinusoidal.
proj_sinusoidal <- "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
pts_proj_sinousoidal = pts %>% st_transform(proj_sinusoidal)

#Continuing the analysis with EPGS4326 points.
pts_coor <- st_coordinates(pts)
pts_coor <- as.data.frame(pts_coor)
pts = st_drop_geometry(pts)
pts <- cbind(pts, pts_coor)

#Remove the column layer
#pts[,layer:=NULL]
#print(pts)

#Reset the coordinates and data rownames to start at 1
rownames(pts) <- 1:nrow(pts)

#Representing the points as a vector layer with crs equal to EPGS:4326.   
pts_sf.wgs84 <- st_as_sf(pts, coords = c("X", "Y"), crs = 4326)

#6.Converting to dara into a matrix of polygons.
#The methodology is extended in comparison to the common one cause matrix should be created from degree crs for equally distanced pixels.
#Then, hte matrix should by reprojected into sinusoidal crs.
pts_sf.wgs84_sp <- sf:::as_Spatial(pts_sf.wgs84)
pA <- spTransform(pts_sf.wgs84_sp,"SpatialPolygonsDataFrame", CRSobj = projection(pts_sf.wgs84_sp))
gridded(pA) <-TRUE
pixels.poly <- as(pA,"SpatialPolygonsDataFrame")

#Obtain pixels in sinusoidal.
proj_sinusoidal <- "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
pixels.poly_sinusoidal <- spTransform(pixels.poly, CRS(proj_sinusoidal))

#Obtain points as sf in the case of needed.
pts.sinusoidal <- as(pixels.poly_sinusoidal, "sf")
pixels_sinusoidal <- as(pts.sinusoidal, 'Spatial')
#pixels_sinusoidal@data <- data.frame(id=as.integer(rownames(pts)))
pixels_sinusoidal <- spTransform(pixels_sinusoidal,"SpatialPointsDataFrame",CRSobj = proj_sinusoidal)

#Plotting with mapview.
mapview(pixels_sinusoidal)


#Exporting vector layer.
shapefile(pixels.poly,"/home/juan-david/Documents/data_science/UGA_phd/documents/Spatial_Data_programming_with_R_course_Michael_Dorman/section_2/files/ERA5_vector_layer.shp", overwrite=TRUE)
