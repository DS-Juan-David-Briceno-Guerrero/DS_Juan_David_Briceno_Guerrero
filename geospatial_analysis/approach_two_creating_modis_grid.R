# Create a reference grid for Israel aligned to the MODIS LST data
# * Mosaic all MODIS Aqua LST tiles for a single date
# * Crop to extent of Israel
# * Set cell x coordinate, y coordinate, row number, and col number
# * Convert cell centroids to SpatialPoints
# * Remove points that are not in the study area
# * Add latitude / longitude and LST 1 km id
# * Save in EPSG: 2039 (Israeli TM Grids)
# * Also save as shapefile and as SpatialPixels in MODIS sinusoidal
#install.packages("mapview")
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

pre_path <-  Sys.info()["sysname"] %>%
  switch(Windows = 'P:/',
         Linux = '/home/juan-david/Documents/data_science/UGA_phd/documents/')
print(pre_path)


#data path of MODIS LST products
data_path <- paste0(pre_path, '/Instructions_for_Earthdatasearch_data_download/Downloaded_data/')
print(data_path)

# Set working directory having MOD data.
setwd(paste0(data_path,'/Israel_2020_1_1_2020_3_1_terra/'))

#list the tiles of a single date (e.g. 2015001)
#Listed files are from 2020 year.
files_ <- list.files(paste0(pre_path,'Instructions_for_Earthdatasearch_data_download/Downloaded_data/Israel_2020_1_1_2020_3_1_terra'),pattern = '.*A202000.*\\.hdf$', full.names = TRUE)
files_split_ = strsplit(files_, "\\.")
tiles_ = sapply(files_split_, "[", 3)
print(length(unique(tiles_)))
print(head(tiles_))
tiles <- files_

#Filtering tiles to cover only the ones involving Israel.
israel_tiles = c("h20v05","h20v06","h21v05","h21v06")
new_tiles_ = vector()
for (t in israel_tiles){
  print(t)
  
  f <- tiles[tiles_== t]
  print(head(f))
  
  new_tiles_ = c(new_tiles_,f)
}
tiles <- new_tiles_

#Check unique tiles.
print(unique(tiles))


#Get the columns (variables) presented in a raster.
print(tiles[1] %>% get_subdatasets()) 

# It's important to understand that rasters here are obtained as objects of the class raster.
# Load the night LST dataset for each tile and clear the values
# The column that is taken from the rasters is "LST_Day_1km".
rasters <- lapply(tiles, function(tile) {
  get_subdatasets(tile) %>%         # list the scientific datasets for the tile
  .[grepl("LST_Day_1km$", .)] %>%   # select the night LST dataset
  raster                        # load the dataset as a raster
#  raster                            # clear the values to speed up mosaicing
})
rm(tiles)

#Printing the first and seconds raster, with their attributes.
#It's important to notice the difference in the tiles h,v values even though the date is the same. 
r_1 = rasters[[1]]
print(st_as_stars(r_1))
print(st_dimensions(st_as_stars(r_1)))
print(names(st_as_stars(r_1)))
print(r_1%>%st_as_stars()%>%st_crs())
print(st_as_stars(r_1)[[1]][1:10,1:10])
print(r_1 %>% projection())

#Change crs of previous raster to geographic.
r_1_mod = st_set_crs(st_as_stars(r_1),4326)
print(r_1_mod%>%st_crs())
print(r_1_mod[[1]][1:10,1:10])
print(projection(as(r_1_mod, "Raster")))

#Print 2 raster.
r_2 = rasters[[2]]
print(st_as_stars(r_2))
print(st_dimensions(st_as_stars(r_2)))
print(names(st_as_stars(r_2)))

#Printing the amount of rasters obtained.
length(rasters)

# Load a shapefile of Israel and project to match the tiles
# EPSG: 32636   WGS84_UTM36N
israel_32636 <-
  file.path(paste0(pre_path,'/Instructions_for_Earthdatasearch_data_download/Shape_files/Israel/'),'israel_borders.shp') %>%
  st_read()

#Print crs of israel_32636.
print(israel_32636%>%st_crs())

#Print projection of israel_32636.
proj.israel_32636 <- projection(israel_32636)

#Obtaining projection from the first raster.
proj.sinu <- projection(rasters[[1]])

#Transforming to sinusoidal CRS the israel_shape file.
israel_sinu <- israel_32636 %>% 
          st_transform(proj.sinu)

#Example
raster_sin = st_as_stars(r_2)
crs_1 = st_crs(raster_sin)
plot(raster_sin)
st_crs(raster_sin)
mapview(raster_sin)
raster_sin[[1]][0:10,1:10]
#terra
#raster_sin = rast(raster_sin)
vector_sin = st_as_sf(raster_sin)
terra_sin <- vect(raster_sin)
#vals


#
crs_2 = st_crs(israel_32636)
raster_wgs84 = st_warp(raster_sin,crs = 4326)
plot(raster_wgs84)
  st_crs(raster_wgs84)
mapview(raster_wgs84)
raster_wgs84[[1]][0:10,1:10]
#terra
#raster_sin = rast(raster_wgs84)
#as sf layers.
vector_wgs84 = st_as_sf(raster_wgs84)
terra_wgs84 <- vectraster_wgs84
#vals


# Mosaic the tiles and clip to Israel.
mos <- do.call(merge, rasters)
plot(mos)
plot(st_as_stars(mos))
mapview(mos)

#Get dataframe representation of raster.
mos_df_1 = as.data.frame(mos,xy=T,na.rm=T)
head(mos_df_1)

#names(mos) <- "QC_Day"
# remove rasters
rm(rasters)

#Obtain object raster as stars object.
#write_stars.
mos_stars = st_as_stars(mos)
print(mos_stars)
plot(mos_stars)
#plot(mos_stars[,,,1])
write_stars(mos_stars, dsn =paste0(pre_path, 'Spatial_Data_programming_with_R_course_Michael_Dorman/section_2/files/MODIS_raster.tif'))


# value ==3  indicates water courses 
#mos[mos!=3] <- 1     # land 
#mos[mos==3] <- 0     # water
#plot(mos)

mos <-  mos %>%   
  mask(israel_sinu)  # crop to Israel
plot(mos)

# Store the cell row and column numbers
mos$row <- rowFromCell(mos, 1:ncell(mos))
mos
mos$col <- colFromCell(mos, 1:ncell(mos))

#Check rasters as stars object.
print(st_as_stars(mos))
print(st_dimensions(st_as_stars(mos)))
print(names(st_as_stars(mos)))
plot(st_as_stars(mos$layer))


df_mos = as.data.frame(mos,xy=T,na.rm=T)
head(df_mos)
unique(df_mos[c("layer")])

# convert raster to data.table, with x,y coordinates, remove NA value
#pts <- as.data.frame(mos,xy=T,na.rm=T) %>% 
#    subset(layer==1) %>%
#      as.data.table()

pts <- as.data.frame(mos,xy=T,na.rm=T) %>%
      as.data.table()
pts

# remove the column layer
#pts[,layer:=NULL]
#print(pts)
#dim(pts)

# Reset the coordinates and data rownames to start at 1
#pts[, id:=.I] 
rownames(pts) <- 1:nrow(pts)

# MODIS Sinusoidal projection 
pts_sf.sinu <-  st_as_sf(pts, coords = c("x", "y"), 
                 crs = proj.sinu)

pts_sf.sinu$id <- 1:nrow(pts_sf.sinu)

# UTM 36N 
pts_sf.utm36 <- pts_sf.sinu %>% st_transform(32636)

#Israeli TM grids  EPSG: 2039
pts_sf.itm <- pts_sf.sinu %>% st_transform(2039)

# Geographic lon/lat   
pts_sf.wgs84 <- pts_sf.sinu %>% st_transform(4326)

grid <-  list(sinu=pts_sf.sinu, utm36=pts_sf.utm36,itm=pts_sf.itm, wgs84=pts_sf.wgs84)
print(grid)

# Save the spatial points dataframe to an rds file
outpath <- file.path(pre_path %>% substr(1, nchar(pre_path)-1),'Spatial_Data_programming_with_R_course_Michael_Dorman/section_2/documents_approach_two_creating_modis_grid')

if (!file.exists(outpath)){
  dir.create(outpath, showWarnings = FALSE)
}

# save different grid in various crs as "grid_lst_1km.rds"
path <- file.path(outpath,"grid_lst_1km.rds")
saveRDS(grid, path)


# save grid centroids [in sinusoidal projection] as a shape file 
path <- file.path(outpath,"grid_lst_1km_sinu_centroids.shp")
st_write(pts_sf.sinu, path, delete_layer =TRUE)

#Important.
# This will facilitate creating rasters of extracted data for visualization
path <- file.path(outpath, "grid_lst_1km_sinu_pixels.rds")
#tail(pts_sf.sinu)
pixels <- as(pts_sf.sinu, 'Spatial')
pixels@data <- data.frame(id=as.integer(rownames(pts)))
gridded(pixels) <- TRUE

# gridded pixels can  illustrate one parameter each time.
saveRDS(pixels, path)


# write the grid to shapefile and rds
path <- file.path(outpath, "grid_lst_1km_sinu_pixels.shp")
pixels.poly <- as(pixels,"SpatialPolygonsDataFrame")
shapefile(pixels.poly,path, overwrite=TRUE)

# save the grid polygons .RDS
path <- file.path(outpath, "grid_lst_1km_sinu_polygons.rds")
saveRDS(pixels.poly,path)


##Final stage.
#Print developped pixels and their properties.
mapview(pixels.poly)
print(pixels.poly)
class(pixels.poly)

#Convert sp object to sf.
pixels.poly_sf = st_as_sf(pixels.poly)
class(pixels.poly_sf)
print(pixels.poly_sf)
mapview(pixels.poly_sf)

#Get attributes.
#Printing 10 first features.
print(pixels.poly_sf[1:10, c("id")])

#st_drop_geometry(pixels.poly_sf)

#Exporting vector layer.
shapefile(pixels.poly,"/home/juan-david/Documents/data_science/UGA_phd/documents/Spatial_Data_programming_with_R_course_Michael_Dorman/section_2/files/MODIS_vector_layer.shp", overwrite=TRUE)
