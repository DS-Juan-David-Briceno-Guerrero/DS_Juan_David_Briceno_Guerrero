#Second practice module to work over Israel LST data.
#Downloading Israel tile rasters for a perdiod interval, and putting them together.

#Calling the libraries' functions.
library(sf)
library(lwgeom)
library(stars)
library(lattice)
library(gdalUtils)
library(magrittr)
library(gtools)

# Set working directory
# setwd("~/Downloads/modis2/")
setwd("/home/juan-david/Documents/data_science/UGA_phd/documents/Instructions_for_Earthdatasearch_data_download/Downloaded_data/Israel_2020_1_1_2020_3_1_terra/")

#1.Prepare files list.
files = list.files(pattern = "^MOD11A1.+\\.hdf$")
files_split = strsplit(files, "\\.")
dates = sapply(files_split, "[", 2)
tiles = sapply(files_split, "[", 3)
dates_tiles = paste0(dates,"_",tiles)
years = substr(dates, 2, 5)

#Checking number files.
dim(array(files))

#Checking unique values in data.
unique(dates)
unique(years)
mixedsort(unique(tiles))

#Plots to understand the distribution of data.
#General info.
info_by_date <- table(dates)
info_by_years <- table(years)
info_by_tiles <- table(tiles)
barplot(info_by_date,main ="Histogram: Dates",xlab ="X",ylab ="Count")
barplot(info_by_years,main ="Histogram: Years", xlab ="X",ylab ="Count")

# Create Extent object for cropping
ext = st_read("/home/juan-david/Documents/data_science/UGA_phd/documents/Instructions_for_Earthdatasearch_data_download/Shape_files/Israel/israel_borders.shp")
mod_proj = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
ext = st_transform(ext, mod_proj)
ext = st_buffer(ext, 10000)
ext = ext %>% st_bbox %>% st_as_sfc
print(ext)

#2.Merging tiles of Israel to create a raster image covering all the territory.
result = list()
dates_unique = unique(dates)
print(dates_unique)

#Mosaicing Israel images from data.
for(d in dates_unique) {
  # Progress
  print(d)
  #Current 4-files group
  current_files = files[dates == d]
  
  tryCatch(
    expr = {
      i = grep("h20v05", current_files, value = TRUE)
      nw = gdal_subdatasets(i)
      nw = read_stars(nw)
      
      i = grep("h21v05", current_files, value = TRUE)
      ne = gdal_subdatasets(i)
      ne = read_stars(ne)
      
      i = grep("h20v06", current_files, value = TRUE)
      sw = gdal_subdatasets(i)
      sw = read_stars(sw)
    
      i = grep("h21v06", current_files, value = TRUE)
      se = gdal_subdatasets(i)
      se = read_stars(se)
    
      n = c(nw, ne, along = "x")
      s = c(sw, se, along = "x")
      r = c(n, s, along = "y")
      
      #Defining rasters CRS.
      r = st_set_crs(r, st_crs(ext))
      
      # Crop
      k = r
      r = r[ext]
      
      # Subset
      # s = r["X.500m.16.days.pixel.reliability."]
      #s = r["X.1.km.monthly.pixel.reliability."]
      # r = r["X.500m.16.days.NDVI."]
      #r = r["X.1.km.monthly.NDVI."]
      #r[s != 0 & s != 1] = NA  # Select only good/marginal data
      
      # Rescale
      r = r * 0.0001
      r = r * 0.0001
      # Combine
      result[[d]] = r
    },
    error = function(e){
      print(paste("Is not possible to create raster image for date: ",d))
    }
  )
}

#3.Printing and plotting characteristics of the mosaicked rasters.
raster_2020_003 = result[["A2020003"]]

#Plotting raster without cropping.
print(k)
plot(k)

#Plotting a raster obtained from the files.
print(raster_2020_003)
plot(raster_2020_003)

#3.Getting dimension and properties.
st_dimensions(raster_2020_003)

#Get names of raster.
names(raster_2020_003)

#Accssesing to rasters values,
print(raster_2020_003[[1]][1:10,1:10])

#Getting the extend.
st_bbox(raster_2020_003)

#Getting the origins.
st_dimensions(raster_2020_003)$x$offset
st_dimensions(raster_2020_003)$y$offset

#Getting resolutions.
st_dimensions(raster_2020_003)$x$delta
st_dimensions(raster_2020_003)$y$delta

#Getting the CRS.
st_crs(raster_2020_003)

#Getting dimension properties.
st_dimensions(raster_2020_003)

#Getting information about vectors in the raster.
str(st_dimensions(f))
class(f)
length(f)
class(f[[1]])