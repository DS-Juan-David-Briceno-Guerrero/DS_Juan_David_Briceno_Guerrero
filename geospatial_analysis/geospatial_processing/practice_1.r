#First practice module to summarize the sections 5 and 6 of the Micheal Dorman's course. 

#Installing implied libraries.
#install.packages('sf')
#install.packages('lwgeom')
#install.packages('stars')
#install.packages('lattice')

#Calling the libraries' functions.
source("/home/juan-david/Documents/data_science/UGA_phd/documents/Spatial_Data_programming_with_R_course_Michael_Dorman/helper_functions_practice_1.r")
library(sf)
library(lwgeom)
library(stars)
library(lattice)

##1.
#Loading the file obtained from the earthdatasearch page. (Polygone covering Lyon.)
file_path = "/home/juan-david/Documents/data_science/UGA_phd/documents/Instructions_for_Earthdatasearch_data_download/Downloaded_data/Israel_2020_2021_terra/MOD11A1.A2021001.h35v10.061.2021040151701.hdf"
sd = gdal_subdatasets(file_path)
#Printing sd shows that it contains 12 rasters.
print(sd)

#Getting first raster.
f = read_stars(sd[[1]])
print(f)

##2.
##Exploration of rasters attributes.
#Get name of raster.
names(f)

#Changing the names for better comprehension.
names(f) <- c("units_1")
#names(f) <- c("units_1","units_2","units_3","units_4","units_5","units_6","units_7","units_8","units_9","units_10","units_11","units_12")

#Get columns and rows.
nrow(f)
ncol(f)

#Get dimensions.
dim(f)

#Getting the extend.
st_bbox(f)

#Getting the origins.
st_dimensions(f)$x$offset
st_dimensions(f)$y$offset

#Getting resolutions.
st_dimensions(f)$x$delta
st_dimensions(f)$y$delta

#Getting the CRS.
st_crs(f)

#Getting dimension properties.
st_dimensions(f)

#Getting information about vectors in the raster.
str(st_dimensions(f))
class(f)
length(f)
class(f[[1]])

##3.Accssesing and working with raster values.
#Accssesing the first 5 cols, and 7 rows of the raster values in the first array file.
print(f$units_1[1:7,1:5])
print(f[[1]][1:7,1:5])

##4.Creating a multiband raster.
#Adding band layers with periods of time.
f_mod = replicate_raster_n_periods(f, 2, "times")
st_dimensions(f_mod)
str(st_dimensions(f_mod))

#Print values of band 2.
print(f_mod[[1]][1:7,1:5,2])

##6.Raster processing and algebra.
#Getting a dataframe from a raster.
df_f_mod = as.data.frame(f_mod)
head(df_f_mod)

#Deploying histogram distribution for first raster band.
hist_raster_band(f_mod, 1,100)

#Filling NA values of 1 and 2 bands' pixles with 1 and 2 through conditionals.
nan_fill_band_1 <- f_mod
nan_fill_band_1[[1]][,,1][is.na(nan_fill_band_1[[1]][,,1])] = 1
nan_fill_band_1[[1]][,,2][is.na(nan_fill_band_1[[1]][,,2])] = 2
print(nan_fill_band_1[[1]][1:7,1:5,2])
st_dimensions(nan_fill_band_1)
hist_raster_band(nan_fill_band_1, 1,100)

#Applying algebra over raster cells by using "st_apply'.
#Adding pixel values and summing by 10 over bands 1 and 2 of a raster.
u = st_apply(X=nan_fill_band_1, MARGIN = 1:2, FUN=function(x) sum(x)+10)
print(u[[1]][1:10,1:10])

#Calculating means aver pixel values..
s = st_apply(X=nan_fill_band_1, MARGIN = 1:2, mean)
print(s[[1]][1:10,1:10])

#Calculating min and max of pixels in a raster with bands.
S_min = st_apply(nan_fill_band_1, 1:2, min2)
S_max = st_apply(nan_fill_band_1, 1:2, max2)
print(S_min[[1]][1:10,1:10])
print(S_max[[1]][1:10,1:10])

#Calculating pixles amplitudes.
S_amplitud = S_max - S_min
print(S_amplitud[[1]][1:10,1:10])

##Operations over layers.
#Layers mean,
layer_mean = st_apply(f_mod, 3, mean, na.rm =TRUE)
print(layer_mean[[1]])

#Layers max.
layer_max = st_apply(f_mod, 3, max, na.rm =TRUE)
print(layer_max[[1]])

#Layers min.
layer_min = st_apply(f_mod, 3, min, na.rm =TRUE)
print(layer_min[[1]])


##7.Plotting.
#Plot simple raster.
plot(f)

#Plot multiband raster.
plot(f_mod)