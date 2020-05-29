# -*- coding: utf-8 -*-
"""
What this file does is to create a dataset for training and validation.
It gets as input the water and ground vector containers, clips the TIF file
and returns a datset with the band values for each of the pixels and 
specifies whether is water or not.
"""

import rasterio
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
from sklearn.utils import shuffle

# Given a raster, put the bands into a pandas dataframe.
def vector_to_df(raster):
    df = pd.DataFrame()
    i = 0
    
    for band in raster:
        i +=1
        df['B'+str(i)] = band.flatten()
    
    return df 

# Given the paths of a shapefile and a GeoTiff file,
# clip the raster and put it into a dataframe.
def shapes_to_dataset(shapepath, rasterpath):
    
    raster = rasterio.open(rasterpath)
    vector = gpd.read_file(shapepath)
    
    out_img, _ = mask(raster, vector['geometry'], crop=True)
    
    df_shp = vector_to_df(out_img)
    
    df_shp = df_shp[(df_shp['B1'] != 0)]
    
    return df_shp


def add_indices(df):
    B03 = df['B3']; B08 = df['B8']; B11 = df['B11']; B04 = df['B4'];
    df['NDWI'] = (B03 - B08)/(B03 + B08)
    df['MNDWI'] = (B03-B11)/(B03+B11)
    df['NDVI'] = (B08-B04)/(B08+B04)
    # ceramic rooftop detection
    df['NDBI'] = (B11 - B08)/(B11 + B08)
    
    return df

def NDWI(df, threshold):
    B03 = df['B3']
    B8A = df['B8A']
    NDWI = (B03-B8A)/(B03+B8A)
    NDWI_t = [1 if x > threshold else 0 for x in NDWI]
    return NDWI, NDWI_t

def AWEI(df, threshold):
    B3 = df['B3']
    B8 = df['B8']
    B12 = df['B12']
    B11 = df['B11']
    AWEI = 4*(B3-B12)- (0.25*B8 + 2.75*B11)
    AWEI_t = [1 if x > threshold else 0 for x in AWEI]
    return AWEI, AWEI_t


# Given three shapefiles and a GeoTiff files, return a dataframe with 
# the band values and the class of every pixel belonging to the shapefiles.
def make_dataset(root, rasterpath):
    
    deeppath = root + 'Deep_Polygon.shp'
    shallowpath = root + 'Shallow_Polygon.shp'
    farmedpath = root + 'Farmed_Polygon.shp'
    fallowpath = root + 'Fallow_Polygon.shp'
    vegpath = root + 'Veg_Polygon.shp'
    urbanpath = root + 'Urban_Polygon.shp'
    
    bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12','B13']
    deep = shapes_to_dataset(deeppath, rasterpath)[bands]
    shallow = shapes_to_dataset(shallowpath, rasterpath)[bands]
    farmed = shapes_to_dataset(farmedpath, rasterpath)[bands]
    fallow = shapes_to_dataset(fallowpath, rasterpath)[bands]
    veg = shapes_to_dataset(vegpath, rasterpath)[bands]
    urban = shapes_to_dataset(urbanpath, rasterpath)[bands]
       
#    deep['Content'] = ['Deep']*len(deep)
#    shallow['Content'] = ['Shallow']*len(shallow)
#    farmed['Content'] = ['Dry']*len(farmed)
#    fallow['Content'] = ['Soil']*len(fallow)
#    veg['Content'] = ['Vegetation']*len(veg)
#    urban['Content'] = ['Urban']*len(urban)
    
    deep['Content'] = [1]*len(deep)
    shallow['Content'] = [2]*len(shallow)
    farmed['Content'] = [3]*len(farmed)
    fallow['Content'] = [4]*len(fallow)
    veg['Content'] = [5]*len(veg)
    urban['Content'] = [6]*len(urban)
    
    df = pd.concat([deep, shallow, farmed, fallow, veg, urban])
    
    return df