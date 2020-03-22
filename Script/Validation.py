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
    
    return df_shp

# Given three shapefiles and a GeoTiff files, return a dataframe with 
# the band values and the class of every pixel belonging to the shapefiles.
def make_dataset(waterpath, groundpath, riverpath, rasterpath):
    
    bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12','B13']
    water = shapes_to_dataset(waterpath, rasterpath)[bands]
    river = shapes_to_dataset(riverpath, rasterpath)[bands]
    ground = shapes_to_dataset(groundpath, rasterpath)[bands]
    
    water['Content'] = [1]*len(water)
    river['Content'] = [2]*len(river)
    ground['Content'] = [3]*len(ground)
    
    ground = ground[(ground['B1'] != 0)]
    water = water[(water['B1'] != 0)]
    river = river[(river['B1'] != 0)]
    
    df = pd.concat([water, ground, river])
    
    return df
