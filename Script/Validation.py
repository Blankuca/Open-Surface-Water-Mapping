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

def vector_to_df(raster):
    df = pd.DataFrame()
    i = 0
    
    for band in raster:
        i +=1
        df['B'+str(i)] = band.flatten()
    
    return df 

def shapes_to_dataset(waterpath, rasterpath):
    
    raster = rasterio.open(rasterpath)
    vector = gpd.read_file(waterpath)
    
    out_img, out_transform = mask(raster, vector['geometry'], crop=True)
    
    df_shp = vector_to_df(out_img)
    
    dim = out_img[0].shape
    
    return {'dim':dim, 'df':df_shp}
    
def make_dataset(waterpath, groundpath, rasterpath):
    
    water_dict = shapes_to_dataset(waterpath, rasterpath)
    ground_dict = shapes_to_dataset(groundpath, rasterpath)
    
    water = water_dict['df']
    ground = ground_dict['df']
    
    water['Content'] = [1]*len(water)
    ground['Content'] = [0]*len(ground)
    
    ground = ground[(ground['B1'] != 0)]
    water = water[(water['B1'] != 0)]
    
    df = pd.concat([water, ground])
    
    return {'df':df,'dim_water':water_dict['dim'],'dim_ground':ground_dict['dim']}
