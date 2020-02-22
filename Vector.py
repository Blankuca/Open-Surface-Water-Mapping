# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 22:23:10 2020

@author: remot
"""

import rasterio
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd

def shape_to_df(raster):
    df = pd.DataFrame()
    i = 0
    
    for band in raster:
        i +=1
        df['B'+str(i)] = band.flatten()
    
    return df 

def dataset(waterpath, rasterpath):
    
    raster = rasterio.open(rasterpath)
    vector = gpd.read_file(waterpath)
    
    out_img, out_transform = mask(raster, vector['geometry'], crop=True)
    
    df_shp = shape_to_df(out_img)
    
    dim = out_img[0].shape
    
    return {'dim':dim, 'df':df_shp}
    
def training(water_dict, ground_dict):
    water = water_dict['df']
    ground = ground_dict['df']
    
    water['Content'] = [1]*len(water)
    ground['Content'] = [0]*len(ground)
    
    ground = ground[(ground['B1'] != 0)]
    water = water[(water['B1'] != 0)]
    
    df = pd.concat([water, ground])
    
    return {'df':df,'dim_water':water_dict['dim'],'dim_ground':ground_dict['dim']}
