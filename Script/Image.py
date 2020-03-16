'''
This script creates a dataframe containing a flat stream the values for each of the bands from Sentinel-2. 
Additionally, water index columns have been computed and added to the dataframe.
'''
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import numpy as np

#%% Create dataframe of image with band values and include water indexes.
def bands_to_df(raster):
    df = pd.DataFrame()
    
    for band in raster.indexes:
        df['B'+str(band)] = raster.read(band).flatten()
    
    return df 
 
def add_NDWI(df):
    B3 = df['B3']
    B8 = df['B8']
    ndwi = (B3-B8)/(B3+B8)
    df['NDWI'] = ndwi
    return df

def add_NDVI(df):
    B4 = df['B4']
    B8 = df['B8']
    ndvi = (B8-B4)/(B8+B4)
    df['NDVI'] = ndvi
    return df

def add_mNDWI(df):
    B3 = df['B3']
    B11 = df['B11']
    mndwi = (B3-B11)/(B3+B11)
    df['mNDWI'] = mndwi
    return df

def make_dataset(filepath):
    raster = rasterio.open(filepath)
    df = bands_to_df(raster)
    df = add_NDWI(df)    
    df = add_NDVI(df)
    df = add_mNDWI(df)
    
    dim = raster.read(1).shape
    
    return {'dim':dim, 'df':df}

#%% For plotting

def show_img(dic, band):
    s = np.array(dic['df'][band]).reshape(dic['dim'])
    plt.figure(figsize=(100,100))
    plt.imshow(s, cmap = 'Greys')
    plt.colorbar()
    
    return s

def show_pred(sh):
    plt.figure(figsize = (100,100))
    plt.imshow(sh, cmap = 'Blues')
    plt.axis()

def threshold(band, dim):
    if band.name == 'NDVI':
        thr_band = np.array([0 if (i > 0) else 1 for i in band]).reshape(dim)
    else:
        thr_band = np.array([1 if (i > 0) else 0 for i in band]).reshape(dim)
    plt.figure(figsize=(100,100))
    plt.imshow(thr_band, cmap = 'Blues')
    
    return thr_band

def RGB(rasterpath):
    raster = rasterio.open(rasterpath)
    R = raster.read(4)
    G = raster.read(3)
    B = raster.read(2)
    plt.imshow(R+G+B)
      
def RGB_hist(df):
    df['B1'].hist(bins=50, alpha= 0.3,color = 'red')
    df['B2'].hist(bins=50, alpha= 0.3,color = 'green')
    df['B3'].hist(bins=50, alpha= 0.3,color = 'blue')
