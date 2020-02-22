'''
This script creates a dataframe containing a flat stream the values for each of the bands from Sentinel-2. Additionally, water index columns have been computed and added to the dataframe.
'''
import pandas as pd
import rasterio

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

def main(filepath):
    raster = rasterio.open(filepath)
    df = bands_to_df(raster)
    df = add_NDWI(df)    
    df = add_NDVI(df)
    df = add_mNDWI(df)
    
    dim = raster.read(1).shape
    
    return {'dim':dim, 'df':df}
