'''
This script creates a dataframe containing a flat stream the values for each of the bands from Sentinel-2. 
Additionally, water index columns have been computed and added to the dataframe.
'''
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
import numpy as np
import gdal
import osr
import matplotlib.image as mpimg
from rasterio.plot import show
from Script.Processing import add_indices
from matplotlib.colors import ListedColormap

households_file_1 = 'Household/Households_coord_1.xls'

village_1 = pd.read_excel(households_file_1,
                            usecols=[0,1,2],
                            index_col=0,
                            nrows=287
                            )

cols = ['B1','B2','B3','B4','B5','B6','B7','B8', 'B8A', 'B9','B10','B11','B12']
#%% Create dataframe of image with band values and include water indexes.
def bands_to_df(raster):
    df = pd.DataFrame()
    
    for band in raster.indexes:
        df['B'+str(band)] = raster.read(band).flatten()
    
    df = df[['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12','B13']]
    df.columns = cols
    return df 

def make_dataset(filepath):
    raster = rasterio.open(filepath)
    df = bands_to_df(raster)
    #df = add_indices(df)
    df = df.fillna(0)
    
    dim = raster.read(1).shape
    
    return {'dim':dim, 'df':df}

#%% For plotting

def show_img(path, show_vill = False):

    if show_vill != False:  
        
        raster = rasterio.open(path)
        switcher = {'NW' : rasterio.open('H:/sentinel2/Ghana/Validation/NW_coord.tif').transform,
                    'SE' : rasterio.open('H:/sentinel2/Ghana/Validation/SE_coord.tif').transform}
        af = switcher[show_vill]
        fig, ax = plt.subplots(1,figsize=(50,50))
        img = show(raster.read(), transform = af, ax = ax)
        vill = ax.scatter(village_1['POINT_X'],village_1['POINT_Y'], color = 'red', marker = 'x')
        plt.tick_params(axis='x', which='major', labelsize=30)
        plt.tick_params(axis='y', which='major', labelsize=30)
        plt.show()
    else:
        img=mpimg.imread(path)        
        plt.figure(figsize=(50,50))
        plt.imshow(img)

def show_pred(sh, filename = None, size = 10):
    colormaps = [ListedColormap(['dodgerblue', 'paleturquoise', 'peru', 'rosybrown', 'forestgreen'])]
    n = len(colormaps)
    data = sh[::-1]
    
    fig, axs = plt.subplots(1, n, figsize=(size,size),
                            constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=1, vmax=5)
        #fig.colorbar(psm, ax=ax, orientation="horizontal")
    plt.show()
    plt.axis('off')
    if filename != None:
        plt.savefig('Figures/{0}.png'.format(filename))
    
def show_pred_5(sh, filename = None, size = 10):
    sh = [[1 if x < 3 else 0 for x in row] for row in sh]
    plt.figure(figsize = (size,size))
    plt.imshow(sh, cmap = 'Blues')
    plt.axis()
    if filename != None:
        plt.savefig('Figures/{0}.png'.format(filename))

def threshold(band, dim):
    if band.name == 'NDVI':
        thr_band = np.array([0 if (i > 0) else 1 for i in band]).reshape(dim)
    else:
        thr_band = np.array([1 if (i > 0) else 0 for i in band]).reshape(dim)
    plt.figure(figsize=(100,100))
    plt.imshow(thr_band, cmap = 'Blues')
    
    return thr_band

def save_as_tiff(new_path, raster_path, array, two_class = False):

    dataset = gdal.Open(raster_path)
    
    if two_class: 
        array = np.array([[1 if pixel < 3 else 0 for pixel in row] for row in array])
    
    cols = array.shape[1]
    rows = array.shape[0]
    originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform() 

    driver = gdal.GetDriverByName('GTiff')

    # set data type to save.
    GDT_dtype = gdal.GDT_Unknown
    GDT_dtype = gdal.GDT_Float32

    # set number of band.
    if array.ndim == 2:
        band_num = 1
    else:
        band_num = array.shape[2]

    outRaster = driver.Create(new_path, cols, rows, band_num, GDT_dtype)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    # Loop over all bands.
    for b in range(band_num):
        outband = outRaster.GetRasterBand(b + 1)
        # Read in the band's data into the third dimension of our array
        if band_num == 1:
            outband.WriteArray(array)
        else:
            outband.WriteArray(array[:,:,b])

    # setteing srs from input tif file.
    prj=dataset.GetProjection()
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    
    
def convert(y):
    new = np.array([])
    
    switcher = {
     'Deep':1,
     'Shallow':2,
     'Vegetation':3,
     'Soil':4,
     'Dry/Urban':5
     }
    
    for i in y:
        new = np.append(new, switcher[i])
        
    return new


#%% Data processing

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
    

def RGB_vs_img(df, path, date):
    plt.figure(figsize=(10,7))

    plt.subplot(2, 2, 1) 
    plt.title('Spectral signature for '+date)
    axes = plt.gca()
    axes.set_xlim([0,2500])
    RGB_hist(df)

    plt.subplot(2, 2, 2)
    plt.title('RGB for '+date)
    img=mpimg.imread(path)
    plt.imshow(img)
    
def class_spectral_signature(training):
    plt.figure(figsize=(10,8))

    plt.subplot(2, 2, 1) 
    plt.title('Spectral signature for Large Water Bodies')
    axes = plt.gca()
    axes.set_xlim([-1000,3000])
    RGB_hist(training.loc[training['Content'] == 1])

    plt.subplot(2, 2, 2) 
    plt.title('Spectral signature for Temporary Water Bodies')
    axes = plt.gca()
    axes.set_xlim([-1000,3000])
    RGB_hist(training.loc[training['Content'] == 2])

    plt.subplot(2, 2, 3) 
    plt.title('Spectral signature for Dry areas')
    axes = plt.gca()
    axes.set_xlim([-1000,3000])
    RGB_hist(training.loc[training['Content'] == 3])
    
    plt.subplot(2, 2, 4) 
    plt.title('Spectral signature for Vegetated areas')
    axes = plt.gca()
    axes.set_xlim([-1000,3000])
    RGB_hist(training.loc[training['Content'] == 4])
    
    plt.show()
    
def class_piechart(training, filename=None):
    
#    deep = training.loc[training['Content'] == 'Deep']
#    shal = training.loc[training['Content'] == 'Shallow']
#    farm = training.loc[training['Content'] == 'Dry/Urban']
#    fall = training.loc[training['Content'] == 'Soil']
#    veg = training.loc[training['Content'] == 'Vegetation']
    
    deep = training.loc[training['Content'] == 1]
    shal = training.loc[training['Content'] == 2]
    farm = training.loc[training['Content'] == 3]
    fall = training.loc[training['Content'] == 4]
    veg = training.loc[training['Content'] == 5]

    n_deep = len(deep); n_shal = len(shal); n_farm = len(farm); n_fall = len(fall); n_veg = len(veg)

    # Data to plot
    colors = ['dodgerblue', 'paleturquoise', 'peru', 'rosybrown', 'forestgreen']
    labels = ['Deep Water Bodies', 'Shallow Water Bodies', 'Dry/Urban area', 'Soil', 'Vegetation']
    sizes = [n_deep, n_shal, n_farm, n_fall, n_veg]

    # Plot
    plt.pie(sizes, labels=labels, colors=colors,
    autopct='%1.1f%%', shadow=True, startangle=140)
    
    plt.legend(bbox_to_anchor=(0,1.02),loc="lower left", borderaxespad=0.)
    plt.axis('equal')
    plt.show()
    if filename != None:
        plt.savefig('Figures/{0}.png'.format(filename))
    

def plot_shapefiles(area,date):
    root = 'H:/sentinel2/Ghana/Validation/' + date + '/'+ area +'/Shapefiles/'
    
    deeppath = root + 'Deep_Polygon.shp'
    shallowpath = root + 'Shallow_Polygon.shp'
    farmedpath = root + 'Farmed_Polygon.shp'
    fallowpath = root + 'Fallow_Polygon.shp'
    vegpath = root + 'Veg_Polygon.shp'
    urbanpath = root + 'Urban_Polygon.shp'
    
    shapepaths = [deeppath,shallowpath,farmedpath,fallowpath,vegpath,urbanpath]
    g = gpd.GeoDataFrame()
    colors = ['dodgerblue', 'paleturquoise', 'peru', 'rosybrown', 'forestgreen']
    
    for shapepath, colorr in zip(shapepaths, colors):
        file = gpd.read_file(shapepath)
        g.plot(color = colorr)
    
    
    
    
    