# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:10:27 2020

@author: remot
"""
import Script.Image as Image
import Script.Validation as Validation
import pandas as pd

# 07/01/2020
filepath1 = "../Sentinel-2/2020_01_07/L2A_07012020_60.tif"
water_path = "../Sentinel-2/2020_01_07/Shapefiles/Water_Polygon.shp"
ground_path = "../Sentinel-2/2020_01_07/Shapefiles/Earth_Polygon.shp"
val_07012020 = Validation.make_dataset(water_path,ground_path,filepath1)

filepath2 = "../Sentinel-2/2020_01_27/L2A_27012020_60.tif"
water_path = "../Sentinel-2/2020_01_27/Shapefiles/Water_Polygon.shp"
ground_path = "../Sentinel-2/2020_01_27/Shapefiles/Earth_Polygon.shp"
val_27012020 = Validation.make_dataset(water_path,ground_path,filepath2)

filepath3 = "../Sentinel-2/2020_01_02/L2A_02012020_60.tif"
water_path = "../Sentinel-2/2020_01_02/Shapefiles/Water_Polygon.shp"
ground_path = "../Sentinel-2/2020_01_02/Shapefiles/Earth_Polygon.shp"
val_02012020 = Validation.make_dataset(water_path,ground_path,filepath3)

training = pd.concat([val_02012020['df'],val_27012020['df']])
training.to_csv(r'CSV/Training.csv')

img_07012020 = Image.make_dataset(filepath1)