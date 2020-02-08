
import pandas as pd
import matplotlib.pyplot as plt
import basemap

households_file_1 = 'Households_coord_1.xls'
households_file_2 = 'Households_coord_2.xlsx'

households_1 = pd.read_excel(households_file_1,
                            usecols=[0,1,2],
                            index_col=0,
                            nrows=287
                            )

households_2 = pd.read_excel(households_file_2,
                            usecols=[0,1,2],
                            index_col=0,
                            )

households = pd.concat([households_1,households_2], ignore_index=True)


fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution=None,
            width=8E6, height=8E6, 
            lat_0=45, lon_0=-100,)
m.etopo(scale=0.5, alpha=0.5)
