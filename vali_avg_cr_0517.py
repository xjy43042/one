# -*- coding: utf-8 -*-

# prepare corresponding CRdata of station to make a validation

import xarray as xr
import os
import pandas as pd
import numpy as np
import time
from joblib import Parallel, delayed, parallel_backend


def find_interval(number):
    lower_bound = int(number // 5) * 5
    upper_bound = lower_bound + 5
    return lower_bound, upper_bound

def process_site_data(site_data):

    slat, slon, spft_id = site_data

    lon_st, lon_en = find_interval(float(slon))
    lat_st, lat_en = find_interval(float(slat))
    sat_names = f'RG_{lat_en}_{lon_st}_{lat_st}_{lon_en}_CR.nc'
    file_path = os.path.join(target_directory, sat_names)

    # 初始化
    cr_all = np.array([])
    gridlat_near = np.array([])
    gridlon_near = np.array([])

    if os.path.exists(file_path) and 1 <= spft_id <= 8: # 并且spft_id不为0，有14个为nan可能是前面数据处理的问题
        with xr.open_dataset(file_path,engine='netcdf4') as ncfile:
                lat_near = ncfile['lat'].sel(lat=slat, method='nearest').values
                lon_near = ncfile['lon'].sel(lon=slon, method='nearest').values
                #---1st resolution 
                # cr_nc = ncfile['CD'].sel(lat=slat, lon=slon, method='nearest').values[spft_id-1].squeeze() # ndarray但没有形状 3.735707242862109
                #---2nd resolution 
                cr_nc = ncfile['CR'].sel(lat=slat, lon=slon, method='nearest').values.squeeze() #(8,)
                cr_nc = cr_nc[spft_id-1] # 3.735707242862109 correctly!
    else:
        cr_nc = np.nan
        lat_near = np.nan
        lon_near = np.nan
     # 如果cr_nc是标量，将其转换为一维数组；否则直接使用
    # cr_all = np.append(np.atleast_1d(cr_all), np.atleast_1d(cr_nc), axis=0)
    cr_all = np.append(cr_all, cr_nc)
    gridlat_near = np.append(gridlat_near, lat_near)
    gridlon_near = np.append(gridlon_near, lon_near)

    return cr_all, gridlat_near, gridlon_near

start_time = time.time()

# crdata           = pd.read_csv('/stu01/xiangjy23/treedata/crowndepth/CrownDepthData_pft_lais_bkp.csv')
crdata = pd.read_csv('/stu01/xiangjy23/treedata/predictor/Tallo_pft_predictors_addlai_addETH.csv')
cols_to_keep = ['latitude', 'longitude', 'crown_radius_m', 'pft']
# cols_to_keep = ['latitude', 'longitude', 'cd', 'pft']

crdata = crdata[cols_to_keep]
# crdata = crdata.head(49)
sitelat = crdata['latitude'].values
sitelon = crdata['longitude'].values

pft_mapping = {
    "MNE": 1,
    "BNE": 2,
    "BND": 3,
    "TBE": 4,
    "MBE": 5,
    "TBD": 6,
    "MBD": 7,
    "BBD": 8
}

crdata['pft_id'] = crdata['pft'].map(pft_mapping)
pft_id = crdata['pft_id'].values
pft_id = pft_id.astype(int)  # 转换成整型
target_directory = "/stu01/xiangjy23/treedata/make_CrownRadius/CR_CD_5x5"
# target_directory = "/stu01/xiangjy23/treedata/make_CrownRadius/China"
file_names = os.listdir(target_directory)

with parallel_backend("multiprocessing"):
    results = Parallel(n_jobs=10)(delayed(process_site_data)(site_data) for site_data in zip(sitelat, sitelon, pft_id))


cr_all, gridlat_near, gridlon_near = map(np.array, zip(*results))

# 将结果添加到crdata
crdata['cr_nc'] = cr_all
crdata['lat_nc'] = gridlat_near
crdata['lon_nc'] = gridlon_near

crdata.to_csv('/stu01/xiangjy23/treedata/make_CrownRadius/vali/crdata_vali_0517.csv', index=False)

end_time = time.time()
total_execution_time = (end_time - start_time) / 60.

print("Total stations processed: ", len(sitelat))
print("Total execution time: {:.3f} minutes".format(total_execution_time))
