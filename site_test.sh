#!/bin/bash -x

# =====================
# 抠点做验证时为什么会出现提取的CR=nan的情况，用于制作的源数据哪一个对应位置是nan
# =====================

cd /stu01/xiangjy23/treedata/make_CrownRadius/vali/extract
# cdo remapnn,lon=-1.33958316/lat=6.6895833 /tera12/yuanhua/mksrf/srf_5x5/RG_10_-5_5_0.MOD2020.nc ./lai_point1.nc
# cdo remapnn,lon=-1.33958316/lat=6.6895833 /stu01/dongwz/urban_data/urban_raw/urban_5x5/ETH_5x5/RG_10_-5_5_0.ETH.nc ./eth_point1.nc
# cdo remapnn,lon=-1.33958316/lat=6.6895833 /stu01/xiangjy23/treedata/make_CrownRadius/rawdata/bio/bio_5x5/RG_10_-5_5_0_bio.nc ./bio_point1.nc
# cdo remapnn,lon=-1.33958316/lat=6.6895833 /stu01/xiangjy23/treedata/make_CrownRadius/CR_CD_5x5/RG_10_-5_5_0_CR.nc ./cr_point1.nc
# cdo remapnn,lon=-1.33958316/lat=6.6895833 /stu01/xiangjy23/treedata/make_CrownRadius/RG_Global_CRCD.nc ./cr_point1.nc

cdo remapnn,lon=-1.31857/lat=6.70442 /stu01/xiangjy23/treedata/make_CrownRadius/RG_Global_CRCD.nc ./cr_point2.nc
cdo remapnn,lon=-52.352/lat=-14.713 /stu01/xiangjy23/treedata/make_CrownRadius/RG_Global_CRCD.nc ./cr_point3.nc
