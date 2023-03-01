#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 13:38:47 2021

@author: alexaputnam
"""

import h5py
import numpy as np
import datetime as dt
from netCDF4 import Dataset
import time
from scipy.stats import skew
from global_land_mask import globe
from matplotlib import pyplot as plt
import datetime
import time


import ocean_tide as ot


def utc2utc_stamp(time_utc):
    gps2utc2 = (datetime.datetime(1985, 1, 1,0,0,0)-datetime.datetime(1980, 1, 6,0,0,0)).total_seconds()
    gps_time = time_utc+gps2utc2+18
    t0 = datetime.datetime(1980,1,6,0,0,0,0)
    leap_seconds = -18 #applicable to everything after 2017-01-01, UTC is currently 18 s behind GPS
    dt = (gps_time + leap_seconds) * datetime.timedelta(seconds=1)
    utc_time = t0+dt
    utc_time_str = np.asarray([str(x) for x in utc_time])
    return utc_time_str

fn = '/Users/alexaputnam/ICESat2/ana_fes2014/reg_atl03_lat_41_lon_n73_newengland_segs_2_100_2000_2020_12_to_2021_03.npz'

#d3s = np.load(f3s,allow_pickle='TRUE', encoding='bytes').item()
#beams3s = d3s.keys()
d3 = np.load(fn)#np.load(fn,allow_pickle='TRUE', encoding='bytes')#np.load(fn)
ssha_fft100 = d3['ssha_fft']
time100 = d3['time']
lon100 = d3['lon']
lat100= d3['lat']
dem100= d3['dem']
ot100 = d3['ocean_tide']
time_utc = utc2utc_stamp(time100) # datetime.datetime.strptime(time_utc2[0], '%Y-%m-%d %H:%M:%S.%f')
N = np.size(lon100)
t1 = time.time()
f14_12 = ot.ocean_tide_replacement(lon100[:N],lat100[:N],time_utc[:N])
print('total time for '+str(N)+' points: '+str(np.round(time.time()-t1)/60)+' min')

fnsv = '/Users/alexaputnam/ICESat2/ana_fes2014/reg_atl03_lat_41_lon_n73_newengland_segs_2_100_2000_2020_12_to_2021_03_fes2014_heights.npz'
np.savez(fnsv,tide_heights=f14_12)
M = N
plt.figure()
plt.plot(ot100[:M])
plt.plot(f14_12[:M])

