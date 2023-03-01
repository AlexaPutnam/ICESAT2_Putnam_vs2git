#!/usr/bin/env python
"""
Created on Fri Nov 19 13:38:47 2021

@author: alexaputnam
"""
import numpy as np
import datetime

import pyTMD.time
import pyTMD.model
from pyTMD.calc_delta_time import calc_delta_time
from pyTMD.infer_minor_corrections import infer_minor_corrections
from pyTMD.predict_tidal_ts import predict_tidal_ts
from pyTMD.read_FES_model import extract_FES_constants






############ Eduard's functions
def gps2utc(gps_time):
    '''
    Converts GPS time that ICESat-2 references to UTC
    '''
    t0 = datetime.datetime(1980,1,6,0,0,0,0)
    leap_seconds = -18 #applicable to everything after 2017-01-01, UTC is currently 18 s behind GPS
    dt = (gps_time + leap_seconds) * datetime.timedelta(seconds=1)
    utc_time = t0+dt
    utc_time_str = np.asarray([str(x) for x in utc_time])
    return utc_time_str

def ocean_tide_replacement(lon,lat,utc_time,model_dir='/Users/alexaputnam/ICESat2/'):
    '''
    #Given a set of lon,lat and utc time, computes FES2014 tidal elevations
    dsot = Dataset('/Users/alexaputnam/ICESat2/fes2014/ocean_tide/2n2.nc')

    import h5py
    f12 = '/Users/alexaputnam/ICESat2/atlcu_v_atl12/ATL12_20220314200251_12611401_005_01.h5'
    d12 = h5py.File(f12, 'r')
    ibms = 'gt1l'
    time_gps = d12['/'+ibms+'/ssh_segments/delta_time'][:]+d12['/ancillary_data/atlas_sdp_gps_epoch'] # mean time of surface photons in segment
    time_utc2 = gps2utc(time_gps) # datetime.datetime.strptime(time_utc2[0], '%Y-%m-%d %H:%M:%S.%f')
    lat_12 = d12['/'+ibms+'/ssh_segments/latitude'][:] # mean lat of surface photons in segment
    lon_12 = d12['/'+ibms+'/ssh_segments/longitude'][:]
    lon,lat,utc_time,model_dir=lon_12[:2],lat_12[:2],time_utc2[:2],'/Users/alexaputnam/ICESat2/'
    '''
    delta_file = pyTMD.utilities.get_data_path(['data','merged_deltat.data'])
    model = pyTMD.model(model_dir,format='netcdf',compressed=False).elevation('FES2014')
    constituents = model.constituents
    time_datetime = np.asarray(list(map(datetime.datetime.fromisoformat,utc_time)))
    unique_date_list = np.unique([a.date() for a in time_datetime])
    tide_heights = np.empty(len(lon),dtype=np.float32)
    for unique_date in unique_date_list: #i.e. 2022-03-14
        print(unique_date)
        idx_unique_date = np.asarray([a.date() == unique_date for a in time_datetime])
        time_unique_date = time_datetime[idx_unique_date]
        lon_unique_date = lon[idx_unique_date]
        lat_unique_date = lat[idx_unique_date]
        YMD = time_unique_date[0].date()
        unique_seconds = np.unique(np.asarray([a.hour*3600+a.minute*60+a.second for a in time_unique_date]))
        seconds = np.arange(np.min(unique_seconds),np.max(unique_seconds)+2)
        seconds_since_midnight = [a.hour*3600 + a.minute*60 + a.second + a.microsecond/1000000 for a in time_unique_date]
        idx_time = np.asarray([np.argmin(abs(t - seconds)) for t in seconds_since_midnight])
        tide_time = pyTMD.time.convert_calendar_dates(YMD.year,YMD.month,YMD.day,second=seconds)
        ## extract_FES_constants takes a lot of time! TYPE=model.type
        amp,ph = extract_FES_constants(np.atleast_1d(lon_unique_date),
                np.atleast_1d(lat_unique_date), model.model_file, TYPE=model.type,
                VERSION=model.version, METHOD='bilinear', EXTRAPOLATE=False,
                SCALE=model.scale, GZIP=model.compressed)
        DELTAT = calc_delta_time(delta_file, tide_time)
        cph = -1j*ph*np.pi/180.0
        hc = amp*np.exp(cph)
        tmp_tide_heights = np.empty(len(lon_unique_date))
        for i in range(len(lon_unique_date)):
            if np.any(amp[i].mask) == True:
                tmp_tide_heights[i] = np.nan
            else:
                TIDE = predict_tidal_ts(np.atleast_1d(tide_time[idx_time[i]]),np.ma.array(data=[hc.data[i]],mask=[hc.mask[i]]),constituents,deltat=DELTAT[idx_time[i]],corrections=model.format)
                MINOR = infer_minor_corrections(np.atleast_1d(tide_time[idx_time[i]]),np.ma.array(data=[hc.data[i]],mask=[hc.mask[i]]),constituents,deltat=DELTAT[idx_time[i]],corrections=model.format)
                TIDE.data[:] += MINOR.data[:]
                tmp_tide_heights[i] = TIDE.data
        tide_heights[idx_unique_date] = tmp_tide_heights
    return tide_heights,DELTAT




