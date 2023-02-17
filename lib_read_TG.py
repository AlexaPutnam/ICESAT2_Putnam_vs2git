#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:15:05 2022

@author: alexaputnam
"""


import numpy as np
#import pandas as pd
from matplotlib import pyplot as plt
from netCDF4 import Dataset
from datetime import date, timedelta, datetime,timezone
import sys

import pyTMD.time
import pyTMD.model
from pyTMD.calc_delta_time import calc_delta_time
from pyTMD.infer_minor_corrections import infer_minor_corrections
from pyTMD.predict_tidal_ts import predict_tidal_ts
from pyTMD.read_FES_model import extract_FES_constants

#import matplotlib as mpl
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#from scipy.stats import kurtosis

import lib_regions as lreg
sys.path.append("/Users/alexaputnam/necessary_functions/")
import plt_bilinear as pbil

LOCDIR = '/Users/alexaputnam/ICESat2/'

'''

The station datum, or zero reference level for a tide gauge time series, 
is defined by its relationship to fixed, land-based benchmarks. Information 
about the benchmarks and historic geodetic field measurements amongst the 
benchmarks and the station datum are available upon request for stations 
within the UHSLC network. For other stations, one must contact the data 
originators. In some cases, the station datum can be a chart datum or 
another specific datum, such as a national geodetic datum, depending on 
how a given agency defines the tide gauge series reference zero. When 
available, information is provided in the station information file 
(metadata) on how to convert the station datum to other datums 
(chart, national, etc.) or benchmarks.
'''
def ymdhms2utc(ymdhms):
    N = np.shape(ymdhms)[0]
    utcT = []#np.empty(N)*np.nan
    for ii in np.arange(N):
        temp_date = datetime(int(ymdhms[ii,0]), int(ymdhms[ii,1]), int(ymdhms[ii,2]), int(ymdhms[ii,3]), int(ymdhms[ii,4]), int(ymdhms[ii,5]))
        utcT.append(str(temp_date.astimezone(timezone.utc)))
    return utcT

def ocean_tide_replacement(lon,lat,utc_time):
    '''
    # https://github.com/EduardHeijkoop/ICESat-2/blob/main/ocean_utils.py
    #Given a set of lon,lat and utc time, computes FES2014 tidal elevations
    '''
    model_dir = '/Users/alexaputnam/External_models/'
    delta_file = pyTMD.utilities.get_data_path(['data','merged_deltat.data'])
    model = pyTMD.model(model_dir,format='netcdf',compressed=False).elevation('FES2014')
    constituents = model.constituents
    time_datetime = np.asarray(list(map(datetime.fromisoformat,utc_time)))
    unique_date_list = np.unique([a.date() for a in time_datetime])
    tide_heights = np.empty(len(lon),dtype=np.float32)
    for unique_date in unique_date_list:
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
        amp,ph = extract_FES_constants(np.atleast_1d(lon_unique_date),
                np.atleast_1d(lat_unique_date), model.model_file, TYPE=model.type,
                VERSION=model.version, METHOD='spline', EXTRAPOLATE=False,
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
    return tide_heights


def lon180_to_lon360(lon_old):
    # Purpose: convert longitude from lon_old (-180 to 180) to lon_new (0 to 360)
    igt = np.where(lon_old<0)[0]
    if np.size(igt)!=0:
        lon_new = np.mod(lon_old,360.)
    else:
        lon_new = np.copy(lon_old)
    return lon_new

def pull_DTU21MSS(lat,lon_old):
    # lat,lon_old = np.arange(-10,10),np.arange(-10,10)
    lon=lon180_to_lon360(lon_old)
    ds = Dataset('/Users/alexaputnam/External_models/DTU21MSS_1min.mss.nc')
    lat_grid = ds['lat'][:]
    lon_grid = ds['lon'][:] #0-360
    mss_grid = ds['mss'][:]
    
    mlon,mlat = np.meshgrid(lon_grid,lat_grid)
    points = np.array( (mlon.flatten(), mlat.flatten()) ).T
    values = mss_grid.flatten()
    plt.figure()
    plt.scatter(points[:,0],points[:,1],c=values,cmap='viridis')
    igeo = np.where((points[:,1]>=np.nanmin(lat)-1)&(points[:,1]<=np.nanmax(lat)+1)&(points[:,0]>=np.nanmin(lon)-1)&(points[:,0]<=np.nanmax(lon)+1))[0]
    plt.scatter
    from scipy.interpolate import griddata
    mss = griddata( points, values, (lon,lat) )
    
    '''
    from scipy import interpolate
    mlon,mlat = np.meshgrid(lon_grid,lat_grid)
    f_mss = interpolate.interp2d(mlon.flatten(), mlat.flatten(), mss_grid, kind='linear')
    mss = np.empty(N)*np.nan
    N = np.size(lat)
    for ii in np.arange(N):
        mss[ii] = f_mss(lat[ii],lon[ii])
    '''
    return mss

def tide_days_1800_to_TS(days_since_1800):
    days_1800 = datetime(1800,1,1,0,0,0)
    N = np.size(days_since_1800)
    ts =  []
    ymdhms = np.empty((N,6))*np.nan
    yrfrac = np.empty(N)*np.nan
    for ii in np.arange(N):
        delta = timedelta(days_since_1800[ii]) 
        timstmp = days_1800 + delta
        ts.append(timstmp)
        ymdhms[ii,0],ymdhms[ii,1],ymdhms[ii,2] = timstmp.year,timstmp.month,timstmp.day
        ymdhms[ii,3],ymdhms[ii,4],ymdhms[ii,5] = timstmp.hour,timstmp.minute,timstmp.second
        d0 = date(timstmp.year-1, 12, 31)
        d1 = date(timstmp.year, timstmp.month, timstmp.day)
        delta = (d1 - d0).total_seconds()
        dt = timedelta(days=0, hours=int(ymdhms[ii,3]), minutes=0, seconds=53).total_seconds()
        delta_fraction = ((delta+dt)/(24.*60.*60.))/365.25
        yrfrac[ii] = timstmp.year+delta_fraction
    return ts,ymdhms,yrfrac

def tide_days_1985_to_TS(days_since_1985):
    days_1985 = datetime(1985,1,1,0,0,0)
    N = np.size(days_since_1985)
    ts =  []
    ymdhms = np.empty((N,6))*np.nan
    yrfrac = np.empty(N)*np.nan
    for ii in np.arange(N):
        if ~np.isnan(days_since_1985[ii]):
            delta = timedelta(days_since_1985[ii]) 
            timstmp = days_1985 + delta
            ts.append(timstmp)
            ymdhms[ii,0],ymdhms[ii,1],ymdhms[ii,2] = timstmp.year,timstmp.month,timstmp.day
            ymdhms[ii,3],ymdhms[ii,4],ymdhms[ii,5] = timstmp.hour,timstmp.minute,timstmp.second
            d0 = date(timstmp.year-1, 12, 31)
            d1 = date(timstmp.year, timstmp.month, timstmp.day)
            delta = (d1 - d0).total_seconds()
            dt = timedelta(days=0, hours=int(ymdhms[ii,3]), minutes=0, seconds=53).total_seconds()
            delta_fraction = ((delta+dt)/(24.*60.*60.))/365.25
            yrfrac[ii] = timstmp.year+delta_fraction
        else:
            ts.append(np.nan)
            ymdhms[ii,0],ymdhms[ii,1],ymdhms[ii,2] = np.nan,np.nan,np.nan
            ymdhms[ii,3],ymdhms[ii,4],ymdhms[ii,5] = np.nan,np.nan,np.nan
            yrfrac[ii] = np.nan
    return ts,ymdhms,yrfrac

def tg_2_is2(lltg,yrfrac_tg,lat_is2,lon_is2,yrfrac_is2,dl=0.1,dh=1.0):
    # lltg,yrfrac_tg,lat_is2,lon_is2,yrfrac_is2,dl,dh=llc2,yrfrac_c2,lat_b,lon_b,yrfrac_b,0.1,1
    N = np.shape(yrfrac_tg)[0]
    ddoy = dh/(364.25*24.0)
    if N==1:
        lat_tg,lon_tg = lltg[0],lltg[1]
        iis2ii = np.where((lat_is2>=lat_tg-dl)&(lat_is2<=lat_tg+dl)&(lon_is2>=lon_tg-dl)&(lon_is2<=lon_tg+dl))[0]
        if np.size(iis2ii)>0:
            Nti = np.size(yrfrac_tg)
            itgii = []
            for jj in np.arange(Nti):
                dt = yrfrac_is2[iis2ii]-yrfrac_tg[ii,jj]
                itgi = np.where(abs(dt)<=ddoy)[0] #4 hours = 0.0004563
                if np.size(itgi)>0:
                    itgii.append(jj)
            if np.size(itgii)>0:
                iis2 = np.copy(iis2ii)
                itg = np.copy(itgii)
    else:
        iis2 = []
        itg = np.empty(np.shape(yrfrac_tg))*np.nan
        for ii in np.arange(N):
            lat_tg,lon_tg = lltg[ii][0],lltg[ii][1]
            iis2ii = np.where((lat_is2>=lat_tg-dl)&(lat_is2<=lat_tg+dl)&(lon_is2>=lon_tg-dl)&(lon_is2<=lon_tg+dl))[0]
            if np.size(iis2ii)>0:
                if np.size(np.shape(yrfrac_tg))==2:
                    Nti = np.size(yrfrac_tg[ii,:])
                    itgii = []
                    for jj in np.arange(Nti):
                        dt = yrfrac_is2[iis2ii]-yrfrac_tg[ii,jj]
                        itgi = np.where(abs(dt)<=ddoy)[0] #4 hours = 0.0004563
                        if np.size(itgi)>0:
                            itgii.append(jj)
                    if np.size(itgii)>0:
                        iis2 = np.hstack((iis2,iis2ii))
                        Ntg = np.size(itgii)
                        itg[ii,:Ntg]=np.copy(itgii)
                else:
                    iis2 = np.hstack((iis2,iis2ii))
                    dt = yrfrac_is2[iis2ii]-yrfrac_tg[ii]
                    itgi = np.where(abs(dt)<=ddoy)[0] #4 hours = 0.0004563
                    if np.size(itgi)>0:
                        itg[ii]=ii
    if np.size(iis2)>0:
        print(np.shape(iis2))
        print(iis2)
        iis2 = np.unique(np.asarray(iis2)).astype(int)
    return iis2,itg



def pull_TG_noaa_sl(FN):
    # https://tidesandcurrents.noaa.gov/waterlevels.html?id=8452660&units=metric&bdate=20200101&edate=20201231&timezone=GMT&datum=MSL&interval=h&action=data
    # FN = '2020_8452660_met_newport_ri.csv'
    import csv
    results = []
    with open(LOCDIR+'tide_gauge/'+FN) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            results.append(row)
    Nres = np.shape(results)[0]
    ymdhms = np.empty((Nres,6))*np.nan
    yrfrac = np.empty(Nres)*np.nan
    msl = np.empty(Nres)*np.nan
    for ii in np.arange(1,Nres):
        ymdhms[ii,0],ymdhms[ii,1],ymdhms[ii,2]=int(results[ii][0][:4]),int(results[ii][0][5:7]),int(results[ii][0][8:])
        ymdhms[ii,3],ymdhms[ii,4]=int(results[ii][1][:2]),int(results[ii][1][3:])
        d0 = date(int(ymdhms[ii,0])-1, 12, 31)
        d1 = date(int(ymdhms[ii,0]), int(ymdhms[ii,1]), int(ymdhms[ii,2]))
        delta = (d1 - d0).total_seconds()
        dt = timedelta(days=0, hours=int(ymdhms[ii,3]), minutes=0, seconds=53).total_seconds()
        delta_fraction = ((delta+dt)/(24.*60.*60.))/365.25
        yrfrac[ii] = ymdhms[ii,0]+delta_fraction
        msl[ii] = results[ii][4]
    return msl,yrfrac,ymdhms

def read_TG_noaa_sl(FN):
    # https://tidesandcurrents.noaa.gov/waterlevels.html?id=8452660&units=metric&bdate=20200101&edate=20201231&timezone=GMT&datum=MSL&interval=h&action=data
    # FN = ['2020_8452660_met_newport_ri.csv']
    # msl,ymdhms,yrfrac=read_TG_noaa_sl(FN)
    Nfn = np.shape(FN)[0]
    if Nfn>1:
        countFN=np.zeros(Nfn)
        for jj in np.arange(Nfn):
            mslii,yrfracii,ymdhmsii=pull_TG_noaa_sl(FN[jj])
            Nl = np.shape(mslii)[0]
            countFN[jj] = Nl+1
        Nc = int(np.nanmax(countFN))
        ymdhms = np.empty((Nfn,Nc,6))*np.nan
        yrfrac = np.empty((Nfn,Nc))*np.nan
        msl = np.empty((Nfn,Nc))*np.nan
        hlt = np.empty((Nfn,Nc))*np.nan
        for jj in np.arange(Nfn):
            mslii,yrfracii,ymdhmsii=pull_TG_noaa_sl(FN[jj])
            Nl = np.shape(mslii)[0]
            ymdhms[jj,:Nl,:] = ymdhmsii
            yrfrac[jj,:Nl] = yrfracii
            msl[jj,:Nl] = mslii
    else:
        mslii,yrfracii,ymdhmsii=pull_TG_noaa_sl(FN[0])
    return msl,ymdhms,yrfrac

def read_TG_surftoday(FN):
    # https://www.surfertoday.com/tide-times
    # https://tidesandcurrents.noaa.gov/noaatideannual.html?id=8452660
    # FN = ['2020_8452660_newport_ri.txt']
    Nfn = np.shape(FN)[0]
    if Nfn>1:
        countFN=np.zeros(Nfn)
        for jj in np.arange(Nfn):
            df = open(LOCDIR+'tide_gauge/'+FN[jj], "r")
            lines = df.readlines()
            Nl = np.shape(lines)[0]
            countFN[jj] = Nl+1
        Nc = int(np.nanmax(countFN))
        ymdhms = np.empty((Nfn,Nc,6))*np.nan
        yrfrac = np.empty((Nfn,Nc))*np.nan
        msl = np.empty((Nfn,Nc))*np.nan
        hlt = np.empty((Nfn,Nc))*np.nan
        for jj in np.arange(Nfn):
            df = open(LOCDIR+'tide_gauge/'+FN[jj], "r")
            lines = df.readlines()
            Nl = np.shape(lines)[0]
            for ii in np.arange(Nl):
                line = lines[ii]
                if line[:4]==FN[jj][:4]:
                    splt = line.strip().split('\t')
                    ymdhms[jj,ii,0],ymdhms[jj,ii,1],ymdhms[jj,ii,2] = int(splt[0][:4]),int(splt[0][5:7]),int(splt[0][8:])
                    ymdhms[jj,ii,3],ymdhms[jj,ii,4] = int(splt[2][:2]),int(splt[2][3:])
                    d0 = date(int(splt[0][:4])-1, 12, 31)
                    d1 = date(int(splt[0][:4]), int(splt[0][5:7]), int(splt[0][8:]))
                    delta = (d1 - d0).total_seconds()
                    dt = timedelta(days=0, hours=int(ymdhms[jj,ii,3]), minutes=0, seconds=53).total_seconds()
                    delta_fraction = ((delta+dt)/(24.*60.*60.))/365.25
                    yrfrac[jj,ii] = int(splt[0][:4])+delta_fraction
                    msl[jj,ii] = float(splt[5])/100.
                    if splt[8]=='H':
                        hlt[jj,ii]=1.0
                    elif splt[8]=='L':
                        hlt[jj,ii]=-1.0
    else:
        FNi = FN[0]
        print(FNi)
        df = open(LOCDIR+'tide_gauge/'+FNi, "r")
        lines = df.readlines()
        Nl = np.shape(lines)[0]
        yr = FNi[:4]
        ymdhms = np.empty((Nl,6))*np.nan
        yrfrac = np.empty(Nl)*np.nan
        msl = np.empty(Nl)*np.nan
        hlt = np.empty(Nl)*np.nan
        for ii in np.arange(Nl):
            line = lines[ii]
            if line[:4]==FNi[:4]:
                splt = line.strip().split('\t')
                ymdhms[ii,0],ymdhms[ii,1],ymdhms[ii,2] = int(splt[0][:4]),int(splt[0][5:7]),int(splt[0][8:])
                ymdhms[ii,3],ymdhms[ii,4] = int(splt[2][:2]),int(splt[2][3:])
                d0 = date(int(splt[0][:4])-1, 12, 31)
                d1 = date(int(splt[0][:4]), int(splt[0][5:7]), int(splt[0][8:]))
                delta = (d1 - d0).total_seconds()
                dt = timedelta(days=0, hours=int(ymdhms[ii,3]), minutes=0, seconds=53).total_seconds()
                delta_fraction = ((delta+dt)/(24.*60.*60.))/365.25
                yrfrac[ii] = int(splt[0][:4])+delta_fraction
                msl[ii] = float(splt[5])/100.
                if splt[8]=='H':
                    hlt[ii]=1.0
                elif splt[8]=='L':
                    hlt[ii]=-1.0
    return msl,hlt,ymdhms,yrfrac
    


def lla2ecef(lat_deg,lon_deg):
    #x,y,z = lla2ecef(lat_deg,lon_deg)
    # WGS84 ellipsoid constants:
    alt = 0.
    a = 6378137. #height above WGS84 ellipsoid (m)
    e = 8.1819190842622e-2
    d2r = np.pi/180.
    N = a/np.sqrt(1.-(e**2)*(np.sin(lat_deg*d2r)**2))
    x = (N+alt)*np.cos(lat_deg*d2r)*np.cos(lon_deg*d2r)
    y = (N+alt)*np.cos(lat_deg*d2r)*np.sin(lon_deg*d2r)
    z = ((1.-e**2)*N+alt)*np.sin(lat_deg*d2r)
    return x,y,z

def lon360_to_lon180(lon_old):
    igt = np.where(lon_old>180)[0] #!!! used to be (prior to may 5 2021) lon_old>=180
    if np.size(igt)!=0:
        lon_new = np.mod((lon_old+180.),360.)-180.
    else:
        lon_new = np.copy(lon_old)
    return lon_new

def lon180_to_lon360(lon_old):
    igt = np.where(lon_old<0)[0]
    if np.size(igt)!=0:
        lon_new = np.mod(lon_old,360.)
    else:
        lon_new = np.copy(lon_old)
    return lon_new

def dist_func(xA,yA,zA,xD,yD,zD):
    # xA,yA,zA,xD,yD,zD = xr,yr,zr,x,y,z
    dist = np.sqrt((np.subtract(xA,xD)**2)+(np.subtract(yA,yD)**2)+(np.subtract(zA,zD)**2))
    return dist

def lla2dist(lat_deg,lon_deg,lat_deg0,lon_deg0):
    # xA,yA,zA,xD,yD,zD = xr,yr,zr,x,y,z
    xA,yA,zA = lla2ecef(lat_deg,lon180_to_lon360(lon_deg))
    xD,yD,zD = lla2ecef(lat_deg0,lon180_to_lon360(lon_deg0))
    dist = np.sqrt((np.subtract(xA,xD)**2)+(np.subtract(yA,yD)**2)+(np.subtract(zA,zD)**2))
    return dist


def narrow_idx(ymdhmsTG,ymdhmsIS,tsTG,tsIS,iis1):
    itg = np.where(ymdhmsTG[:,0]==ymdhmsIS[iis1,0])[0]
    Ntg = np.size(itg)
    dTg = np.empty(Ntg)*np.nan
    for ii in np.arange(Ntg):
        dtii = (tsTG[itg[ii]]-tsIS[iis1])
        dTg[ii] = (dtii.days*24.0)+(dtii.seconds/(60.0*60.0))+(dtii.microseconds/3600000000.0)
    itg_close_pre = np.where(np.abs(dTg)==np.nanmin(np.abs(dTg)))[0]
    if np.size(itg_close_pre)>1:
        print('np.size(itg_close_pre): '+str(np.size(itg_close_pre)))
        raise('Come up with something.')
    itg_close = itg[itg_close_pre][0]
    return itg_close,dTg[itg_close_pre]
    

def find_is2_rel_tg(latTG,lonTG,tsTG,ymdhmsTG,latIS,lonIS,tsIS,ymdhmsIS,dist_max=2000,hr_max=6):
    # latTG,lonTG,tsTG,ymdhmsTG,latIS,lonIS,tsIS,ymdhmsIS = lat_tg,lon_tg,ts_tg,ymdhms_tg,lat_200,lon_200,tsI,ymdhmsI
    lon360 = lon180_to_lon360(lonIS)
    x,y,z = lla2ecef(latIS,lon360)
    lon360ref = lon180_to_lon360(lonTG)
    
    xr,yr,zr = lla2ecef(latTG,lon360ref)
    dist = dist_func(xr,yr,zr,x,y,z)
    iis = np.where(np.abs(dist)<=dist_max)[0]
    #diis = np.diff(iis)
    #igt1 = np.where(np.abs(diis)>1)[0]
    itg_close = np.empty(np.size(iis))*np.nan
    dTg = np.empty(np.size(iis))*np.nan
    for ii in np.arange(np.size(iis)):
        itg_close[ii],dTg[ii] = narrow_idx(ymdhmsTG,ymdhmsIS,tsTG,tsIS,iis[ii])
    return iis,itg_close.astype(int),dTg

def select_region(lat_deg1,lon_deg1,lat_deg2,lon_deg2,dlat=1,dlon=1): #dlat=0.0625,dlon=0.0625,dlat=0.4,dlon=0.4
    '''
    #lat_deg1,lon_deg1,lat_deg2,lon_deg2=lat_tg,lon_tg,lat_200,lon_200
    idx = np.where((latIS>=latTG-dlat)&(latIS<=latTG+dlat)&(lonIS>=lonTG-dlon)&(lonIS<=lonTG+dlon))[0]
    inc_deg = 0.02
    while np.size(idx)<10:
        dlat+=inc_deg
        dlon+=inc_deg
        idx = np.where((latIS>=latTG-dlat)&(latIS<=latTG+dlat)&(lonIS>=lonTG-dlon)&(lonIS<=lonTG+dlon))[0]
    print('dlat = '+str(dlat)+', dlon = '+str(dlon))
    '''
    x1,y1,z1 = lla2ecef(lat_deg1,lon180_to_lon360(lon_deg1))
    x2,y2,z2 = lla2ecef(lat_deg2,lon180_to_lon360(lon_deg2))
    dist = dist_func(x1,y1,z1,x2,y2,z2)/1000.
    wgt = 1.0/((dist)**(1./3.))
    idist = np.where((dist)>100)[0]
    wgt[idist]=0
    return wgt,dist

def lse_swh(x,y):
    inn = np.where(~np.isnan(x+y))[0]
    N = np.shape(inn)[0]
    H = np.ones((N,2))
    H[:,1] = x[inn]
    ce = np.linalg.inv(H.T.dot(H)).dot(H.T.dot(y[inn]))
    cer = np.round(ce,3)
    print('SWH_adj = '+str(cer[0])+' + '+str(cer[1])+'*SWH')
    swh_adj = cer[0]+(cer[1]*x)
    return swh_adj,cer

def month_2_month_grid(ymdhms,ssha,yrs,lat,lon,IS2=True,LATLON=[],wgt=[],dm=1):
    lat_minmax=[np.nanmin(lat),np.nanmax(lat)]
    lon_minmax=[np.nanmin(lon),np.nanmax(lon)]
    inn = np.where(~np.isnan(ssha))[0]
    ymdhms,ssha = ymdhms[inn,:],ssha[inn]
    mnths = np.arange(1,13,dm)
    if IS2 ==True:
        minnum = 1#1000
    else:
        minnum = 1
    kk = 0
    time_grid=[]
    for yy in np.arange(np.size(yrs)):
        for mm in np.arange(np.size(mnths)):
            d0 = date(yrs[yy]-1, 12, 31)
            d1 = date(yrs[yy], mnths[mm], 15)
            delta = d1 - d0
            delta_fraction = delta.days/365.25
            idx = np.where((ymdhms[:,0]==yrs[yy])&(ymdhms[:,1]>=mnths[mm])&(ymdhms[:,1]<mnths[mm]+dm))[0]
            if np.size(idx)>minnum:
                if kk==0:
                    time_grid.append(yrs[yy]+delta_fraction)
                    lat_grid,lon_grid,ssha_grid,ssha_grid_var = gridded(ssha[idx],lat[idx],lon[idx],lat_minmax=lat_minmax,lon_minmax=lon_minmax)
                    #print('shape grid kk: '+str(np.shape(ssha_grid)))
                else:
                    time_grid.append(yrs[yy]+delta_fraction)
                    lat_grid_kk,lon_grid_kk,ssha_grid_kk,ssha_grid_kk_var = gridded(ssha[idx],lat[idx],lon[idx],lat_minmax=lat_minmax,lon_minmax=lon_minmax)
                    #print('shape grid kk: '+str(np.shape(ssha_grid_kk)))
                    ssha_grid = np.dstack((ssha_grid,ssha_grid_kk))
                kk+=1
    return lat_grid,lon_grid,np.asarray(time_grid),ssha_grid

def month_2_month(ymdhms,ssha,yrs,IS2=True,LATLON=[],wgt=[],d100=False):
    # ymdhms,ssha,yrs,IS2,LATLON,wgt=ymdhms_tg2,sl_tg2,yrs_mm,False,[],[]
    inn = np.where(~np.isnan(ssha))[0]
    ymdhms,ssha = ymdhms[inn,:],ssha[inn]
    dm=1
    mnths = np.arange(1,12+dm,dm)
    mean_ssha = []
    var_ssha = []
    mean_time = []
    if IS2 ==True:
        minnum = 10#1000
    else:
        minnum = 10
    for yy in np.arange(np.size(yrs)):
        for mm in np.arange(np.size(mnths)):
            d0 = date(yrs[yy]-1, 12, 31)
            if dm>1 and mnths[mm]>1:
                dmnth = int(mnths[mm]-(dm/2))
            else:
                dmnth = mnths[mm]
            d1 = date(yrs[yy], dmnth, 15)
            delta = d1 - d0
            delta_fraction = delta.days/365.25
            idx = np.where((ymdhms[:,0]==yrs[yy])&(ymdhms[:,1]>=mnths[mm])&(ymdhms[:,1]<mnths[mm]+dm))[0]
            if np.size(idx)>minnum:
                if np.size(wgt)==0:
                    if d100==False:
                        mean_ssha.append(np.nanmean(ssha[idx]))
                        var_ssha.append(np.nanvar(ssha[idx]))
                    elif d100==True:
                        iarg = np.argsort(ssha[idx])[::-1]
                        Narg = int(np.size(idx)/(3.0))
                        mean_ssha.append(np.nanmean(ssha[idx[:Narg+1]]))
                        var_ssha.append(np.nanvar(ssha[idx]))
                else:
                    mean_ssha.append(np.nansum(ssha[idx]*wgt[idx])/np.nansum(wgt[idx]))
                    var_ssha.append(np.nanvar(ssha[idx]))
                mean_time.append(yrs[yy]+delta_fraction)#np.nanmean(t[idx]))
                if np.size(LATLON)!=0:
                    pbil.groundtracks_multi(LATLON[idx,0],LATLON[idx,1],ssha[idx]*100.,'year/month = '+str(yy)+'/'+str(mm),'ssha [cm]',
                                       cm='RdYlGn_r',vmin=-50,vmax=50,FN=[],proj=180.,fc='0.1')
            else:
                mean_ssha.append(np.nan)
                var_ssha.append(np.nan)
                mean_time.append(np.nan)
    mean_ssha,var_ssha,mean_time = np.array(mean_ssha),np.array(var_ssha),np.array(mean_time)
    return mean_ssha,var_ssha,mean_time
                
def hist_cdf(x,bins=30):
    # histogram
    count, bins_count = np.histogram(x, bins=bins)
    dbin = np.diff(bins_count)[0]
    binz = bins_count[:-1]+(dbin/2.0)
    # pdf
    pdf = count / sum(count)
    # cdf
    cdf = np.cumsum(pdf)*100
    return binz,count,pdf,cdf


def hist_stats_old(t,x,bins=30):
    ce = np.polyfit(t, x, 1)
    fit = ce[1]+(ce[0]*t)
    anom = fit-x
    sd,mn = np.nanstd(anom),np.nanmean(anom)
    binz,count,pdf,cdf = hist_cdf(anom,bins=bins)
    # fit slope
    buff = 15
    ilin = np.where((cdf>=buff)&(cdf<=100-buff))[0]
    ce = np.polyfit(binz[ilin], cdf[ilin], 1)
    fit = ce[1]+(ce[0]*binz)
    # find min/max
    Z=1.5
    xmin = mn-(Z*sd) #-ce[1]/ce[0]
    xmax = mn+(Z*sd) #(100-ce[1])/ce[0]
    # figure
    '''
    plt.figure()#(figsize=(4,6))    
    plt.plot(binz,cdf)
    plt.plot(binz,fit)
    plt.grid()
    plt.ylim(cdf.min(),cdf.max())
    plt.axhline(y=50,color='black')
    plt.axvline(x=xmin,color='black')
    plt.axvline(x=xmax,color='black')
    '''
    return binz,count,pdf,cdf,xmin,xmax    

def atl03_regional(t,x,Z=1.5):
    ce = np.polyfit(t, x, 1)
    fit = ce[1]+(ce[0]*t)
    anom = fit-x
    sd,mn = np.nanstd(anom),np.nanmean(anom)
    Z=1.5
    buff = 0.2
    xmin = mn-(Z*sd) #-ce[1]/ce[0]
    xmax = mn+(Z*sd) #(100-ce[1])/ce[0]
    idx_valid = np.where((anom>=xmin)&(anom<=xmax))[0]
    plt.figure()
    binz = np.arange(-1,1.2,0.2)
    plt.hist(x,bins=binz,label='unfiltered')
    plt.hist(x[idx_valid],bins=binz,alpha=0.5,label='filtered')
    plt.grid()
    plt.title('Regional filter')
    plt.legend()
    return idx_valid,xmin,xmax

def gridded(ssha,lat,lon,lat_minmax=[],lon_minmax=[],dl=0.2): #dl = 0.1#0.0625
    # lat_grid,lon_grid,ssha_grid = gridded(ssha,lat,lon,lat_minmax=[],lon_minmax=[])
    #lon,lat,ssha=lon_200,lat_200,ssha_200
    from scipy.interpolate import griddata
    if np.size(lat_minmax)==0:
        print('DEFINE GRID BOUNDARIES')
        lat_minmax=[np.nanmin(lat),np.nanmax(lat)]
        lon_minmax=[np.nanmin(lon),np.nanmax(lon)]
    
        if lat.max()-lat.min()<=dl or lon.max()-lon.min()<=dl:
            mindl = np.asarray([lat.max()-lat.min(),lon.max()-lon.min()])
            dl = mindl.min()/3.0
            print('dl = '+str(dl))
    lat_grid = np.arange(lat_minmax[0],lat_minmax[1]+dl,dl) 
    lon_grid = np.arange(lon_minmax[0],lon_minmax[1]+dl,dl)
    mlon,mlat = np.meshgrid(lon_grid,lat_grid)
    ssha_grid = np.empty(np.shape(mlon))*np.nan
    ssha_grid_var = np.empty(np.shape(mlon))*np.nan
    print('template ssha size: '+str(np.shape(ssha_grid)))
    for xx in np.arange(np.size(lon_grid)):
        for  yy in np.arange(np.size(lat_grid)):
            ixy = np.where((lon>=lon_grid[xx]-dl/2)&(lon<lon_grid[xx]+dl/2)&(lat>=lat_grid[yy]-dl/2)&(lat<lat_grid[yy]+dl/2))[0]
            if np.size(ixy)>=1:#10:
                ssha_grid[yy,xx]=np.nanmean(ssha[ixy])
                ssha_grid_var[yy,xx]=np.nanvar(ssha[ixy])
    #points = np.vstack((lon,lat)).T
    #ssha_grid = griddata(points, ssha, (mlon, mlat), method='linear')
    return mlat,mlon,ssha_grid,ssha_grid_var

def  pull_hawaii_tg(FN):  
    N = np.size(FN)
    count = np.empty(N)*np.nan
    uhslc_id = np.empty(N)*np.nan
    lat_tg = np.empty(N)*np.nan
    lon_tg = np.empty(N)*np.nan
    
    for ii in np.arange(N):
        ds = Dataset(LOCDIR+'tide_gauge/'+FN[ii])
        sl_tg = ds['sea_level'][:].data.squeeze()/1000.
        isl_tg = np.where(np.abs(sl_tg)<20)[0]
        lat_tg[ii] = ds['lat'][:].data[0]
        lon_tg[ii] = lon360_to_lon180(ds['lon'][:].data[0])
        uhslc_id[ii] = ds['uhslc_id'][:].data[0]
        count[ii]=np.size(isl_tg)
    count = count.astype(int)
    if N>1:
        sl_tg = np.empty((np.nanmax(count),N))*np.nan
        days_since_1985_tg = np.empty((np.nanmax(count),N))*np.nan
        ymdhms_tg = np.empty((np.nanmax(count),N,6))*np.nan
        yrfrac_tg = np.empty((np.nanmax(count),N))*np.nan
        for ii in np.arange(N):
            ds = Dataset(LOCDIR+'tide_gauge/'+FN[ii])
            sl_tgi = ds['sea_level'][:].data.squeeze()/1000.
            isl_tg = np.where(np.abs(sl_tgi)<20)[0]
            sl_tg[:count[ii],ii] = sl_tgi[isl_tg]
            days_since_1800 = ds['time'][:].data[isl_tg]
            diff_days_1800_1985 = (datetime(1800,1,1,0,0,0)-datetime(1985,1,1,0,0,0)).days
            days_since_1985_tg[:count[ii],ii] = days_since_1800-np.abs(diff_days_1800_1985)
            ts_tg,ymdhms_tg[:count[ii],ii,:],yrfrac_tg[:count[ii],ii] = tide_days_1800_to_TS(days_since_1800)
    else:
        sl_tgi = ds['sea_level'][:].data.squeeze()/1000.
        isl_tg = np.where(np.abs(sl_tgi)<20)[0]
        sl_tg = sl_tgi[isl_tg]
        days_since_1800 = ds['time'][:].data[isl_tg]
        diff_days_1800_1985 = (datetime(1800,1,1,0,0,0)-datetime(1985,1,1,0,0,0)).days
        days_since_1985_tg = days_since_1800-np.abs(diff_days_1800_1985)
        ts_tg,ymdhms_tg,yrfrac_tg = tide_days_1800_to_TS(days_since_1800)
        #station_name = ds['station_name'][:].data
        #station_country = ds['station_country'][:].data
        #station_country_code = ds['station_country_code'][:].data
        #record_id = ds['record_id'][:].data
        #gloss_id = ds['gloss_id'][:].data
        #ssc_id = ds['ssc_id'][:].data
        '''
        if '_fd.nc' not in FNtg:
            ref_off_tg = ds['reference_offset'][:].data
            ref_code_tg = ds['reference_code'][:].data
            dec_meth_tg = ds['decimation_method'][:].data
            version = ds['version'][:].data
        else:
            last_rq_date = ds['last_rq_date'][:].data
        '''
    return sl_tg,lat_tg,lon_tg,days_since_1985_tg,ymdhms_tg,uhslc_id,yrfrac_tg

def pull_altimetry(FN,llmm=[],dll=0.25):
    ds_alt = Dataset(LOCDIR+'tide_gauge_match/'+FN)
    lat_alt = ds_alt['lat'][:]
    lon_alt = ds_alt['lon'][:]
    ssha_alt = ds_alt['sla'][:]
    swh_alt = ds_alt['swh'][:]
    time_alt = ds_alt['time'][:]
    time_ymdhms_alt = ds_alt['time_ymdhms'][:]
    days_since_1985_alt = time_alt/86400.
    tsA,ymdhmsA,yrfracA = tide_days_1985_to_TS(days_since_1985_alt)
    if np.size(llmm)!=0:
        idx = np.where((lat_alt>=llmm[0]-dll)&(lat_alt<=llmm[0]+dll)&(lon_alt>=llmm[1]-dll)&(lon_alt<=llmm[1]+dll))[0]
        ssha_alt,lat_alt,lon_alt,days_since_1985_alt,ymdhmsA,tsA,swh_alt,yrfracA = ssha_alt[idx],lat_alt[idx],lon_alt[idx],days_since_1985_alt[idx],ymdhmsA[idx],tsA[idx],swh_alt[idx],yrfracA[idx]
    return ssha_alt,lat_alt,lon_alt,days_since_1985_alt,ymdhmsA,tsA,swh_alt,yrfracA
    
def pull_icesat(FN,SEG=100,pth=LOCDIR+'tide_gauge_match/',llmm_fix=[],dll=0.1):
    ds2 = np.load(pth+FN,allow_pickle=True)#'reg_atl03_lat_41_lon_n73_newengland_2018_12_to_2019_12.npz'
    kys = list(ds2.keys())
    print(kys)
    swell_hf = []
    swell_lf = []
    swell = []
    ip_hf = []
    ip_lf = []
    ip = []
    wl = []
    wsteep = []
    OT = []
    if SEG==100:
        ATCH = ''
    elif SEG==2:
        ATCH = 'S'
    elif SEG==2000:
        ATCH = 'M'
    ssha = ds2['ssha'+ATCH]#ds2['ssha'+ATCH]# ds2['ssha'+ATCH+'_md'] ##ds2['ssha'+ATCH]# ds2['ssha'+ATCH+'_md'] #
    time = ds2['time'+ATCH]
    loni = ds2['lon'+ATCH]
    lati = ds2['lat'+ATCH]
    #ssha5k = ds2['sshaM']
    #sd = np.nanstd(ssha5k)
    #mn = np.nanmean(ssha5k)
    days_since_1985 = time/86400.
    #idx_valid,xmin_200,xmax_200=atl03_regional(days_since_1985,ssha,Z=3.0)
    zval =  2
    llmm = [40.9,41.2,-71.2,-70.6]
    if np.size(llmm_fix)!=0:
        idx = np.where((lati>=llmm_fix[0]-dll)&(lati<=llmm_fix[0]+dll)&(loni>=llmm_fix[1]-dll)&(loni<=llmm_fix[1]+dll))[0]
        if  np.size(idx)>0:
            print('area constrained')
        else:
            idx = np.arange(np.size(loni))
    else:
        idx = np.arange(np.size(loni))
    #idx_valid = np.where((lati>=llmm[0])&(lati<=llmm[1])&(loni>=llmm[2])&(loni<=llmm[3]))[0]#np.arange(np.shape(lati)[0])##np.where((ssha>=mn-(zval*sd))&(ssha<=mn+(zval*sd)))[0]
    ssha = ssha[idx]
    ssha_fft = ds2['ssha_fft'+ATCH][idx]
    time = time[idx]
    days_since_1985 = time/86400.
    lon = ds2['lon'+ATCH][idx]
    lat = ds2['lat'+ATCH][idx]
    tsI,ymdhmsI,yrfrac = tide_days_1985_to_TS(days_since_1985)
    beam = ds2['beam'+ATCH][idx]
    if ATCH=='S':
        swell_hf = ds2['swell_hf'+ATCH][idx]
        swell_lf = ds2['swell_lf'+ATCH][idx]
        swell = ds2['swell'+ATCH][idx]
        swh = ds2['swh'+ATCH][idx]
        swh66 = ds2['swh'+ATCH][idx]
        skew = ds2['skew'+ATCH][idx]
        ip_lf,ip_hf,ip = ds2['ip_lf'+ATCH][idx],ds2['ip_hf'+ATCH][idx],ds2['ip'+ATCH][idx]
        #'''
    elif ATCH=='':
        OT = ds2['ocean_tide'+ATCH][idx]
        swh = 4.0*np.sqrt(ds2['var'+ATCH][idx]) #ds2['swh_ip_lf'+ATCH]+ds2['swh_ip_hf'+ATCH]#+4.0*np.sqrt(ds2['var_fft'+ATCH]) 
        swh66 = 4.0*np.sqrt(ds2['var_fft'+ATCH])#ds2['swh_ip'+ATCH][idx]#+ds2['rng_ip'+ATCH]#+2.0*np.sqrt(ds2['var_fft'+ATCH])  #4.0*np.sqrt(ds2['var'+ATCH]) #
        skew = ds2['skew_fft'+ATCH][idx]
        #'''
    else:
        swh = 4.0*np.sqrt(ds2['var'+ATCH][idx]) #ds2['swh_ip_lf'+ATCH]+ds2['swh_ip_hf'+ATCH]#+4.0*np.sqrt(ds2['var_fft'+ATCH]) 
        swh66 = 4.0*np.sqrt(ds2['var_fft'+ATCH])#ds2['swh_ip'+ATCH][idx]#+3.0*np.sqrt(ds2['var_fft'+ATCH])  #4.0*np.sqrt(ds2['var'+ATCH]) #
        skew = ds2['skew_fft'+ATCH][idx]
        #if SEG==100:
        #    swh = 4.0*np.sqrt(ds2['var'+ATCH]) #ds2['swh_ip_lf'+ATCH]+ds2['swh_ip_hf'+ATCH]+4.0*np.sqrt(ds2['var_fft'+ATCH]) 
        #swh = 4.0*np.sqrt(ds2['var'+ATCH]) #ds2['swh'+ATCH][idx_valid]
        wl = ds2['wl'+ATCH][idx]#np.vstack((ds2['wl_lf'+ATCH],ds2['wl_hf'+ATCH]))
        wsteep = []#np.vstack((ds2['swh_ip_lf'+ATCH]/ds2['wl_lf'+ATCH],ds2['swh_ip_hf'+ATCH]/ds2['wl_hf'+ATCH]))
    N = ds2['N'+ATCH][idx]
    if SEG!=2:
        slope = ds2['slope'+ATCH][idx]
    else:
        slope = np.nan
    '''
    if np.size(llmm_fix)!=0:
        idx = np.where((lat>=llmm_fix[0]-dll)&(lat<=llmm_fix[0]+dll)&(lon>=llmm_fix[1]-dll)&(lon<=llmm_fix[1]+dll))[0]
        if  np.size(idx)>0:
            print('area constrained')
            ssha,ssha_fft,swell_hf,swell_lf,swell,lat,lon,days_since_1985=ssha[idx],ssha_fft[idx],swell_hf[idx],swell_lf[idx],swell[idx],lat[idx],lon[idx],days_since_1985[idx]
            ymdhmsI,tsI,beam,swh,swh66,N,slope,skew,yrfrac,wl,wsteep,ip_lf,ip_hf,ip,OT = ymdhmsI[idx,:],tsI[idx],beam[idx],swh[idx],swh66[idx],N[idx],slope[idx],skew[idx],yrfrac[idx],wl[idx],wsteep[idx],ip_lf[idx],ip_hf[idx],ip[idx],OT[idx]
        else:
            print('no points on region')
    '''
    return ssha,ssha_fft,swell_hf,swell_lf,swell,lat,lon,days_since_1985,ymdhmsI,tsI,beam,swh,swh66,N,slope,skew,yrfrac,wl,wsteep,ip_lf,ip_hf,ip,OT
def tg_coordinates(TG):
    if 'buzzard' in TG:
        ll = [41.74166, -70.61666]
    elif 'fallriver' in TG:
        ll = []
def file4_is2_alt_tg(REG):
    FNtg = []
    FNtg2 = []
    lltg2 = []
    FNj = []
    FNc2 = []
    FNs3 = []
    if REG=='newengland':
        # New England: 
        FNtg = ['h253_newport_ri_fd.nc']#,'h743a_nantucket_ma.nc','h742a_woodshole_ma.nc','h744a_newlondon_ct.nc','h279a_montauk_ny.nc','h743a_nantucket_ma.nc'] #['h253_newport_ri_fd.nc'] #,'h741a_boston.nc'
        ##FNtg2 = ['2020_8447270_buzzardsbay_ma.txt','2020_8447386_fallriver_ma.txt','2020_8447930_woodshole_ma.txt','2020_8448558_edgartown_ma.txt',
        ##'2020_8448725_menemshaharbor_ma.txt','2020_8449130_nantucket_ma.txt',
        ##'2020_8452660_newport_ri.txt','2020_8452944_conimicutlight_ri.txt','2020_8454000_providence_ri.txt','2020_8454049_quonsetpoint_ri.txt',
        ##'2020_8459681_blockisland_ri.txt','2020_8461490_newlondon_ct.txt','2020_8510560_montauk_ny.txt','2020_8510719_silvereel_ny.txt',
        ##'2020_8512668_mattituckinlet_ny.txt','2020_8512735_southjamesport_ny.txt']
        FNtg2 = ['2020_8447386_met_fallriver_ma.csv','2020_8447930_met_woodshole_ma.csv','2020_8449130_met_nantucket_ma.csv',
        '2020_8452660_met_newport_ri.csv','2020_8452944_met_conimicutlight_ri.csv','2020_8454000_met_providence_ri.csv','2020_8454049_met_quonsetpoint_ri.csv',
        '2020_8461490_met_newlondon_ct.csv','2020_8510560_met_montauk_ny.csv']
        lltg2 = [[41.705, -71.16333],[41.52333, -70.67166],[41.28499, -70.09666],
        [41.505, -71.32666],[41.71666, -71.345],[41.80666, -71.4],[41.58666, -71.41],
        [41.37166, -72.095],[41.04833, -71.95999]]
        FNj = 'j3_dir_newengland_2021.nc'#'j3_dir_newengland_2019.nc' 
        #FNc2 = 'c2_dir_newengland_2021.nc' #'c2_dir_newengland_2019.nc' 
        FNs3 = 's3_dir_newengland_2021.nc'
        FNis2 = 'reg_atl03_lat_41_lon_n73_newengland_segs_2_100_2000_2018_10_to_2022_10.npz'#'reg_atl03_lat_41_lon_n73_newengland_segs_2_100_2000_2020_12_to_2021_12.npz'#'reg_atl03_lat_41_lon_n73_newengland_segs_2_100_2000_2018_12_to_2019_12.npz'#'reg_atl03_lat_41_lon_n73_newengland_submaj_2018_12_to_2021_12.npz'#'reg_atl03_lat_41_lon_n73_newengland_final_2018_12_to_2020_12.npz' #'reg_atl03_lat_41_lon_n73_newengland_2018_12_to_2020_12.npz' # 'reg_atl03_lat_41_lon_n73_newengland_beamfilt_2018_12_to_2020_12.npz' #
    elif REG=='hawaii':
        # Hawaii: 
        FNtg = ['h057_honolulu_hi_fd.nc','h058_nawiliwili_hi_fd.nc','h059_kahului_fd.nc','h060_hilo_hi_fd.nc','h061_mokuoloe_hi_fd.nc','h552_kawaihae_hi_fd.nc']
        FNj = 'j3_dir_hawaii_2019_21.nc'
        FNis2 = 'reg_atl03_lat_19_lon_n160_hawaii_2018_12_to_2020_12.npz' 
    elif REG == 'antarctica':
        FNtg = ['h730_base_prat_chile_fd.nc']
        FNj = 'j3_dir_antarctica_2019_21.nc'
        FNis2 = 'reg_atl03_lat_n64_lon_n62_antarctica_final_2018_12_to_2020_12.npz'
    elif REG == 'greenland':
        FNtg = ['h299_ qaqortoq_greenland_fd.nc']
        FNj = 'j3_dir_greenland_2019_21.nc'
        FNis2 = 'reg_atl03_lat_60_lon_n48_greenland_submaj_2018_12_to_2021_12.npz'
    elif REG == 'norway':
        FNtg = ['h800_andenes_norway_fd.nc']
        FNis2 = 'reg_atl03_lat_69_lon_16_norway_submaj_2018_12_to_2021_12.npz' #'reg_atl03_lat_69_lon_16_norway_final_2018_12_to_2020_12.npz'
    elif REG == 'japan':
        FNtg = ['h362_ nagasaki_japan_fd.nc','h354_ aburatsu_japan_fd.nc','h363_ nishinoomote_japan_fd.nc','h345_ nakano_shima_japan_fd.nc']
        FNj = 'j3_dir_japan_2019_21.nc'
        FNis2 = 'reg_atl03_lat_29_lon_129_japan_submaj_2018_12_to_2021_12.npz'#'reg_atl03_lat_29_lon_129_japan_final_2018_12_to_2021_12.npz'
    elif REG == 'french_antarctic':
        FNtg = ['h180_ kerguelen_france_fd.nc']
        FNj = 'j3_dir_french_antarctic_2019_21.nc'
        FNis2 = 'reg_atl03_lat_n50_lon_70_french_antarctic_final_2018_12_to_2021_12.npz'
    elif REG == 'lagos':
        FNis2 = 'reg_atl03_lat_6_lon_3_lagos_segs_2_100_2000_2019_10_to_2022_10.npz'#'reg_atl03_lat_6_lon_3_lagos_segs_2_100_2000_2018_10_to_2019_10.npz'
        #FNs3 = 's3_dir_lagos_2018.nc'
    elif REG=='ittoqqortoormiit_winter':
        FNtg = ['h809_ittoqqortoormiit_hi_fd.nc']
        FNis2 = 'reg_atl03_lat_69_lon_n26_ittoqqortoormiit_segs_100_2000_2018_12_to_2019_03.npz'
    elif REG=='ittoqqortoormiit_summer':
        FNtg = ['h809_ittoqqortoormiit_hi_fd.nc']
        FNis2 = 'reg_atl03_lat_69_lon_n26_ittoqqortoormiit_segs_100_2000_2019_06_to_2019_09.npz'
    elif REG=='ittoqqortoormiit':
        FNtg = ['h809_ittoqqortoormiit_hi_fd.nc']
        FNis2 = 'reg_atl03_lat_69_lon_n26_ittoqqortoormiit_segs_100_2000_2018_12_to_2021_12.npz'
    elif REG=='north_atlantic':
        FNj = 'j3_dir_north_atlantic_2021.nc'#'j3_dir_north_atlantic_2020_21.nc'#'j3_dir_north_atlantic_2019.nc'
        #FNc2 = 'c2_dir_north_atlantic_2020_21.nc'#'c2_dir_north_atlantic_2019.nc'#
        FNs3 = 's3_dir_north_atlantic_2021.nc'#'s3_dir_north_atlantic_2020_21.nc'
        FNis2 = 'reg_atl03_lat_65_lon_n8_north_atlantic_segs_2_100_2000_2020_12_to_2021_12.npz'#'reg_atl03_lat_65_lon_n8_north_atlantic_segs_2_100_2000_2019_12_to_2021_12.npz'#'reg_atl03_lat_65_lon_n8_north_atlantic_segs_2_100_2000_2018_12_to_2019_12.npz'
    elif REG=='gom':
        FNj = 'j3_dir_gom_2021.nc'#'j3_dir_gom_2020.nc'
        FNs3 = 's3_dir_gom_2021.nc'
        #FNc2 = 'c2_dir_gom_2020.nc'#
        FNis2 = 'reg_atl03_lat_23_lon_n88_gom_segs_2_100_2000_2020_12_to_2021_06.npz'#'reg_atl03_lat_23_lon_n88_gom_segs_2_100_2000_2020_08_to_2020_08.npz'
    elif REG=='mumbai':
        FNs3 = 's3_dir_mumbai_2021.nc'
        FNis2 = 'reg_atl03_lat_18_lon_72_mumbai_segs_2_100_2000_2020_12_to_2021_12.npz'
    elif REG=='benghazi':
        FNj = 'j3_dir_benghazi_2021.nc'
        FNs3 = 's3_dir_benghazi_2021.nc'
        FNis2 = 'reg_atl03_lat_32_lon_19_benghazi_segs_2_100_2000_2020_12_to_2021_03.npz'
    elif REG=='hunga_tonga':
        FNtg = ['h038_nukualofa_tonga_fd.nc']
        FNj = 'j3_dir_hunga_tonga_2021.nc'
        FNs3='s3_dir_hunga_tonga_2021.nc'
        FNis2 = 'reg_atl03_lat_n22_lon_n176_hunga_tonga_segs_2_100_2000_2020_12_to_2022_05.npz'
    elif REG=='bowman_island':
        FNis2 = 'reg_atl03_lat_n66_lon_102_bowman_island_segs_2_100_2000_2020_12_to_2021_06.npz'
        FNj = 'j3_dir_bowman_island_2021.nc'
        FNs3='s3_dir_bowman_island_2021.nc'
    elif REG=='carolina':
        FNis2 = 'reg_atl03_lat_33_lon_n79_carolina_segs_2_100_2000_2021_10_to_2021_12.npz'
        FNj = 'j3_dir_carolina_2021.nc'
        FNs3='s3_dir_carolina_2021.nc'
    return FNtg,FNtg2,FNj,FNc2,FNs3,FNis2,lltg2
