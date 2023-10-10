#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-03-02

@author: alexaputnam
"""
import lib_read_TG as lTG

import time
import numpy as np
from matplotlib import pyplot as plt
import geopandas as gpd
import pandas as pd
import os
import numpy as np
from netCDF4 import Dataset
import xarray as xr
from glob import glob
from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator
from scipy import interpolate
import datetime as dt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.feature as ft
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import colors
from scipy.interpolate import NearestNDInterpolator as nn_interp
from scipy.ndimage import uniform_filter
import scipy.fftpack
import sys
import requests

import rasterio as rio
from rasterio.plot import show
#sys.path.append("/Users/alexaputnam/necessary_functions/")
#import plt_bilinear as pbil


def groundtracks_multi(lon1,lat1,gt1,TIT,LAB,cm='viridis',vmin=[],vmax=[],FN=[],proj=180.,lon2=[],lat2=[],gt2=[],lon3=[],lat3=[],gt3=[],s1=3,ss2 = 100,figsize=[12,9],TB=[0.9,0.3],zorderL=3,EXPAND=[]):
    # var,LAB = b_geo,'b_geo'
    #vmin=1
    #vmax=254
    ss = 200
    ss2 = 100
    cp = np.zeros(np.shape(gt1))
    fig = plt.figure(figsize=(figsize[0],figsize[1]))#plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EckertIV(central_longitude=proj))
    plt.subplots_adjust(top=TB[0],bottom=TB[1])
    plt.title(TIT,fontsize=15)
    #ax.coastlines()
    ax.outline_patch.set_visible(True)
    ax.outline_patch.set_edgecolor('0.5')
    #ax.add_feature(cfeature.OCEAN, facecolor=cfeature.COLORS['water'])
    #ax.scatter(lon_grid,lat_grid,c=cp,s=3,cmap='seismic',transform=ccrs.PlateCarree(),vmin=0,vmax=1)
    if np.size(EXPAND)!=0:
        lonEx = np.asarray([np.nanmin(lon1)-EXPAND[1],np.nanmax(lon1)+EXPAND[1]])
        latEx = np.asarray([np.nanmin(lat1)-EXPAND[0],np.nanmax(lat1)+EXPAND[0]])
        sm = ax.scatter(lonEx,latEx,c=np.empty(2)*np.nan,s=s1,cmap=cm,transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax,zorder=2)
    sm = ax.scatter(lon1,lat1,c=gt1,s=s1,cmap=cm,transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax,zorder=2)
    if np.size(lon2)!=0:
        sm2 = ax.scatter(lon2,lat2,c=gt2,s=ss2,cmap=cm,transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax,edgecolors='blue')#edgecolors='blue',linewidths=3)
    if np.size(lon3)!=0:
        sm3 = ax.scatter(lon3,lat3,c=gt3,s=ss2,cmap=cm,transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax,edgecolors='gray')
        #sm = ax.scatter(lon_fix,lat_fix,c='black',s=12,cmap=cm,transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS['land'],zorder=zorderL)#facecolor=fc,zorder=2) # facecolor='0.1' #facecolor='0.75''0.9' cfeature.COLORS['land']
    ax.add_feature(cfeature.COASTLINE)
    #cb=fig.colorbar(sm,cax=plt.axes([0.265,0.35,0.5,0.02]),spacing='proportional',orientation='horizontal')#,label=LAB).outline.remove()
    cb=fig.colorbar(sm,cax=plt.axes([0.265,0.2,0.5,0.02]),spacing='proportional',orientation='horizontal')
    cb.ax.tick_params(labelsize=15)
    cb.set_label(LAB,size=15)
    gl = ax.gridlines(draw_labels=True,color='cadetblue')
    gl.top_labels = False
    gl.right_labels = False
    if np.size(FN)==0:
        plt.show()
    else:
        fig.savefig(FN)
        plt.close()


def utc2yrfrac(utc):
    N = np.shape(utc)[0]
    utc_yr = utc/(60.*60.*24.*365.25)
    yrfrac = utc_yr+1985
    return yrfrac

def yrfrac2utc(yrfrac):
    utc_yr=yrfrac-1985
    utc = utc_yr*(60.*60.*24.*365.25)
    return utc

def convert_partial_year(number):
    # number = t_frac_moments13
    #yr,mnt,dy,hr,mi,date = convert_partial_year(number)
    from datetime import timedelta, datetime
    N = np.shape(number)[0]
    yr,mnt,dy,hr,mi = np.empty(N)*np.nan,np.empty(N)*np.nan,np.empty(N)*np.nan,np.empty(N)*np.nan,np.empty(N)*np.nan
    dtm=[]
    yf_chk = np.empty(N)*np.nan
    for ii in np.arange(N):
        year = int(np.floor(number[ii]))
        d = timedelta(days=(number[ii] - year)*365.25)#timedelta(days=(number - year)*(365.25 + is_leap(year)))
        day_one = datetime(year,1,1)
        date = d + day_one
        yr[ii] = date.year
        mnt[ii] = date.month
        dy[ii] = date.day
        hr[ii] = date.hour
        mi[ii] = date.minute
        yf_chk[ii] = date.year+((dt.datetime(date.year,date.month,date.day,date.hour,date.minute)-dt.datetime(date.year,1,1,0,0)).total_seconds()/(365.25*24*60*60))
        dtm.append(str(int(yr[ii]))+'-'+"{:02d}".format(int(mnt[ii]))+'-'+"{:02d}".format(int(dy[ii]))+'-'+"{:02d}".format(int(hr[ii]))+'-'+"{:02d}".format(int((mi[ii]))))
    return yr,mnt,dy,hr,mi,dtm

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
    dist = np.sqrt((np.subtract(xA,xD)**2)+(np.subtract(yA,yD)**2)+(np.subtract(zA,zD)**2))
    return dist

def geo2dist(lat,lon,lat0,lon0):
    # lat,lon,lat0,lon0=flati[idx],floni[idx],latTG,lonTG
    lon360 = lon180_to_lon360(lon)
    x,y,z = lla2ecef(lat,lon360)
    lon360ref = lon180_to_lon360(lon0)
    xr,yr,zr = lla2ecef(lat0,lon360ref)
    dist = dist_func(xr,yr,zr,x,y,z)
    idx = np.where((lat-lat0)<0)[0]
    if np.size(idx)>1:
        dist[idx]=dist[idx]*-1.0
    elif np.size(idx)==1:
        dist=dist*-1.0
    return dist

def get_lonlat_geometry(geom):
    '''
    Returns lon/lat of all exteriors and interiors of a Shapely geomery:
        -Polygon
        -MultiPolygon
        -GeometryCollection
    '''
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    if geom.geom_type == 'Polygon':
        lon_geom,lat_geom = get_lonlat_polygon(geom)
        lon = np.append(lon,lon_geom)
        lat = np.append(lat,lat_geom)
    elif geom.geom_type == 'MultiPolygon':
        polygon_list = [p for p in geom.geoms if p.geom_type == 'Polygon']
        for polygon in polygon_list:
            lon_geom,lat_geom = get_lonlat_polygon(polygon)
            lon = np.append(lon,lon_geom)
            lat = np.append(lat,lat_geom)
    elif geom.geom_type == 'GeometryCollection':
        polygon_list = [p for p in geom.geoms if p.geom_type == 'Polygon']
        for polygon in polygon_list:
            lon_geom,lat_geom = get_lonlat_polygon(polygon)
            lon = np.append(lon,lon_geom)
            lat = np.append(lat,lat_geom)
    return lon,lat

def get_lonlat_polygon(polygon):
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    exterior_xy = np.asarray(polygon.exterior.xy)
    lon = np.append(lon,exterior_xy[0,:])
    lon = np.append(lon,np.nan)
    lat = np.append(lat,exterior_xy[1,:])
    lat = np.append(lat,np.nan)
    for interior in polygon.interiors:
        interior_xy = np.asarray(interior.coords.xy)
        lon = np.append(lon,interior_xy[0,:])
        lon = np.append(lon,np.nan)
        lat = np.append(lat,interior_xy[1,:])
        lat = np.append(lat,np.nan)
    return lon,lat

def get_lonlat_gdf(gdf):
    '''
    Returns lon/lat of all exteriors and interiors of a GeoDataFrame.
    '''
    lon = np.empty([0,1],dtype=float)
    lat = np.empty([0,1],dtype=float)
    for geom in gdf.geometry:
        lon_geom,lat_geom = get_lonlat_geometry(geom)
        lon = np.append(lon,lon_geom)
        lat = np.append(lat,lat_geom)
    return lon,lat

def remove_outliers(x,zval=3,MAXIT=10):
    if MAXIT>1:
        idx=[1]
        ss = 0
        while np.size(idx)!=0:
            mn = np.nanmean(x)
            sd = np.nanstd(x)
            idx = np.where((x<mn-zval*sd)|(x>mn+zval*sd))[0]
            if np.size(idx)!=0:
                x[idx]=np.nan
            if ss>MAXIT:
                idx=[]
            ss+=1
    else:
        mn = np.nanmean(x)
        sd = np.nanstd(x)
        idx = np.where((x<mn-zval*sd)|(x>mn+zval*sd))[0]
        x[idx]=np.nan
    return x

def idx_outliers(x,MAX=50):
    xi = np.copy(x)
    xi[np.abs(xi)>MAX]=np.nan
    mn = np.nanmean(xi)
    sd = np.nanstd(xi)
    isig = np.where((xi<mn-3*sd)|(xi>mn+3*sd))[0]
    xi[isig]=np.nan
    idx = np.where(np.isnan(xi))[0]
    return idx


def open_sonel_file(sonel_file):
    # https://www.sonel.org/-GPS-24-.html?lang=fr
    fileS = open(sonel_file, 'r')
    LinesS = fileS.readlines()
    NlS = np.shape(LinesS)[0]
    yfr_sonel = []
    h_sonel = []
    sh_sonel = []
    lat_sonel = float(LinesS[12].split()[3])
    lon_sonel = float(LinesS[13].split()[3])
    h0_sonel = float(LinesS[14].split()[3])
    for ii in np.arange(NlS-1):
        sLine1 = LinesS[ii].split()
        if '#' not in sLine1:
            sLine2 = LinesS[ii+1].split()
            yfr_sonel.append(float(sLine1[0]))
            h_sonel.append(float(sLine2[3]))
            sh_sonel.append(float(sLine2[-1]))
    yfr_sonel = np.asarray(yfr_sonel)   
    h_sonel = np.asarray(h_sonel)   #[m]
    sh_sonel = np.asarray(sh_sonel)   #[m]
    return yfr_sonel,h_sonel,sh_sonel


def dist_point2line(latP,lonP,lat1,lon1,lat2,lon2):
    x1,y1,z1 = lla2ecef(lat1,lon1)
    x2,y2,z2 = lla2ecef(lat2,lon2)
    xp,yp,zp = lla2ecef(latP,lonP)
    dx21 = x2-x1
    dy1p = y1-yp
    dx1p = x1-xp
    dy21 = y2-y1
    num = np.abs((dx21*dy1p)-(dx1p*dy21))
    den = np.sqrt((dx21**2)+(dy21**2))
    dist = num/den
    return dist

def search_deg_bin(lat0,lon0,lat_bindist,lon_bindist):
    #lat0,lon0,dLAT,dLON = 84,356,300,300
    alat = np.arange(0.00001,0.101,0.00001)
    alon = np.arange(0.00001,0.101,0.00001)
    Na = np.shape(alat)[0]
    dist_lat = np.empty(Na)*np.nan
    dist_lon = np.empty(Na)*np.nan
    for ii in np.arange(Na):
        dist_lat[ii] = geo2dist(lat0+alat[ii],lon0,lat0,lon0)
    for ii in np.arange(Na):
        dist_lon[ii] = geo2dist(lat0,lon0+alon[ii],lat0,lon0)
    ddlat = np.abs(dist_lat-lat_bindist)
    ddlon = np.abs(dist_lon-lon_bindist)
    ilat = np.where(ddlat==np.nanmin(ddlat))[0]
    ilon = np.where(ddlon==np.nanmin(ddlon))[0]
    print('lat dist: ='+str(dist_lat[ilat]))
    print('lon dist: ='+str(dist_lon[ilon]))
    lat_bindeg = alat[ilat]
    lon_bindeg = alon[ilon]
    return lat_bindeg,lon_bindeg


def make_grid(LLMM,lat_bindist,lon_bindist):
    # lat_grid,lon_grid= make_grid(LLMM,lat_bindist,lon_bindist)
    dLAT = LLMM[1]-LLMM[0]
    dLON = LLMM[3]-LLMM[2]
    lat0 = LLMM[0]+(dLAT/2.0)
    lon0 = LLMM[2]+(dLON/2.0)
    lat_bindeg,lon_bindeg = search_deg_bin(lat0,lon0,lat_bindist,lon_bindist)
    lat_grid = np.arange(LLMM[0],LLMM[1]+lat_bindeg,lat_bindeg)
    lon_grid = np.arange(LLMM[2],LLMM[3]+lon_bindeg,lon_bindeg)
    #mlon_grid,mlat_grid = np.meshgrid(lon_grid,lat_grid)
    #flon_grid,flat_grid = mlon_grid.flatten(),mlat_grid.flatten()
    return lat_grid,lon_grid

def mean_grid(lat,lon,lat_grid,lon_grid,var):
    # med_temp,mean_temp,var_temp,occ_temp=mean_grid(lat,lon,lat_grid,lon_grid,var)
    # find bins
    occ_temp, lonbins, latbins = np.histogram2d(lon, lat, bins = [lon_grid, lat_grid])
    sum_temp, lonbins, latbins = np.histogram2d(lon, lat, bins = [lon_grid, lat_grid], weights=var)
    occ_temp = occ_temp.T
    sum_temp = sum_temp.T
    in0 = np.where(occ_temp!=0)
    # Compute mean
    mean_temp = np.ones(np.shape(occ_temp))*np.nan
    med_temp = np.ones(np.shape(occ_temp))*np.nan
    var_temp = np.ones(np.shape(occ_temp))*np.nan
    N = np.shape(in0)[1]
    #print('shape in0: '+str(np.shape(in0)))
    if N>0:
        #print('shape in0: '+str(np.shape(in0)))
        for ii in np.arange(N):
            idata = np.where((lat>=lat_grid[in0[0][ii]])&(lat<lat_grid[in0[0][ii]+1])&(lon>=lon_grid[in0[1][ii]])&(lon<lon_grid[in0[1][ii]+1]))[0]
            #print('size idata: '+str(np.size(idata)))
            if np.size(idata)>5:
                data_tmp = np.copy(var[idata])
                mn_tmp = np.nanmedian(data_tmp)
                sd_tmp = np.nanstd(data_tmp)
                idx_tmp = np.where((data_tmp>=mn_tmp-(3.0*sd_tmp))&(data_tmp<=mn_tmp+(3.0*sd_tmp)))[0]
                mean_temp[in0[0][ii],in0[1][ii]] = np.nanmean(data_tmp[idx_tmp])
                med_temp[in0[0][ii],in0[1][ii]] = np.nanmedian(data_tmp[idx_tmp])
                var_temp[in0[0][ii],in0[1][ii]] = np.nanvar(data_tmp[idx_tmp])
    return med_temp,mean_temp,var_temp,occ_temp

def groundtracks(lon_grid,lat_grid,gt,TIT,LAB,cm,vmin=[],vmax=[],FN=[],proj=180.,fc='0.1',lon_fix=[],lat_fix=[],lon_fix2=[],lat_fix2=[]):
    # var,LAB = b_geo,'b_geo'
    #vmin=1
    #vmax=254
    cp = np.zeros(np.shape(gt))
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EckertIV(central_longitude=proj))
    plt.subplots_adjust(top=0.9,bottom=0.3)
    plt.title(TIT,fontsize=15)
    #ax.coastlines()
    ax.outline_patch.set_visible(True)
    ax.outline_patch.set_edgecolor('0.5')
    ax.add_feature(cfeature.OCEAN, facecolor=cfeature.COLORS['water'])
    #ax.scatter(lon_grid,lat_grid,c=cp,s=3,cmap='seismic',transform=ccrs.PlateCarree(),vmin=0,vmax=1)
    sm = ax.scatter(lon_grid,lat_grid,c=gt,s=3,cmap=cm,transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax)
    if np.size(lon_fix)!=0:
        #sm = ax.scatter(lon_fix,lat_fix,c=gt_fix,s=30,cmap=cm,transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax)
        sm2 = ax.scatter(lon_fix,lat_fix,c='black',s=12,transform=ccrs.PlateCarree())
    if np.size(lon_fix2)!=0:
        #sm = ax.scatter(lon_fix,lat_fix,c=gt_fix,s=30,cmap=cm,transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax)
        sm2 = ax.scatter(lon_fix2,lat_fix2,c='black',s=12,transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS['land_alt1'],zorder=2)#facecolor=fc,zorder=2) # facecolor='0.1' #facecolor='0.75''0.9' cfeature.COLORS['land']
    ax.add_feature(cfeature.COASTLINE)
    cb=fig.colorbar(sm,cax=plt.axes([0.265,0.2,0.5,0.02]),spacing='proportional',orientation='horizontal')#,label=LAB).outline.remove()
    cb.ax.tick_params(labelsize=15)
    cb.set_label(LAB,size=15)
    gl = ax.gridlines(draw_labels=True,color='cadetblue')
    gl.top_labels = False
    gl.right_labels = False
    if np.size(FN)==0:
        plt.show()
    else:
        fig.savefig(FN)
        plt.close()

def tide_pull(F):
    f_tg = open(F,'r')
    LinesH = f_tg.readlines()
    NlH = np.shape(LinesH)[0]
    yrf = []
    ssh = []
    for ii in np.arange(13,NlH):
        sLine1 = LinesH[ii].split()
        if '2023' in sLine1[0]:
            yr = int(sLine1[0][:4])
            mn = int(sLine1[0][5:7])
            dy = int(sLine1[0][8:10])
            hr = int(sLine1[2][:2])
            mi = int(sLine1[2][3:5])
            
            #yr.append(float(sLine1[0][:4]))
            #mn.append(float(sLine1[0][5:7]))
            #dy.append(float(sLine1[0][8:10]))
            #hr.append(float(sLine1[2][:2]))
            #mi.append(float(sLine1[2][3:5]))
            ssh.append(float(sLine1[3]))
            ti_date = dt.datetime(yr, mn, dy,hr,mi)
            t0_date = dt.datetime(yr, 1, 1)
            dtime = ti_date-t0_date
            yrf.append(((dtime.days+(dtime.seconds/(60*60*24)))/365.25)+yr)
    yrf = np.asarray(yrf)   
    ssh = np.asarray(ssh)   
    return yrf,ssh

def data2bin(xbins,xdata,ydata):
    #xbins,xdata,ydata = t_bin,yrf1,ssh1
    dbin = np.abs(np.diff(xbins)[0]/2.0)
    N = np.size(xbins)
    ybins = np.empty(N)*np.nan
    SZ = np.empty(N)*np.nan
    for ii in np.arange(N):
        idx = np.where((xdata>=xbins[ii]-dbin)&(xdata<xbins[ii]+dbin))[0]
        SZ[ii]=np.size(idx)
        if np.size(idx)>0:
            ybins[ii] = np.nanmean(ydata[idx])
    return ybins

def find_phase_diff(t,data1_in,data2_in):
    #t,data1_in,data2_in = t_binsAS,mn_sel,mn_anc
    #t,data1_in,data2_in = t_bin,ssh1B,ssh2B
    iin =np.where((~np.isnan(data1_in))&(~np.isnan(data2_in)))[0]
    data1 = data1_in[iin] 
    data2 = data2_in[iin] 
    dti = np.diff(t)[0]
    delay = np.arange(0,(12.0/(365.25*24))+dti,dti)
    N = np.shape(delay)[0]
    cc = np.empty(N)*np.nan
    A = np.empty(N)*np.nan
    rms = np.empty(N)*np.nan
    for ii in np.arange(N):
        if ii ==0:
            cc[ii] = np.corrcoef(data1,data2)[1,0]
            A[ii] = np.nanstd((data2-np.nanmean(data2))-(data1-np.nanmean(data1)))
            rms[ii] = np.sqrt(np.nanmean((data2-data1)**2))
        else:
            cc[ii] = np.corrcoef(data1[:-ii],data2[ii:])[1,0]
            A[ii] = np.nanstd((data2[ii:]-np.nanmean(data2[ii:]))-(data1[:-ii]-np.nanmean(data1[:-ii])))
            rms[ii] = np.sqrt(np.nanmean((data2[ii:]-data1[ii:])**2))
    idx = np.where(cc==np.nanmax(cc))[0][0]
    #idx = np.where(rms==np.nanmin(rms))[0][0]
    return delay[idx],A[idx]


def mean_depth_line(LOC,ptB,ptE):
    # LOC,ptB,ptE = 'seattle',lin_pts[ii],lin_pts[ii+1]
    M = (ptB[0]-ptE[0])/(ptB[1]-ptE[1])
    B = ptB[0]-(M*ptB[1])
    lat,lon,elev,ocean = pull_gebco(LOC)
    if ptB[0]<ptE[0]:
        idx = np.where((lat>=ptB[0])&(lat<=ptE[0]))[0]
    else:
        idx = np.where((lat>=ptE[0])&(lat<=ptB[0]))[0]
    lat_line = np.copy(lat[idx])
    lon_line = (lat_line-B)/M
    N = np.size(idx)
    Xdist = np.empty(N)*np.nan
    Xbath = np.empty(N)*np.nan
    Ldist = np.empty(N)*np.nan
    ##
    for ii in np.arange(N):
        lat0,lon0 = lat_line[ii],lon_line[ii]
        dist_i = geo2dist(np.ones(np.size(lon))*lat0,lon,lat0,lon0)/1000.0
        # determine direction of grid cells with respect to point
        direction = lon180_to_lon360(lon)-lon180_to_lon360(lon0)
        direction[direction<0]=-1
        direction[direction>0]=1
        # pull ocean segement
        ocean_lon_i = ocean[idx[ii],:]
        # find fjord boundry to the East of the point
        idirP = np.where((direction>=0))[0]
        ocean_p = np.copy(ocean_lon_i[idirP])
        inp = np.where(np.isnan(ocean_p))[0]
        if np.size(inp)>3:
            dinp = np.diff(np.diff(inp))
            i0p = np.where(dinp==0)[0]
            if np.size(i0p)>1:
                istop_p = idirP[inp[i0p[0]+1]]
            else:
                istop_p = idirP[inp[-1]]
        else:
            istop_p = np.nanmax(np.where(np.abs(dist_i[idirP])<20)[0])
        # find fjord boundry to the West of the point
        idirN = np.where((direction<0))[0]
        ocean_n = np.copy(ocean_lon_i[idirN])
        inn = np.where(np.isnan(ocean_n))[0]
        if np.size(inn)>3:
            dinn = np.diff(np.diff(inn))
            i0n = np.where(dinn==0)[0]
            if np.size(i0n)>1:
                istop_n = idirN[inn[i0n[-1]-1]]
            else:
                istop_n = idirN[inn[-1]]
        else:
            istop_n = np.nanmax(np.where(np.abs(dist_i[idirN])<20)[0])
        # distance of each lon grid point with respect to line point
        if istop_p>=istop_n:
            dist_seg = (dist_i*direction)[istop_n:istop_p+1]
            Xdist[ii] = dist_seg[-1]-dist_seg[0] # [km]
            Oi = ocean_lon_i[istop_n:istop_p+1]
            Di = dist_i[istop_n:istop_p+1]
            Oi[np.abs(Di)>30] = np.nan
            Wi = (1.0/(Di**2))#np.ones(np.size(Di))# 
            Xbath[ii] = np.nansum(Oi*Wi)/np.nansum(Wi)#np.nanmean(ocean_lon_i[istop_n:istop_p+1])
            Ldist[ii] = np.abs(geo2dist(lat0,lon0,lat_line[0],lon_line[0])/1000.0)
        else:
            Xdist[ii] = np.nan
            Xbath[ii] = np.nan
            Ldist[ii] = np.nan
    return [M,B],lat_line,lon_line,Xdist,Xbath,Ldist



def pull_gebco(LOC):
    # https://download.gebco.net
    if LOC in ['skagway','anchorage','seattle','vancouver','juneau','skagway2','juneau2']:
        fn='GEBCO/alaska/gebco_2023_n61.3954_s47.0534_w-155.5567_e-121.0439.nc'
        fnT='GEBCO/alaska/gebco_2023_tid_n61.3954_s47.0534_w-155.5567_e-121.0439.nc'
    ds=Dataset(fn)
    dsT=Dataset(fnT)
    lat=ds['lat'][:]
    lon=ds['lon'][:]
    elev=ds['elevation'][:]
    #iout=np.where(elev<-30000)
    #elev[iout]=np.nan
    land_id=dsT['tid'][:]
    ocean = np.empty(np.shape(elev))*np.nan
    idL = np.where(elev<=0)
    ocean[idL]=elev[idL]
    return lat,lon,elev,ocean

def gebco_bathymetry(LOC,pts):
    lat,lon,elev,ocean = pull_gebco(LOC)
    plt.figure(figsize=(25,10))
    plt.pcolormesh(lon,lat,elev,cmap='terrain',vmin=-5000,vmax=4000)
    plt.colorbar()

    plt.figure(figsize=(25,10))
    plt.pcolormesh(lon,lat,ocean,cmap='terrain',vmin=-5000,vmax=4000)
    plt.colorbar()

    ibath_lat = np.where((lat>=pts[0][0])&(lat<=pts[1][0]))[0]
    ibath_lon = np.where((lon>=pts[1][1])&(lon<=pts[0][1]))[0]
    LLMM = [np.nanmin(ibath_lat),np.nanmax(ibath_lat),np.nanmin(ibath_lon),np.nanmax(ibath_lon)]
    mean_bath = np.nanmean(ocean[LLMM[0]:LLMM[1],LLMM[2]:LLMM[3]])
    '''
    f = scipy.interpolate.interp2d(lon, lat, elev, kind='cubic')
    N = np.shape(coord)[0]
    bath = np.empty(N)*np.nan
    for ii in np.arange(N):
        bath[ii] = f(coord[ii][1],coord[ii][0])
    '''


def bin_data(var_new,t2,vr_min,bin_days,N_MIN):
    '''
    var0 = h2-tide_total
    bin_days=30
    vr_min=1
    '''
    binz2 = np.arange(-25,25,0.1)
    dti=bin_days/365.25
    t_bins = np.arange(2018,2024+dti,dti)#t_tg[t_tg>2018]#np.arange(2018,2024+dt,dt)
    Nt = np.size(t_bins)
    h2mt_new = np.empty(np.size(var_new))*np.nan
    for tt in np.arange(Nt-1):
        idt = np.where((t2>=t_bins[tt])&(t2<t_bins[tt+1]))[0]
        var_idt = np.copy(var_new[idt])
        if np.size(var_idt[~np.isnan(var_idt)])>N_MIN:
            h2mti = remove_outliers(var_idt,zval=2,MAXIT=10)
            if np.size(h2mti[~np.isnan(h2mti)])>N_MIN:
                if np.nanvar(h2mti)<vr_min:
                    inn = np.where(~np.isnan(h2mti))[0]
                    if np.size(inn)>0:
                        '''
                        plt.figure()
                        plt.title('size: '+str(np.size(var_idt[~np.isnan(var_idt)]))+'\n var: '+str(np.round(np.nanvar(var_idt),3)))
                        plt.hist(var_new[idt][~np.isnan(var_new[idt])],bins=binz2,alpha=0.5)
                        plt.hist(h2mti,bins=binz2,alpha=0.5)
                        '''
                        h2mt_new[idt]=h2mti
    return h2mt_new


def mode_filter(var,max_frac=0.2,dbin=0.1):
    # filter
    # var0 = h2-tide_total
    # var = h2-tide_total
    inn = np.where(~np.isnan(var))[0]
    dist = np.arange(0,10+dbin,dbin)
    Ndist = np.size(dist)
    cum_occ = np.empty(Ndist)*np.nan
    N_dist = np.empty(Ndist)*np.nan
    mdV = np.nanmedian(var)
    sdV = np.nanstd(var)
    bmin = mdV-(3*sdV)
    bmax = mdV+(3*sdV)
    if np.size(inn)>10:
        occ,edg = np.histogram(var[inn],bins=np.arange(bmin,bmax+dbin,dbin))
        ipks = np.argsort(occ)[::-1][:5]
        pk = edg[ipks][0]
        dedg = edg[:-1]-pk
        iin = np.where(np.abs(dedg)<10)[0]
        sum_occ = np.nansum(occ[iin])
        for ii in np.arange(Ndist-1):
            idx = np.where((np.abs(dedg)>dist[ii])&(np.abs(dedg)<=dist[ii+1]))[0]
            cum_occ[ii]=(np.nansum(occ[idx])/(1.0*sum_occ))*100.
            N_dist[ii]=np.size(idx)
        #'''
        plt.figure()
        plt.plot(dist,cum_occ/np.nanmax(cum_occ),'.-')
        plt.grid()
        #'''
        cum_occ_max = cum_occ/np.nanmax(cum_occ)
        icut = np.where(cum_occ_max>max_frac)[0] # only consider points that have greater than 20% 
        dist_max = np.max(dist[icut])
        dvar = np.abs(var-pk)
        var_new = np.copy(var)
        var_new[dvar>dist_max]=np.nan
    else:
        var_new = np.empty(np.size(var))*np.nan
    return var_new

def generate_tide_corrections(utc2,LON_tide,LAT_tide):
        import ana_fes2014_tide_reg_pyTMD as atide 
        tide_total = np.empty(np.size(utc2))*np.nan 
        inn = np.where((~np.isnan(utc2)))[0]
        LLMM_tides = np.asarray([LAT_tide-1,LAT_tide+1,LON_tide-1,LON_tide+1])
        FN = sorted(os.listdir('regional_constants/')) # lists all files in directory
        if 'regional_fes2014_constants_ocean.nc' not in FN:
            atide.prelim_ocean_tide_replacement(np.asarray(LLMM_tides),False)
            atide.prelim_ocean_tide_replacement(np.asarray(LLMM_tides),True)
        else:
            f1O = 'regional_constants/regional_fes2014_constants_ocean.nc'
            f1T = 'regional_constants/regional_fes2014_constants_load.nc'
            dsO = Dataset(f1O)
            dsT = Dataset(f1T)
            latO,lonO = dsO['lat'][:],dsO['lon'][:]
            latT,lonT = dsT['lat'][:],dsT['lon'][:]
            LON_tide = lon180_to_lon360(LON_tide)
            PASS=0
            if np.nanmin(lonO)>LON_tide:
                PASS=1
            if np.nanmax(lonO)<LON_tide:
                PASS=1
            if np.nanmin(latO)>LAT_tide:
                PASS=1
            if np.nanmax(latO)<LAT_tide:
                PASS=1
            
            if np.nanmin(lonT)>LON_tide:
                PASS=1
            if np.nanmax(lonT)<LON_tide:
                PASS=1
            if np.nanmin(latT)>LAT_tide:
                PASS=1
            if np.nanmax(latT)<LAT_tide:
                PASS=1

            if PASS==1:
                atide.prelim_ocean_tide_replacement(np.asarray(LLMM_tides),False)
                atide.prelim_ocean_tide_replacement(np.asarray(LLMM_tides),True)

        time_SF = atide.utc2utc_stamp(utc2[inn])
        lon_tides,lat_tides = np.ones(np.size(inn))*LON_tide,np.ones(np.size(inn))*LAT_tide

        tide_ocean = atide.ocean_tide_replacement(lon_tides,lat_tides,time_SF,LOAD=False,method='spline')
        tide_load = atide.ocean_tide_replacement(lon_tides,lat_tides,time_SF,LOAD=True,method='spline')
        tide_total[inn] = (tide_ocean+tide_load).squeeze()
        return tide_total


def coastline():
    """
    ---------------------------------------------------------------------------------------
    COASLINE
    ---------------------------------------------------------------------------------------
    """
    working_dir = '/Users/alexaputnam/ICESat2/VLM/'
    coastline_file = f'{working_dir}gshhg-shp/GSHHS_shp/l/GSHHS_l_L1.shp'
    gdf_coast = gpd.read_file(coastline_file)
    gdf_antarctica = gpd.read_file(coastline_file.replace('L1','L6'))
    tmp_df = pd.DataFrame({'id':[5],'level':[6],'source':['ANT-G'],'parent_id':[-1],'sibling_id':[5],'area':[gdf_antarctica['area'][0]+gdf_antarctica['area'][1]]})
    tmp_gdf = gpd.GeoDataFrame(tmp_df,geometry=[gdf_antarctica.geometry[0].union(gdf_antarctica.geometry[1])],crs='EPSG:4326')
    gdf_antarctica = gpd.GeoDataFrame(pd.concat([tmp_gdf,gdf_antarctica],ignore_index=True),crs='EPSG:4326')
    gdf_antarctica = gdf_antarctica.drop(gdf_antarctica.index[[1,2]]).reset_index(drop=True)
    lon_coast,lat_coast = get_lonlat_gdf(gdf_coast)
    lon_ant,lat_ant = get_lonlat_gdf(gdf_antarctica)
    lon_coast = np.concatenate([lon_coast,lon_ant])
    lat_coast = np.concatenate([lat_coast,lat_ant])
    return lon_coast,lat_coast

def external_data(tg_id,NAME):
    """
    ---------------------------------------------------------------------------------------
    PSMSL
    ---------------------------------------------------------------------------------------
    """
    working_dir = '/Users/alexaputnam/ICESat2/VLM/'
    sonel_file = []
    psmsl_dir = f'{working_dir}rlr_monthly/'
    psmsl_file_list = f'{psmsl_dir}filelist.txt'
    psmsl_mtl_msl_conversion_file = f'{psmsl_dir}mtl_msl_corrections.csv'
    N_missing_days_threshold = 15
    df_tg = pd.read_csv(psmsl_file_list,header=None,sep=';',
        names = ['TG_ID','latitude','longitude','TG_name','coastline','station_code','flag'])#dtype={'TG_ID':'int','latitude':'float','longitude':'float','TG_name':'str','coastline':'int','station_code':'int','flag':'str'})
    df_tg.TG_name = [a.strip() for a in df_tg.TG_name]
    df_tg.flag = [a.strip() for a in df_tg.flag]
    df_tg = df_tg[df_tg.flag == 'N'].reset_index(drop=True)
    df_tg = df_tg.drop(columns=['flag'])
    df_mtl_msl_conversion = pd.read_csv(psmsl_mtl_msl_conversion_file) #this file has a header so names will be automatically detected
    df_mtl_msl_conversion.rename(columns={'STATION':'station','PERIOD START':'t_start','PERIOD END':'t_end','MTL-MSL (MM)':'delta_mtl_msl'}, inplace=True)
    df_tg.describe()
    if np.size(tg_id)==0:
        for ii in np.arange(np.shape(df_tg.TG_ID)[0]):
            if NAME in df_tg.TG_name[ii]:
                print(df_tg.TG_name[ii])
                print('ID: '+str(df_tg.TG_ID[ii]))
                print(ii)
                tg_id=df_tg.TG_ID[ii]

    lat0 = np.asarray(df_tg.latitude[df_tg.TG_ID==tg_id])[0]
    lon0 = np.asarray(df_tg.longitude[df_tg.TG_ID==tg_id])[0]
    tg_data_file = f'{psmsl_dir}data/{tg_id}.rlrdata'
    df_individual_tg = pd.read_csv(tg_data_file,header=None,sep=';',
            names=['time','sea_level','N_missing_days','flag_for_attention'],
            dtype={'time':'float','sea_level':'int','N_missing_days':'int','flag_for_attention':'str'})
    if np.any(df_mtl_msl_conversion.station == tg_id) == True:
        idx_id = np.argwhere(np.asarray(df_mtl_msl_conversion.station) == tg_id)[0][0]
        delta_mtl_msl = df_mtl_msl_conversion.delta_mtl_msl[idx_id]
        if delta_mtl_msl != -99999:
            df_individual_tg.loc[df_individual_tg.flag_for_attention.str[1] == '1','sea_level'] = df_individual_tg.sea_level[df_individual_tg.flag_for_attention.str[1] == '1'] + delta_mtl_msl
            df_individual_tg.flag_for_attention = df_individual_tg.flag_for_attention.str[0] + df_individual_tg.flag_for_attention.str[1].replace('1','0') + df_individual_tg.flag_for_attention.str[2]
    idx_nodata = df_individual_tg.sea_level == -99999
    idx_missing_days = df_individual_tg.N_missing_days > N_missing_days_threshold
    idx_flag = df_individual_tg.flag_for_attention != '000'
    idx_filter = np.any((idx_nodata,idx_missing_days,idx_flag),axis=0)
    df_individual_tg = df_individual_tg[~idx_filter].reset_index(drop=True)
    t_tg = np.asarray(df_individual_tg.time)
    sl_tg = np.asarray(df_individual_tg.sea_level)/1000.

    if np.size(sonel_file)!=0:
        ### SONEL GPS
        yfr_sonel,h_sonel,sh_sonel = open_sonel_file(sonel_file)
    else:
        yfr_sonel,h_sonel = np.copy(t_tg),np.empty(np.size(t_tg))*np.nan
    '''
    plt.figure()
    plt.plot(yfr_sonel,h_sonel)
    plt.plot(t_tg,sl_tg,label=NAME)
    plt.xlim(2018,2024)
    plt.legend()

    plt.figure()
    plt.plot(t_tg,sl_tg,label=NAME)
    plt.plot(yfr_sonel,h_sonel)
    plt.xlim(2018,2024)
    plt.legend()
    '''
    """
    ---------------------------------------------------------------------------------------
    Load MIDAS VLM
    ---------------------------------------------------------------------------------------
    """
    midas_file = f'{working_dir}midas.IGS14.txt'
    df_midas = pd.read_csv(midas_file,header=None,delim_whitespace=True,
        names=['station_id','midas_version','t_start','t_end','t_duration','N_epochs','N_good_epochs','N_vel_pairs',
            'v_east','v_north','v_up','v_uncertainty_east','v_uncertainty_north','v_uncertainty_up',
            'offset_first_epoch_east','offset_first_epoch_north','offset_first_epoch_up','frac_outliers_east','frac_outliers_north','frac_outliers_up',
            'std_vel_pair_east','std_vel_pair_north','std_vel_pair_up','N_steps',
            'latitude','longitude','height'],
        dtype={'station_id':'str','midas_version':'str','t_start':'float','t_end':'float','t_duration':'float','N_epochs':'int','N_good_epochs':'int','N_vel_pairs':'int',
            'v_east':'float','v_north':'float','v_up':'float','v_uncertainty_east':'float','v_uncertainty_north':'float','v_uncertainty_up':'float',
            'offset_first_epoch_east':'float','offset_first_epoch_north':'float','offset_first_epoch_up':'float','frac_outliers_east':'float','frac_outliers_north':'float','frac_outliers_up':'float',
            'std_vel_pair_east':'float','std_vel_pair_north':'float','std_vel_pair_up':'float','N_steps':'float',
            'latitude':'float','longitude':'float','height':'float'})

    df_midas.loc[df_midas.longitude < -180,'longitude'] = df_midas.longitude[df_midas.longitude < -180] + 360
    df_midas_orig = df_midas.copy()
    df_midas = df_midas[df_midas.v_east != 99.999999].reset_index(drop=True)
    df_midas = df_midas[df_midas.v_north != 99.999999].reset_index(drop=True)
    df_midas = df_midas[df_midas.v_up != 99.999999].reset_index(drop=True)
    #df_midas = df_midas[df_midas.v_uncertainty_up < v_uncertainty_up_threshold].reset_index(drop=True)
    #df_midas = df_midas[df_midas.frac_outliers_east < frac_outlier_threshold].reset_index(drop=True)
    #df_midas = df_midas[df_midas.frac_outliers_north < frac_outlier_threshold].reset_index(drop=True)
    #df_midas = df_midas[df_midas.frac_outliers_up < frac_outlier_threshold].reset_index(drop=True)
    #df_midas = df_midas[df_midas.t_duration >= min_duration_gps].reset_index(drop=True)
    #df_midas = df_midas[df_midas.t_start > 1990].reset_index(drop=True)
    #df_midas = df_midas[df_midas.N_good_epochs / df_midas.N_epochs > good_epochs_threshold].reset_index(drop=True)
    #df_midas = df_midas[np.abs(df_midas.v_up) < max_velocity].reset_index(drop=True)
    midas_vlm = np.asarray(df_midas.v_up) #m/y
    midas_t_duration = np.asarray(df_midas.t_duration) #years
    midas_t_start = np.asarray(df_midas.t_start) #decimal year
    midas_id = np.asarray(df_midas.station_id)
    print('MIDAS dataset after filtering:')
    df_midas.describe()

    dist = np.abs(np.asarray(geo2dist(df_midas.latitude,df_midas.longitude,lat0,lon0)))#np.asarray(great_circle_distance(df_tg.longitude[i],df_tg.latitude[i],df_midas.longitude,df_midas.latitude))
    dist_closest_gps = np.min(dist)
    imidas = np.where((dist==dist_closest_gps))[0]
    t_gps_str = midas_t_start[imidas]
    t_gps_dur = midas_t_duration[imidas]
    vlm_gps = midas_vlm[imidas]

    """
    ---------------------------------------------------------------------------------------
    Load Hammond VLM data
    ---------------------------------------------------------------------------------------
    """
    hammond_vlm_file = f'{working_dir}hammond_vlm.txt'
    fileH = open(hammond_vlm_file, 'r')
    LinesH = fileH.readlines()
    NlH = np.shape(LinesH)[0]
    TGid_H = []
    lat_H = []
    lon_H = []
    vlm_H = []
    evlm_H = []
    for ii in np.arange(NlH):
        sLine1 = LinesH[ii].split()
        if '>>' in sLine1:
            sLine2 = LinesH[ii+1].split()
            TGid_H.append(float(sLine1[1]))
            lat_H.append(float(sLine2[2]))
            lon_H.append(float(sLine2[1]))
            vlm_H.append(float(sLine2[3]))
            evlm_H.append(float(sLine2[4]))
    TGid_H = np.asarray(TGid_H)   
    lat_H = np.asarray(lat_H)   
    lon_H = np.asarray(lon_H)   
    vlm_H = np.asarray(vlm_H)   # mm/y
    evlm_H = np.asarray(evlm_H)   #mm/y
    dist_hamm = np.abs(np.asarray(geo2dist(lat_H,lon_H,lat0,lon0)))#np.asarray(great_circle_distance(df_tg.longitude[i],df_tg.latitude[i],df_midas.longitude,df_midas.latitude))
    dist_closest_hamm = np.min(dist_hamm)
    ihamm = np.where((dist_hamm==dist_closest_hamm))[0]
    vlm_Hi = vlm_H[ihamm]/1000. #mm/y
    evlm_Hi = evlm_H[ihamm] #mm/y

    plt.figure()
    vlm_fit = vlm_Hi*(t_tg-t_tg[0])
    plt.plot(t_tg,sl_tg+vlm_fit,label=NAME+' corrected')
    plt.plot(t_tg,sl_tg,label=NAME+' not corrected')
    plt.legend()
    plt.grid()
    return t_tg,sl_tg,yfr_sonel,h_sonel,t_gps_str,vlm_gps,vlm_Hi

def pull_csv(pthNOAA): #convert_partial_year
    # pthNOAA ='/Users/alexaputnam/ICESat2/Fjord/cook_inlet/anchorage/'
    # yf,meas,pred = pull_csv(pth)
    # https://tidesandcurrents.noaa.gov/waterlevels.html?id=8452660&units=metric&bdate=20200101&edate=20201231&timezone=GMT&datum=MSL&interval=h&action=data
    # FN = '2020_8452660_met_newport_ri.csv'
    import datetime as dt
    filenames = glob(pthNOAA+'pred*.csv') #filenames = glob('/Users/alexaputnam/ICESat2/Fjord/nikiski/*.csv')
    N = np.shape(filenames)[0]
    yr,mn,dy,hr,mi=[],[],[],[],[]
    sigma,meas,yf=[],[],[]
    #dateTG = []
    for ii in np.arange(N):
        FN = filenames[ii]
        with open(FN) as csvfile:
            dfh = pd.read_csv(FN,skiprows=[0],skip_blank_lines=False,header=None)#
            if np.shape(dfh)[1]>1: #'Wrong' not in dfh[0][0]:
                #datei = dfh[0]
                #timei = dfh[1]
                datetimei = dfh[0]
                #ikpi = np.asarray([ii for ii in np.arange(np.size(dfh[4])) if isinstance(dfh[4][ii], (int,float))])
                ikpi = np.asarray([ii for ii in np.arange(np.size(dfh[1])) if isinstance(dfh[1][ii], (int,float))])
                if np.size(ikpi)!=0:
                    Ni=np.shape(datetimei)[0]
                    yri = np.asarray([int(datetimei[jj][:4]) for jj in np.arange(Ni)])
                    mni = np.asarray([int(datetimei[jj][5:7]) for jj in np.arange(Ni)])
                    dyi = np.asarray([int(datetimei[jj][8:10]) for jj in np.arange(Ni)])
                    hri = np.asarray([int(datetimei[jj][11:13]) for jj in np.arange(Ni)])#np.asarray([int(timei[jj][:2]) for jj in np.arange(Ni)])
                    mii = np.asarray([int(datetimei[jj][14:16]) for jj in np.arange(Ni)])#np.asarray([int(timei[jj][3:5]) for jj in np.arange(Ni)])
                    yfi = np.asarray([yri[jj]+((dt.datetime(yri[jj],mni[jj],dyi[jj],hri[jj],mii[jj])-dt.datetime(yri[jj],1,1,0,0)).total_seconds()/(365.25*24*60*60)) for jj in np.arange(Ni)])
                    yr = np.hstack((yr,yri[ikpi]))
                    mn = np.hstack((mn,mni[ikpi]))
                    dy = np.hstack((dy,dyi[ikpi]))
                    hr = np.hstack((hr,hri[ikpi]))
                    mi = np.hstack((mi,mii[ikpi]))
                    meas = np.hstack((meas,dfh[1][ikpi]))#np.hstack((meas,dfh[4][ikpi]))
                    #sigma = np.hstack((sigma,dfh[2][ikpi]))#np.hstack((pred,dfh[2][ikpi]))
                    yf = np.hstack((yf,yfi))
    sigma = np.empty(np.size(meas))*np.nan
    ikp = np.where((~np.isnan(meas)))[0]
    print('size(ikp) = '+str(np.size(ikp)))
    dateTG = [str(int(yr[jj]))+'-'+"{:02d}".format(int(mn[jj]))+'-'+"{:02d}".format(int(dy[jj]))+'-'+"{:02d}".format(int(hr[jj]))+'-'+"{:02d}".format(int(mi[jj])) for jj in ikp]
    return yf[ikp],meas[ikp],sigma[ikp],dateTG

def pull_from_noaa():
    # https://github.com/GClunies/py_noaa
    """
    Function to get data from NOAA CO-OPS API and convert it to a pandas
    dataframe for convenient analysis.

    Info on the NOOA CO-OPS API can be found at https://tidesandcurrents.noaa.gov/api/,
    the arguments listed below generally follow the same (or a very similar) format.

    Arguments:
    begin_date -- the starting date of request (yyyyMMdd, yyyyMMdd HH:mm, MM/dd/yyyy, or MM/dd/yyyy HH:mm), string
    end_date -- the ending date of request (yyyyMMdd, yyyyMMdd HH:mm, MM/dd/yyyy, or MM/dd/yyyy HH:mm), string
    stationid -- station at which you want data, string
    product -- the product type you would like, string
    datum -- the datum to be used for water level data, string  (default None)
    bin_num -- the bin number you would like your currents data at, int (default None)
    interval -- the interval you would like data returned, string
    units -- units to be used for data output, string (default metric)
    time_zone -- time zone to be used for data output, string (default gmt)
    """
    import py_noaa as pne
    from py_noaa import coops
    df_currents = coops.get_data(
        begin_date="20230101",
        end_date="20230102",
        stationid="9447130",
        product="water_level",
        datum="MLLW",
        units="metric",
        time_zone="gmt")


def tide_estimation(utc2,h2,LON_tide,LAT_tide,t_delay_min=60,max_dt_sec=60,Return_moments=False,A=1.0):
    '''
    tide_estimation(utc2,h2,LON_tide,LAT_tide,t_delay_min=60,max_dt_sec=60,Return_moments=False,A=1.0)
    t_delay_min=60
    max_dt_sec=2
    '''
    isrt = np.argsort(utc2)
    cutc2 = np.copy(utc2)[isrt]
    arr_dt = np.abs(np.diff(cutc2))
    it2 = np.where(arr_dt>max_dt_sec)[0]
    t_moments = cutc2[it2+1]
    h_moments = np.empty(np.size(t_moments))*np.nan
    if np.size(LON_tide)!=0:
        tide_totali=generate_tide_corrections(t_moments-(t_delay_min*60.0),LON_tide,LAT_tide)*A
        tide_total = np.empty(np.size(utc2))*np.nan
        for ii in np.arange(np.size(it2)):
            dutc = np.abs(utc2-t_moments[ii])
            idx = np.where(dutc<=max_dt_sec)[0]
            tide_total[idx] = tide_totali[ii]
            h_moments[ii] = np.nanmean(h2[idx]-tide_total[idx])
    else:
        tide_total = np.zeros(np.size(utc2))   
        tide_totali = np.empty(np.size(t_moments))*np.nan 
    if Return_moments == False:
        return tide_total
    else:
        return tide_total,t_moments,tide_totali,h_moments


def pull_is2(fn):
    Nfn = np.size(fn)
    if Nfn==1:
        ds_var = np.load(fn) #'reg_atl03_lat_18_lon_n65_stj_segs_2_200_2000_2018_10_to_2023_06.npz'
        kys = list(ds_var.keys())
        ti = utc2yrfrac(ds_var['time_utc_mean_cu2'])
        innT = np.where(~np.isnan(ti))[0]
        tO=ds_var['tide_ocean_g48_mean_cu2'][innT]
        tL=ds_var['tide_load_g48_mean_cu2'][innT]
        tE=ds_var['tide_equilibrium_mean_cu2'][innT]
        dac2= ds_var['dac_mean_cu2'][innT]
        lat2 = ds_var['lat_mean_cu2'][innT]
        lon2 = ds_var['lon_mean_cu2'][innT]
        dem2 = ds_var['dem_mean_cu2'][innT]
        geoid2 = ds_var['geoid_mean_cu2'][innT]
        N2 = ds_var['N_cu2'][innT]
        h2 = ds_var['h_mean_cu2'][innT]+tL
        utc2=ds_var['time_utc_mean_cu2'][innT]
        t2 = utc2yrfrac(ds_var['time_utc_mean_cu2'][innT])
        h2[np.abs(h2)>100]=np.nan   
        if 'solar_elv_cu2' in kys:
            solEv = ds_var['solar_elv_cu2'][innT]
            solAz = ds_var['solar_azi_cu2'][innT]
    else:
        tO,tL,tE,dac2,lat2,lon2,dem2,geoid2,N2,h2,t2,utc2,solEv,solAz=[],[],[],[],[],[],[],[],[],[],[],[],[],[]
        for ii in np.arange(Nfn):
            ds_var = np.load(fn[ii]) #'reg_atl03_lat_18_lon_n65_stj_segs_2_200_2000_2018_10_to_2023_06.npz'
            #kys = list(ds_var.keys())
            ti = utc2yrfrac(ds_var['time_utc_mean_cu2'])
            innT = np.where(~np.isnan(ti))[0]

            tO = np.hstack((tO,ds_var['tide_ocean_g48_mean_cu2'][innT]))
            tL=np.hstack((tL,ds_var['tide_load_g48_mean_cu2'][innT]))
            tE=np.hstack((tE,ds_var['tide_equilibrium_mean_cu2'][innT]))
            dac2= np.hstack((dac2,ds_var['dac_mean_cu2'][innT]))
            lat2 = np.hstack((lat2,ds_var['lat_mean_cu2'][innT]))
            lon2 = np.hstack((lon2,ds_var['lon_mean_cu2'][innT]))
            dem2 = np.hstack((dem2,ds_var['dem_mean_cu2'][innT]))
            geoid2 = np.hstack((geoid2,ds_var['geoid_mean_cu2'][innT]))
            N2 = np.hstack((N2,ds_var['N_cu2'][innT]))
            h2 =np.hstack((h2,(ds_var['h_mean_cu2']+ds_var['tide_load_g48_mean_cu2'])[innT]))
            utc2=np.hstack((utc2,ds_var['time_utc_mean_cu2'][innT]))
            t2 = np.hstack((t2,utc2yrfrac(ds_var['time_utc_mean_cu2'][innT])))
            if 'solar_elv_cu2' in kys:
                solEv = np.hstack((solEv,ds_var['solar_elv_cu2'][innT]))
                solAz = np.hstack((solAz,ds_var['solar_azi_cu2'][innT]))
    if 'solar_elv_cu2' not in kys:
            solEv = np.empty(np.size(tL))*np.nan 
            solAz = np.empty(np.size(tL))*np.nan 
    h2c = h2-dac2
    return tO,tL,tE,dac2,lat2,lon2,dem2,geoid2,N2,h2c,utc2,t2,solEv,solAz


def correct_is2_ssh(h2,utc2,LON_tide,LAT_tide,t_delay_min=0,A=1,aSolEv=30,solElv=[]):
    h2_out = np.copy(h2)
    tide_total = np.empty(np.size(h2))*np.nan
    inn = np.where(~np.isnan(h2_out))[0]
    # Estimate tide
    tide_total[inn] = tide_estimation(utc2[inn],h2[inn],LON_tide,LAT_tide,t_delay_min=t_delay_min)
    if np.size(solElv)!=0:
        isol1 = np.where(solElv<=aSolEv)[0]
        tide_total[isol1] = tide_total[isol1]*A[0]
        isol2 = np.where(solElv>aSolEv)[0]
        tide_total[isol2] = tide_total[isol2]*A[1]
    else:
        tide_total = tide_total*A
    print('size of non-nan interpolated dataset '+str(np.size(np.where(~np.isnan(tide_total[inn])[0]))))
    # Remove all points greater than 100 m w.r.t. ellipsoid
    h2t = h2_out-tide_total
    h2t[np.abs(h2t)>100]=np.nan
    h2_out[np.abs(h2t)>100]=np.nan
    return h2_out,h2t,tide_total


def filter_is2_ssh(h2t,t2,MODE=True,VR_MAX=2,N_MIN=50):
    #h2t_i2 = filter_is2_ssh(h2t,t2)
    # Filter-out non-ocean points
    #h2t_i2= mode_filter(h2t_i1,max_frac=0.2)
    if MODE==True:
        h2t_i1= mode_filter(h2t,max_frac=0.2,dbin=0.1)
    else:
        h2t_i1 = np.copy(h2t)
    h2t_i2 = bin_data(h2t_i1,t2,VR_MAX,bin_days=1,N_MIN=N_MIN)
    #h2t_i3= mode_filter(h2t_i2,max_frac=0.3,dbin=0.3)
    binz = np.arange(-25,25,0.1)
    '''
    plt.figure()
    plt.hist(h2t_i1,bins=binz)
    plt.hist(h2t_i2,bins=binz,alpha=1)
    plt.hist(h2t_i3,bins=binz,alpha=0.7)
    plt.xlim(-10,10)
    '''
    #h2_fin=bin_data(h2_fini,0.1,bin_days=15)
    return h2t_i2

def reference_frames(ref):
    # https://kb.osu.edu/bitstream/handle/1811/51274/Geometric_Reference_Systems_2012.pdf
    if ref=='wgs84':
        a = 6378137.0
        f = 1.0/298.257223563
    elif ref=='topex': # IERS standard
        a = 6378136.300
        f = 1.0/298.257
    return a,f

def ellipsoid_transformation_2step(latA,lonA,hA,refA,refB):
    # https://nsidc.org/sites/default/files/documents/technical-reference/comparisonusersguide_v006.pdf
    '''
    #Example from site (page 9)
    latA = flat_sec#np.ones(100)*47
    lonA = flon_sec#np.ones(100)*15
    hA = fmss_sec#np.ones(100)*1200
    refA = 'wgs84'
    refB = 'topex'
    latB,lonB,hB = ellipsoid_transformation_2step(latA,lonA,hA,refA,refB)
    '''
    latAf = latA*(np.pi/180.0)
    lonAf = lonA*(np.pi/180.0)
    aA,fA = reference_frames(refA)
    aB,fB = reference_frames(refB)
    eA = 1.0-((1.0-fA)**2)
    eB = 1.0-((1.0-fB)**2)
    # A fram
    NA = aA/np.sqrt(1.0-((eA**2)*(np.sin(latAf)**2)))
    X = (NA+hA)*np.cos(latAf)*np.cos(lonAf)
    Y = (NA+hA)*np.cos(latAf)*np.sin(lonAf)
    Z = (((1.0-(eA**2))*NA)+hA)*np.sin(latAf)
    # B frame
    lonB = np.arctan(Y/X)*(180.0/np.pi)
    p = (((X**2)+(Y**2))/(aB**2))
    q = ((1.0-(eB**2))/(aB**2))*(Z**2)
    r = (p+q+(eB**4))/6.0
    s = (eB**4)*((p*q)/(4.0*(r**3)))
    t = (1.0+s+np.sqrt(s*(2.0+s)))**(1.0/3.0)
    u = r*(1.0+t+(1.0/t))
    v = np.sqrt((u**2)+((eB**4)*q))
    w = (eB**2)*((u+v-q)/(2.0*v))
    k = np.sqrt(u+v+(w**2))-w
    D = (k*np.sqrt((X**2)+(Y**2)))/(k+(eB**2))
    latB = (2.0*np.arctan(Z/(D+np.sqrt((D**2)+(Z**2)))))*(180.0/np.pi)
    hB = ((k+(eB**2)-1.0)/k)*np.sqrt((D**2)+(Z**2))
    return hB


def mss_model(lat,lon,MODEL,RETURN_COORD=False,mss_in=[],TP2WGS=True,IS2DATE=False,tm=[]):
    # mss = mss_model(lat,lon,MODEL,RETURN_COORD=False)
    # flat_sec,flon_sec,fmss_sec = mss_model(lat,lon,MODEL,RETURN_COORD=True)
    #lon,lat,mss_in = lon180_to_lon360(lon13[inn]),lat13[inn],ssh13_new[inn]
    #lat,lon,MODEL = lat2[inn],lon2[inn],'dtu21'
    #RETURN_COORD,mss_in,TP2WGS,IS2DATE = False,[],True,False
    N = np.shape(lat)[0]
    lon = lon180_to_lon360(lon)
    if MODEL=='cnes15':
        mean_year = 2003 #(spans 1993-2012)
        ds = Dataset('/Users/alexaputnam/External_models/mss_cnes_cls15.nc')
        mss_grid = ds['mss'][:][0,:,:] #(lon x lat) = (21600 x 9811)
        lat_grid,lon_grid = ds['NbLatitudes'][:],ds['NbLongitudes'][:]
    elif MODEL=='dtu21':
        mean_year = 2003 #(spans 1993-2012)
        ds = Dataset('/Users/alexaputnam/External_models/DTU21MSS_1min.mss.nc')
        mss_grid = ds['mss'][:].T #(lon x lat) = (21600 x 10800)
        lat_grid,lon_grid = ds['lat'][:],ds['lon'][:]
    # Section
    dlat,dlon = np.nanmax(np.diff(lat_grid))*2,np.nanmax(np.diff(lon_grid))*2
    LLMM = np.asarray([np.nanmin(lat)-dlat,np.nanmax(lat)+dlat,np.nanmin(lon)-dlon,np.nanmax(lon)+dlon])
    ilat = np.where((lat_grid>=LLMM[0])&(lat_grid<=LLMM[1]))[0]
    ilon = np.where((lon_grid>=LLMM[2])&(lon_grid<=LLMM[3]))[0]
    lat_sec,lon_sec = lat_grid[ilat],lon_grid[ilon]
    mss_seci = mss_grid[ilon[0]:ilon[-1]+1,ilat[0]:ilat[-1]+1]
    mlat_sec,mlon_sec = np.meshgrid(lat_sec,lon_sec)
    flon_sec,flat_sec = mlon_sec.flatten('F'),mlat_sec.flatten('F')
    fmss_seci = mss_seci.flatten('F')
    if TP2WGS==True:
        #mlat_grid,mlon_grid = np.meshgrid(lat_grid,lon_grid)
        #flon_grid,flat_grid = mlon_grid.flatten('F'),mlat_grid.flatten('F')
        #fmss_grid = mss_grid.flatten('F')
        #fmss_grid_wgs = np.empty(np.shape(fmss_grid))*np.nan
        fmss_sec = ellipsoid_transformation_2step(flat_sec,lon360_to_lon180(flon_sec),fmss_seci,'topex','wgs84')
        mss_sec = np.reshape(fmss_sec,np.shape(mss_seci),order='F')
        if IS2DATE ==True:
            dyear = 2021-mean_year #mean ICESat-2 year
            sl_rate = 3.4 #[mm/y] from CU Boulder sea level
            mss_sec = mss_sec+(dyear*sl_rate)
    else:
        fmss_sec = np.copy(fmss_seci)
        mss_sec = np.copy(mss_seci)
    if RETURN_COORD==False:
        f_mss = interpolate.interp2d(lat_sec, lon_sec, mss_sec, kind='linear')
        if np.size(np.shape(lat))==1:
            mss = np.empty(N)*np.nan
            for ii in np.arange(N):
                mss[ii] = f_mss(lat[ii],lon[ii])
        else:
            mss = np.empty((N,2))*np.nan
            for ii in np.arange(N):
                mss[ii,0] = f_mss(lat[ii,0],lon[ii,0])
                mss[ii,1] = f_mss(lat[ii,1],lon[ii,1])
        mss[mss>10000]=np.nan
        return mss
    else:
        if np.size(mss_in)==0:
            return flat_sec,lon360_to_lon180(flon_sec),fmss_sec
        else:
            #dLL = 0.01666667
            #dlat,dlon = dLL*1,dLL*1 #np.diff(mlat_sec)[0]*3,np.diff(mlon_sec)[0]*3
            #lat_sec_new,lon_sec_new = np.arange(LLMM[0],LLMM[1]+dlat,dlat),np.arange(LLMM[2],LLMM[3]+dlon,dlon)
            #mlat_sec_new,mlon_sec_new = np.meshgrid(lat_sec_new,lon_sec_new)
            mss_grid_new,mss_var,N_grid = mss_grid_est(lat,lon,mss_in,lat_sec,lon_sec,tm)
            '''
            plt.figure()
            plt.pcolormesh(mlon_sec_new, mlat_sec_new, mss_grid_new)
            plt.colorbar()
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            plt.title('Overlay: Scatter and Grid')
            plt.show()
            '''
            '''
            Z = griddata(points = (lon, lat),values = mss_in,xi = (mlon_sec, mlat_sec),method = 'cubic')
            f_mss = interpolate.interp2d(lat, lon, mss_in, kind='linear')
            mss = f_mss(flat_sec,flon_sec)
            '''
            return flat_sec,lon360_to_lon180(flon_sec),mss_grid_new.flatten('F'),mss_var.flatten('F'),N_grid.flatten('F')


def mss_grid_est(lat,lon,mss,lat_grid,lon_grid,tm):
    Nlat = np.size(np.unique(lat_grid))
    Nlon = np.size(np.unique(lon_grid))
    dlat = np.diff(lat_grid)[0]/2.0
    dlon = np.diff(lon_grid)[0]/2.0
    mss_grid = np.empty((Nlon,Nlat))*np.nan
    mss_var = np.empty((Nlon,Nlat))*np.nan
    N_grid = np.zeros((Nlon,Nlat))
    for ii in np.arange(Nlon):
        lonGi = lon_grid[ii]
        for jj in np.arange(Nlat):
            latGi = lat_grid[jj]
            idx = np.where((lat>=latGi-dlat)&(lat<latGi+dlat)&(lon>=lonGi-dlon)&(lon<lonGi+dlon))[0]
            if np.size(idx)>0:
                '''
                plt.figure(figsize=(8,8))
                plt.subplot(211)
                plt.hist(mss[idx],bins=30)
                plt.subplot(212)
                plt.plot(tm[idx],mss[idx],'.')
                '''
                mss_grid[ii,jj] = np.nanmean(mss[idx])
                mss_var[ii,jj] = np.nanvar(mss[idx])
                N_grid[ii,jj] = np.size(idx)
    return mss_grid,mss_var,N_grid

def t_mean(t_alt,h_alt,dt_days=30,N_MIN=50,PLOT=False):
    # t_bins,mn_h,vr_h = t_mean(t_alt,lat_alt,lon_alt,h_alt,dt_days=30,N_MIN=50)
    dti=dt_days/365.25
    t_bins = np.arange(2018,2024+dti,dti)#t_tg[t_tg>2018]#np.arange(2018,2024+dt,dt)
    Nt = np.size(t_bins)
    '''
    #Gridding
    LLMM = np.asarray([np.nanmin(lat_alt),np.nanmax(lat_alt),np.nanmin(lon_alt),np.nanmax(lon_alt)])
    lat_bindist,lon_bindist = 100,100
    lat_grid,lon_grid= make_grid(LLMM,lat_bindist,lon_bindist)
    mlon,mlat=np.meshgrid(lon_grid[:-1],lat_grid[:-1])
    flon,flat = mlon.flatten(),mlat.flatten()
    Nlat,Nlon = np.size(lat_grid),np.size(lon_grid)
    h_mn_grid = np.empty((Nlat-1,Nlon-1,Nt))*np.nan
    '''
    mn_h = np.empty(Nt)*np.nan
    vr_h = np.empty(Nt)*np.nan
    for tt in np.arange(Nt-1):
        idt = np.where((t_alt>=t_bins[tt])&(t_alt<t_bins[tt+1]))[0]
        if ~np.isnan(np.nanmean(h_alt[idt])):
            if np.size(h_alt[idt][~np.isnan(h_alt[idt])])>N_MIN:# and np.nanvar(h2m[idt])<0.1:
                    mn_h[tt] = np.nanmean(h_alt[idt])
                    vr_h[tt] = np.nanvar(h_alt[idt])
                    if PLOT==True:
                        plt.figure(figsize=(8,8))
                        plt.subplot(211)
                        plt.hist(h_alt[idt],bins=30)
                        plt.subplot(212)
                        plt.plot(t_alt[idt],h_alt[idt],'.')
    return t_bins,mn_h,vr_h

def radar_alt(ALT,LLMM=[-90,90,-180,180]):
    # tA,latA,lonA,hA=radar_alt(ALT,LLMM=[-90,90,-180,180])
    if ALT=='j2':
        # Jason-2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        dsA = Dataset('j3_nuuk.nc')
        idx = np.where((dsA['lat'][:]>=LLMM[0])&(dsA['lat'][:]<=LLMM[1])&(dsA['lon'][:]>=LLMM[2])&(dsA['lon'][:]<=LLMM[3]))
        tA = dsA['time'][:][idx]
        latA = dsA['lat'][:][idx]
        lonA = dsA['lon'][:][idx]
        hA = dsA['ssha'][:][idx]+dsA['mss'][:][idx]
    elif ALT=='c2':
        # CryoSat-2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        dsA = Dataset('c2_nuuk.nc')
        idx = np.where((dsA['lat'][:]>=LLMM[0])&(dsA['lat'][:]<=LLMM[1])&(dsA['lon'][:]>=LLMM[2])&(dsA['lon'][:]<=LLMM[3]))
        tA = dsA['time'][:][idx]
        latA = dsA['lat'][:][idx]
        lonA = dsA['lon'][:][idx]
        hA = dsA['ssha'][:][idx]+dsA['mss'][:][idx]
    elif ALT=='3a':
        # Sentinel-3A ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        dsA = Dataset('s3a_nuuk.nc')
        idx = np.where((dsA['lat'][:]>=LLMM[0])&(dsA['lat'][:]<=LLMM[1])&(dsA['lon'][:]>=LLMM[2])&(dsA['lon'][:]<=LLMM[3]))
        tA = dsA['time'][:][idx]
        latA = dsA['lat'][:][idx]
        lonA = dsA['lon'][:][idx]
        hA = dsA['ssha'][:][idx]+dsA['mss'][:][idx]
    return tA,latA,lonA,hA

def tide_estimation_tg(yf_IS,h_IS,lat_IS,lon_IS,yf_TG,h_TG,lat_TG,lon_TG,max_dt_sec=60,mxD=30):
    '''
    tide_TG = tide_estimation_tg(t13,sshDS13,lat13,lon13,yf_tg_anc,H_tg_anc,LL_anc[0],LL_anc[1],max_dt_sec=60)
    yf_IS,h_IS,lat_IS,lon_IS,yf_TG,h_TG,lat_TG,lon_TG,max_dt_sec = t13,sshDS13,lat13,lon13,yf_tg_anc,H_tg_anc,LL_anc[0],LL_anc[1],60
    t_delay_min=60
    max_dt_sec=2
    '''
    max_dt_sec=60
    sec_TG = yf_TG*(365.25*24.*60.*60.)
    dist = np.abs((geo2dist(lat_IS,lon_IS,lat_TG,lon_TG))/1000.)
    ikp = np.where((dist<=mxD)&(~np.isnan(h_IS)))[0]
    
    sec_IS = yf_IS[ikp]*(365.25*24.*60.*60.)
    isrt = np.argsort(sec_IS)
    csec_IS = np.copy(sec_IS)[isrt]
    arr_dt = np.abs(np.diff(csec_IS))
    it2 = np.where(arr_dt>max_dt_sec)[0]
    N = np.size(it2)
    t_moments = csec_IS[it2+1]
    yf_moments = np.empty(N)*np.nan
    tide_moments = np.empty(N)*np.nan
    tide_tg = np.empty(np.size(h_IS))*np.nan
    AidxIS = []
    AtidesIS = []
    AdistIS = []
    for ii in np.arange(N):
        dIS = np.abs(sec_IS-t_moments[ii])
        dTG = np.abs(sec_TG-t_moments[ii])
        idxIS = np.where(dIS<=max_dt_sec)[0]
        idxTG = np.where(dTG<=(2*max_dt_sec))[0]
        if np.size(idxTG)!=0:
            tide_moments[ii]=np.nanmean(h_TG[idxTG])
            #print('idxIS: '+str(np.size(idxIS)))
            if np.size(idxIS)>0:
                AidxIS=np.hstack((AidxIS,idxIS))
                AtidesIS=np.hstack((AtidesIS,np.ones(np.size(idxIS))*np.nanmean(h_TG[idxTG])))
                val=np.ones(np.size(idxIS))*np.nanmean(h_TG[idxTG])
                #print('idxIS: '+str(np.size(idxIS)))
                #print(val)
    
    if np.size(AidxIS)>0:
        AidxIS = AidxIS.astype(int)
        tide_tg[AidxIS]=AtidesIS
        sshi = h_IS-tide_tg
        ssh = filter_is2_ssh(sshi,yf_IS,MODE=False,VR_MAX=2,N_MIN=20)
    else:
        ssh = np.empty(np.size(h_IS))*np.nan
    return tide_tg,ssh

def comp_tg_is2(t_tg,dt_tg,H_tg,LLtg,H_is2,HDS_is2,t_is2,DT_is2,lat_is2,lon_is2,TG,mxD=20000):
    # t_tg,dt_tg,H_tg,LLtg,H_is2,HDS_is2,t_is2,DT_is2,lat_is2,lon_is2,TG=yf_tg_anc,dt_tg_anc,H_tg_anc,LL_anc,tideT13,tideDS13,t13,date13,lat13,lon13,'Anchorage'
    dist_is2 = geo2dist(lat_is2,lon_is2,LLtg[0],LLtg[1])
    iloc = np.where((np.abs(dist_is2)<=mxD)&(~np.isnan(HDS_is2)))[0]
    #tB,mnH_is2,vr_is2= t_mean(t_is2[iloc],H_is2[iloc],dt_days=dt_days2,N_MIN=5)
    #tB,mnHDS_is2,vrDS_is2= t_mean(t_is2[iloc],HDS_is2[iloc],dt_days=dt_days2,N_MIN=5)
    ##isrt = np.argsort(t_is2[iloc])
    ##cutc2 = np.copy(t_is2[iloc])[isrt]
    ##max_DT_is2 = .1/(365.25*24)#15.0/(365.25*24*60) == 0.1 = 6min
    ##max_dt_tg = .25/(365.25*24)
    ##arr_dt = np.abs(np.diff(t_is2[iloc]))
    ##it2 = np.where(arr_dt>max_DT_is2)[0]
    #t_moments = cutc2[it2+1]
    Nt = np.size(DT_is2)
    mnH_is2 = np.empty(Nt)*np.nan
    mnHDS_is2 = np.empty(Nt)*np.nan
    mnH_tg = np.empty(Nt)*np.nan
    vrH_is2 = np.empty(Nt)*np.nan
    vrHDS_is2 = np.empty(Nt)*np.nan
    vrH_tg = np.empty(Nt)*np.nan
    N_is2 = np.empty(Nt)*np.nan
    N_tg = np.empty(Nt)*np.nan
    dT_tg = np.empty(Nt)*np.nan
    yrTG = np.asarray([int(dt_tg[jj][:4]) for jj in np.arange(np.size(t_tg))])
    mnTG = np.asarray([int(dt_tg[jj][5:7]) for jj in np.arange(np.size(t_tg))])
    dyTG = np.asarray([int(dt_tg[jj][8:10]) for jj in np.arange(np.size(t_tg))])
    hrTG = np.asarray([int(dt_tg[jj][11:13]) for jj in np.arange(np.size(t_tg))])
    miTG = np.asarray([int(dt_tg[jj][14:16]) for jj in np.arange(np.size(t_tg))])
    H_is2_TGcor = np.empty(np.size(HDS_is2))
    tide_TG = np.empty(np.size(HDS_is2))
    yrI,mnI,dyI,hrI,miI,datI = convert_partial_year(t_is2)
    tt=0
    for ii in np.arange(Nt):
        dsi = DT_is2[ii]
        yri,mni,dyi=int(dsi[:4]),int(dsi[5:7]),int(dsi[8:10])
        hri,mii = int(dsi[11:13]),int(dsi[14:16])
        #dIS = np.abs(t_is2[iloc]-t_moments[ii])
        #dTG = np.abs(t_tg-t_moments[ii])
        dTG = np.abs(miTG-mii)
        iTG = np.where((yrTG==yri)&(mnTG==mni)&(dyTG==dyi)&(hrTG==hri)&(dTG<3))[0]  #&(miTG==mii)iTG =np.where(dTG<=max_DT_is2)[0]
        iIS = np.where((yrI[iloc]==yri)&(mnI[iloc]==mni)&(dyI[iloc]==dyi)&(hrI[iloc]==hri)&(miI[iloc]==mii))[0]  #np.where(dIS<=max_DT_is2)[0]
        scl=1
        '''
        while np.size(iTG)==0:
            iTG =np.where(dTG<=max_DT_is2*scl)[0]
            scl+=1
            if scl>15:
                iTG=np.arange(19)*np.nan
        '''
        if np.size(iTG)>0:# and scl<=5:
            mnH_tg[ii]=np.nanmean(H_tg[iTG])
            vrH_tg[ii]=np.nanvar(H_tg[iTG])
            N_tg[ii] = np.size(iTG)
            #dT_tg[ii] = max_DT_is2*scl            
        if np.size(iIS)>0:
            mnH_is2[ii]=np.nanmean(H_is2[iloc][iIS])
            vrH_is2[ii]=np.nanvar(H_is2[iloc][iIS])
            mnHDS_is2[ii]=np.nanmean(HDS_is2[iloc][iIS])
            vrHDS_is2[ii]=np.nanvar(HDS_is2[iloc][iIS])
            H_is2_TGcor[iloc][iIS] = (H_is2[iloc][iIS])-np.nanmean(H_tg[iTG])
            tide_TG[iloc][iIS] = mnH_tg[ii]
            N_is2[ii] = np.size(iIS)
    inn = np.where(~np.isnan(mnH_tg*mnH_is2*mnHDS_is2))[0]
    dtg_0 = mnH_is2-mnH_tg
    dtg_DS = mnHDS_is2-mnH_tg
    d0_DS = mnH_is2-mnHDS_is2
    stg0 = '\n FES-TG: $\mu \pm \sigma$ = '+str(np.round(np.nanmean(dtg_0),2))+' $\pm$ '+str(np.round(np.nanstd(dtg_0),2))+' m'
    stgDS = '\n Est-TG: $\mu \pm \sigma$ = '+str(np.round(np.nanmean(dtg_DS),2))+' $\pm$ '+str(np.round(np.nanstd(dtg_DS),2))+' m'
    s0DS = '\n FES-Est: $\mu \pm \sigma$ = '+str(np.round(np.nanmean(d0_DS),2))+' $\pm$ '+str(np.round(np.nanstd(d0_DS),2))+' m'
    t_moments = np.asarray([int(DT_is2[jj][:4])+((dt.datetime(int(DT_is2[jj][:4]),int(DT_is2[jj][5:7]),int(DT_is2[jj][8:10]),int(DT_is2[jj][11:13]),int(DT_is2[jj][14:16]))-dt.datetime(int(DT_is2[jj][:4]),1,1,0,0)).total_seconds()/(365.25*24*60*60)) for jj in np.arange(Nt)])
    plt.figure()
    plt.title(TG+stg0+stgDS+s0DS)
    plt.plot(t_moments[inn],mnH_tg[inn],'.-',color='black',label='tide gauge')
    plt.plot(t_moments[inn],mnH_is2[inn],'.-',color='red',label='FES 2014b OTC')
    plt.plot(t_moments[inn],mnHDS_is2[inn],'.-',label='estimated OTC')
    plt.xlabel('Time (year fraction)')
    plt.ylabel('Sea surface height [m]')
    plt.legend()

    ilocnan = np.where(~np.isnan(H_is2_TGcor*HDS_is2))[0]
    plt.figure()
    plt.title(TG+stg0+stgDS+s0DS)
    plt.plot(t_is2[ilocnan],H_is2[ilocnan],'.',color='black',label='no OTC')
    plt.plot(t_is2[ilocnan],HDS_is2[ilocnan],'.',color='red',label='estimated OTC)')
    plt.plot(t_is2[ilocnan],H_is2_TGcor[ilocnan],'.',color='green',label='tide gauge OTC')
    plt.xlabel('Time (year fraction)')
    plt.ylabel('Sea surface height [m]')
    plt.legend()

    plt.figure()
    binz= np.arange(10,10.2,0.2)
    plt.title(TG+stg0+stgDS+s0DS)
    plt.hist(H_is2[ilocnan]-H_is2_TGcor[ilocnan],color='black',label='no OTC-tide gauge',bins=binz)
    plt.hist(HDS_is2[ilocnan]-H_is2_TGcor[ilocnan],color='red',label='estimated-tide gauge OTC)',alpha=0.7,bins=binz)
    plt.ylabel('frequency')
    plt.xlabel('Sea surface height [m]')
    plt.legend()
    return t_moments,mnH_is2,mnHDS_is2,mnH_tg,vrH_is2,vrHDS_is2,vrH_tg,N_is2,N_tg,dT_tg,H_is2_TGcor
