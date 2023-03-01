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


def pull_is2_atl12_beams(fn_is2_12,BEAM=[],LLMM=[]):
    # fn_is2_03 = 'is2_atl03_nsidc_tst1_lat_30_60_lon_n170_n150.h5'
    ds = h5py.File(fn_is2_12, 'r')
    beam_list = ['gt1','gt2','gt3']
    sc_orient = ds['/orbit_info/sc_orient']
    sc_orient = sc_orient[0]
    # determine SV orientation
    if sc_orient == 1:
        ADD = ['r','l']
        beam_orient = [ii+'r' for ii in beam_list]
        beam_off = [ii+'l' for ii in beam_list]
    elif sc_orient == 0:
        ADD = ['l','r']
        beam_orient = [ii+'l' for ii in beam_list]
        beam_off = [ii+'r' for ii in beam_list]
    elif sc_orient == 2:
        raise('need sc_orient to be 0 (backward) or 1 (forward)')
    DR = 0
    if np.size(BEAM)==0:
        N = np.shape(beam_orient)[0]
        ds_atl12 = {}
        ds_atl12_off = {}
        for ii in np.arange(N):
            #ds_atl12[beam_list[ii]]={}
            ds_atl12 = pull_is2_atl12_var(ds,beam_orient[ii],LLMM,ds_var=ds_atl12)
            #ds_atl12_off = pull_is2_atl12_var(ds,beam_off[ii],LLMM,ds_var=ds_atl12_off,ORIENT=False)
            if np.size(ds_atl12[beam_orient[ii]]['lat'])>1:
                DR=DR+1
    else:
        ds_atl12 = {}
        ds_atl12_off = {}
        ds_atl12 = pull_is2_atl12_var(ds,BEAM+ADD[0],LLMM,ds_var=ds_atl12)
        #ds_atl12_off = pull_is2_atl12_var(ds,BEAM+ADD[1],LLMM,ds_var=ds_atl12_off,ORIENT=False)
    return ds_atl12,ds_atl12_off,ADD
    
def pull_is2_atl12_var(ds,beam,LLMM,ds_var={},ORIENT=True):
    '''
    beam=beam_list[ii]
    ds=ds_is2
    if np.size(ds_var) == 0:
        ds_var = {}
    '''
    ##### gtx/ssh_segments/
    print(beam)
    ds_var[beam]={}
    kys = ds['/'+beam+'/ssh_segments/'].keys()
    print(kys)
    kys
    # filter by region
    if np.size(LLMM)!=0:
        print('Min/Max latitude: '+str(LLMM[0])+' / '+str(LLMM[1]))
        print('Min/Max longitude: '+str(LLMM[2])+' / '+str(LLMM[3]))
        lat_ph = ds['/'+beam+'/ssh_segments/latitude'][:]
        lon_ph = ds['/'+beam+'/ssh_segments/longitude'][:]
        iph = np.where((lat_ph>=LLMM[0])&(lat_ph<=LLMM[1])&(lon_ph>=LLMM[2])&(lon_ph<=LLMM[3]))[0]
        print('% photon data reduction: '+str(np.round(100.*(1.0-((np.size(iph)*1.0)/(np.size(lat_ph)))))))
    else:
        iph = np.arange(np.shape(ds['/'+beam+'/ssh_segments/latitude'][:])[0])

    if np.size(iph)!=0:
        gps2utc = (dt.datetime(1985, 1, 1,0,0,0)-dt.datetime(1980, 1, 6,0,0,0)).total_seconds()
        ds_var[beam]['lat'] = ds['/'+beam+'/ssh_segments/latitude'][:][iph] # mean lat of surface photons in segment
        ds_var[beam]['lon'] = ds['/'+beam+'/ssh_segments/longitude'][:][iph] # mean lon of surface photons in segment
        ds_var[beam]['delta_time'] = ds['/'+beam+'/ssh_segments/delta_time'][:][iph] # mean time of surface photons in segment
        ds_var[beam]['time_gps'] = ds['/'+beam+'/ssh_segments/delta_time'][:][iph]+ds['/ancillary_data/atlas_sdp_gps_epoch'] # mean time of surface photons in segment
        ds_var[beam]['time_utc'] = ds_var[beam]['time_gps']-gps2utc-18
        #ds_var[beam]['epoch'] = ds['/ancillary_data/atlas_sdp_gps_epoch'][:] 
        ds_var[beam]['dtime'] = ds['/'+beam+'/ssh_segments/delt_seg'][:][iph] # mean time of surface photons in segment
        ##### gtx/ssh_segments/heights
        ds_var[beam]['bin_ssbias'] = ds['/'+beam+'/ssh_segments/heights/bin_ssbias'][:][iph] # sea state bias (correlation between photon return rate with avg surface height)
        ds_var[beam]['dxbar'] = ds['/'+beam+'/ssh_segments/heights/dxbar'][:] [iph]# mean distance between reflected photons
        ds_var[beam]['dxskew'] = ds['/'+beam+'/ssh_segments/heights/dxskew'][:][iph] # Skewness of the distribution of distance between surface reflected photons
        ds_var[beam]['dxvar'] = ds['/'+beam+'/ssh_segments/heights/dxvar'][:][iph] # variance of the distribution of distance between reflected photons
        ds_var[beam]['ssh'] = ds['/'+beam+'/ssh_segments/heights/h'][:][iph] # mean SSH relative to WGS84 ellipsoid
        ds_var[beam]['ssh_kurt'] = ds['/'+beam+'/ssh_segments/heights/h_kurtosis'][:][iph] # excess kurtosis of  ssh hist
        ds_var[beam]['ssh_skew'] = ds['/'+beam+'/ssh_segments/heights/h_skewness'][:][iph] # skewness of photon ssh hist
        ds_var[beam]['ssh_uncer'] = ds['/'+beam+'/ssh_segments/heights/h_uncrtn'][:][iph] # uncertainty in the mean ssh over an ocean segement
        ds_var[beam]['ssh_var'] = ds['/'+beam+'/ssh_segments/heights/h_var'][:][iph] # variance of best fit pdf (m^2)
        ds_var[beam]['length_seg'] = ds['/'+beam+'/ssh_segments/heights/length_seg'][:][iph] # length of segment
        # Significant wave height estimated as 4 times the standard deviation of along track 10-m bin averaged surface height
        ds_var[beam]['swh'] = ds['/'+beam+'/ssh_segments/heights/swh'][:][iph]
        ds_var[beam]['wl'] = 1.0/ds['/'+beam+'/ssh_segments/heights/wn'][:][iph] # wavelengths for each harmonic component in harmonic analysis of heights (5.3.3.2)
        ds_var[beam]['y'] = ds['/'+beam+'/ssh_segments/heights/y'][:][iph,:] # pdf of photon surface height
        ds_var[beam]['ykurt'] = ds['/'+beam+'/ssh_segments/heights/ykurt'][:][iph] # excess kurtosis = (4th moment of y)/(yvar^3)-3
        ds_var[beam]['ymean'] = ds['/'+beam+'/ssh_segments/heights/ymean'][:][iph] # mean = 1st moment of y (~0=h-meanoffit2)
        ds_var[beam]['yskew'] = ds['/'+beam+'/ssh_segments/heights/yskew'][:][iph] # skewness = (3rd moment of y)/(yvar^(3/2))
        ds_var[beam]['yvar'] = ds['/'+beam+'/ssh_segments/heights/yvar'][:][iph] # variance = 2nd moment of y
        ##### gtx/ssh_segments/stats
        ds_var[beam]['depth_ocn_seg'] = ds['/'+beam+'/ssh_segments/stats/depth_ocn_seg'][:][iph] # avg depth of ocean segment
        ds_var[beam]['n_photon'] = ds['/'+beam+'/ssh_segments/stats/n_photons'][:][iph] # number of surface photons found for the segment
        ds_var[beam]['n_ttl_photon'] = ds['/'+beam+'/ssh_segments/stats/n_ttl_photon'][:][iph] # number of photons in the +/- 15 m ocean downlink band
        ds_var[beam]['solar_az_sg'] = ds['/'+beam+'/ssh_segments/stats/solar_azimuth_seg'][:][iph] # I local ENU frame (angle = measured from north and positive towards east)
        ds_var[beam]['solar_el_sg'] = ds['/'+beam+'/ssh_segments/stats/solar_elevation_seg'][:][iph] # I local ENU frame (angle = measured from east-north plane and positive towards up)
        ds_var[beam]['ss_corr'] = ds['/'+beam+'/ssh_segments/stats/ss_corr'][:][iph] # subsurface scattering correction
        ds_var[beam]['ss_corr_stdev'] = ds['/'+beam+'/ssh_segments/stats/ss_corr_stdev'][:][iph] # estimated error for subsurface scattering correction
        ds_var[beam]['dac'] = ds['/'+beam+'/ssh_segments/stats/dac_seg'][:][iph] # dynamic atmospheric corrections (includes IB)
        ds_var[beam]['tide_ocean'] = ds['/'+beam+'/ssh_segments/stats/tide_ocean_seg'][:][iph] # ocean tides (includes diurnal and semi-diurnal harmomic analysis)

        ds_var[beam]['tide_pole'] = ds['/'+beam+'/ssh_segments/stats/tide_pole_seg'][:][iph] # ocean tides (includes diurnal and semi-diurnal harmomic analysis)
        ds_var[beam]['tide_load'] = ds['/'+beam+'/ssh_segments/stats/tide_load_seg'][:][iph] # ocean tides (includes diurnal and semi-diurnal harmomic analysis)
        ds_var[beam]['tide_solid'] = ds['/'+beam+'/ssh_segments/stats/tide_earth_seg'][:][iph] # ocean tides (includes diurnal and semi-diurnal harmomic analysis)
        ds_var[beam]['tide_oc_pole'] = ds['/'+beam+'/ssh_segments/stats/tide_oc_pole_seg'][:][iph] # ocean tides (includes diurnal and semi-diurnal harmomic analysis)
        ds_var[beam]['tide_earth_free2mean'] = ds['/'+beam+'/ssh_segments/stats/tide_earth_free2mean_seg'][:][iph] # ocean tides (includes diurnal and semi-diurnal harmomic analysis)
        ds_var[beam]['geoid'] = ds['/'+beam+'/ssh_segments/stats/geoid_seg'][:][iph] # 
        ds_var[beam]['geoid_free2mean'] = ds['/'+beam+'/ssh_segments/stats/geoid_free2mean_seg'][:][iph]
        ##### 10-m bins
        ds_var[beam]['hty_10'] = ds['/'+beam+'/ssh_segments/heights/htybin'][:][iph].flatten('C') # 10-m detrended SSH
        ds_var[beam]['lat_10'] = ds['/'+beam+'/ssh_segments/heights/latbind'][:][iph].flatten('C') # latitude of 10 m segemnts
        ds_var[beam]['lon_10'] = ds['/'+beam+'/ssh_segments/heights/lonbind'][:][iph].flatten('C') # longitude of  of 10 m segemnts
        ds_var[beam]['xbind_10'] = ds['/'+beam+'/ssh_segments/heights/xbind'][:][iph].flatten('C') # 10-m avg of along-track distance
        ds_var[beam]['xrbin_10'] = ds['/'+beam+'/ssh_segments/heights/xrbin'][:][iph].flatten('C') # 10-m photon data rate per meter
        # The percentages of each surf_type of the photons in the ocean segment as a 5-element variable with each element corresponding to the percentage of photons from each of the 5 surface types
        ds_var[beam]['surf_type_prcnt'] = ds['/'+beam+'/ssh_segments/stats/surf_type_prcnt'][:][iph,:]
        ds_var[beam]['orbit_number'] = ds['/'+beam+'/ssh_segments/stats/orbit_number'][:][iph]
        ds_var[beam]['photon_rate'] = ds['/'+beam+'/ssh_segments/stats/photon_rate'][:][iph]
        ds_var[beam]['photon_noise_rate'] = ds['/'+beam+'/ssh_segments/stats/photon_noise_rate'][:][iph]
        ds_var[beam]['xrbin'] = ds['/'+beam+'/ssh_segments/heights/xrbin'][:][iph,:]
        ds_var[beam]['dot'] = ds_var[beam]['ssh']-(ds_var[beam]['geoid']+ds_var[beam]['geoid_free2mean'])-ds_var[beam]['tide_earth_free2mean']
        # determine ascending and descending passes
        #dlat = np.diff(ds_var[beam]['lat'])
        #ad_is2_pre = np.copy(dlat)
        #ad_is2_pre[ad_is2_pre<0]=0
        #ad_is2_pre[ad_is2_pre>0]=1
        #ds_var[beam]['pass_0D_1A'] = np.insert(ad_is2_pre,0,ad_is2_pre[0]) 
        # ds_var[beam]['ssha_ph'] = ds_var[beam]['h']-ds_var[beam]['tide_ocean_ph']-ds_var[beam]['dac_ph']-ds_var[beam]['tide_equilibrium_ph']-ds_var[beam]['dem_ph']#-ds_var[beam]['tropo_ph']
    else:
        ds_var[beam]['lat'] = 99999
    return ds_var




def gps2utc(gps_time):
    '''
    Converts GPS time that ICESat-2 references to UTC
    gps_time = time_gps
    '''
    t0 = datetime.datetime(1980,1,6,0,0,0,0)
    leap_seconds = -18 #applicable to everything after 2017-01-01, UTC is currently 18 s behind GPS
    dt = (gps_time + leap_seconds) * datetime.timedelta(seconds=1)
    utc_time = t0+dt
    utc_time_str = np.asarray([str(x) for x in utc_time])
    return utc_time_str

def utc2utc_stamp(time_utc):
    gps2utc2 = (datetime.datetime(1985, 1, 1,0,0,0)-datetime.datetime(1980, 1, 6,0,0,0)).total_seconds()
    gps_time = time_utc+gps2utc2+18
    t0 = datetime.datetime(1980,1,6,0,0,0,0)
    leap_seconds = -18 #applicable to everything after 2017-01-01, UTC is currently 18 s behind GPS
    dt = (gps_time + leap_seconds) * datetime.timedelta(seconds=1)
    utc_time = t0+dt
    utc_time_str = np.asarray([str(x) for x in utc_time])
    return utc_time_str


f12 = '/Users/alexaputnam/ICESat2/atlcu_v_atl12/ATL12_20220314200251_12611401_005_01.h5'
f3s = '/Users/alexaputnam/ICESat2/atlcu_v_atl12/ATL03_20220314230424_12621414_005_01_filtered_on_diego_garcia.npy'
f3w = '/Users/alexaputnam/ICESat2/atlcu_v_atl12/ATL03_20220314230424_12621414_005_01_filtered_off_diego_garcia.npy'
f3 = '/Users/alexaputnam/ICESat2/atlcu_v_atl12/reg_atl03_lat_n8_lon_72_diego_garcia_segs_2_100_2000_2022_03_to_2022_03_MSS.npz'
f3m = '/Users/alexaputnam/ICESat2/atlcu_v_atl12/reg_atl03_lat_n8_lon_72_diego_garcia_segs_2_100_2000_2022_03_to_2022_03.npz'

#d3s = np.load(f3s,allow_pickle='TRUE', encoding='bytes').item()
#beams3s = d3s.keys()
d12 = h5py.File(f12, 'r')
ibms = 'gt1l'
time_gps = d12['/'+ibms+'/ssh_segments/delta_time'][:]+d12['/ancillary_data/atlas_sdp_gps_epoch'] # mean time of surface photons in segment
gps2utc2 = (dt.datetime(1985, 1, 1,0,0,0)-dt.datetime(1980, 1, 6,0,0,0)).total_seconds()
time_utc = time_gps-gps2utc2-18
time_utc=utc2utc_stamp(time_utc)
time_utc2 = gps2utc(time_gps) # datetime.datetime.strptime(time_utc2[0], '%Y-%m-%d %H:%M:%S.%f')
lat_12 = d12['/'+ibms+'/ssh_segments/latitude'][:] # mean lat of surface photons in segment
lon_12 = d12['/'+ibms+'/ssh_segments/longitude'][:]
N = 100

t1 = time.time()
f14_12 = ot.ocean_tide_replacement(lon_12[:N],lat_12[:N],time_utc2[:N])
print('total time for '+str(N)+' points: '+str(np.round(time.time()-t1)/60)+' min')

# check: (datetime.datetime.strptime(time_utc2[0], '%Y-%m-%d %H:%M:%S.%f')-(dt.datetime(1985, 1, 1,0,0,0))).total_seconds()
d3 = np.load(f3)
d3m = np.load(f3m)
kys3 = list(d3.keys())
ssha2 = d3['sshaS']#ds2['ssha'+ATCH]# ds2['ssha'+ATCH+'_md'] #
time2 = d3['timeS']
lon2 = d3['lonS']
lat2= d3['latS']
ssha100 = d3['ssha']#ds2['ssha'+ATCH]# ds2['ssha'+ATCH+'_md'] #
#sshha100dem = 
ssha_fft100 = d3['ssha_fft']
time100 = d3['time']
lon100 = d3['lon']
lat100= d3['lat']
dem100= d3['dem']
mss100= d3['mss']
ssha100m = d3m['ssha']#ds2['ssha'+ATCH]# ds2['ssha'+ATCH+'_md'] #
ssha_fft100m = d3m['ssha_fft']
time100m = d3m['time']
lon100m = d3m['lon']
lat100m= d3m['lat']
LLMM = [np.nanmin(lat2),np.nanmax(lat2),np.nanmin(lon2),np.nanmax(lon2)]

f14_100 = ot.ocean_tide_replacement(lon100,lat100,stime100)


plt.figure()
plt.plot(time100-time100[0],dem100,'.')
plt.plot(time100-time100[0],mss100,'.')

mrk = '.'
plt.figure()
plt.plot((time100-time100[0])*7000.,ssha100,mrk)
plt.plot((time100m-time100[0])*7000.,ssha100m,mrk)
plt.xlim(0,2000)

plt.figure()
plt.plot((time100-time100[0])*7000.,ssha_fft100,mrk)
plt.plot((time100m-time100[0])*7000.,ssha_fft100m,mrk)
plt.xlim(0,2000)

'''
d12 = h5py.File(f12, 'r')
d12_on,d12_off,ADD12 = pull_is2_atl12_beams(f12,BEAM=[],LLMM=LLMM)
bms12 = d12_on.keys()
ibms = 'gt1l'
time7k = d12_on[ibms]['time_utc']
lat7k = d12_on[ibms]['lat']
lon7k = d12_on[ibms]['lon']
ssha7k = d12_on[ibms]['ssh']
ssha10 = d12_on[ibms]['hty_10']
lat10 = d12_on[ibms]['lat_10']
lon10 = d12_on[ibms]['lon_10']
#ssha10 = d12_on[ibms]['xbind_10']

plt.figure()
plt.plot(lat10[ssha10<500],ssha10[ssha10<500],',')
#plt.plot(lat7k[ssha7k<500],ssha7k[ssha7k<500])
plt.plot(lat100,ssha100,',')
plt.plot(lat100,ssha_fft100,',')

'''



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

def data_stats(x):
    # mod,med,mn,skw = data_stats(x)
    mod = [np.nan]#mod,cnt = stats.mode(hist)
    med = np.nanmedian(x)
    mn = np.nanmean(x)
    skw = skew(x)
    var = np.nanvar(x)
    return mod[0],med,mn,skw,var
    
def fit_filter(t,x,Zmax=4):
    ce = np.polyfit(t[~np.isnan(x)], x[~np.isnan(x)], 1)
    fit = ce[1]+(ce[0]*t)
    anom = fit-x
    mn = np.nanmean(anom)
    sd = np.nanstd(anom)
    print('mean = '+str(mn))
    print('std = '+str(sd))
    Z = ((anom-mn)/sd)
    ikp = np.where((anom>=(-sd*Zmax))&(anom<=(sd*Zmax)))[0] #np.abs(Z)<=Zmax)[0]
    return ikp

def major_filter(t,x):
    ce = np.polyfit(t, x, 1)[::-1]
    sd = np.nanstd(x)
    mn_diff = np.nanmean(np.sqrt(np.diff(x)**2))
    fit = ce[0]+(ce[1]*t)
    #anom = fit-x
    if sd>0.5: # if 68.2% of my data deviates by more than 0.5 m from the mean
        DITCH = True
    elif np.abs(ce[1])>=0.01 and np.abs(fit[-1]-fit[0])>=2.0: # 1 cm/m threshold (2 m ssha trend over 200 m)
        DITCH = True
    elif mn_diff>0.5: # if the mean difference between measurements is > 1/2 m
        DITCH = True
    else:
        DITCH = False
    return DITCH

def iterative_outlier_function(x_in,zval):
    # x_new,iout,ikp = iterative_outlier_function(x_in,zval)
    outliers = 1.
    x_keep = np.copy(x_in)
    x_rmv = np.ones(np.shape(x_in))
    i=0
    while outliers > 0:
        mn = np.mean(x_keep[~np.isnan(x_keep)])
        sig = np.std(x_keep[~np.isnan(x_keep)])
        sig5 = zval*sig
        ciL,ciU= mn-sig5,mn+sig5
        iRmL = np.where(x_keep[:]<ciL)[0] #&(x_keep[:]>ciU))[0]
        iRmU = np.where(x_keep[:]>ciU)[0]
        iRm = np.append(iRmL,iRmU)
        outliers = np.size(iRm)
        i += 1
        if outliers > 0:
            x_keep[iRm] = np.empty(outliers)*np.nan
            x_rmv[iRm] = np.zeros(outliers)
    ix = np.where(~np.isnan(x_keep))[0]
    iout = np.where(np.isnan(x_keep))[0]
    ikpi = np.where(x_rmv==1)[0]
    ikp = np.intersect1d(ix,ikpi)
    x_new = x_keep[ikp]
    Nkp = np.shape(ikp)[0]
    Nout = np.shape(iout)[0]
    Nin = np.shape(x_in)[0]
    if Nin != Nkp+Nout:
        raise('Nin != Nkp+Nout in iterative_outlier_function')
    return x_new,iout,ikp

def outlier_conv(tph,sshaph,Zmax=5.0):
    kernel_size = 3
    kernel = np.ones(kernel_size) / kernel_size
    tph_conv = np.convolve(tph[~np.isnan(sshaph)], kernel, mode='same')
    sshaph_conv = np.convolve(sshaph[~np.isnan(sshaph)], kernel, mode='same')
    std_sshaph_conv = np.nanstd(sshaph_conv)
    m2, b2 = np.polyfit(tph_conv, sshaph_conv, 1)
    fitph2 = (m2*tph)+b2
    res = fitph2-sshaph
    ikp = np.where((res>=-Zmax*std_sshaph_conv)&(res<=Zmax*std_sshaph_conv))[0]
    return ikp

def atl03_filter(ds_var,beam,ORIENT):
    # STEP-1
    #####  Indices
    # flag_signal_conf_ph_ocean: photon confidence row=1 is ocean (0=noise, 1=background, 2=low, 3=med, 4=high)
    t1=time.time()
    if ORIENT is True:
        ci = 2 #1
    else:
        ci = 1#2
    idx_h = np.where((ds_var[beam]['flag_signal_conf_ph_ocean']>=ci)&(ds_var[beam]['flag_quality_ph']==0))[0]#np.where(ds_var[beam]['flag_quality_ph']==0)[0]#
    ##### New filter
    kys = ['tropo','dac','dem_h','tide_ocean']#['tropo','neutat_ht','dac','tide_ocean','tide_earth','tide_load','tide_pole','dem_h']
    Nk = np.shape(kys)[0]
    idx_k=np.arange(np.shape(ds_var[beam]['dac'])[0])
    for ii in np.arange(Nk):
        idx_ki = np.where(np.abs(ds_var[beam][kys[ii]])<1000)[0]
        idx_k = np.intersect1d(idx_k,idx_ki)#np.unique(np.append(idx_k,idx_ki))
    ##### Photon rate # h has been corrected for: solid earth tide, solid earth pole tide, ocean pole tide, ocean loading and atmospheric range delays
    ds_var[beam]['ph_nan'] = np.empty(np.shape(ds_var[beam]['h']))*np.nan
    ds_var[beam]['ph_nan'][idx_h] = np.zeros(np.shape(idx_h))
    Np = np.shape(ds_var[beam]['ph_nan'])[0]
    ##### Segmented rate
    ##### segment (Section 3.2 pg.40/215)
    ds_var[beam]['flag_dem_ph'] = np.interp(ds_var[beam]['delta_time_ph'],ds_var[beam]['delta_time_ref'],ds_var[beam]['flag_dem'])
    ds_var[beam]['flag_pod_ph'] = np.interp(ds_var[beam]['delta_time_ph'],ds_var[beam]['delta_time_ref'],ds_var[beam]['flag_pod'])
    ds_var[beam]['flag_surf_type_ocean_ph'] = np.interp(ds_var[beam]['delta_time_ph'],ds_var[beam]['delta_time_ref'],ds_var[beam]['flag_surf_type_ocean'])
    # Make sure that we aren't interpolating invalid values of tropo,dac,dem_h and tide_ocean
    # Apply reference flags to photons: idx_flag_seg = np.where((ds_var[beam]['flag_dem']==3)&(ds_var[beam]['flag_pod']==0)&(ds_var[beam]['flag_surf_type_ocean']==1))[0]
    idx_ref2ph = np.where((ds_var[beam]['flag_dem_ph']>=2.75)&(ds_var[beam]['flag_pod_ph']<=0.25)&(ds_var[beam]['flag_surf_type_ocean_ph']>=0.75))[0] #&(idx_k_10_ph<0.5))[0]
    ds_var[beam]['ref2ph_nan'] = np.empty(Np)*np.nan
    ds_var[beam]['ref2ph_nan'][idx_ref2ph] = np.zeros(np.shape(idx_ref2ph)[0])
    ds_var[beam]['ref_ph_nan'] = ds_var[beam]['ref2ph_nan']+ds_var[beam]['ph_nan']
    print('seconds to atl03_filter :'+str(time.time()-t1))
    return ds_var

def atl03_time(ds_var,beam):
    # STEP-2
    #---------------------------------------------------------------------#
    ##### gtx/heights/ (section 10.2.1. pg. 198/215)
    t1=time.time()
    gps2utc = (dt.datetime(1985, 1, 1,0,0,0)-dt.datetime(1980, 1, 6,0,0,0)).total_seconds()
    ds_var[beam]['time_gps_ph'] = ds_var[beam]['delta_time_ph'] + ds_var[beam]['atlas_sdp_gps_epoch']
    ds_var[beam]['time_utc_ph'] = ds_var[beam]['time_gps_ph']-gps2utc-18
    print('seconds to alt03_time :'+str(time.time()-t1))
    return ds_var

def atl03_ssha(ds_var,beam):
    #STEP-3
    #---------------------------------------------------------------------#
    t1=time.time()
    inn = np.where(np.abs(ds_var[beam]['tide_ocean'])<1000)[0]
    if np.size(inn)>0:
        ds_var[beam]['tide_ocean_ph'] = np.interp(ds_var[beam]['delta_time_ph'],ds_var[beam]['delta_time_ref'][inn],ds_var[beam]['tide_ocean'][inn])
    else:
        ds_var[beam]['tide_ocean_ph'] = np.empty(np.shape(ds_var[beam]['delta_time_ph']))*np.nan
    inn = np.where(np.abs(ds_var[beam]['dac'])<1000)[0]
    if np.size(inn)!=0:
        ds_var[beam]['dac_ph'] = np.interp(ds_var[beam]['delta_time_ph'],ds_var[beam]['delta_time_ref'][inn],ds_var[beam]['dac'][inn])
    else:
        ds_var[beam]['dac_ph'] = np.empty(np.shape(ds_var[beam]['delta_time_ph']))*np.nan
    inn = np.where(np.abs(ds_var[beam]['dem_h'])<1000)[0]
    if np.size(inn)!=0:
        ds_var[beam]['dem_ph'] = np.interp(ds_var[beam]['delta_time_ph'],ds_var[beam]['delta_time_ref'][inn],ds_var[beam]['dem_h'][inn])
    else:
        ds_var[beam]['dem_ph'] = np.empty(np.shape(ds_var[beam]['delta_time_ph']))*np.nan
    inn = np.where(np.abs(ds_var[beam]['tropo'])<1000)[0]
    if np.size(inn)!=0:
        ds_var[beam]['tropo_ph'] = np.interp(ds_var[beam]['delta_time_ph'],ds_var[beam]['delta_time_ref'][inn],ds_var[beam]['tropo'][inn])
    else:
        ds_var[beam]['tropo_ph'] = np.empty(np.shape(ds_var[beam]['delta_time_ph']))*np.nan
    inn = np.where(np.abs(ds_var[beam]['geoid'])<1000)[0]
    if np.size(inn)!=0:
        ds_var[beam]['geoid_ph'] = np.interp(ds_var[beam]['delta_time_ph'],ds_var[beam]['delta_time_ref'][inn],ds_var[beam]['geoid'][inn])
    else:
        ds_var[beam]['geoid_ph'] = np.empty(np.shape(ds_var[beam]['delta_time_ph']))*np.nan
    inn = np.where(np.abs(ds_var[beam]['geoid_free2mean'])<1000)[0]
    if np.size(inn)!=0:
        ds_var[beam]['geoid_free2mean_ph'] = np.interp(ds_var[beam]['delta_time_ph'],ds_var[beam]['delta_time_ref'][inn],ds_var[beam]['geoid_free2mean'][inn])
    else:
        ds_var[beam]['geoid_free2mean_ph'] = np.empty(np.shape(ds_var[beam]['delta_time_ph']))*np.nan
    inn = np.where(np.abs(ds_var[beam]['tide_equilibrium'])<1000)[0]
    if np.size(inn)!=0:
        ds_var[beam]['tide_equilibrium_ph'] = np.interp(ds_var[beam]['delta_time_ph'],ds_var[beam]['delta_time_ref'][inn],ds_var[beam]['tide_equilibrium'][inn])
    else:
        ds_var[beam]['tide_equilibrium_ph'] = np.empty(np.shape(ds_var[beam]['delta_time_ph']))*np.nan

    ds_var[beam]['ssha_ph'] = ds_var[beam]['h']-ds_var[beam]['tide_ocean_ph']-ds_var[beam]['dac_ph']-ds_var[beam]['tide_equilibrium_ph']-ds_var[beam]['dem_ph']#-ds_var[beam]['tropo_ph']
    #---------------------------------------------------------------------#
    # outlier detector
    #---------------------------------------------------------------------#
    MAX_SSHA = 10 # match common radar altimetry limit
    #x_new,iout,ikp1 = iterative_outlier_function(ds_var[beam]['ssha_ph']+ds_var[beam]['ref_ph_nan'],zval) # used to be ds_var[beam]['ph_nan']
    ssha_n = ds_var[beam]['ssha_ph']+ds_var[beam]['ref_ph_nan']
    ikp1 = np.where(np.abs(ssha_n)<=MAX_SSHA)[0]
    print('size(no nan and |ssha|<= '+str(MAX_SSHA)+'): '+str(np.size(ikp1)))
    if np.size(ikp1)!=0:
        #x_new,iout,ikp2 = iterative_outlier_function(ssha_n[ikp1],zval) # used to be ds_var[beam]['ph_nan'] izs = fit_filter(ds_var[beam]['time_utc_ph'][ikp1],ssha_n[ikp1],Zmax=3)
        
        ikp2=outlier_conv(ds_var[beam]['delta_time_ph'][ikp1],ssha_n[ikp1],Zmax=5.0)
        ikp = np.copy(ikp1[ikp2])
        ds_var[beam]['flag_ph_outlier'] = np.empty(np.shape(ds_var[beam]['ph_nan']))*np.nan
        ds_var[beam]['flag_ph_outlier'][ikp] = np.zeros(np.shape(ikp))
        ds_var[beam]['ref_ph_nan'] = ds_var[beam]['ref_ph_nan']+ds_var[beam]['flag_ph_outlier']
    else:
        ds_var[beam]['flag_ph_outlier'] = np.empty(np.shape(ds_var[beam]['ph_nan']))*np.nan
        ds_var[beam]['ref_ph_nan'] = ds_var[beam]['ref_ph_nan']+ds_var[beam]['flag_ph_outlier']
    #---------------------------------------------------------------------#
    print('seconds to alt03_ssha :'+str(time.time()-t1))
    return ds_var

def land_or_not(lat,lon,ssha,scl=3):
    std = np.nanstd(ssha)
    mn = np.nanmean(ssha)
    limU = mn+(scl*std)
    limL = mn-(scl*std)
    N = np.size(lat)
    mask =  np.zeros(N)
    for ii in np.arange(N):
        TF = globe.is_land(lat[ii],lon[ii])
        if TF == True:
            if ssha[ii]>limL:
                mask[ii] = 1
            if ssha[ii]<limU:
                mask[ii] = 1
    return mask

def atlcu_segment(ds_var,beam,seg_foot=2.):
    t1=time.time()
    # set parameters
    # seg_foot = segement footprint in [m]
    # call key variables
    dist_n = ds_var[beam]['dist_ph']
    ssha_n = ds_var[beam]['ssha_ph']+ds_var[beam]['ref_ph_nan'] #ds_var[beam]['ph_nan']
    # reduce data amount by removing NaNs and outliers
    innzs = np.where((~np.isnan(ssha_n)))[0]
    print('size inn: '+str(np.size(innzs)))
    #imsk = land_or_not(ds_var[beam]['lat'][inn],ds_var[beam]['lon'][inn],ssha_n[inn],scl=3)
    SF = str(int(seg_foot))
    ds_var[beam]['filter_'+SF+'_ph'] = np.empty(np.shape(dist_n))*np.nan
    if np.size(innzs)!=0:
        dist = np.copy(dist_n[innzs])
        ssha = np.copy(ssha_n[innzs])
        dTime=ds_var[beam]['delta_time_ph'][innzs]
        time_utc=ds_var[beam]['time_utc_ph'][innzs]
        lat=ds_var[beam]['lat'][innzs]
        lon=ds_var[beam]['lon'][innzs]
        h_ph = ds_var[beam]['h'][innzs]
        tide_ocean = ds_var[beam]['tide_ocean_ph'][innzs]
        dac = ds_var[beam]['dac_ph'][innzs]
        tide_eq = ds_var[beam]['tide_equilibrium_ph'][innzs]
        dem = ds_var[beam]['dem_ph'][innzs]
        geoid = ds_var[beam]['geoid_ph'][innzs]
        geoid_f2m = ds_var[beam]['geoid_free2mean_ph'][innzs]
        # define segments
        SV_vel = 7000. # ICESat-2 vel = 7km/s
        dT_bin = np.round(seg_foot/SV_vel,4) # segment_footprint/7000.
        dT = ds_var[beam]['delta_time_ph'][innzs]-ds_var[beam]['delta_time_ph'][innzs][0]
        aT = np.arange(dT[0],dT[-1]+dT_bin,dT_bin)
        NT = np.shape(aT)[0]
        # Declare files to save
        ds_var[beam]['swh_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['N_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['i_beg_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['i_end_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['dist_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['ssha_mn_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['ssha_md_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['ssha_vr_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['ssha_sk_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['dT_beg_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['dT_end_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['time_utc_beg_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['time_utc_end_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['lat_beg_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['lon_beg_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['lat_end_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['lon_end_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['dT_mean_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['time_utc_mean_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['lat_mean_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['lon_mean_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['h_mean_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['tide_ocean_mean_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['dac_mean_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['tide_equilibrium_mean_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['dem_mean_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['geoid_mean_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['geoid_free2mean_mean_cu'+SF] = np.empty(NT)*np.nan
        ds_var[beam]['filter_'+SF+'_ph'][innzs] = np.zeros(np.size(innzs))
        for ii in np.arange(NT-1):
            # segment index
            iseg = np.where((dT>=aT[ii])&(dT<aT[ii+1]))[0]
            if ii%10000==0:
                print('time '+str(ii)+': '+str(time.time()-t1))
            if np.size(iseg)>=1:
                ds_var[beam]['filter_'+SF+'_ph'][iseg] = np.ones(np.size(iseg))
                ssha_ss = ssha[iseg]
                mod,med,mn,skw,var = data_stats(ssha_ss)
                ds_var[beam]['swh_cu'+SF][ii] = 4.0*np.sqrt(var)
                ds_var[beam]['dT_beg_cu'+SF][ii] = dTime[iseg[0]]
                ds_var[beam]['time_utc_beg_cu'+SF][ii] = time_utc[iseg[0]]
                ds_var[beam]['dT_end_cu'+SF][ii] = dTime[iseg[-1]]
                ds_var[beam]['time_utc_end_cu'+SF][ii] = time_utc[iseg[-1]]
                ds_var[beam]['lat_beg_cu'+SF][ii] = lat[iseg[0]]
                ds_var[beam]['lon_beg_cu'+SF][ii] = lon[iseg[0]]
                ds_var[beam]['lat_end_cu'+SF][ii] = lat[iseg[-1]]
                ds_var[beam]['lon_end_cu'+SF][ii] = lon[iseg[-1]]
                ds_var[beam]['N_cu'+SF][ii] = np.size(iseg)
                ds_var[beam]['dist_cu'+SF][ii] = np.abs(dist[iseg[-1]]-dist[iseg[0]])
                ds_var[beam]['i_beg_cu'+SF][ii] = innzs[iseg][0]
                ds_var[beam]['i_end_cu'+SF][ii] = innzs[iseg][-1]
                ds_var[beam]['ssha_md_cu'+SF][ii] = med
                ds_var[beam]['ssha_mn_cu'+SF][ii] = mn
                ds_var[beam]['ssha_sk_cu'+SF][ii] = skw
                ds_var[beam]['ssha_vr_cu'+SF][ii] = var
                ds_var[beam]['dT_mean_cu'+SF][ii] = np.nanmean(dTime[iseg])
                ds_var[beam]['time_utc_mean_cu'+SF][ii] = np.nanmean(time_utc[iseg])
                ds_var[beam]['lat_mean_cu'+SF][ii] = np.nanmean(lat[iseg])
                if np.size(iseg)==1:
                    ds_var[beam]['lon_mean_cu'+SF][ii] = lon[iseg]
                else:
                    if np.nanmax(np.abs(np.diff(lon[iseg])))>=90:
                        ds_var[beam]['lon_mean_cu'+SF][ii] = np.nanmedian(lon[iseg])
                    else:
                        ds_var[beam]['lon_mean_cu'+SF][ii] = np.nanmean(lon[iseg])
                ds_var[beam]['h_mean_cu'+SF][ii] = np.nanmean(h_ph[iseg])
                ds_var[beam]['tide_ocean_mean_cu'+SF][ii] = np.nanmean(tide_ocean[iseg])
                ds_var[beam]['dac_mean_cu'+SF][ii] = np.nanmean(dac[iseg])
                ds_var[beam]['tide_equilibrium_mean_cu'+SF][ii] = np.nanmean(tide_eq[iseg])
                ds_var[beam]['dem_mean_cu'+SF][ii] = np.nanmean(dem[iseg])
                ds_var[beam]['geoid_mean_cu'+SF][ii] = np.nanmean(geoid[iseg])
                ds_var[beam]['geoid_free2mean_mean_cu'+SF][ii] = np.nanmean(geoid_f2m[iseg])
    # turn to array format and save
    print('Time to create '+SF+' m segments: '+str(time.time()-t1))
    return ds_var

def atlcu_segment_major(ds_var,beam,seg_foot=2.0,segmaj_foot=2000.0,segmaj_foot2=100.0):    
    t1 = time.time()
    # set parameter
    #seg_foot = 500. # segment footprint = 5000 m
    # define segment
    SV_vel = 7000. # ICESat-2 vel = 7km/s
    SF = str(int(seg_foot))
    SF2 = str(int(segmaj_foot))
    SF3 = str(int(segmaj_foot2))
    # define bin size (in time)
    dT_maj = np.round(segmaj_foot/SV_vel,4) # 0.0029 s segments = 20 m bin segments
    dT_maj2 = np.round(segmaj_foot2/SV_vel,4)
    # define new variables
    kys_new = ['N_cu','i_beg_cu','i_end_cu','dist_cu','ssha_mn_cu','ssha_md_cu','ssha_vr_cu','ssha_sk_cu','swh_cu','dT_beg_cu','slope_cu','time_utc_beg_cu','dT_end_cu','time_utc_end_cu','lat_beg_cu','lon_beg_cu','lat_end_cu',
               'lon_end_cu','dT_mean_cu','time_utc_mean_cu','lat_mean_cu','lon_mean_cu','h_mean_cu','tide_ocean_mean_cu','dac_mean_cu','tide_equilibrium_mean_cu','dem_mean_cu','geoid_mean_cu','geoid_free2mean_mean_cu']
    Nky = np.shape(kys_new)[0]
    for ii in np.arange(Nky):
        ds_var[beam][kys_new[ii]+SF2]=[]
        ds_var[beam][kys_new[ii]+SF3]=[]
    # remove outliers
    ssha_n = ds_var[beam]['ssha_mn_cu'+SF]
    inn=np.where(~np.isnan(ssha_n))[0]
    if np.size(inn)>100: #(segmaj_foot/2.0):
        ds_var[beam]['filter_'+SF2+'_'+SF] = np.empty(np.shape(ds_var[beam]['dT_beg_cu'+SF]))*np.nan
        ds_var[beam]['filter_'+SF2+'_'+SF][inn] = np.zeros(np.size(inn))
        print('Size of dT_beg_cu'+SF+': '+str(np.size(inn)))
        dTbeg = ds_var[beam]['dT_beg_cu'+SF][inn]-ds_var[beam]['delta_time_ph'][0]
        dTend = ds_var[beam]['dT_end_cu'+SF][inn]-ds_var[beam]['delta_time_ph'][0]
        aT = np.arange(dTbeg[0],dTend[-1]+dT_maj,dT_maj)
        NT = np.shape(aT)[0]
        # Declare files to save
        for ii in np.arange(NT-1):
          # 2000-m segment index
          iseg5000 = np.where((dTbeg>=aT[ii])&(dTbeg<aT[ii+1]))[0]
          if np.size(iseg5000)>100:
            #isegfilt = fit_filter(dTbeg[iseg5000],ssha_n[inn][iseg5000],Zmax=4) #
            iseg = iseg5000#[isegfilt]
            print('size(inn): '+str(np.size(inn)))
            print('size(iseg): '+str(np.size(iseg)))
            if np.size(iseg)>=int(np.ceil(segmaj_foot/(10))):            
                ds_var[beam]['filter_'+SF2+'_'+SF][iseg] = np.ones(np.size(iseg))
                ssha_ss = ds_var[beam]['ssha_mn_cu'+SF][inn][iseg]
                dT_dist = dTbeg[iseg]*SV_vel #(ds_var[beam]['dT_mean_cu'+SF][inn][iseg])*SV_vel
                DITCH = False#major_filter(dT_dist,ssha_ss)
                print(DITCH)
                if DITCH==False:
                    mod,med,mn,skw,var = data_stats(ssha_ss)
                    ds_var[beam]['ssha_md_cu'+SF2].append(med)
                    ds_var[beam]['ssha_mn_cu'+SF2].append(mn)
                    ds_var[beam]['ssha_sk_cu'+SF2].append(skw)
                    ds_var[beam]['ssha_vr_cu'+SF2].append(var)
                    ds_var[beam]['swh_cu'+SF2].append(4.0*np.sqrt(var))
                    ds_var[beam]['dT_beg_cu'+SF2].append(ds_var[beam]['dT_beg_cu'+SF][inn][iseg[0]])
                    ce = np.polyfit(ds_var[beam]['dT_beg_cu'+SF][inn][iseg], ssha_ss, 1)
                    ds_var[beam]['slope_cu'+SF2].append(ce[0])
                    ds_var[beam]['time_utc_beg_cu'+SF2].append(ds_var[beam]['time_utc_beg_cu'+SF][inn][iseg[0]])
                    ds_var[beam]['dT_end_cu'+SF2].append(ds_var[beam]['dT_end_cu'+SF][inn][iseg[-1]])
                    ds_var[beam]['time_utc_end_cu'+SF2].append(ds_var[beam]['time_utc_end_cu'+SF][inn][iseg[-1]])
                    ds_var[beam]['lat_beg_cu'+SF2].append(ds_var[beam]['lat_beg_cu'+SF][inn][iseg[0]])
                    ds_var[beam]['lon_beg_cu'+SF2].append(ds_var[beam]['lon_beg_cu'+SF][inn][iseg[0]])
                    ds_var[beam]['lat_end_cu'+SF2].append(ds_var[beam]['lat_end_cu'+SF][inn][iseg[-1]])
                    ds_var[beam]['lon_end_cu'+SF2].append(ds_var[beam]['lon_end_cu'+SF][inn][iseg[-1]])
                    ds_var[beam]['dist_cu'+SF2].append(np.abs(ds_var[beam]['dist_cu'+SF][inn][iseg[-1]]-ds_var[beam]['dist_cu'+SF][inn][iseg[0]]))
                    ds_var[beam]['N_cu'+SF2].append(np.size(iseg))
                    ds_var[beam]['dT_mean_cu'+SF2].append(np.nanmean(ds_var[beam]['dT_mean_cu'+SF][inn][iseg]))
                    ds_var[beam]['time_utc_mean_cu'+SF2].append(np.nanmean(ds_var[beam]['time_utc_mean_cu'+SF][inn][iseg]))
                    ds_var[beam]['lat_mean_cu'+SF2].append(np.nanmean(ds_var[beam]['lat_mean_cu'+SF][inn][iseg]))
                    if np.nanmax(np.abs(np.diff(ds_var[beam]['lon_mean_cu'+SF][inn][iseg])))>=90:
                        ds_var[beam]['lon_mean_cu'+SF2].append(np.nanmedian(ds_var[beam]['lon_mean_cu'+SF][inn][iseg]))
                    else:
                        ds_var[beam]['lon_mean_cu'+SF2].append(np.nanmean(ds_var[beam]['lon_mean_cu'+SF][inn][iseg]))
                    ds_var[beam]['h_mean_cu'+SF2].append(np.nanmean(ds_var[beam]['h_mean_cu'+SF][inn][iseg]))
                    ds_var[beam]['tide_ocean_mean_cu'+SF2].append(np.nanmean(ds_var[beam]['tide_ocean_mean_cu'+SF][inn][iseg]))
                    ds_var[beam]['dac_mean_cu'+SF2].append(np.nanmean(ds_var[beam]['dac_mean_cu'+SF][inn][iseg]))
                    ds_var[beam]['tide_equilibrium_mean_cu'+SF2].append(np.nanmean(ds_var[beam]['tide_equilibrium_mean_cu'+SF][inn][iseg]))
                    ds_var[beam]['dem_mean_cu'+SF2].append(np.nanmean(ds_var[beam]['dem_mean_cu'+SF][inn][iseg]))
                    ds_var[beam]['geoid_mean_cu'+SF2].append(np.nanmean(ds_var[beam]['geoid_mean_cu'+SF][inn][iseg]))
                    ds_var[beam]['geoid_free2mean_mean_cu'+SF2].append(np.nanmean(ds_var[beam]['geoid_free2mean_mean_cu'+SF][inn][iseg]))
                    ds_var[beam]['i_beg_cu'+SF2].append(iseg[0])
                    ds_var[beam]['i_end_cu'+SF2].append(iseg[-1])
                    ## 100-m segment index
                    aT2 = np.arange(dTbeg[iseg][0],dTend[iseg][-1]+dT_maj2,dT_maj2)
                    NT2 = np.shape(aT2)[0]
                    for jj in np.arange(NT2-1):
                        iseg100 = np.where((dTbeg[iseg]>=aT2[jj])&(dTbeg[iseg]<aT2[jj+1]))[0]
                        print('size(iseg100): '+str(np.size(iseg100)))
                        if np.size(iseg100)>=int(np.ceil(segmaj_foot2/(10))): #int(np.ceil((segmaj_foot2/(seg_foot*1.0)))):   
                            DITCH = False #major_filter(dT_dist[iseg100],ssha_ss[iseg100])
                            print(DITCH)
                            if DITCH==False:
                                mod2,med2,mn2,skw2,var2 = data_stats(ssha_ss[iseg100])
                                ds_var[beam]['ssha_md_cu'+SF3].append(med2)
                                ds_var[beam]['ssha_mn_cu'+SF3].append(mn2)
                                ds_var[beam]['ssha_sk_cu'+SF3].append(skw2)
                                ds_var[beam]['ssha_vr_cu'+SF3].append(var2)
                                ds_var[beam]['swh_cu'+SF3].append(4.0*np.sqrt(var))
                                ds_var[beam]['dT_beg_cu'+SF3].append(ds_var[beam]['dT_beg_cu'+SF][inn][iseg[iseg100][0]])
                                ds_var[beam]['time_utc_beg_cu'+SF3].append(ds_var[beam]['time_utc_beg_cu'+SF][inn][iseg[iseg100][0]])
                                ds_var[beam]['dT_end_cu'+SF3].append(ds_var[beam]['dT_end_cu'+SF][inn][iseg[iseg100][-1]])
                                ce2 = np.polyfit(ds_var[beam]['dT_beg_cu'+SF][inn][iseg][iseg100], ssha_ss[iseg100], 1)
                                ds_var[beam]['slope_cu'+SF3].append(ce2[0])
                                ds_var[beam]['time_utc_end_cu'+SF3].append(ds_var[beam]['time_utc_end_cu'+SF][inn][iseg[iseg100][-1]])
                                ds_var[beam]['lat_beg_cu'+SF3].append(ds_var[beam]['lat_beg_cu'+SF][inn][iseg[iseg100][0]])
                                ds_var[beam]['lon_beg_cu'+SF3].append(ds_var[beam]['lon_beg_cu'+SF][inn][iseg[iseg100][0]])
                                ds_var[beam]['lat_end_cu'+SF3].append(ds_var[beam]['lat_end_cu'+SF][inn][iseg[iseg100][-1]])
                                ds_var[beam]['lon_end_cu'+SF3].append(ds_var[beam]['lon_end_cu'+SF][inn][iseg[iseg100][-1]])
                                ds_var[beam]['dist_cu'+SF3].append(np.abs(ds_var[beam]['dist_cu'+SF][inn][iseg[iseg100][-1]]-ds_var[beam]['dist_cu'+SF][inn][iseg[iseg100][0]]))
                                ds_var[beam]['N_cu'+SF3].append(np.size(iseg[iseg100]))
                                ds_var[beam]['dT_mean_cu'+SF3].append(np.nanmean(ds_var[beam]['dT_mean_cu'+SF][inn][iseg[iseg100]]))
                                ds_var[beam]['time_utc_mean_cu'+SF3].append(np.nanmean(ds_var[beam]['time_utc_mean_cu'+SF][inn][iseg[iseg100]]))
                                ds_var[beam]['lat_mean_cu'+SF3].append(np.nanmean(ds_var[beam]['lat_mean_cu'+SF][inn][iseg[iseg100]]))
                                if np.nanmax(np.abs(np.diff(ds_var[beam]['lon_mean_cu'+SF][inn][iseg[iseg100]])))>=90:
                                    ds_var[beam]['lon_mean_cu'+SF3].append(np.nanmedian(ds_var[beam]['lon_mean_cu'+SF][inn][iseg[iseg100]]))
                                else:
                                    ds_var[beam]['lon_mean_cu'+SF3].append(np.nanmean(ds_var[beam]['lon_mean_cu'+SF][inn][iseg[iseg100]]))
                                ds_var[beam]['h_mean_cu'+SF3].append(np.nanmean(ds_var[beam]['h_mean_cu'+SF][inn][iseg[iseg100]]))
                                ds_var[beam]['tide_ocean_mean_cu'+SF3].append(np.nanmean(ds_var[beam]['tide_ocean_mean_cu'+SF][inn][iseg[iseg100]]))
                                ds_var[beam]['dac_mean_cu'+SF3].append(np.nanmean(ds_var[beam]['dac_mean_cu'+SF][inn][iseg[iseg100]]))
                                ds_var[beam]['tide_equilibrium_mean_cu'+SF3].append(np.nanmean(ds_var[beam]['tide_equilibrium_mean_cu'+SF][inn][iseg[iseg100]]))
                                ds_var[beam]['dem_mean_cu'+SF3].append(np.nanmean(ds_var[beam]['dem_mean_cu'+SF][inn][iseg[iseg100]]))
                                ds_var[beam]['geoid_mean_cu'+SF3].append(np.nanmean(ds_var[beam]['geoid_mean_cu'+SF][inn][iseg[iseg100]]))
                                ds_var[beam]['geoid_free2mean_mean_cu'+SF3].append(np.nanmean(ds_var[beam]['geoid_free2mean_mean_cu'+SF][inn][iseg[iseg100]]))
                                ds_var[beam]['i_beg_cu'+SF3].append(iseg[iseg100][0])
                                ds_var[beam]['i_end_cu'+SF3].append(iseg[iseg100][-1])
        # turn to array format and save
        for ii in np.arange(Nky):
            if np.size(ds_var[beam][kys_new[ii]+SF2])!=0:
                ds_var[beam][kys_new[ii]+SF2]=np.asarray(ds_var[beam][kys_new[ii]+SF2])
            if np.size(ds_var[beam][kys_new[ii]+SF3])!=0:
                ds_var[beam][kys_new[ii]+SF3]=np.asarray(ds_var[beam][kys_new[ii]+SF3])
        print('Time to create '+SF2+' m and '+SF3+' m segments: '+str(time.time()-t1))
        return ds_var

    
def pull_is2_atl03_var(ds,beam,LLMM,SEG,ds_var={},ORIENT=True):
    t1 = time.time()
    print(beam)
    ds_var[beam]={}
    # filter by region
    if np.size(LLMM)!=0:
        print('Min/Max latitude: '+str(LLMM[0])+' / '+str(LLMM[1]))
        print('Min/Max longitude: '+str(LLMM[2])+' / '+str(LLMM[3]))
        lat_ph = ds['/'+beam+'/heights/lat_ph'][:]
        lon_ph = ds['/'+beam+'/heights/lon_ph'][:]
        iph = np.where((lat_ph>=LLMM[0])&(lat_ph<=LLMM[1])&(lon_ph>=LLMM[2])&(lon_ph<=LLMM[3]))[0]
        lat_ref = ds['/'+beam+'/geolocation/reference_photon_lat'][:]
        lon_ref = ds['/'+beam+'/geolocation/reference_photon_lon'][:]
        iref = np.where((lat_ref>=LLMM[0])&(lat_ref<=LLMM[1])&(lon_ref>=LLMM[2])&(lon_ref<=LLMM[3]))[0]
        print('% photon data reduction: '+str(np.round(100.*(1.0-((np.size(iph)*1.0)/(np.size(lat_ph)))))))
        print('% reference data reduction: '+str(np.round(100.*(1.0-((np.size(iref)*1.0)/(np.size(lat_ref)))))))
    else:
        iph = np.arange(np.shape(ds['/'+beam+'/heights/lat_ph'][:])[0])
        iref = np.arange(np.shape(ds['/'+beam+'/geolocation/reference_photon_lon'][:])[0])
        
    if np.size(iph)!=0 and np.size(iref)!=0:
        # epoch and time
        # pg. 199/215: data_start_utc + (delta_time - start_delta_time)
        ds_var[beam]['atlas_sdp_gps_epoch'] = ds['/ancillary_data/atlas_sdp_gps_epoch'][:] # GPS time of photon
        ds_var[beam]['delta_time_ph'] = ds['/'+beam+'/heights/delta_time'][:][iph] # transmit time of photon
        ds_var[beam]['delta_time_ref'] = ds['/'+beam+'/geophys_corr/delta_time'][:][iref] # transmit time of reference photon from atlas_sdp_gps_epoch
        
        # GPS epoch (1980-01-06T00:00:000000Z UTC)
        ds_var[beam]['flag_signal_conf_ph_ocean'] = ds['/'+beam+'/heights/signal_conf_ph'][:][:,1][iph] # photon confidence row=1 is ocean (0=noise, 1=background, 2=low, 3=med, 4=high) #!!!
        ds_var[beam]['flag_quality_ph'] = ds['/'+beam+'/heights/quality_ph'][:][iph] # photon quality (0=nominal, 1=afterpulse, 2=impulse responde effect, 3=tep) #!!!
        ds_var[beam]['lat'] = ds['/'+beam+'/heights/lat_ph'][:][iph] # latitude of photon
        ds_var[beam]['lon'] = ds['/'+beam+'/heights/lon_ph'][:][iph] # longitude of photon
        ds_var[beam]['h'] = ds['/'+beam+'/heights/h_ph'][:][iph] # heigh of photon (no ocean tide, DAC or geoid)
        
    
        ##### gtx/geolocation (one segment is approx. 20 m along-track ~ 27 pulses)
        ds_var[beam]['ref_ph_idx'] = ds['/'+beam+'/geolocation/reference_photon_index'][:][iref] # transmit time of reference photon
        ds_var[beam]['ph_idx_beg'] = ds['/'+beam+'/geolocation/ph_index_beg'][:][iref] # index of ref photon within segment. 
        ds_var[beam]['ref_idx_arr'] = (ds_var[beam]['ph_idx_beg']+ds_var[beam]['ref_ph_idx'])-1
        # position of reference photon within the photon-rate arrays = ds_var[beam]['ref_ph_idx'] + ds_var[beam]['ph_idx_beg'] - 1 # 0 for no photons in segment
        ds_var[beam]['lat_ref'] = ds['/'+beam+'/geolocation/reference_photon_lat'][:][iref] # latitude of reference photon
        ds_var[beam]['lon_ref'] = ds['/'+beam+'/geolocation/reference_photon_lon'][:][iref] # longitude of reference photon
        ds_var[beam]['tropo'] = ds['/'+beam+'/geolocation/neutat_delay_total'][:][iref] # neutral atmosphere (wet tropo + dry tropo)
        ds_var[beam]['neutat_ht'] = ds['/'+beam+'/geolocation/neutat_ht'][:][iref] # height corrected for tropo
        ds_var[beam]['h_seg'] = ds_var[beam]['neutat_ht']+ds_var[beam]['tropo']
        ds_var[beam]['sigma_h'] = ds['/'+beam+'/geolocation/sigma_h'][:][iref] # height uncertainty for ref point
        ds_var[beam]['flag_surf_type_ocean'] = ds['/'+beam+'/geolocation/surf_type'][:][:,1][iref] # 5xN: 0=not type, 1=is type (land, ocean, sea ice, land ice, inland water) #!!!
        ds_var[beam]['seg_id'] = ds['/'+beam+'/geolocation/segment_id'][:][iref] # seven-digit along-track segment id
        ds_var[beam]['seg_len'] = ds['/'+beam+'/geolocation/segment_length'][:][iref] # along-track segment length
        ds_var[beam]['seg_cnt'] = ds['/'+beam+'/geolocation/segment_ph_cnt'][:][iref] # along-track segment photon count
        ds_var[beam]['flag_pod'] = ds['/'+beam+'/geolocation/podppd_flag'][:][iref] # POD/PPD flag = quality of geolocation (0=nominal, 1=pod degraded, 2=ppd degraded, 3=both degraded) #!!!
        ds_var[beam]['ref_elev'] = ds['/'+beam+'/geolocation/ref_elev'][:][iref] # elevation of the unit pointing vector (local ENU, radians)
        ds_var[beam]['ref_azim'] = ds['/'+beam+'/geolocation/ref_azimuth'][:][iref] # azimuth of the unit pointing vector (local ENU, radians)
        ##### gtx/geophys_corr
        ds_var[beam]['dac'] = ds['/'+beam+'/geophys_corr/dac'][:][iref] # dynamic atmospheric correction (+- 5 cm) NOT YET APPLIED.
        ds_var[beam]['tide_earth'] = ds['/'+beam+'/geophys_corr/tide_earth'][:][iref] # solid earth tides (+- 40 cm max)
        ds_var[beam]['tide_load'] = ds['/'+beam+'/geophys_corr/tide_load'][:][iref] # local displacement due to ocean loading (-6 to 0 cm)
        ds_var[beam]['tide_ocean'] = ds['/'+beam+'/geophys_corr/tide_ocean'][:][iref] # diurnal and semi-diurnal ocean tides (+- 4 m). NOT YET APPLIED.
        ds_var[beam]['tide_equilibrium'] = ds['/'+beam+'/geophys_corr/tide_equilibrium'][:][iref] # long-period equilibrium tide self-consistent with ocean tide model tide (+- 0.07 m). NOT YET APPLIED.
        ds_var[beam]['tide_pole'] = ds['/'+beam+'/geophys_corr/tide_pole'][:][iref] # solid earth rotational deformation due to polar motion ((+- 1.5 cm))
        ds_var[beam]['tide_oc_pole'] = ds['/'+beam+'/geophys_corr/tide_oc_pole'][:][iref] # oceanic rotational deformation due to polar motion ((+- 2 mm))
        ds_var[beam]['geoid'] = ds['/'+beam+'/geophys_corr/geoid'][:][iref] # geoid height above WGS-84 reference ellipsoid in tide free system (-107 to 86 m). NOT YET APPLIED.
        ds_var[beam]['geoid_free2mean'] = ds['/'+beam+'/geophys_corr/geoid_free2mean'][:][iref] # added value to convert geoid heights from tide-free to mean-tide system (geoid + geoid_free2mean). NOT YET APPLIED.
        ds_var[beam]['flag_dem'] = ds['/'+beam+'/geophys_corr/dem_flag'][:][iref] # source of DEM height (0=none, 1=Arctic, 2=MERIT, 3=MSS, 4=Antarctic) #!!!
        ds_var[beam]['dem_h'] = ds['/'+beam+'/geophys_corr/dem_h'][:][iref] # [m] best available DEM 
        
        #####/gtx/bckgrd_atlas/
        #ds_var[beam]['bckgrd_counts'] = ds['/'+beam+'/bckgrd_atlas/bckgrd_counts'][:] # 
        
        ##### Additional
        lon360 = lon180_to_lon360(ds_var[beam]['lon'])
        x,y,z = lla2ecef(ds_var[beam]['lat'],lon360)
        lon360ref = lon180_to_lon360(ds_var[beam]['lon_ref'])
        xr,yr,zr = lla2ecef(ds_var[beam]['lat_ref'],lon360ref)
        ds_var[beam]['dist_ph'] = dist_func(x[0],y[0],z[0],x,y,z)
        ds_var[beam]['dist_ref'] = dist_func(x[0],y[0],z[0],xr,yr,zr)
        
        ##### /orbit_info
        ds_var[beam]['cyc_number'] = ds['/orbit_info/cycle_number'][:] # number of 91-day cycle in the mission beginning with 01
        # unique orbit number = (cycle_number - 1)*1387+rgt
        ds_var[beam]['rgt'] = ds['/orbit_info/rgt'][:] # reference ground track
        ds_var[beam]['orbit_number'] = ds['/orbit_info/orbit_number'][:] #
        ds_var[beam]['sc_orient'] = ds['/orbit_info/sc_orient'][:] # forward or backward sc orientation (backward=0, forward=1, transition=2)
            
        # determine ascending and descending passes
        dlat = np.diff(ds_var[beam]['lat_ref'])
        ad_is2_pre = np.copy(dlat)
        ad_is2_pre[ad_is2_pre<0]=0
        ad_is2_pre[ad_is2_pre>0]=1
        ds_var[beam]['pass_0D_1A'] = np.insert(ad_is2_pre,0,ad_is2_pre[0]) 
        print('seconds to call variables :'+str(time.time()-t1))
        ds_var = atl03_filter(ds_var,beam,ORIENT)
        ds_var = atl03_time(ds_var,beam)
        ds_var = atl03_ssha(ds_var,beam) # longish (20 sec?)
        ds_var = atlcu_segment(ds_var,beam,seg_foot=SEG) # longish (30 sec)
        if 'ssha_mn_cu'+str(SEG) in ds_var[beam].keys():
            inn=np.where(~np.isnan(ds_var[beam]['ssha_mn_cu'+str(SEG)]))[0] #+ds_var[beam]['beam_nan_cu'+str(SEG)]))[0]
            if np.size(inn)>100:
                ds_var = atlcu_segment_major(ds_var,beam,seg_foot=SEG,segmaj_foot=2000,segmaj_foot2=100)
        print('seconds to create library for '+beam+' :'+str(time.time()-t1))
    else:
        ds_var[beam]['lat'] = 99999        
    return ds_var


def geo2dist(lat,lon,lat0,lon0):
    lon360 = lon180_to_lon360(lon)
    x,y,z = lla2ecef(lat,lon360)
    lon360ref = lon180_to_lon360(lon0)
    xr,yr,zr = lla2ecef(lat0,lon360ref)
    dist = dist_func(xr,yr,zr,x,y,z)
    return dist

def beam_slope(ds,BEAM,SEG):
    rse_max = 0.4
    
    sshaSEG_1 = ds[BEAM[0]]['ssha_mn_cu'+str(SEG)][~np.isnan(ds[BEAM[0]]['ssha_mn_cu'+str(SEG)])]# 'ssha_mn_cu'+str(SEG) 'ssha_mn_cu100'
    sshaSEG_2 = ds[BEAM[1]]['ssha_mn_cu'+str(SEG)][~np.isnan(ds[BEAM[1]]['ssha_mn_cu'+str(SEG)])]
    sshaSEG_3 = ds[BEAM[2]]['ssha_mn_cu'+str(SEG)][~np.isnan(ds[BEAM[2]]['ssha_mn_cu'+str(SEG)])]
    dTSEG_1 = ds[BEAM[0]]['dT_beg_cu'+str(SEG)][~np.isnan(ds[BEAM[0]]['ssha_mn_cu'+str(SEG)])]# time_utc_beg_cu10
    dTSEG_2 = ds[BEAM[1]]['dT_beg_cu'+str(SEG)][~np.isnan(ds[BEAM[1]]['ssha_mn_cu'+str(SEG)])]
    dTSEG_3 = ds[BEAM[2]]['dT_beg_cu'+str(SEG)][~np.isnan(ds[BEAM[2]]['ssha_mn_cu'+str(SEG)])]
    
    # Interpolate to estimate timeseries for each beam given SSHA from other beams 
    int_ssha1_2 = np.interp(dTSEG_2,dTSEG_1,sshaSEG_1)
    int_ssha1_3 = np.interp(dTSEG_3,dTSEG_1,sshaSEG_1)
    
    int_ssha2_1 = np.interp(dTSEG_1,dTSEG_2,sshaSEG_2)
    int_ssha2_3 = np.interp(dTSEG_3,dTSEG_2,sshaSEG_2)
    
    int_ssha3_1 = np.interp(dTSEG_1,dTSEG_3,sshaSEG_3)
    int_ssha3_2 = np.interp(dTSEG_2,dTSEG_3,sshaSEG_3)
    
    # Take the difference between the interpolated values and measured values
    dif_ssha1_12 = int_ssha1_2-sshaSEG_2
    dif_ssha1_13 = int_ssha1_3-sshaSEG_3
    
    dif_ssha2_12 = int_ssha2_1-sshaSEG_1
    dif_ssha2_23 = int_ssha2_3-sshaSEG_3
    
    dif_ssha3_13 = int_ssha3_1-sshaSEG_1
    dif_ssha3_23 = int_ssha3_2-sshaSEG_2
    
    # Find the RSE between the interpolated values and measured values
    rse_ssha1_12 = np.sqrt((dif_ssha1_12)**2)
    rse_ssha1_13 = np.sqrt((dif_ssha1_13)**2)
    
    rse_ssha2_12 = np.sqrt((dif_ssha2_12)**2)
    rse_ssha2_23 = np.sqrt((dif_ssha2_23)**2)
    
    rse_ssha3_13 = np.sqrt((dif_ssha3_13)**2)
    rse_ssha3_23 = np.sqrt((dif_ssha3_23)**2)
    
    # Detremine where rse is greater than some max allowable error
    idx1_12 = np.where(rse_ssha1_12>rse_max)[0]
    idx1_13 = np.where(rse_ssha1_13>rse_max)[0]

    idx2_12 = np.where(rse_ssha2_12>rse_max)[0]
    idx2_23 = np.where(rse_ssha2_23>rse_max)[0]
    
    idx3_13 = np.where(rse_ssha3_13>rse_max)[0]
    idx3_23 = np.where(rse_ssha3_23>rse_max)[0]
    
    # Find where the measured values are most likely NOT valid ocean data
    out_1 = np.intersect1d(idx2_12,idx3_13)
    out_2 = np.intersect1d(idx1_12,idx3_23)
    out_3 = np.intersect1d(idx1_13,idx2_23)
    
    # Isolate valid ocean data
    idx_1 = np.delete(np.arange(np.size(sshaSEG_1)),out_1)
    idx_2 = np.delete(np.arange(np.size(sshaSEG_2)),out_2)
    idx_3 = np.delete(np.arange(np.size(sshaSEG_3)),out_3)
    
    # Update dataset
    ds[BEAM[0]]['beam_nan_cu'+str(SEG)] = np.empty(np.size(ds[BEAM[0]]['ssha_mn_cu'+str(SEG)]))*np.nan
    ds[BEAM[0]]['beam_nan_cu'+str(SEG)][idx_1] = np.zeros(np.size(idx_1))
    
    ds[BEAM[1]]['beam_nan_cu'+str(SEG)] = np.empty(np.size(ds[BEAM[1]]['ssha_mn_cu'+str(SEG)]))*np.nan
    ds[BEAM[1]]['beam_nan_cu'+str(SEG)][idx_2] = np.zeros(np.size(idx_2))
    
    ds[BEAM[2]]['beam_nan_cu'+str(SEG)] = np.empty(np.size(ds[BEAM[2]]['ssha_mn_cu'+str(SEG)]))*np.nan
    ds[BEAM[2]]['beam_nan_cu'+str(SEG)][idx_3] = np.zeros(np.size(idx_3))
    return ds



