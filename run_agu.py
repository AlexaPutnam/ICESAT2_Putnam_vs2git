#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:15:05 2022

@author: alexaputnam
"""


import numpy as np
from matplotlib import pyplot as plt
from datetime import date, timedelta, datetime
import sys
import cartopy.crs as ccrs

import lib_read_TG as lTG
import lib_regions as lreg
sys.path.append("/Users/alexaputnam/necessary_functions/")
import plt_bilinear as pbil

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
def find_averages(binz,x,y,z=[],zconst=[],imin=2,MED=False,SKIP=False):
    Nbin = np.size(binz)
    db = np.diff(binz)[0]/2.0
    if np.size(zconst)<=2:
        my = np.empty(Nbin)*np.nan
        N = np.empty(Nbin)*np.nan
        for ii in np.arange(Nbin):
            if np.size(zconst)==0:
                ibin = np.where((x>=binz[ii]-db)&(x<binz[ii]+db))[0]
            else:
                ibin = np.where((x>=binz[ii]-db)&(x<binz[ii]+db)&(z>=zconst[0])&(z<=zconst[1]))[0]
            if np.size(ibin)>imin:
                idx = np.argsort(np.abs(y[ibin]))
                Ni = np.size(ibin)
                if SKIP==True:
                    ibin=ibin[idx[int(Ni/2):]]
                if MED == False:
                    my[ii] = np.nanmean(y[ibin])
                else:
                    my[ii] = np.nanmedian(y[ibin])
            N[ii] = np.size(ibin)
    else:
        Nz = np.size(zconst)
        dz = np.diff(zconst)[0]/2
        my = np.empty((Nbin,Nz))*np.nan
        N = np.empty((Nbin,Nz))*np.nan
        for ii in np.arange(Nbin):
            for jj in np.arange(Nz):
                ibin = np.where((x>=binz[ii]-db)&(x<binz[ii]+db)&(z>=zconst[jj]-dz)&(z<=zconst[jj]+dz))[0]
            if np.size(ibin)>imin:
                if MED == False:
                    my[ii,jj] = np.nanmean(y[ibin])
                else:
                    my[ii,jj] = np.nanmedian(y[ibin])
            N[ii,jj] = np.size(ibin)
    return my,N


def dist_func(xA,yA,zA,xD,yD,zD):
    dist = np.sqrt((np.subtract(xA,xD)**2)+(np.subtract(yA,yD)**2)+(np.subtract(zA,zD)**2))
    return dist
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

def dist_sv2pt(latSV,lonSV,beamSV,latPt,lonPt,maxKM=2000,beam=[]):
    xA,yA,zA = lla2ecef(latSV,lonSV)
    xP,yP,zP = lla2ecef(latPt,lonPt)
    dist = dist_func(xA,yA,zA,xP,yP,zP)
    if np.size(beam)!=0:
        i20 = np.where((dist>=0)&(dist<=maxKM)&(beamSV==beam))[0]
    else:
        i20i = np.where((dist>=0)&(dist<=maxKM))[0]
        Uni_beam,beam_cnt = np.unique(beamSV[i20i],return_counts=True)
        beam_max = int(Uni_beam[beam_cnt==np.nanmax(beam_cnt)])
        i20 = np.where((dist>=0)&(dist<=maxKM)&(beamSV==beam_max))[0]
    return i20,dist

def fft2signal(x,y,Nf=20):
    import scipy
    # https://scipy-lectures.org/intro/scipy/auto_examples/plot_fftpack.html
    # The FFT of the signal
    ce = np.polyfit(x, y, 1)[::-1]
    fity = ce[0]+ce[1]*x
    sig = y-fity
    time_vec = x#days_since_1985_a[i20A2]
    dt = np.diff(time_vec)
    time_step = np.nanmedian(dt)
    sig_fft = scipy.fftpack.fft(sig)
    # And the power (sig_fft is of complex dtype)
    power = np.abs(sig_fft)**2
    # The corresponding frequencies
    sample_freq = scipy.fftpack.fftfreq(sig.size, d=time_step)
    # Find the peak frequency: we can focus on only the positive frequencies
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]
    # remove all the high frequencies
    idx = np.argsort(np.abs(sample_freq))
    high_freq_fft = sig_fft.copy()
    high_freq_fft[idx[Nf+1:]] = 0#high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
    filtered_sig = scipy.fftpack.ifft(high_freq_fft)
    #ceB,filtered_sig_no0 = sinfit(x,filtered_sig,numharm=10)
    # dominant frequencies
    new_freq_arr = sample_freq[idx[:Nf+1]]#np.copy(sample_freq[np.abs(sample_freq) < peak_freq])
    print(new_freq_arr)
    print(high_freq_fft[idx[:Nf+1]])
    return filtered_sig.real,new_freq_arr

def pull_day(ymdhms,y=[],m=[],d=[],hr=[],mi=[],se=[],elem=0):
    if elem==0:
        dig_ymd = np.round(ymdhms[:,0]*10000)+(ymdhms[:,1]*100)+(ymdhms[:,2])
        days_cnt, cnt_days = np.unique(dig_ymd, return_counts=True)
        icnt = np.where(cnt_days==np.nanmax(cnt_days))
        elem = days_cnt[icnt]
        imax_day = np.where(dig_ymd==elem)
    else:
        dig_ymd = np.round(ymdhms[:,0]*10000)+(ymdhms[:,1]*100)+(ymdhms[:,2])
        imax_day = np.where(dig_ymd==elem)
    return imax_day,elem

def find_max_pull_day_beam(cnt_days_pre,days_cnt_pre,days_cnt):
    cnt_days = np.empty(np.shape(days_cnt))*np.nan
    for ii in np.arange(np.size(days_cnt)):
        idx = np.where(days_cnt_pre==days_cnt[ii])[0]
        cnt_days[ii]=cnt_days_pre[idx]
    return cnt_days


def pull_day_beam(ymdhms,beams):
    # day_max,i1,i2,i3,i10,i20,i30=pull_day_beam(ymdhms,beams)
    dig_ymd = np.round(ymdhms[:,0]*10000)+(ymdhms[:,1]*100)+(ymdhms[:,2])
    days_cnt1, cnt_days1 = np.unique(dig_ymd[beams==1], return_counts=True)
    days_cnt2, cnt_days2 = np.unique(dig_ymd[beams==2], return_counts=True)
    days_cnt3, cnt_days3 = np.unique(dig_ymd[beams==3], return_counts=True)
    days_cnt10, cnt_days10 = np.unique(dig_ymd[beams==10], return_counts=True)
    days_cnt20, cnt_days20 = np.unique(dig_ymd[beams==20], return_counts=True)
    days_cnt30, cnt_days30 = np.unique(dig_ymd[beams==30], return_counts=True)

    days_cntR=np.intersect1d(days_cnt1,np.intersect1d(days_cnt2,days_cnt3))
    days_cntL=np.intersect1d(days_cnt10,np.intersect1d(days_cnt20,days_cnt30))
    days_cnt = np.intersect1d(days_cntR,days_cntL)

    cnt_days1_f = find_max_pull_day_beam(cnt_days1,days_cnt1,days_cnt)
    cnt_days2_f = find_max_pull_day_beam(cnt_days2,days_cnt2,days_cnt)
    cnt_days3_f = find_max_pull_day_beam(cnt_days3,days_cnt3,days_cnt)

    cnt_days10_f = find_max_pull_day_beam(cnt_days10,days_cnt10,days_cnt)
    cnt_days20_f = find_max_pull_day_beam(cnt_days20,days_cnt20,days_cnt)
    cnt_days30_f = find_max_pull_day_beam(cnt_days30,days_cnt30,days_cnt)

    sum_cnt_days = cnt_days1_f+cnt_days2_f+cnt_days3_f+cnt_days10_f+cnt_days20_f+cnt_days30_f
    idx = np.where(sum_cnt_days==np.nanmax(sum_cnt_days))[0]

    day_max = days_cnt[idx]
    i1 = np.where((dig_ymd==day_max)&(beams==1))[0]
    i2 = np.where((dig_ymd==day_max)&(beams==2))[0]
    i3 = np.where((dig_ymd==day_max)&(beams==3))[0]
    i10 = np.where((dig_ymd==day_max)&(beams==10))[0]
    i20 = np.where((dig_ymd==day_max)&(beams==20))[0]
    i30 = np.where((dig_ymd==day_max)&(beams==30))[0]

    return day_max,i1,i2,i3,i10,i20,i30


def spectral_ssha(x,y):
    #x,y=t_a,ssha_a[icut]
    ce = np.polyfit(x, y, 1)[::-1]
    fity = ce[0]+ce[1]*x
    fitAno0,fA = fft2signal(x,y,Nf=60)
    lim=2
    plt.figure(figsize=(20,5))
    plt.subplots_adjust(top=0.9,hspace=0.4,wspace=0.4)
    plt.suptitle('Swell and Sea')
    plt.subplot(131)
    plt.plot(x*7000.0,y,label='2 m SSHA')
    plt.plot(x*7000.,fitAno0,label='wave signal')
    plt.xlabel('distance [m]')
    plt.ylabel('SSHA [m]')
    plt.legend()
    plt.ylim(-lim,lim)
    plt.grid()
    plt.subplot(132)
    plt.plot(x*7000.0,y-fitAno0,label='2 m SSHA - wave signal')
    plt.xlabel('distance [m]')
    plt.ylabel('$\Delta$SSHA [m]')
    plt.legend()
    plt.ylim(-lim,lim)
    plt.grid()
    plt.subplot(133)
    from scipy import signal
    fs = 1.0/(np.nanmedian(np.diff(x))) #sampling frequency
    f, Pxx_den = signal.periodogram(y, fs)
    #plt.semilogy(f[1:], Pxx_den[1:],'-')
    plt.plot(f[1:], Pxx_den[1:])
    #plt.plot(fA,np.ones(np.size(fA))*10)
    #plt.ylim([1e-7, 1e2])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.axvspan(0,110, alpha=0.5, color='green')
    #plt.axvline(x=66)
    plt.xlim(0,250)
    plt.grid()
    #plt.axvline(x=66,color='red')
    plt.show()

def subseg_to_seg2(subseg,Nsubseg,Nseg):
    # subseg,Nsubseg,Nseg=ssha_fft_b[icut_b],100,2000
    fS = int(np.floor(Nseg/Nsubseg))
    N = np.size(subseg)
    idx = np.arange(0,N+fS,fS)
    seg = np.empty(np.size(idx))*np.nan

def subseg_to_seg(subseg,tsubseg,tseg):
    # subseg,tsubseg,tseg=ssha_b[icut_b],days_since_1985_b[icut_b],days_since_1985_c[icut_c]
    N = np.size(tseg)
    seg = np.empty(N)*np.nan
    buff = ((2000.0/7000.0)/2)/(60.*60.*24.)#np.nanmedian(np.diff(tseg))/2 #((2000.0/7000.0)/2)/(60.*60.*24.)
    for ii in np.arange(0,N):
        idx = np.where((tsubseg>=tseg[ii]-buff)&(tsubseg<tseg[ii]+buff))[0]
        if np.size(idx)>0:
            seg[ii] = np.nanmean(subseg[idx])
    return seg


###################################
########### INPUT ###########
###################################
# Region/Time
REG = 'newengland' #'mumbai'#'north_atlantic'#'ittoqqortoormiit' #'newengland','hawaii','antarctica', 'japan'
if REG=='hunga_tonga':
    yrs_mm = [2021,2022]
elif REG=='newengland':
    yrs_mm = [2018,2019,2020,2021,2022]
else:
    yrs_mm = [2021]
fp_a = 2
fp_b = 100
fp_c = 2000

SEG2 = True
lat_min,lat_max,lon_min,lon_max = lreg.regions(REG)
FNtg,FNtg2,FNj3,FNc2,FNs3,FNis2,lltg2=lTG.file4_is2_alt_tg(REG)

###################################
########### DATA ###########
###################################
clr = ['black','tab:orange','tab:green','tab:purple','tab:red','tab:blue','tab:olive','indianred']
# Tide Gauge
if np.size(FNtg)!=0:
    sl_tg,lat_tg,lon_tg,days_since_1985_tg,ymdhms_tg,uhslc_id,yrfrac_tg = lTG.pull_hawaii_tg(FNtg)
    if np.shape(sl_tg)[0]!= np.size(sl_tg):
        Ntg=np.shape(sl_tg)[1]
        count_mn_tg = np.empty(Ntg)*np.nan
        for ii in np.arange(Ntg):
            mn_ssha_tg,vr_ssha_tg,mn_time_tg = lTG.month_2_month(ymdhms_tg[:,:,ii],sl_tg[:,ii],yrs_mm,IS2=False)
            count_mn_tg[ii] = np.size(mn_ssha_tg)
    else:
        mn_ssha_tg,vr_ssha_tg,mn_time_tg = lTG.month_2_month(ymdhms_tg,sl_tg,yrs_mm,IS2=False)
    if np.shape(sl_tg)[0]!= np.size(sl_tg):
        count_mn_tg = count_mn_tg.astype(int)
        mn_ssha_tg,mn_time_tg = np.empty((np.nanmax(count_mn_tg),Ntg))*np.nan,np.empty((np.nanmax(count_mn_tg),Ntg))*np.nan
        for ii in np.arange(Ntg):
            mn_ssha_tg[:,ii],vr_ssha_tg[:,ii],mn_time_tg[:,ii] = lTG.month_2_month(ymdhms_tg[:,:,ii],sl_tg[:,ii],yrs_mm,IS2=False)

# Tide Gauge(from Surftoday)
if np.size(FNtg2)!=0:
    sl_tg2,ymdhms_tg2,yrfrac_tg2=lTG.read_TG_noaa_sl(FNtg2)
    #sl_tg2,hlt_tg2,ymdhms_tg2,yrfrac_tg2 = lTG.read_TG_surftoday(FNtg2)
    if np.shape(sl_tg2)[0]!= np.size(sl_tg2):
        Ntg2=np.shape(FNtg2)[0]
        count_mn_tg2 = np.empty(Ntg2)*np.nan
        for ii in np.arange(Ntg2):
            mn_ssha_tg2,vr_ssha_tg2,mn_time_tg2 = lTG.month_2_month(ymdhms_tg2[ii,:,:],sl_tg2[ii,:],yrs_mm,IS2=False)
            count_mn_tg2[ii] = np.size(mn_ssha_tg2)
    else:
        mn_ssha_tg2,vr_ssha_tg2,mn_time_tg2 = lTG.month_2_month(ymdhms_tg2,sl_tg2,yrs_mm,IS2=False)
    if np.shape(sl_tg2)[0]!= np.size(sl_tg2):
        count_mn_tg2 = count_mn_tg2.astype(int)
        mn_ssha_tg2,mn_time_tg2,vr_ssha_tg2 = np.empty((Ntg2,np.nanmax(count_mn_tg2)))*np.nan,np.empty((Ntg2,np.nanmax(count_mn_tg2)))*np.nan,np.empty((Ntg2,np.nanmax(count_mn_tg2)))*np.nan
        for ii in np.arange(Ntg2):
            mn_ssha_tg2[ii,:],vr_ssha_tg2[ii,:],mn_time_tg2[ii,:] = lTG.month_2_month(ymdhms_tg2[ii,:,:],sl_tg2[ii,:],yrs_mm,IS2=False)
    lon_tg2 = np.empty(np.shape(lltg2)[0])*np.nan
    lat_tg2 = np.empty(np.shape(lltg2)[0])*np.nan
    ss_tg2 = np.nanmean(sl_tg2,axis=1)
    for ii in np.arange(np.shape(lltg2)[0]):
        lon_tg2[ii] = lltg2[ii][1]
        lat_tg2[ii] = lltg2[ii][0]
llmm = []#[lat_tg,lon_tg]
# Jason-3
if np.size(FNj3)!=0:
    ssha_alt,lat_alt,lon_alt,days_since_1985_alt,ymdhmsA,tsA,swh_alt,yrfracA = lTG.pull_altimetry(FNj3,llmm=llmm)
    mn_ssha_alt,vr_ssha_alt,mn_time_alt = lTG.month_2_month(ymdhmsA,ssha_alt,yrs_mm,IS2=False)
    mn_swh_alt,vr_swh_alt,mn_time_alt = lTG.month_2_month(ymdhmsA,swh_alt,yrs_mm,IS2=False)
    mn_ts_alt,mn_ymdhms_alt,mn_yrfrac_alt = lTG.tide_days_1985_to_TS(mn_time_alt)

# CryoSat-2
if np.size(FNc2)!=0:
    ssha_c2,lat_c2,lon_c2,days_since_1985_c2,ymdhms_c2,tsA,swh_c2,yrfrac_c2 = lTG.pull_altimetry(FNc2,llmm=llmm)
    mn_ssha_c2,vr_ssha_v2,mn_time_c2 = lTG.month_2_month(ymdhms_c2,ssha_c2,yrs_mm,IS2=False)
    mn_swh_c2,vr_swh_c2,mn_time_c2 = lTG.month_2_month(ymdhms_c2,swh_c2,yrs_mm,IS2=False)
    mn_ts_c2,mn_ymdhms_c2,mn_yrfrac_c2 = lTG.tide_days_1985_to_TS(mn_time_c2)

# Sentinel-3
if np.size(FNs3)!=0:
    ssha_s3,lat_s3,lon_s3,days_since_1985_s3,ymdhms_s3,tsA,swh_s3,yrfrac_s3 = lTG.pull_altimetry(FNs3,llmm=llmm)
    mn_ssha_s3,vr_ssha_v2,mn_time_s3 = lTG.month_2_month(ymdhms_s3,ssha_s3,yrs_mm,IS2=False)
    mn_swh_s3,vr_swh_s3,mn_time_s3 = lTG.month_2_month(ymdhms_s3,swh_s3,yrs_mm,IS2=False)
    mn_ts_s3,mn_ymdhms_s3,mn_yrfrac_s3 = lTG.tide_days_1985_to_TS(mn_time_s3)

llmm = []#[lat_tg[0],lon_tg[0]]
# IceSat2
if SEG2==True:
    ## 2-m segment 
    ssha_a,ssha_fft_a,swell_hf_a,swell_lf_a,swell_a,lat_a,lon_a,days_since_1985_a,ymdhmsI_a,tsI_a,beam_a,swh_a,swh66_a,N_a,slope_a,skew_a,yrfrac_a,wl_seg_a,wsteep_seg_a,ip_lf_a,ip_hf_a,ip_a,OT_a = lTG.pull_icesat(FNis2,SEG=2,llmm_fix=llmm)
    iS_a = np.where((lat_a>=41.55)&(lat_a<=41.7)&(lon_a>=-71)&(lon_a<=-70.5))[0]
    swh_aadj = 0.056 + 7.83*(swh_a/4.0)
    #mn_ssha_a,vr_ssha_a,mn_time_a = lTG.month_2_month(ymdhmsI_a,ssha_a,yrs_mm)
    #mn_swh_a,vr_swh_a,mn_time_a = lTG.month_2_month(ymdhmsI_a,swh_a,yrs_mm)
    #mn_swh_aadj,vr_swh_aadj,mn_time_a = lTG.month_2_month(ymdhmsI_a,swh_aadj,yrs_mm)

## 100-m segment
ssha_b,ssha_fft_b,swell_hf_b,swell_lf_b,swell_b,lat_b,lon_b,days_since_1985_b,ymdhmsI_b,tsI_b,beam_b,swh_b,swh66_b,N_b,slope_b,skew_b,yrfrac_b,wl_seg_b,wsteep_seg_b,ip_lf_b,ip_hf_b,ip_b,OT_b = lTG.pull_icesat(FNis2,SEG=fp_b,llmm_fix=llmm)
#ssha_fft_b = ssha_fft_b+OT_b
utcT_b = lTG.ymdhms2utc(ymdhmsI_b)
#ot_b = lTG.ocean_tide_replacement(lon_b,lat_b,utcT_b)
swh_badj = 0.523 + 5.393*(swh_b/4.0)
#wgt_b,dist_b = lTG.select_region(lat_b,lon_b,lat_tg[0],lon_tg[0],dlat=1,dlon=1)
#mn_ssha_b,vr_ssha_b,mn_time_b = lTG.month_2_month(ymdhmsI_b,ssha_b,yrs_mm)#,LATLON = np.vstack((lat_b,lon_b)).T) #ymdhmsI_b[idx_b],ssha_b[idx_b],days_since_1985_b[idx_b]
#mn_ssha_fft_b,vr_ssha_fft_b,mn_time_fft_b = lTG.month_2_month(ymdhmsI_b,ssha_fft_b,yrs_mm)#
#mn_swh_b,vr_swh_b,mn_time_b = lTG.month_2_month(ymdhmsI_b,swh_b,yrs_mm,d100=True)
#mn_swh_badj,vr_swh_badj,mn_time_b = lTG.month_2_month(ymdhmsI_b,swh_badj,yrs_mm)
#mn_slope_b,vr_slope_b,mn_time_b = lTG.month_2_month(ymdhmsI_b,slope_b,yrs_mm)
## 2000-m segment
ssha_c,ssha_fft_c,swell_hf_c,swell_lf_c,swell_c,lat_c,lon_c,days_since_1985_c,ymdhmsI_c,tsI_c,beam_c,swh_c,swh66_c,N_c,slope_c,skew_c,yrfrac_c,wl_seg_c,wsteep_seg_c,ip_lf_c,ip_hf_c,ip_c,OT_c = lTG.pull_icesat(FNis2,SEG=fp_c,llmm_fix=llmm)
#mn_swh_c,vr_swh_c,mn_time_c = lTG.month_2_month(ymdhmsI_c,swh_c,yrs_mm)

            
#############

def sv_v_tg(lat_tg,lon_tg,ymdhms_tg,lat_b,lon_b,ymdhms_b,max_dist,max_hr):
    dist_b2tg = lTG.lla2dist(lat_b,lon_b,lat_tg[0],lon_tg[0])
    id_b2tg =  np.where(dist_b2tg<max_dist)[0]
    itg = []
    isv = []
    dist = []
    dtime = []
    Ni = np.size(id_b2tg)
    if Ni>0:
        for ii in np.arange(Ni):
            itg_yr = np.where(ymdhms_tg[:,0]==ymdhms_b[ii,0])[0]
            Nyr = np.size(itg_yr)
            if Nyr>0:
                for jj in np.arange(Nyr):
                    a = datetime(int(ymdhms_b[ii,0]),int(ymdhms_b[ii,1]),int(ymdhms_b[ii,2]),int(ymdhms_b[ii,3]),int(ymdhms_b[ii,4]),int(ymdhms_b[ii,5]))
                    b = datetime(int(ymdhms_tg[jj,0]),int(ymdhms_tg[jj,1]),int(ymdhms_tg[jj,2]),int(ymdhms_tg[jj,3]),int(ymdhms_tg[jj,4]),int(ymdhms_tg[jj,5]))
                    dt = (b-a).total_seconds()
                    dh = dt/(60.*60.)
                    if dh<=max_hr:
                        itg.append(itg_yr[jj])
                        isv.append(id_b2tg[ii])
                        dist.append(dist_b2tg[ii])
                        dtime.append(dt)
    itg,isv,dist,dtime = np.asarray(itg),np.asarray(isv),np.asarray(dist),np.asarray(dtime)
    unq,counts = np.unique(itg,return_counts=True)
    iN = np.where(counts>1)[0]
    itg_f = np.copy(itg)
    isv_f = np.copy(isv)
    dist_f = np.copy(dist)
    dtime_f= np.copy(dtime)
    for ii in np.arange(np.size(iN)):
        iunq = np.where(itg==unq[iN[ii]])[0]
        idd = np.where(dtime[iunq]==np.nanmin(dtime[iunq]))[0]
        itg_f[iunq]=np.empty(np.shape(iunq))*np.nan
        isv_f[iunq]=np.empty(np.shape(iunq))*np.nan
        dist_f[iunq]=np.empty(np.shape(iunq))*np.nan
        dtime_f[iunq]=np.empty(np.shape(iunq))*np.nan
        if np.size(idd)>1:
            itg_f[iunq][idd[0]]=np.copy(itg[iunq][idd[0]])
            isv_f[iunq][idd[0]]=np.copy(isv[iunq][idd][0])
            dist_f[iunq][idd[0]]=np.copy(dist[iunq][idd][0])
            dtime_f[iunq][idd[0]]=np.copy(dtime[iunq][idd][0])
        else:
            itg_f[iunq][idd]=np.copy(itg[iunq][idd])
            isv_f[iunq][idd]=np.copy(isv[iunq][idd])
            dist_f[iunq][idd]=np.copy(dist[iunq][idd])
            dtime_f[iunq][idd]=np.copy(dtime[iunq][idd])

    itg_f,isv_f,dist_f,dtime_f = itg_f[~np.isnan(itg_f)],isv_f[~np.isnan(itg_f)],dist_f[~np.isnan(itg_f)],dtime_f[~np.isnan(itg_f)]
    return itg,isv,dist,dtime#itg_f,isv_f,dist_f,dtime_f
#itg,isv,dist_f,dtime_f = sv_v_tg(lat_tg,lon_tg,ymdhms_tg,lat_b,lon_b,ymdhmsI_b,2000,.1)



#############
#############
## Available Beams
#############
#############
plt.figure()
plt.plot(yrfrac_b,beam_b,'.')
plt.xlabel('time')
plt.ylabel('beams')
plt.grid()

binz = np.arange(-1,1.1,0.1)
binz0,count0,pdf0,cdf0 = lTG.hist_cdf(ssha_b,bins=binz)
binz1,count1,pdf1,cdf1 = lTG.hist_cdf(ssha_b[beam_b==1],bins=binz)
binz2,count2,pdf2,cdf2 = lTG.hist_cdf(ssha_b[beam_b==2],bins=binz)
binz3,count3,pdf3,cdf3 = lTG.hist_cdf(ssha_b[beam_b==3],bins=binz)
binz10,count10,pdf10,cdf10 = lTG.hist_cdf(ssha_b[beam_b==10],bins=binz)
binz20,count20,pdf20,cdf20 = lTG.hist_cdf(ssha_b[beam_b==20],bins=binz)
binz30,count30,pdf30,cdf30 = lTG.hist_cdf(ssha_b[beam_b==30],bins=binz)

plt.figure()
#plt.plot(binz0,count0,label='FULL')
plt.plot(binz1,count1,label='R1')
plt.plot(binz2,count2,label='R2')
plt.plot(binz3,count3,label='R3')
plt.plot(binz10,count10,label='L1')
plt.plot(binz20,count20,label='L2')
plt.plot(binz30,count30,label='L3')
plt.legend()
plt.grid()
plt.ylabel('occurrences')
plt.xlabel('ssha [m] \n '+str(fp_b)+' m segment')


SEG=1
if SEG==0:
    yrfrac=np.copy(yrfrac_a)
    ymdhms=np.copy(ymdhmsI_a)
    beam=np.copy(beam_a)
    ssha=np.copy(ssha_a) #_fft
    ssha_fft=np.copy(ssha_fft_a) #
    swell = np.copy(swell_a)
    lat = np.copy(lat_a)
    dayz = days_since_1985_a#*(24*60*60)*7000.
    sSEG = '2 m' 
elif SEG==1:
    yrfrac=np.copy(yrfrac_b)
    ymdhms=np.copy(ymdhmsI_b)
    beam=np.copy(beam_b)
    ssha=np.copy(ssha_b)
    ssha_fft=np.copy(ssha_fft_b) #
    swell = np.copy(swell_b)
    lat = np.copy(lat_b)
    dayz = days_since_1985_b#*(24*60*60)*7000.
    sSEG = '100 m' 
elif SEG==2:
    yrfrac=np.copy(yrfrac_c)
    ymdhms=np.copy(ymdhmsI_c)
    beam=np.copy(beam_c)
    ssha=np.copy(ssha_c)
    ssha_fft=np.copy(ssha_fft_c) #
    swell = np.copy(swell_c)
    lat = np.copy(lat_c)
    dayz = days_since_1985_c#*(24*60*60)*7000.
    sSEG = '2 km' 
day_max,i1,i2,i3,i10,i20,i30=pull_day_beam(ymdhms,beam)
min_day = np.nanmin(np.asarray([np.nanmin(dayz[i1]),np.nanmin(dayz[i2]),np.nanmin(dayz[i3]),np.nanmin(dayz[i10]),np.nanmin(dayz[i20]),np.nanmin(dayz[i30])]))
ddayz = ((dayz-min_day)*(24*60*60)*7000.)/1000.
mrk='.-'
xlim = [0,110]
plt.figure(figsize=(10,12))
plt.subplots_adjust(top=0.85,hspace=0.4,wspace=0.4)
plt.suptitle('ICESat-2 '+sSEG+' segments')
plt.subplot(311)
plt.plot(ddayz[i10],ssha[i10],':',label='gt1l')
plt.plot(ddayz[i1],ssha[i1],':',label='gt1r')
plt.plot(ddayz[i10],ssha_fft[i10],mrk,label='gt1l w/o SW')
plt.plot(ddayz[i1],ssha_fft[i1],mrk,label='gt1r w/o SW')
plt.legend()
plt.xlabel('distance [km]')
plt.ylabel('SSHA [m]')
plt.grid()
plt.title('gt1')
plt.xlim(xlim[0],xlim[1])
plt.subplot(312)
plt.plot(ddayz[i20],ssha[i20],mrk,label='gt2l')
plt.plot(ddayz[i2],ssha[i2],mrk,label='gt2r')
plt.plot(ddayz[i20],ssha_fft[i20],':',label='gt2l w/o SW')
plt.plot(ddayz[i2],ssha_fft[i2],':',label='gt2r w/o SW')
plt.legend()
plt.xlabel('distance [km]')
plt.ylabel('SSHA [m]')
plt.grid()
plt.title('gt2')
plt.xlim(xlim[0],xlim[1])
plt.subplot(313)
plt.plot(ddayz[i30],ssha[i30],mrk,label='gt3l')
plt.plot(ddayz[i3],ssha[i3],mrk,label='gt3r')
plt.plot(ddayz[i30],ssha_fft[i30],':',label='gt3l w/o SW')
plt.plot(ddayz[i3],ssha_fft[i3],':',label='gt3r w/o SW')
plt.legend()
plt.xlabel('distance [km]')
plt.ylabel('SSHA [m]')
plt.grid()
plt.title('gt3')
plt.xlim(xlim[0],xlim[1])


xlim = [10,102]
ylim = [-0.3,0.75]
plt.figure(figsize=(14,8))
plt.subplots_adjust(top=0.85,hspace=0.5,wspace=0.4)
plt.suptitle('ICESat-2 '+sSEG+' segments')
plt.subplot(321)
plt.plot(ddayz[i1],ssha[i1],mrk,label='SSHA')
plt.plot(ddayz[i1],ssha_fft[i1],mrk,label='SSHA w/o SW')
plt.legend()
plt.xlabel('distance [km]')
plt.ylabel('SSHA [m]')
plt.grid()
plt.title('gt1r')
plt.xlim(xlim[0],xlim[1])
plt.ylim(ylim[0],ylim[1])
plt.subplot(323)
plt.plot(ddayz[i2],ssha[i2],mrk,label='SSHA')
plt.plot(ddayz[i2],ssha_fft[i2],mrk,label='SSHA w/o SW')
plt.legend()
plt.xlabel('distance [km]')
plt.ylabel('SSHA [m]')
plt.grid()
plt.title('gt2r')
plt.xlim(xlim[0],xlim[1])
plt.ylim(ylim[0],ylim[1])
plt.subplot(325)
plt.plot(ddayz[i3],ssha[i3],mrk,label='SSHA')
plt.plot(ddayz[i3],ssha_fft[i3],mrk,label='SSHA w/o SW')
plt.legend()
plt.xlabel('distance [km]')
plt.ylabel('SSHA [m]')
plt.grid()
plt.title('gt3r')
plt.xlim(xlim[0],xlim[1])
plt.ylim(ylim[0],ylim[1])
plt.subplot(322)
plt.plot(ddayz[i10],ssha[i10],mrk,label='SSHA')
plt.plot(ddayz[i10],ssha_fft[i10],mrk,label='SSHA w/o SW')
plt.legend()
plt.xlabel('distance [km]')
plt.ylabel('SSHA [m]')
plt.grid()
plt.title('gt1l')
plt.xlim(xlim[0],xlim[1])
plt.ylim(ylim[0],ylim[1])
plt.subplot(324)
mn20,sig20 = np.nanmean(ssha_fft[i20]),np.nanstd(ssha_fft[i20])
i20p = np.where((ssha_fft[i20]>=mn20-(4*sig20))&(ssha_fft[i20]<=mn20+(4*sig20)))[0]
plt.plot(ddayz[i20],ssha[i20],mrk,label='SSHA')
#plt.plot(ddayz[i20][i20p],ssha[i20][i20p],mrk,label='SSHA cut')
plt.plot(ddayz[i20],ssha_fft[i20],mrk,label='SSHA w/o SW')
plt.legend()
plt.xlabel('distance [km]')
plt.ylabel('SSHA [m]')
plt.grid()
plt.title('gt2l')
plt.xlim(xlim[0],xlim[1])
plt.ylim(ylim[0],ylim[1])
plt.subplot(326)
plt.plot(ddayz[i30],ssha[i30],mrk,label='SSHA')
plt.plot(ddayz[i30],ssha_fft[i30],mrk,label='SSHA w/o SW')
plt.legend()
plt.xlabel('distance [km]')
plt.ylabel('SSHA [m]')
plt.grid()
plt.title('gt3l')
plt.xlim(xlim[0],xlim[1])
plt.ylim(ylim[0],ylim[1])

if SEG==0:
    ylim = [-6,6]
    xlim = [100,110]
    mrk = '-'
    plt.figure(figsize=(14,8))
    plt.subplots_adjust(top=0.85,hspace=0.5,wspace=0.4)
    plt.suptitle('ICESat-2 '+sSEG+' segments')
    plt.subplot(321)
    plt.plot(ddayz[i1],ssha[i1],mrk,label='SSHA')
    plt.plot(ddayz[i1],swell[i1],mrk,label='SW')
    plt.legend()
    plt.xlabel('distance [km]')
    plt.ylabel('SSHA [m]')
    plt.grid()
    plt.title('gt1r')
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])
    plt.subplot(323)
    plt.plot(ddayz[i2],ssha[i2],mrk,label='SSHA')
    plt.plot(ddayz[i2],swell[i2],mrk,label='SW')
    plt.legend()
    plt.xlabel('distance [km]')
    plt.ylabel('SSHA [m]')
    plt.grid()
    plt.title('gt2r')
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])
    plt.subplot(325)
    plt.plot(ddayz[i3],ssha[i3],mrk,label='SSHA')
    plt.plot(ddayz[i3],swell[i3],mrk,label='SW')
    plt.legend()
    plt.xlabel('distance [km]')
    plt.ylabel('SSHA [m]')
    plt.grid()
    plt.title('gt3r')
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])
    plt.subplot(322)
    plt.plot(ddayz[i10],ssha[i10],mrk,label='SSHA')
    plt.plot(ddayz[i10],swell[i10],mrk,label='SW')
    plt.legend()
    plt.xlabel('distance [km]')
    plt.ylabel('SSHA [m]')
    plt.grid()
    plt.title('gt1l')
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])
    plt.subplot(324)
    mn20,sig20 = np.nanmean(ssha_fft[i20]),np.nanstd(ssha_fft[i20])
    i20p = np.where((ssha_fft[i20]>=mn20-(4*sig20))&(ssha_fft[i20]<=mn20+(4*sig20)))[0]
    plt.plot(ddayz[i20],ssha[i20],mrk,label='SSHA')
    #plt.plot(ddayz[i20][i20p],ssha[i20][i20p],mrk,label='SSHA cut')
    plt.plot(ddayz[i20],swell[i20],mrk,label='SW')
    plt.legend()
    plt.xlabel('distance [km]')
    plt.ylabel('SSHA [m]')
    plt.grid()
    plt.title('gt2l')
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])
    plt.subplot(326)
    plt.plot(ddayz[i30],ssha[i30],mrk,label='SSHA')
    plt.plot(ddayz[i30],swell[i30],mrk,label='SW')
    plt.legend()
    plt.xlabel('distance [km]')
    plt.ylabel('SSHA [m]')
    plt.grid()
    plt.title('gt3l')
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])



#############
#############
## SWH
#############
#############
TF=False
if TF==True:
    AVG = 'median'
else:
    AVG = 'mean'
dday = 24*91
dyr = np.round((dday/24.)/365.25,5)#(96./24.)
imin=2
BEAMLIM = 100
yrf_bins = np.arange(yrs_mm[0],yrs_mm[-1]+1+dyr,dyr)
ssha_yrf_b,N_yrf_mn2_b = find_averages(yrf_bins,yrfrac_b[beam_b<BEAMLIM],ssha_fft_b[beam_b<BEAMLIM],imin=imin,MED=TF)
ssha_yrf_c,N_yrf_mn2_c = find_averages(yrf_bins,yrfrac_c[beam_c<BEAMLIM],ssha_fft_c[beam_c<BEAMLIM],imin=imin,MED=TF)
ssha_ws_yrf_b,N_ws_yrf_mn2_b = find_averages(yrf_bins,yrfrac_b[beam_b<BEAMLIM],ssha_b[beam_b<BEAMLIM],imin=imin,MED=TF)
ssha_ws_yrf_c,N_ws_yrf_mn2_c = find_averages(yrf_bins,yrfrac_c[beam_c<BEAMLIM],ssha_c[beam_c<BEAMLIM],imin=imin,MED=TF)
if np.size(FNj3)!=0:
    ssha_yrf_j3,N_yrf_j3 = find_averages(yrf_bins,yrfracA,ssha_alt,imin=imin,MED=TF)
if np.size(FNs3)!=0:
    ssha_yrf_s3,N_yrf_s3 = find_averages(yrf_bins,yrfrac_s3,ssha_s3,imin=imin,MED=TF)
if np.size(FNtg)!=0:
    ssha_yrf_tg,N_yrf_tg = find_averages(yrf_bins,yrfrac_tg,sl_tg,imin=imin,MED=TF)

swh_yrf_b,N_yrf_mn2_b = find_averages(yrf_bins,yrfrac_b[beam_b<BEAMLIM],swh_b[beam_b<BEAMLIM],imin=imin,MED=TF)
swh_yrf_c,N_yrf_mn2_c = find_averages(yrf_bins,yrfrac_c[beam_c<BEAMLIM],swh_c[beam_c<BEAMLIM],imin=imin,MED=TF)

swh66_yrf_b,N66_yrf_mn2_b = find_averages(yrf_bins,yrfrac_b[beam_b<BEAMLIM],swh66_b[beam_b<BEAMLIM],imin=imin,MED=TF)
swh66_yrf_c,N66_yrf_mn2_c = find_averages(yrf_bins,yrfrac_c[beam_c<BEAMLIM],swh66_c[beam_c<BEAMLIM],imin=imin,MED=TF)

plt.figure(figsize=(22,6))
FS=18
plt.suptitle('ICESat-2 vs. tide gauge (Newport, RI h253) \n '+str(int(dday/24))+' day averages',fontsize=FS) 
plt.plot(yrf_bins,ssha_yrf_b,'-',label='IS2 (100 m)',color='red',linewidth=2)#,alpha=0.5)
#plt.plot(yrf_bins[innB_tg],ssha_ws_yrf_b[innB_tg],'o-',label='IS2 100 m (WS)')
plt.xlabel('year fraction',fontsize=FS)
plt.ylabel('SSHA [m]',fontsize=FS)
#plt.fill_betweenx([yrfrac_start,yrfrac_end],color='red',alpha=0.5)
plt.grid()
plt.legend(fontsize=FS)
plt.tick_params(axis='both', which='major', labelsize=FS)

if np.size(FNtg)!=0:
    yrf_bins_tg = np.arange(1960,yrs_mm[-1]+1+dyr,dyr)
    ssha_yrf_tg1,N_yrf_tg = find_averages(yrf_bins_tg,yrfrac_tg,sl_tg,imin=imin,MED=TF)
    yrf_bins_tg2 = np.arange(2010,yrs_mm[-1]+1+dyr,dyr)
    ssha_yrf_tg2,N_yrf_tg2 = find_averages(yrf_bins_tg2,yrfrac_tg,sl_tg,imin=imin,MED=TF)
    dl = 1
    idx_tgB = np.where((beam_b<BEAMLIM)&(lat_b>=lat_tg-dl)&(lat_b<=lat_tg+dl)&(lon_b>=lon_tg-dl)&(lon_b<=lon_tg+dl))[0]
    ssha_yrf_b2,N_yrf_mn2_b2 = find_averages(yrf_bins,yrfrac_b[idx_tgB],ssha_fft_b[idx_tgB],imin=imin,MED=TF)
    #innB_tg = np.where((~np.isnan(ssha_yrf_b))&(~np.isnan(ssha_yrf_tg)))[0]

    plt.figure(figsize=(22,6))
    FS=18
    plt.suptitle('ICESat-2 vs. tide gauge (Newport, RI h253) \n '+str(int(dday/24))+' day averages',fontsize=FS) 
    plt.plot(yrf_bins,ssha_yrf_b,'-',label='IS2 (100 m)',color='red',linewidth=2)#,alpha=0.5)
    #plt.plot(yrf_bins[innB_tg],ssha_ws_yrf_b[innB_tg],'o-',label='IS2 100 m (WS)')
    bias_b_tg = np.nanmean(sl_tg[yrfrac_tg>=yrs_mm[0]])-np.nanmean(ssha_fft_b)#np.nanmedian(ssha_yrf_tg[yrf_bins_tg>=yrs_mm[0]])-np.nanmedian(ssha_yrf_b)
    plt.plot(yrf_bins_tg,ssha_yrf_tg1-bias_b_tg,'-',label='tide gauge',color='black')
    plt.xlabel('year fraction',fontsize=FS)
    plt.ylabel('SSHA [m]',fontsize=FS)
    #plt.fill_betweenx([yrfrac_start,yrfrac_end],color='red',alpha=0.5)
    plt.grid()
    plt.legend(fontsize=FS)
    plt.tick_params(axis='both', which='major', labelsize=FS)

if REG=='hunga_tonga':
    d0 = date(2020, 12, 31)
    d1 = date(2021, 12, 20)
    delta = (d1 - d0).total_seconds()
    delta_fraction = ((delta)/(24.*60.*60.))/365.25
    yrfrac_start = 2021+delta_fraction
    d0 = date(2021, 12, 31)
    d1 = date(2022, 1, 15)
    delta = (d1 - d0).total_seconds()
    delta_fraction = ((delta)/(24.*60.*60.))/365.25
    yrfrac_end = 2022+delta_fraction
    #plt.axvline(x=yrfrac_start,color='black')
    #plt.axvline(x=yrfrac_end,color='black')

    d0 = date(2020, 12, 31)
    d1 = date(2021, 11, 20)
    delta = (d1 - d0).total_seconds()
    delta_fraction = ((delta)/(24.*60.*60.))/365.25
    yrfrac_b4 = 2021+delta_fraction
    d0 = date(2021, 12, 31)
    d1 = date(2022, 2, 15)
    delta = (d1 - d0).total_seconds()
    delta_fraction = ((delta)/(24.*60.*60.))/365.25
    yrfrac_after = 2022+delta_fraction
    yrfrac_b4 = 2022+delta_fraction
    iHT_B4eruption = np.where((yrfrac_b>=yrfrac_b4)&(yrfrac_b<=yrfrac_start))[0]
    iHT_eruption = np.where((yrfrac_b>=yrfrac_start)&(yrfrac_b<=yrfrac_end))[0]
    iHT_AFeruption = np.where((yrfrac_b>=yrfrac_end)&(yrfrac_b<=yrfrac_after))[0]
    mm=50
    cmap='RdYlGn_r'#'RdYlGn_r'#'coolwarm'
    s1=6
    LEG = 'ssha [cm]'
    '''
    pbil.groundtracks_multi(lon_b[iHT_B4eruption],lat_b[iHT_B4eruption],(ssha_fft_b[iHT_B4eruption])*100.,'ICESat-2 ('+str(fp_b)+' m  footprint) wave signals removed. '+REG+' '+str(yrs_mm),LEG,
                cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',s1=s1)

    pbil.groundtracks_multi(lon_b[iHT_eruption],lat_b[iHT_eruption],(ssha_fft_b[iHT_eruption])*100.,'ICESat-2 ('+str(fp_b)+' m  footprint) wave signals removed. '+REG+' '+str(yrs_mm),LEG,
                    cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',s1=s1)

    pbil.groundtracks_multi(lon_b[iHT_AFeruption],lat_b[iHT_AFeruption],(ssha_fft_b[iHT_AFeruption])*100.,'ICESat-2 ('+str(fp_b)+' m  footprint) wave signals removed. '+REG+' '+str(yrs_mm),LEG,
                cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',s1=s1)
    '''

if REG=='carolina':
    d0 = date(2021, 1, 1)
    d1 = date(2021, 11, 6)
    delta = (d1-d0).total_seconds()
    delta_fraction = ((delta)/(24.*60.*60.))/365.25
    yrfrac_b4 = 2021+delta_fraction
    d1 = date(2021, 11, 8)
    delta = (d1-d0).total_seconds()
    delta_fraction = ((delta)/(24.*60.*60.))/365.25
    yrfrac_after = 2021+delta_fraction
    iHT_B4 = np.where(yrfrac_b<yrfrac_b4)[0]
    iHT_dur = np.where((yrfrac_b>=yrfrac_b4)&(yrfrac_b<=yrfrac_after))[0]
    iHT_AF = np.where(yrfrac_b>=yrfrac_after)[0]
    mm=50
    cmap='RdYlGn_r'#'RdYlGn_r'#'coolwarm'
    s1=6
    LEG = 'ssha [cm]'
    plt.figure()
    plt.subplot(131)
    plt.plot(lat_b[iHT_B4],(ssha_fft_b[iHT_B4])*100.)
    plt.subplot(132)
    plt.plot(lat_b[iHT_dur],(ssha_fft_b[iHT_dur])*100.)
    plt.subplot(133)
    plt.plot(lat_b[iHT_AF],(ssha_fft_b[iHT_AF])*100.)
    #'''
    pbil.groundtracks_multi(lon_b[iHT_B4],lat_b[iHT_B4],(ssha_fft_b[iHT_B4])*100.,'ICESat-2 ('+str(fp_b)+' m  footprint) wave signals removed. '+REG+' '+str(yrs_mm),LEG,
                cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',s1=s1)

    pbil.groundtracks_multi(lon_b[iHT_dur],lat_b[iHT_dur],(ssha_fft_b[iHT_dur])*100.,'ICESat-2 ('+str(fp_b)+' m  footprint) wave signals removed. '+REG+' '+str(yrs_mm),LEG,
                    cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',s1=s1)

    pbil.groundtracks_multi(lon_b[iHT_AF],lat_b[iHT_AF],(ssha_fft_b[iHT_AF])*100.,'ICESat-2 ('+str(fp_b)+' m  footprint) wave signals removed. '+REG+' '+str(yrs_mm),LEG,
                cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',s1=s1)
    #'''



if np.size(FNj3)!=0:
    swh_yrf_j3,N_yrf_j3 = find_averages(yrf_bins,yrfracA,swh_alt,imin=imin,MED=TF)
    innB_j3 = np.where((~np.isnan(ssha_yrf_c))&(~np.isnan(ssha_yrf_j3)))[0]

    Nsum=N_yrf_j3[innB_j3]+N_yrf_mn2_c[innB_j3]
    mnJ3 = np.empty(np.size(innB_j3))*np.nan
    mnIS2 = np.empty(np.size(innB_j3))*np.nan
    plt.figure(figsize=(16,8))
    plt.subplots_adjust(top=0.85,hspace=0.4,wspace=0.5)
    plt.suptitle('IS2 2 km SWH estimates vs. Jason-3 SWH (<= 1 hour separation) \n j3 = square, is2 = circle',fontsize=18) 
    if dyr == np.round((1./24.)/365.25,5):
        for ii in np.arange(np.size(innB_j3)):
            plt.subplot(2,4,ii+1)
            isum = ii#np.where(Nsum==np.nanmax(Nsum))[0][0]
            iday_b = np.where((yrfrac_b>=yrf_bins[innB_j3][isum]-dyr/2.0)&(yrfrac_b<yrf_bins[innB_j3][isum]+dyr/2.0)&(beam_b<10))[0]
            iday_c = np.where((yrfrac_c>=yrf_bins[innB_j3][isum]-dyr/2.0)&(yrfrac_c<yrf_bins[innB_j3][isum]+dyr/2.0)&(beam_c<10))[0]
            iday_j3 = np.where((yrfracA>=yrf_bins[innB_j3][isum]-dyr/2.0)&(yrfracA<yrf_bins[innB_j3][isum]+dyr/2.0))[0]
            cmap='viridis'
            #plt.scatter(lon_b[iday_b],swh66_b[iday_b])
            vmin,vmax = 0,4#np.nanmean(swh66_c[iday_c])-0.05,np.nanmean(swh_alt[iday_j3])+0.05
            Xt = np.round(np.abs(np.nanmean(swh_alt[iday_j3])/np.nanmean(swh66_c[iday_c])),3)
            SEP = np.round(np.abs(np.nanmean(yrfracA[iday_j3])-np.nanmean(yrfrac_c[iday_c]))*365*24,3)
            mJ3,mIS = np.round(np.nanmean(swh_alt[iday_j3]),2),np.round(np.nanmean(swh66_c[iday_c]),2)
            mnJ3[ii],mnIS2[ii]= mJ3,mIS 
            plt.scatter(lon_c[iday_c],lat_c[iday_c],c=swh66_c[iday_c],cmap=cmap,vmin=vmin,vmax=vmax,marker='o')
            plt.scatter(lon_alt[iday_j3],lat_alt[iday_j3],c=swh_alt[iday_j3],cmap=cmap,vmin=vmin,vmax=vmax,marker='s')
            cb = plt.colorbar()
            fs = 14
            if ii in [3,7]:
                cb.set_label('wave height',fontsize=fs)
            if ii>3:
                plt.xlabel('longitude [deg]',fontsize=fs)
            if ii in [0,4]:
                plt.ylabel('latitude [deg]',fontsize=fs)
            plt.title('SWH$_{j3}$ = '+str(Xt)+'*SWH$_{is2}$ \n j3('+str(mJ3)+' m), IS2('+str(mIS)+' m)') 
            #plt.title('j3('+str(mJ3)+' m), IS2('+str(mIS)+' m)')
    else:
        for ii in np.arange(np.size(innB_j3)):
            isum = ii#np.where(Nsum==np.nanmax(Nsum))[0][0]
            iday_b = np.where((yrfrac_b>=yrf_bins[innB_j3][isum]-dyr/2.0)&(yrfrac_b<yrf_bins[innB_j3][isum]+dyr/2.0)&(beam_b<10))[0]
            iday_c = np.where((yrfrac_c>=yrf_bins[innB_j3][isum]-dyr/2.0)&(yrfrac_c<yrf_bins[innB_j3][isum]+dyr/2.0)&(beam_c<10))[0]
            iday_s3 = np.where((yrfrac_s3>=yrf_bins[innB_j3][isum]-dyr/2.0)&(yrfrac_s3<yrf_bins[innB_j3][isum]+dyr/2.0))[0]
            mj3,mIS = np.round(np.nanmean(swh_s3[iday_s3]),2),np.round(np.nanmean(swh66_c[iday_c]),2)
            mnJ3[ii],mnIS2[ii]= mj3,mIS 
    '''
    plt.figure()
    plt.title('mean IS2 2 km SWH estimates vs. mean Jason-3 SWH (<= 1 hour separation)') 
    plt.plot(mnJ3,mnIS2,'o')
    plt.plot(np.arange(0,np.ceil(np.nanmax(mnIS2))+1),np.arange(0,np.ceil(np.nanmax(mnIS2))+1),color='black')
    plt.xlabel('mean J3 SWH [m]')
    plt.ylabel('mean IS2 SWH [m]')
    plt.grid()
    plt.ylim(0,np.ceil(np.nanmax(mnIS2)))
    plt.xlim(0,np.ceil(np.nanmax(mnIS2)))

    plt.figure()
    plt.title('mean IS2 2 km SWH estimates vs. mean Jason-3 SWH ('+str(int(dday))+' hour separation)') 
    plt.plot(yrf_bins[innB_j3],ssha_yrf_b[innB_j3],'.-',label='IS2 100 m')
    #plt.plot(yrf_bins[innB_j3],ssha_ws_yrf_b[innB_j3],'o-',label='IS2 100 m (WS)')
    bias_b_j3 = np.nanmean(ssha_yrf_j3[innB_j3]-ssha_yrf_b[innB_j3])
    plt.plot(yrf_bins[innB_j3],ssha_yrf_j3[innB_j3],'.-',label='Jason-3',color='grey')
    plt.xlabel('year fraction')
    plt.ylabel('SSHA [m]')
    #plt.fill_betweenx([yrfrac_start,yrfrac_end],color='red',alpha=0.5)
    plt.grid()
    plt.legend()
    

    Nj3i = np.size(innB_j3)
    for ii in np.arange(Nj3i):
        iseg_j3_b = np.where((yrfracA>=yrf_bins[innB_j3][ii]-dyr/2.0)&(yrfracA<=yrf_bins[innB_j3][ii]+dyr/2.0))[0]
        iseg_b_j3 = np.where((yrfrac_b>=yrf_bins[innB_j3][ii]-dyr/2.0)&(yrfrac_b<=yrf_bins[innB_j3][ii]+dyr/2.0))[0]
        pbil.groundtracks_multi(lon_b[iseg_b_j3],lat_b[iseg_b_j3],(ssha_fft_b[iseg_b_j3])*100.,'ICESat-2 ('+str(fp_b)+' m  footprint fft) with Jason-3. '+REG+' '+str(yrs_mm),LEG,
                        cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                        lon3=lon_alt[iseg_j3_b],lat3=lat_alt[iseg_j3_b],gt3=(ssha_alt[iseg_j3_b])*100.)
    '''

if np.size(FNs3)!=0:
    swh_yrf_s3,N_yrf_s3 = find_averages(yrf_bins,yrfrac_s3,swh_s3,imin=imin,MED=TF)
    innB_s3 = np.where((~np.isnan(ssha_yrf_c))&(~np.isnan(ssha_yrf_s3)))[0]
    Nsum=N_yrf_s3[innB_s3]+N_yrf_mn2_c[innB_s3]
    mnS3 = np.empty(np.size(innB_s3))*np.nan
    mnIS2 = np.empty(np.size(innB_s3))*np.nan
    plt.figure(figsize=(12,5))
    plt.subplots_adjust(top=0.7,hspace=0.4,wspace=0.5)
    plt.suptitle('IS2 2 km SWH estimates vs. Sentinel-3 SWH (<= 1 hour separation) \n s3 = square, is2 = circle',fontsize=18) 
    if dyr == np.round((1./24.)/365.25,5):
        for ii in np.arange(np.size(innB_s3)):
            plt.subplot(1,3,ii+1)
            isum = ii#np.where(Nsum==np.nanmax(Nsum))[0][0]
            iday_b = np.where((yrfrac_b>=yrf_bins[innB_s3][isum]-dyr/2.0)&(yrfrac_b<yrf_bins[innB_s3][isum]+dyr/2.0)&(beam_b<10))[0]
            iday_c = np.where((yrfrac_c>=yrf_bins[innB_s3][isum]-dyr/2.0)&(yrfrac_c<yrf_bins[innB_s3][isum]+dyr/2.0)&(beam_c<10))[0]
            iday_s3 = np.where((yrfrac_s3>=yrf_bins[innB_s3][isum]-dyr/2.0)&(yrfrac_s3<yrf_bins[innB_s3][isum]+dyr/2.0))[0]
            cmap='viridis'
            #plt.scatter(lon_b[iday_b],swh66_b[iday_b])
            vmin,vmax = 0,4#np.nanmean(swh66_c[iday_c])-0.05,np.nanmean(swh_alt[iday_s3])+0.05
            Xt = np.round(np.abs(np.nanmean(swh_s3[iday_s3])/np.nanmean(swh66_c[iday_c])),3)
            SEP = np.round(np.abs(np.nanmean(yrfracA[iday_s3])-np.nanmean(yrfrac_c[iday_c]))*365*24,3)
            ms3,mIS = np.round(np.nanmean(swh_s3[iday_s3]),2),np.round(np.nanmean(swh66_c[iday_c]),2)
            mnS3[ii],mnIS2[ii]= ms3,mIS 
            plt.scatter(lon_c[iday_c],lat_c[iday_c],c=swh66_c[iday_c],cmap=cmap,vmin=vmin,vmax=vmax,marker='o')
            plt.scatter(lon_s3[iday_s3],lat_s3[iday_s3],c=swh_s3[iday_s3],cmap=cmap,vmin=vmin,vmax=vmax,marker='s')
            cb = plt.colorbar()
            fs = 14
            if ii in [2]:
                cb.set_label('wave height',fontsize=fs)
            plt.xlabel('longitude [deg]',fontsize=fs)
            if ii in [0,4]:
                plt.ylabel('latitude [deg]',fontsize=fs)
            plt.title('SWH$_{s3}$ = '+str(Xt)+'*SWH$_{is2}$ \n s3('+str(ms3)+' m), IS2('+str(mIS)+' m)') 
            #plt.title('s3('+str(ms3)+' m), IS2('+str(mIS)+' m)')
    else:
        for ii in np.arange(np.size(innB_s3)):
            isum = ii#np.where(Nsum==np.nanmax(Nsum))[0][0]
            iday_b = np.where((yrfrac_b>=yrf_bins[innB_s3][isum]-dyr/2.0)&(yrfrac_b<yrf_bins[innB_s3][isum]+dyr/2.0)&(beam_b<10))[0]
            iday_c = np.where((yrfrac_c>=yrf_bins[innB_s3][isum]-dyr/2.0)&(yrfrac_c<yrf_bins[innB_s3][isum]+dyr/2.0)&(beam_c<10))[0]
            iday_s3 = np.where((yrfrac_s3>=yrf_bins[innB_s3][isum]-dyr/2.0)&(yrfrac_s3<yrf_bins[innB_s3][isum]+dyr/2.0))[0]
            ms3,mIS = np.round(np.nanmean(swh_s3[iday_s3]),2),np.round(np.nanmean(swh66_c[iday_c]),2)
            mnS3[ii],mnIS2[ii]= ms3,mIS 
    '''
    plt.figure()
    plt.title('mean IS2 2 km SWH estimates vs. mean Senitnel-3 SWH (<= 1 hour separation)') 
    plt.plot(mnS3,mnIS2,'o')
    plt.plot(np.arange(0,np.ceil(np.nanmax(mnS3))+1),np.arange(0,np.ceil(np.nanmax(mnS3))+1),color='black')
    plt.xlabel('mean S3 SWH [m]')
    plt.ylabel('mean IS2 SWH [m]')
    plt.grid()
    plt.ylim(0,np.ceil(np.nanmax(mnS3)))
    plt.xlim(0,np.ceil(np.nanmax(mnS3)))
    

    Ns3i = np.size(innB_s3)
    for ii in np.arange(Ns3i):
        iseg_s3_b = np.where((yrfrac_s3>=yrf_bins[innB_s3][ii]-dyr/2.0)&(yrfrac_s3<=yrf_bins[innB_s3][ii]+dyr/2.0))[0]
        iseg_b_s3 = np.where((yrfrac_b>=yrf_bins[innB_s3][ii]-dyr/2.0)&(yrfrac_b<=yrf_bins[innB_s3][ii]+dyr/2.0))[0]
        pbil.groundtracks_multi(lon_b[iseg_b_s3],lat_b[iseg_b_s3],(ssha_fft_b[iseg_b_s3])*100.,'ICESat-2 ('+str(fp_b)+' m  footprint fft) with Jason-3. '+REG+' '+str(yrs_mm),LEG,
                        cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                        lon3=lon_s3[iseg_s3_b],lat3=lat_s3[iseg_s3_b],gt3=(ssha_s3[iseg_s3_b])*100.)
    '''




mrk='.-'
'''
swh66_yrf_b_B1,N66_yrf_mn2_b_B1 = find_averages(yrf_bins,yrfrac_b[beam_b==1],swh66_b[beam_b==1],imin=imin,MED=TF)
swh66_yrf_c_B1,N66_yrf_mn2_c_B1 = find_averages(yrf_bins,yrfrac_c[beam_c==1],swh66_c[beam_c==1],imin=imin,MED=TF)

swh66_yrf_b_B2,N66_yrf_mn2_b_B2 = find_averages(yrf_bins,yrfrac_b[beam_b==2],swh66_b[beam_b==2],imin=imin,MED=TF)
swh66_yrf_c_B2,N66_yrf_mn2_c_B2 = find_averages(yrf_bins,yrfrac_c[beam_c==2],swh66_c[beam_c==2],imin=imin,MED=TF)

swh66_yrf_b_B3,N66_yrf_mn2_b_B3 = find_averages(yrf_bins,yrfrac_b[beam_b==3],swh66_b[beam_b==3],imin=imin,MED=TF)
swh66_yrf_c_B3,N66_yrf_mn2_c_B3 = find_averages(yrf_bins,yrfrac_c[beam_c==3],swh66_c[beam_c==3],imin=imin,MED=TF)

swh66_yrf_b_B10,N66_yrf_mn2_b_B10 = find_averages(yrf_bins,yrfrac_b[beam_b==10],swh66_b[beam_b==10],imin=imin,MED=TF)
swh66_yrf_c_B10,N66_yrf_mn2_c_B10 = find_averages(yrf_bins,yrfrac_c[beam_c==10],swh66_c[beam_c==10],imin=imin,MED=TF)

swh66_yrf_b_B20,N66_yrf_mn2_b_B20 = find_averages(yrf_bins,yrfrac_b[beam_b==20],swh66_b[beam_b==20],imin=imin,MED=TF)
swh66_yrf_c_B20,N66_yrf_mn2_c_B20 = find_averages(yrf_bins,yrfrac_c[beam_c==20],swh66_c[beam_c==20],imin=imin,MED=TF)

swh66_yrf_b_B30,N66_yrf_mn2_b_B30 = find_averages(yrf_bins,yrfrac_b[beam_b==30],swh66_b[beam_b==30],imin=imin,MED=TF)
swh66_yrf_c_B30,N66_yrf_mn2_c_B30 = find_averages(yrf_bins,yrfrac_c[beam_c==30],swh66_c[beam_c==30],imin=imin,MED=TF)

plt.figure(figsize=(10,11))
plt.subplots_adjust(top=0.85,hspace=0.4,wspace=0.4)
plt.suptitle('SWH comparison between beams')
plt.subplot(211)
plt.plot(yrf_bins,swh66_yrf_b_B1,mrk,label='gt1r',color='red')
plt.plot(yrf_bins,swh66_yrf_b_B2,mrk,label='gt2r',color='green')
plt.plot(yrf_bins,swh66_yrf_b_B3,mrk,label='gt3r',color='blue')
plt.plot(yrf_bins,swh66_yrf_b_B10,mrk,label='gt1l',color='indianred')
plt.plot(yrf_bins,swh66_yrf_b_B20,mrk,label='gt2l',color='mediumseagreen')
plt.plot(yrf_bins,swh66_yrf_b_B30,mrk,label='gt3l',color='cornflowerblue')
plt.legend()
plt.xlabel('year fraction')
plt.ylabel('SWH [m]')
plt.grid()
plt.title('IS2 100 m SWH')
plt.subplot(212)
plt.plot(yrf_bins,swh66_yrf_c_B1,mrk,label='gt1r',color='red')
plt.plot(yrf_bins,swh66_yrf_c_B2,mrk,label='gt2r',color='green')
plt.plot(yrf_bins,swh66_yrf_c_B3,mrk,label='gt3r',color='blue')
plt.plot(yrf_bins,swh66_yrf_c_B10,mrk,label='gt1l',color='indianred')
plt.plot(yrf_bins,swh66_yrf_c_B20,mrk,label='gt2l',color='mediumseagreen')
plt.plot(yrf_bins,swh66_yrf_c_B30,mrk,label='gt3l',color='cornflowerblue')
plt.legend()
plt.xlabel('year fraction')
plt.ylabel('SWH [m]')
plt.grid()
plt.title('IS2 2 km SWH')
'''



#############
#############
## Morrison improvement figure
#############
#############
iyf=np.where(np.round(yrfrac_a,1)==2021.2)[0]
BEAM = 2
sBEAM = 'gt'+str(BEAM)+'r'
ist=0#300000#swell=500000, seas=300000+600
ien=ist+800#2000
#icut,dist_a = dist_sv2pt(lat_a,lon_a,beam_a,lat_a[ist],lon_a[ist],maxKM=2000,beam=[])#np.where((~np.isnan(ssha_a)))[0][ist:ien]
icut =np.where((~np.isnan(ssha_a))&(beam_a==BEAM))[0][ist:ien]
dist_a=(days_since_1985_a[icut]-days_since_1985_a[icut][0])*(24*60*60)*7000.
mm=2#mm=10
lw=3
pmax = 200

'''
def fft2signal(x,y,Nf=66):
    # https://scipy-lectures.org/intro/scipy/auto_examples/plot_fftpack.html
    # Detrend signal
    #x,y,Nf=t_a,ssha_a[icut],110
    import scipy
    ce = np.polyfit(x, y, 1)[::-1]
    fity = ce[0]+ce[1]*x
    sig = y-fity
    # The FFT of the signal
    time_vec = x#days_since_1985_a[i20A2]
    dt = np.diff(time_vec)#*(24*60*60))
    time_step = np.nanmedian(dt)
    sig_fft = scipy.fftpack.fft(sig)
    # And the power (sig_fft is of complex dtype)
    power = np.abs(sig_fft)**2
    # The corresponding frequencies
    sample_freq = scipy.fftpack.fftfreq(sig.size, d=time_step)
    # Find the peak frequency: we can focus on only the positive frequencies
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]
    # remove all the high frequencies
    idx = np.argsort(np.abs(sample_freq))
    high_freq_fft = sig_fft.copy()
    idxF = np.where(np.abs(sample_freq)>Nf)[0]
    high_freq_fft[idxF]=0#[idx[Nf+1:]] = 0#high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
    filtered_sig = scipy.fftpack.ifft(high_freq_fft)
    # dominant frequencies
    idxF2 = np.where(np.abs(sample_freq)<=Nf)[0]
    new_freq_arr = sample_freq[idxF2]#[idx[:Nf+1]]#np.copy(sample_freq[np.abs(sample_freq) < peak_freq])
    #print(new_freq_arr)
    #print(high_freq_fft[idx[:Nf+1]])
    return filtered_sig.real,new_freq_arr

fitAno0,fA = fft2signal(t_a,ssha_a[icut],Nf=110)
plt.figure()
plt.plot(dist_a,ssha_a[icut])
plt.plot(dist_a,fitAno0)
plt.figure()
plt.plot(dist_a,ssha_a[icut]-fitAno0)
'''
t_a = (days_since_1985_a[icut]-days_since_1985_a[icut][0])*(24*60*60)
spectral_ssha(t_a,ssha_a[icut])

plt.figure(figsize=(14,10))
plt.subplots_adjust(top=0.9,hspace=0.4,wspace=0.4)
plt.suptitle('ICESat-2 ('+sBEAM+') ocean profile')
plt.subplot(221)
plt.title('2 m SSHA means \n std(ssha) = '+str(np.round(np.nanstd(ssha_a[icut]),4)))
plt.plot(dist_a,ssha_a[icut],'-',label='ssha')
plt.xlabel('distance [m]')
plt.ylabel('ssha [m]')
plt.legend()
plt.grid()
plt.ylim(-mm,mm)
plt.xlim(0,pmax)
plt.subplot(222)
plt.title('2 m SSHA means with SW')
plt.plot(dist_a,ssha_a[icut],'-',label='ssha')
plt.plot(dist_a,swell_a[icut],'-',label='SW',linewidth=lw)
imm=np.where(ip_a[icut]==1)[0]
plt.plot(dist_a[imm],swell_a[icut][imm],'o',label='max/min',color='red')
plt.xlabel('distance [m]')
plt.ylabel('ssha [m]')
plt.legend()
plt.grid()
plt.ylim(-mm,mm)
plt.xlim(0,pmax)
plt.subplot(223)
plt.title('2 m SSHA means-SW \n std(ssha-LFSW) = '+str(np.round(np.nanstd(ssha_a[icut]-swell_a[icut]),4)))
plt.plot(dist_a,ssha_a[icut]-swell_a[icut],'-',label='ssha-wave signal')
plt.xlabel('distance [m]')
plt.ylabel('ssha [m]')
plt.legend()
plt.grid()
plt.ylim(-mm,mm)
plt.xlim(0,pmax)
plt.subplot(224)
plt.title('point distributions')
binz = np.arange(-2,2.05,0.05)
from scipy import stats
h1=ssha_a[icut]
pdf1 = stats.norm.pdf(x = binz, loc=np.nanmean(h1), scale=np.nanstd(h1))
plt.plot(binz,pdf1,label='ssha')
h2=ssha_a[icut]-swell_a[icut]
pdf2 = stats.norm.pdf(x = binz, loc=np.nanmean(h2), scale=np.nanstd(h2))
plt.plot(binz,pdf2,label='ssha-SW')
plt.xlabel('ssha [m]')
plt.ylabel('frequency')
plt.legend()
plt.grid()





icut_b1,max_dayb = pull_day(ymdhmsI_b,y=[],m=[],d=[],hr=[],mi=[],se=[],elem=0) 
icut_b =np.intersect1d(np.where((beam_b==BEAM)&(~np.isnan(ssha_b)))[0],icut_b1)
dist_b=(days_since_1985_b[icut_b]-days_since_1985_b[icut_b][0])*(24*60*60)*7000.
icut_c1,max_dayc = pull_day(ymdhmsI_c,y=[],m=[],d=[],hr=[],mi=[],se=[],elem=max_dayb) 
icut_c =np.intersect1d(np.where((beam_c==BEAM)&(~np.isnan(ssha_c)))[0],icut_c1)
dist_c=(days_since_1985_c[icut_c]-days_since_1985_c[icut_c][0])*(24*60*60)*7000.
icut_a1,max_daya = pull_day(ymdhmsI_a,y=[],m=[],d=[],hr=[],mi=[],se=[],elem=max_dayb) 
icut_a =np.intersect1d(np.where((beam_a==BEAM)&(~np.isnan(ssha_a)))[0],icut_a1)
dist_a=(days_since_1985_a[icut_a]-days_since_1985_a[icut_a][0])*(24*60*60)*7000.

mm = 5
scl=1
plt.figure(figsize=(14,5))
plt.subplots_adjust(top=0.84,hspace=0.4,wspace=0.4)
plt.suptitle('ICESat-2 ('+sBEAM+') ocean profile')
plt.subplot(131)
plt.title('2 m measurement distributions')
binz = np.arange(-5,5.05,0.05)
from scipy import stats
h1=ssha_a
pdf1 = stats.norm.pdf(x = binz, loc=np.nanmean(h1), scale=np.nanstd(h1))
plt.plot(binz,pdf1,label='ssha')
h2=ssha_a-swell_a
pdf2 = stats.norm.pdf(x = binz, loc=np.nanmean(h2), scale=np.nanstd(h2))
plt.plot(binz,pdf2,label='ssha-wave signal')
plt.xlabel('ssha [m]')
plt.ylabel('frequency')
plt.legend()
plt.grid()
plt.xlim(-mm,mm)
plt.subplot(132)
plt.title('100 m measurement distributions')
binz = np.arange(-5,5.05,0.05)
h1=ssha_b
pdf1 = stats.norm.pdf(x = binz, loc=np.nanmean(h1), scale=np.nanstd(h1))
plt.plot(binz,pdf1,label='ssha')
h2=ssha_fft_b
pdf2 = stats.norm.pdf(x = binz, loc=np.nanmean(h2), scale=np.nanstd(h2))
plt.plot(binz,pdf2,label='ssha-wave signal')
plt.xlabel('ssha [m]')
plt.ylabel('frequency')
plt.legend()
plt.grid()
plt.xlim(-mm,mm)
plt.subplot(133)
plt.title('2 km measurement distributions')
binz = np.arange(-5,5.05,0.05)
h1=ssha_c
pdf1 = stats.norm.pdf(x = binz, loc=np.nanmean(h1), scale=np.nanstd(h1))
plt.plot(binz,pdf1,label='ssha')
h2=ssha_fft_c
pdf2 = stats.norm.pdf(x = binz, loc=np.nanmean(h2), scale=np.nanstd(h2))
plt.plot(binz,pdf2,label='ssha-wave signal')
plt.xlabel('ssha [m]')
plt.ylabel('frequency')
plt.legend()
plt.grid()
plt.xlim(-mm,mm)

plt.figure(figsize=(8,5))
plt.subplots_adjust(top=0.7,hspace=0.5,wspace=0.4)
plt.suptitle('IS2 (strong beams) \n J3 footprint >(3 x 3 $km^2$) \n S3 footprint ~(1.64 x 1.64 km$^2$)')
plt.subplot(111)
plt.title('2 m SSHA means ('+sBEAM+') with overlaid wave signal')
plt.plot(dist_a,ssha_a[icut_a],'-',label='ssha')
plt.plot(dist_a,swell_a[icut_a],'-',label='wave signal',linewidth=1)
imm=np.where(ip_a[icut_a]==1)[0]
plt.plot(dist_a[imm],swell_a[icut_a][imm],'o',label='max/min',color='red')
plt.xlabel('distance [m]')
plt.ylabel('ssha [m]')
plt.legend()
plt.grid()
mm = 2 
plt.xlim(0,pmax)



if np.size(FNj3)!=0 and np.size(FNs3)!=0:
    innB = np.where((~np.isnan(ssha_yrf_c))&(~np.isnan(ssha_yrf_s3))&(~np.isnan(ssha_yrf_j3)))[0]
    plt.figure(figsize=(10,10))
    plt.subplots_adjust(top=0.80,hspace=0.5,wspace=0.4)
    plt.suptitle('IS2 (strong beams) \n S3 footprint ~(1.64 x 1.64 km$^2$) \n J3 footprint >(3.5 x 3.5 $km^2$)')
    plt.subplot(211)
    plt.plot(yrf_bins[innB_s3],swh_yrf_b[innB_s3],mrk,label='IS2 100 m (4$\sigma$)',color='red')#(4$\sigma$)
    plt.plot(yrf_bins[innB_s3],swh66_yrf_b[innB_s3],mrk,label='IS2 100 m (SW)',color='green')
    plt.plot(yrf_bins[innB_s3],swh_yrf_s3[innB_s3],mrk,label='S3',color='black')
    plt.plot(yrf_bins[innB_j3],swh_yrf_j3[innB_j3],mrk,label='J3',color='gray')
    plt.legend()
    plt.xlabel('year fraction')
    plt.ylabel('SWH [m]')
    plt.grid()
    plt.title(AVG+' SWH  vs. year fraction \n 4 day separation maximum')
    plt.subplot(212)
    plt.plot(yrf_bins[innB_s3],swh_yrf_c[innB_s3],mrk,label='2 km (4$\sigma$)',color='red')
    plt.plot(yrf_bins[innB_s3],swh66_yrf_c[innB_s3],mrk,label='IS2 2 km (SW)',color='green')
    plt.plot(yrf_bins[innB_s3],swh_yrf_s3[innB_s3],mrk,label='S3',color='black')
    plt.plot(yrf_bins[innB_j3],swh_yrf_j3[innB_j3],mrk,label='J3',color='gray')
    plt.legend()
    plt.xlabel('year fraction')
    plt.ylabel('SWH [m]')
    plt.grid()
    plt.title(AVG+' SWH  vs. year fraction \n 4 day separation maximum')

    plt.figure(figsize=(10,10))
    plt.subplots_adjust(top=0.87,hspace=0.5,wspace=0.4)
    plt.suptitle('IS2 (strong beams) \n S3 footprint ~(1.64 x 1.64 km$^2$) \n J3 footprint >(3.5 x 3.5 $km^2$)')
    plt.subplot(211)
    plt.plot(yrf_bins[innB_s3],ssha_yrf_b[innB_s3],mrk,label='IS2 100 m w/o SW',color='red')#(4$\sigma$)
    plt.plot(yrf_bins[innB_s3],ssha_ws_yrf_b[innB_s3],mrk,label='IS2 100 m ',color='green')
    plt.plot(yrf_bins[innB_s3],ssha_yrf_s3[innB_s3],mrk,label='S3',color='black')
    plt.plot(yrf_bins[innB_j3],ssha_yrf_j3[innB_j3],mrk,label='J3',color='gray')
    plt.legend()
    plt.xlabel('year fraction')
    plt.ylabel('SSHA [m]')
    plt.grid()
    plt.title(AVG+' SSHA  vs. year fraction \n 4 day separation maximum')
    plt.subplot(212)
    plt.plot(yrf_bins[innB_s3],ssha_yrf_c[innB_s3],mrk,label='IS2 2 km w/o SW',color='red')
    plt.plot(yrf_bins[innB_s3],ssha_ws_yrf_c[innB_s3],mrk,label='IS2 2 km',color='green')
    plt.plot(yrf_bins[innB_s3],ssha_yrf_s3[innB_s3],mrk,label='S3',color='black')
    plt.plot(yrf_bins[innB_j3],ssha_yrf_j3[innB_j3],mrk,label='J3',color='gray')
    plt.legend()
    plt.xlabel('year fraction')
    plt.ylabel('SSHA [m]')
    plt.grid()
    plt.title(AVG+' SSHA  vs. year fraction \n 4 day separation maximum')

icheck = np.where((yrfrac_c>2021.7)&(yrfrac_c<2021.9))[0]
plt.figure()
plt.plot(yrfrac_c[icheck],ssha_c[icheck],'.')
plt.plot(yrfrac_c[icheck],ssha_fft_c[icheck],'.')


db=10
binz = np.arange(-400,400+db,db)
hista,beb=np.histogram(ssha_a[icut_a]*100,bins=binz)
histb,beb=np.histogram(ssha_b[icut_b]*100,bins=binz)
histc,bec=np.histogram(ssha_c[icut_c]*100,bins=binz)
hista_norm = hista/np.max(hista)
histb_norm = histb/np.max(histb)
histc_norm = histc/np.max(histc)
histaF,beb=np.histogram(ssha_fft_a[icut_a]*100,bins=binz)
histbF,beb=np.histogram(ssha_fft_b[icut_b]*100,bins=binz)
histcF,bec=np.histogram(ssha_fft_c[icut_c]*100,bins=binz)
histaF_norm = histaF/np.max(histaF)
histbF_norm = histbF/np.max(histbF)
histcF_norm = histcF/np.max(histcF)


pmax=10
plt.figure(figsize=(14,6))
plt.subplots_adjust(top=0.7,hspace=0.4,wspace=0.4)
plt.suptitle('validation: removing wave signal improves \n SSHA ('+sBEAM+') estimates of short segments')
ssha_fft_b_c = subseg_to_seg(ssha_fft_b[icut_b],days_since_1985_b[icut_b],days_since_1985_c[icut_c])
ssha_b_c = subseg_to_seg(ssha_b[icut_b],days_since_1985_b[icut_b],days_since_1985_c[icut_c])
plt.subplot(121)
plt.title('2km ATLCU vs. subsets')
plt.plot(dist_c[1:-1]/1000.,(ssha_c[icut_c[1:-1]])*100.,'-',label='2km ATLCU')
plt.plot(dist_c[1:-1]/1000.,(ssha_b_c[1:-1])*100.,'-',label='2km subset from 100 m (wave signal)')
plt.plot(dist_c[1:-1]/1000.,(ssha_fft_b_c[1:-1])*100.,'-',label='2km subset from 100 m (no wave signal)')
plt.xlabel('distance [km]')
plt.ylabel('ssha [cm]')
plt.legend()
plt.grid()
plt.subplot(122)
rms_b_c = np.sqrt(np.sum(np.square((ssha_c[icut_c[1:-1]]-ssha_b_c[1:-1])*100.)))
RMSbc = 'rms($\Delta_{Waves}$) = '+str(np.round(rms_b_c,3))+' cm'
rms_fft_b_c = np.sqrt(np.sum(np.square((ssha_c[icut_c[1:-1]]-ssha_fft_b_c[1:-1])*100.)))
RMSfftbc = 'rms($\Delta_{NoWaves}$) = '+str(np.round(rms_fft_b_c,3))+' cm'
plt.title('2km ATLCU - subsets \n '+RMSbc+' \n ' +RMSfftbc)
plt.plot(dist_c[1:-1]/1000.,(ssha_c[icut_c[1:-1]]-ssha_b_c[1:-1])*100.,'-',label='2km ATLCU - 2km subset from 100 m (wave signal)')
plt.plot(dist_c[1:-1]/1000.,(ssha_c[icut_c[1:-1]]-ssha_fft_b_c[1:-1])*100.,'-',label='2km ATLCU - 2km subset from 100 m (no wave signal)')
plt.xlabel('distance [km]')
plt.ylabel('$\Delta$ssha [cm]')
plt.legend()
plt.grid()
plt.ylim(-2,2)
#plt.xlim(0,pmax)



#############
#############
## SSHA dependencies (SWH, steepness and wavelength))
#############
#############
plt.figure()
plt.grid()
swh_bins = np.arange(0,12,0.25)
ce =  np.polyfit(swh_c,ssha_fft_c,1)[::-1]
fit = ce[0]+(ce[1]*swh_bins)#+(ce[2]*swh_bins**2)+(ce[3]*swh_bins**3)
ce66 =  np.polyfit(swh66_c,ssha_fft_c,1)[::-1]
fit66 = ce66[0]+(ce66[1]*swh_bins)
plt.plot(swh_c,ssha_fft_c,'.',label='SWH from LFSW & HFSW',color='cornflowerblue')
plt.plot(swh66_c,ssha_fft_c,'.',label='SWH from SW',color='indianred')
plt.plot(swh_bins,fit,color='blue')
plt.plot(swh_bins,fit66,color='red')
plt.xlabel('SWH [m]')
plt.ylabel('SSHA [m]')
plt.title('SSHA vs. SWH')
plt.legend()




#############
#############
## monthly SSHA means vs. tide gauges
#############
#############
if np.size(FNtg2)!=0:
    dl,dh = 0.1,6
    iis2,itg = lTG.tg_2_is2(lltg2,yrfrac_tg2,lat_b,lon_b,yrfrac_b,dl=dl,dh=dh)
    mn_ssha_b_cut,vr_ssha_b_cut,mn_time_b_cut = lTG.month_2_month(ymdhmsI_b[iis2,:],ssha_b[iis2],yrs_mm)#,LATLON = np.vstack((lat_b,lon_b)).T) #ymdhmsI_b[idx_b],ssha_b[idx_b],days_since_1985_b[idx_b]
    if np.shape(sl_tg2)[0]== np.size(sl_tg2):
        itg_int = itg[~np.isnan(itg)].astype(int)
        mn_ssha_tg2_cut,vr_ssha_tg2_cut,mn_time_tg2_cut = lTG.month_2_month(ymdhms_tg2[itg_int,:],sl_tg2[itg_int],yrs_mm,IS2=False)
    elif np.shape(sl_tg2)[0]!= np.size(sl_tg2):
        mn_ssha_tg2_cut,mn_time_tg2_cut,vr_ssha_tg2_cut = np.empty(np.shape(mn_ssha_tg2))*np.nan,np.empty(np.shape(mn_ssha_tg2))*np.nan,np.empty(np.shape(mn_ssha_tg2))*np.nan
        for ii in np.arange(Ntg2):
            itg_int = itg[ii,:][~np.isnan(itg[ii,:])].astype(int)
            mn_ssha_tg2_cut[ii,:],vr_ssha_tg2_cut[ii,:],mn_time_tg2_cut[ii,:] = lTG.month_2_month(ymdhms_tg2[ii,itg_int,:],sl_tg2[ii,itg_int],yrs_mm,IS2=False)

    mn_mn_ssha_tg2_cut = np.nanmean(mn_ssha_tg2_cut,axis=0)
    plt.figure(figsize=(12,8))
    bias_is2_2_tg = np.nanmean(mn_ssha_b_cut-mn_mn_ssha_tg2_cut)
    plt.plot(mn_time_b_cut,np.array(mn_ssha_b_cut-bias_is2_2_tg)*100.,'.-',label='IS2 100 m (bias='+str(np.round(bias_is2_2_tg*100,3))+' cm)')
    plt.plot(mn_time_tg2[0,:],np.array(mn_mn_ssha_tg2_cut)*100.,'.-',label='mean of all TG2')
    for ii in np.arange(np.shape(mn_ssha_tg2)[0]):
        lab = FNtg2[ii][13:-4]
        plt.plot(mn_time_tg2[0,:],np.array(mn_ssha_tg2_cut[ii,:])*100.,'s',label=lab)
    plt.legend()
    plt.grid()
    plt.xlabel('year fraction')
    plt.ylabel('monthly SSHA means [cm]')
    plt.title(REG) 

    plt.figure(figsize=(18,18))
    for ii in np.arange(np.shape(mn_ssha_tg2)[0]):
        plt.subplot(3,3,ii+1)
        itg_int = itg[ii,:][~np.isnan(itg[ii,:])].astype(int)
        lab = FNtg2[ii][13:-4]
        lat_tg,lon_tg = lltg2[ii][0],lltg2[ii][1]
        iis2_ll = np.where((lat_b[iis2]>=lat_tg-dl)&(lat_b[iis2]<=lat_tg+dl)&(lon_b[iis2]>=lon_tg-dl)&(lon_b[iis2]<=lon_tg+dl))[0]
        plt.plot(yrfrac_b[iis2][iis2_ll],(ssha_b[iis2][iis2_ll]-bias_is2_2_tg)*100.,'>',label='IS2 100 m (bias='+str(np.round(bias_is2_2_tg*100,3))+' cm)',color='cornflowerblue')
        plt.plot(yrfrac_tg2[ii,itg_int],(sl_tg2)[ii,itg_int]*100.,'<',label='Tide Gauge',color='indianred')
        plt.plot(mn_time_b_cut,np.array(mn_ssha_b_cut-bias_is2_2_tg)*100.,'.-',label='mean IS2 100 m (bias='+str(np.round(bias_is2_2_tg*100,3))+' cm)',color='navy')
        plt.plot(mn_time_tg2[0,:],np.array(mn_mn_ssha_tg2_cut)*100.,'.-',label='mean of all TG2',color='maroon')
        plt.legend()
        plt.grid()
        plt.xlabel('year fraction')
        plt.ylabel('monthly SSHA means [cm]')
        plt.title(lab)#(REG) 
        #plt.xlim(2020.43,2020.45)#(2020,2021)
        #plt.ylim(-130,130)
    '''
    LEG = 'ssha [cm]'
    frac = np.append(2020,mn_time_b)#np.arange(2020,2021.1,0.1)
    Nfrac = np.size(frac)
    sp=0
    cmap='RdYlGn_r'
    mm=50
    proj=180.
    sp = Nfrac-1
    if sp%2==0:
        ax_sp = [sp/2,2]
        fs = [14,22]
    elif sp%3==0:
        ax_sp = [sp/3,3]
        fs=[12,18]
    fig=plt.figure(figsize=(int(fs[0]),int(fs[1])))
    plt.subplots_adjust(hspace=0.4)
    plt.suptitle('ICESat-2 ('+str(fp_b)+' m  footprint) with NOAA TGs. '+REG,fontsize=20)
    for ii in np.arange(Nfrac-1):
        iyrf_c2 = np.where((yrfrac_c2>=frac[ii])&(yrfrac_c2<frac[ii+1]))[0]
        iyrf_b = np.where((yrfrac_b>=frac[ii])&(yrfrac_b<frac[ii+1]))[0]
        TIT=str(np.round(frac[ii+1],2))#'ICESat-2 ('+str(fp_b)+' m  footprint) with NOAA TGs. \n '+REG+' '+str(np.round(frac[ii],1))
        ax=plt.subplot(int(ax_sp[0]),int(ax_sp[1]),ii+1, projection=ccrs.EckertIV(central_longitude=proj))
        if np.size(iyrf_c2)>0:   
            pbil.groundtracks_multi_subplot(fig,ax,lon_b[iyrf_b],lat_b[iyrf_b],(ssha_b[iyrf_b]-bias_is2_2_tg)*100.,TIT,LEG,
                    cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                    lon2=lon_tg2,lat2=lat_tg2,gt2=ss_tg2*100,ss=20)
        else:
            pbil.groundtracks_multi_subplot(fig,ax,lon_b[iyrf_b],lat_b[iyrf_b],(ssha_b[iyrf_b]-bias_is2_2_tg)*100.,TIT,LEG,
                    cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                    lon2=lon_tg2,lat2=lat_tg2,gt2=ss_tg2*100,ss=20)
    '''

if np.size(FNc2)!=0:
    ll = np.vstack((lat_c2,lon_c2)).T
    llc2 = []
    for ii in np.arange(np.size(lat_c2)):
        llc2.append([lat_c2[ii],lon_c2[ii]])
    iis2c2,ic2 = lTG.tg_2_is2(llc2,yrfrac_c2,lat_b,lon_b,yrfrac_b,dl=1,dh=36)
    ic2_int = ic2[~np.isnan(ic2)].astype(int)
    mn_ssha_b_c2_cut,vr_ssha_b_c2_cut,mn_time_b_c2_cut = lTG.month_2_month(ymdhmsI_b[iis2c2,:],ssha_b[iis2c2],yrs_mm)#,LATLON = np.vstack((lat_b,lon_b)).T) #ymdhmsI_b[idx_b],ssha_b[idx_b],days_since_1985_b[idx_b]
    mn_ssha_c2_cut,vr_ssha_c2_cut,mn_time_c2_cut = lTG.month_2_month(ymdhms_c2[ic2_int],ssha_c2[ic2_int],yrs_mm)



#############
#############
## IS2 vs. other missions
#############
#############

Nbins = np.size(swh_bins)
dswh = np.diff(swh_bins)[0]/2.
dyr = np.round(30/365.25,4)
imin=0
yrf_bins = np.arange(2021,2022+dyr,dyr)
mn_i2b_bins,N_mn_i2b_bins = find_averages(swh_bins,swh_b,ssha_fft_b,imin=imin)#,z=yrfrac_b,zconst=yrf_bins,imin=imin)
mn_i2c_bins,N_mn_i2c_bins = find_averages(swh_bins,swh_c,ssha_fft_c,imin=imin)#,z=yrfrac_c,zconst=yrf_bins,imin=imin)
if np.size(FNj3)!=0:
    mn_j3_bins,N_mn_j3_bins = find_averages(swh_bins,swh_alt,ssha_alt,imin=imin)#,z=yrfracA,zconst=yrf_bins,imin=imin)
if np.size(FNs3)!=0:
    mn_s3_bins,N_mn_s3_bins = find_averages(swh_bins,swh_s3,ssha_s3,imin=imin)#,z=yrfrac_s3,zconst=yrf_bins,imin=imin)

plt.figure(figsize=(8,6))
plt.plot(swh_bins,mn_i2b_bins,'-',label='is2 (100 m)')
plt.plot(swh_bins,mn_i2c_bins,'-',label='is2 (2 km))')
if np.size(FNj3)!=0:
    plt.plot(swh_bins,mn_j3_bins,'-',label='j3')
if np.size(FNs3)!=0:
    plt.plot(swh_bins,mn_s3_bins,'-',label='s3')
#plt.plot(swh_bins,np.nanmean(mn_i2b_bins,axis=1),'-',label='is2 (100 m)')
#plt.plot(swh_bins,np.nanmean(mn_i2c_bins,axis=1),'-',label='is2 (2 km))')
#plt.plot(swh_bins,np.nanmean(mn_j3_bins,axis=1),'-',label='j3')
#plt.plot(swh_bins,np.nanmean(mn_s3_bins,axis=1),'-',label='s3')
plt.xlabel('SWH [m]')
plt.ylabel('SSHA [cm]')
plt.legend()
plt.grid()
plt.title('Mean SSHA vs. SWH')
plt.xlim(0,12)





########### MONTHLY AVERAGES ###########
if np.shape(FNtg)[0]==1:
    cc,rr=1,1
elif np.shape(FNtg)[0]%2 == 0:
    cc,rr=int(np.shape(FNtg)[0]/2),2
elif np.shape(FNtg)[0]%3 == 0:
    cc,rr=int(np.shape(FNtg)[0]/3),3
elif np.shape(FNtg)[0]>3:
    cc,rr=int(np.shape(FNtg)[0]/3)+1,3
else:
    raise('come up with something new')


### Improve montly averages with SSB
issb = np.where((swh_c>=1)&(swh_c<=6))[0]
SSB1 =  np.polyfit(swh_c,ssha_fft_c,1)[::-1]
ssb_c_fit = (SSB1[1]*swh_bins)
ssb_b = ((SSB1[1])*swh_b)
ssb_c = ((SSB1[1])*swh_c)
mn_ssha_fft_ssb_b,vr_ssha_fft_ssb_b,mn_time_fft_ssb_b = lTG.month_2_month(ymdhmsI_b,ssha_fft_b+ssb_b,yrs_mm)
plt.figure()
plt.grid()
swh_bins = np.arange(0,12,0.25)
plt.plot(swh_c,ssha_fft_c,',',label='C fft')
plt.plot(swh_bins,ssb_c_fit,color='black')
plt.xlabel('SWH [m]')
plt.ylabel('SSHA [m]')
plt.legend()


'''
plt.figure(figsize=(8,6))
plt.plot(mn_time_b,np.array(mn_ssha_fft_ssb_b)*100.,'.-',label='IS2 wave signal and SSB',color=clr[1])
plt.plot(mn_time_b,np.array(mn_ssha_fft_b)*100.,'o-',label='IS2 wave signal removed',color=clr[7])
if np.size(FNj3)!=0:
    plt.plot(mn_time_alt,np.array(mn_ssha_alt)*100.,'.-',label='J3',color=clr[2])
if np.size(FNc2)!=0:
    plt.plot(mn_time_c2,np.array(mn_ssha_c2)*100.,'.-',label='C2',color=clr[3])
if np.size(FNs3)!=0:
    plt.plot(mn_time_s3,np.array(mn_ssha_s3)*100.,'.-',label='S3',color=clr[6])
if np.size(FNtg)!=0:
    plt.plot(mn_time_tg,np.array(mn_ssha_tg-np.nanmean(mn_ssha_tg-mn_ssha_alt))*100.,'.-',label='TG',color=clr[4])
if np.size(FNtg)!=0:
    mn_mn_ssha_tg2 = np.nanmean(mn_ssha_tg2,axis=0)
    plt.plot(mn_time_tg,np.array(mn_mn_ssha_tg2)*100.,'.-',label='TG2',color=clr[5])
plt.legend()
plt.grid()
plt.xlabel('year fraction')
plt.ylabel('monthly SSHA means [cm]')
plt.title(REG) 


if np.size(FNtg)!=0:
    plt.figure(figsize=(8,8))
    plt.plot(mn_time_b,np.array(mn_ssha_b)*100.,'o-',label='IS2 100 m')
    plt.plot(mn_time_tg,np.array(mn_mn_ssha_tg2)*100.,'o-',label='mean of all TG2')
    for ii in np.arange(np.shape(mn_ssha_tg2)[0]):
        lab = FNtg2[ii][13:-4]
        plt.plot(mn_time_tg,np.array(mn_ssha_tg2[ii,:])*100.,'-',label=lab)
    plt.legend()
    plt.grid()
    plt.xlabel('year fraction')
    plt.ylabel('monthly SSHA means [cm]')
    plt.title(REG) 
'''

plt.figure(figsize=(5,8))
plt.subplots_adjust(hspace=0.4)
bins = np.arange(-5,5.2,0.2)
plt.subplot(311)
plt.hist(ssha_a,bins=bins,label='a')
plt.hist(ssha_fft_a,bins=bins,label='a fft',alpha=0.5)
plt.grid()
plt.xlabel('ssha [m]')
plt.ylabel('occurrences')
plt.legend()
plt.subplot(312)
plt.hist(ssha_b,bins=bins,label='b')
plt.hist(ssha_fft_b,bins=bins,label='b fft',alpha=0.5)
plt.grid()
plt.xlabel('ssha [m]')
plt.ylabel('occurrences')
plt.legend()
plt.xlim(-2,2)
plt.subplot(313)
plt.hist(ssha_c,bins=bins,label='c')
plt.hist(ssha_fft_c,bins=bins,label='c fft',alpha=0.5)
plt.grid()
plt.xlabel('ssha [m]')
plt.ylabel('occurrences')
plt.legend()
plt.xlim(-1,1)

plt.figure(figsize=(8,6))
mk=','
plt.plot(swh_b,np.array(ssha_b)*100.,mk,label='IS2 '+str(fp_b)+' m',color=clr[1])
plt.plot(swh_c,np.array(ssha_c)*100.,mk,label='IS2 '+str(fp_c)+' m',color=clr[4])
if np.size(FNj3)!=0:
    plt.plot(swh_alt,np.array(ssha_alt)*100.,mk,label='J3',color=clr[2])
if np.size(FNc2)!=0:
    plt.plot(swh_c2,np.array(ssha_c2)*100.,mk,label='C2',color=clr[3])
if np.size(FNs3)!=0:
    plt.plot(swh_s3,np.array(ssha_s3)*100.,mk,label='S3',color=clr[6])
plt.legend()
plt.grid()
plt.xlabel('SWH [m]')
plt.ylabel('SSHAs [cm]')
plt.title(REG) 



# TREND?
#ce_mnb = np.polyfit(mn_time_b, mn_ssha_b, 1)[::-1]
ce_b = np.polyfit(days_since_1985_b,ssha_b, 1)[::-1]



########### GRIDDED AVERAGES ###########
mm = 50
dl=0.2
dl2 = 0.05
cmap = 'RdYlGn_r'#'RdYlGn_r' #'coolwarm'

if REG=='mumbai':
    lat_minmax_b = [np.nanmin(lat_b),np.nanmax(lat_b)]
    lon_minmax_b=[72.6,np.nanmax(lon_b)]
else:
    lat_minmax_b = [np.nanmin(lat_b),np.nanmax(lat_b)]
    lon_minmax_b=[np.nanmin(lon_b),np.nanmax(lon_b)]

lat_grid,lon_grid,ssha_grid,ssha_grid_var = lTG.gridded(ssha_b*100,lat_b,lon_b,lat_minmax=lat_minmax_b,lon_minmax=lon_minmax_b,dl=dl)
lat_grid,lon_grid,ssha_fft_grid,ssha_fft_grid_var = lTG.gridded(ssha_fft_b*100,lat_b,lon_b,lat_minmax=lat_minmax_b,lon_minmax=lon_minmax_b,dl=dl)
lat_grid,lon_grid,ssha_fft_grid_c,ssha_fft_grid_var_c = lTG.gridded(ssha_fft_c*100,lat_c,lon_c,lat_minmax=lat_minmax_b,lon_minmax=lon_minmax_b,dl=dl)
#lat_grid,lon_grid,time_grid,ssha_grid3d = lTG.month_2_month_grid(ymdhmsI_b,ssha_b,yrs_mm,lat_b,lon_b,dm=3)
if np.size(FNj3)!=0:
    lat_gridA,lon_gridA,ssha_gridA,ssha_gridA_var = lTG.gridded(ssha_alt*100,lat_alt,lon_alt,lat_minmax=lat_minmax_b,lon_minmax=lon_minmax_b,dl=dl)
    pbil.groundtracks_contour(np.unique(lon_gridA),np.unique(lat_gridA),ssha_gridA,'Jason-3 gridded. '+REG+' '+str(yrs_mm),'ssha [cm]',
                      cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                      LEV=np.arange(-0.5,0.5+dl2,dl2)*100.)
if np.size(FNs3)!=0:
    lat_grid_s3,lon_grid_s3,ssha_grid_s3,ssha_grid_s3_var = lTG.gridded(ssha_s3*100,lat_s3,lon_s3,lat_minmax=lat_minmax_b,lon_minmax=lon_minmax_b,dl=dl)
    pbil.groundtracks_contour(np.unique(lon_grid_s3),np.unique(lat_grid_s3),ssha_grid_s3,'Senitnel-3 gridded. '+REG+' '+str(yrs_mm),'ssha [cm]',
                      cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                      LEV=np.arange(-0.5,0.5+dl2,dl2)*100.)
if np.size(FNc2)!=0:
    lat_gridC,lon_gridC,ssha_gridC,ssha_gridC_var = lTG.gridded(ssha_c2*100,lat_c2,lon_c2,lat_minmax=lat_minmax_b,lon_minmax=lon_minmax_b,dl=dl)
    pbil.groundtracks_contour(np.unique(lon_gridC),np.unique(lat_gridC),ssha_gridC,'CryoSat-2 gridded. '+REG+' '+str(yrs_mm),'ssha [cm]',
                      cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                      LEV=np.arange(-0.5,0.5+dl2,dl2)*100.)
pbil.groundtracks_contour(np.unique(lon_grid),np.unique(lat_grid),ssha_grid,'ICESat-2 gridded product ('+str(fp_b)+' m  footprint). '+REG+' '+str(yrs_mm),'ssha [cm]',
                    cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',LEV=np.arange(-0.5,0.5+dl2,dl2)*100.,figsize=[5,12],TB=[0.9,0.4])
pbil.groundtracks_contour(np.unique(lon_grid),np.unique(lat_grid),ssha_fft_grid,'ICESat-2 gridded product ('+str(fp_b)+' m  footprint fft). '+REG+' '+str(yrs_mm),'ssha [cm]',
                    cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',LEV=np.arange(-0.5,0.5+dl2,dl2)*100.,figsize=[5,12],TB=[0.9,0.4])
pbil.groundtracks_contour(np.unique(lon_grid),np.unique(lat_grid),(ssha_fft_grid_var-ssha_grid_var),'ICESat-2 variance reduction wave signal removed - not removed \n ('+str(fp_b)+' m  footprint fft). '+REG+' '+str(yrs_mm),'$\Delta$ var(ssha) [cm$^2$]',
                    cm='inferno',vmin=-1800,vmax=0,FN=[],proj=180.,fc='0.1',LEV=np.arange(-1800,50,50),figsize=[5,12],TB=[0.9,0.4])
MONTH_GRID = False
if MONTH_GRID == True:
    for ii in np.arange(np.shape(ssha_grid3d)[2]):
        pbil.groundtracks_contour(np.unique(lon_grid),np.unique(lat_grid),ssha_grid3d[:,:,ii],str(time_grid[ii])+' | ICESat-2 gridded product ('+str(fp_b)+' m  footprint). '+REG+' '+str(yrs_mm),'ssha [cm]',
                          cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',LEV=np.arange(-0.5,0.55,0.05)*100.)
########### SPATIAL PLOTS ###########
mm=50
cmap='RdYlGn_r'#'RdYlGn_r'#'coolwarm'
s1=6
if np.size(FNj3)!=0:
    #'''
    LEG = 'ssha [cm] \n gray outline = Jason-3'
    pbil.groundtracks_multi(lon_b,lat_b,(ssha_b)*100.,'ICESat-2 ('+str(fp_b)+' m  footprint) with Jason-3. '+REG+' '+str(yrs_mm),LEG,
                       cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                       lon3=lon_alt,lat3=lat_alt,gt3=(ssha_alt)*100.)
    pbil.groundtracks_multi(lon_b,lat_b,(ssha_fft_b)*100.,'ICESat-2 ('+str(fp_b)+' m  footprint fft) with Jason-3. '+REG+' '+str(yrs_mm),LEG,
                       cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                       lon3=lon_alt,lat3=lat_alt,gt3=(ssha_alt)*100.)
    #'''
    LEG = 'ssha [cm]' 
    pbil.groundtracks_multi(lon_alt,lat_alt,(ssha_alt)*100.,'Jason-3. '+REG+' '+str(yrs_mm),LEG,
                       cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',s1=s1)
if np.size(FNs3)!=0:
    LEG = 'ssha [cm]'
    pbil.groundtracks_multi(lon_s3,lat_s3,(ssha_s3)*100.,'Sentinel-3. '+REG+' '+str(yrs_mm),LEG,
                       cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',s1=s1)
if np.size(FNc2)!=0:
    LEG = 'ssha [cm] \n gray outline = CryoSat-2'
    pbil.groundtracks_multi(lon_c2,lat_c2,(ssha_c2)*100.,'CryoSat-2. '+REG+' '+str(yrs_mm),LEG,
                       cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',s1=100)
if np.size(FNtg2)!=0:
    LEG = 'ssha [cm] \n blue outline = NOAA TGs'
    pbil.groundtracks_multi(lon_b,lat_b,(ssha_b)*100.,'ICESat-2 ('+str(fp_b)+' m  footprint) with TG and Jason-3. '+REG+' '+str(yrs_mm),LEG,
                       cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                       lon2=lon_tg2,lat2=lat_tg2,gt2=ss_tg2*100,s1=s1)

LEG = '$\Delta var(ssha)$'
var1 = str(np.round(np.nanstd((ssha_b)*100.),2))+' cm'
pbil.groundtracks_multi(lon_b,lat_b,(ssha_b)*100.,'ICESat-2 ('+str(fp_b)+' m  footprint).  $\sigma$ = '+var1,LEG,
                    cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',s1=s1)
var2 = str(np.round(np.nanstd((ssha_fft_b)*100.),2))+' cm'
pbil.groundtracks_multi(lon_b,lat_b,(ssha_fft_b)*100.,'ICESat-2 ('+str(fp_b)+' m  footprint) wave signals removed. $\sigma$ = '+var2,LEG,
                    cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',s1=s1) #REG+' '+str(yrs_mm)

var2 = str(np.round(np.nanstd((ssha_fft_b)*100.),2))+' cm'
ichk = np.where((yrfrac_b>=2021)&(yrfrac_b<2022))[0]
pbil.groundtracks_multi(lon_b[ichk],lat_b[ichk],((swh66_b[ichk]/4)**2-(swh_b[ichk]/4)**2),'ICESat-2 variance reduction of 100 m SSHA segments with wave signal removed vs. not removed \n blue indicates improvement with wave signal removal','$\Delta$ var(ssha) [m$^2$]',
                    cm='coolwarm',vmin=-.1,vmax=.1,FN=[],proj=180.,fc='0.1',s1=10,figsize=[12,5],TB=[0.85,0.4]) #REG+' '+str(yrs_mm)
dlat_grid,dlon_grid,dssha_grid_var,dssha_grid_varXX = lTG.gridded(((swh66_b[ichk]/4)**2-(swh_b[ichk]/4)**2),lat_b[ichk],lon_b[ichk],lat_minmax=lat_minmax_b,lon_minmax=lon_minmax_b,dl=0.25)
pbil.groundtracks_contour(np.unique(dlon_grid),np.unique(dlat_grid),dssha_grid_var,'ICESat-2 variance reduction of 100 m SSHA segments with wave signal removed vs. not removed \n blue indicates improvement with wave signal removal','$\Delta$ var(ssha) [m$^2$]',
                    cm='coolwarm',vmin=-0.1,vmax=0.1,FN=[],proj=180.,fc='0.1',LEV=np.arange(-0.1,0.101,.001),figsize=[12,8],TB=[0.9,0.4])

plt.figure()
plt.hist((swh_b[ichk]/4),bins=50)
plt.hist((swh66_b[ichk]/4),bins=50,alpha=0.5)

dday = 24*30
dyr = np.round((dday/24.)/365.25,5)#(96./24.)
yrf_bins = np.arange(yrs_mm[0],yrs_mm[-1]+1+dyr,dyr)
for ii in np.arange(np.size(yrf_bins)):
    iyr = np.where((yrfrac_b>=yrf_bins[ii]-dyr/2.0)&(yrfrac_b<yrf_bins[ii]+dyr/2.0))[0]
    if np.size(iyr)>0:
        pbil.groundtracks_multi(lon_b[iyr],lat_b[iyr],(ssha_fft_b[iyr])*100.,'ICESat-2 ('+str(fp_b)+' m  footprint) wave signals removed. $\sigma$ = '+var2,LEG,
                cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',s1=s1) #REG+' '+str(yrs_mm)

'''
if SEG2==True:
    LEG = 'ssha [cm]'
    pbil.groundtracks_multi(lon_a,lat_a,(ssha_a)*100.,'ICESat-2 ('+str(fp_a)+' m  footprint) with TG and ALT. '+REG+' '+str(yrs_mm),LEG,
                       cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1')
'''

########### GRIDDED AVERAGES - EDDY ###########
def surf3d(lat_grid,lon_grid,ssha_grid,TIT,vmin=-45,vmax=45,D3=True):
    from matplotlib.ticker import LinearLocator
    # Plot the surface.
    if D3==True:
        fig, ax = plt.subplots(figsize=(10,8),subplot_kw={"projection": "3d"})
        plt.suptitle(TIT)
        surf = ax.plot_surface(lat_grid,lon_grid,ssha_grid, cmap=plt.cm.coolwarm,
                            linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(vmin, vmax)
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')
    else:
        fig, ax = plt.subplots(figsize=(4,4))
        plt.suptitle(TIT)
        surf=plt.scatter(lat_grid,lon_grid,c=ssha_grid, cmap='RdYlBu_r',vmin=vmin,vmax=vmax,marker='s')
    # Add a color bar which maps values to colors.
    cb =fig.colorbar(surf, shrink=0.5, aspect=5)
    cb.set_label('SSHA [cm]')
    plt.xlabel('latitude [deg]')
    plt.ylabel('longitude [deg]')
    plt.show()



surf3d(lat_grid,lon_grid,ssha_fft_grid,'ICESat-2 100 m 6-month regional averages over Gulf of Mexico')
surf3d(lat_grid,lon_grid,ssha_fft_grid_c,'ICESat-2 2 km 6-month regional averages over Gulf of Mexico')
surf3d(lat_grid,lon_grid,ssha_fft_grid-ssha_fft_grid_c,'ICESat-2 100 m - 2 km 6-month regional averages over Gulf of Mexico',vmin=-5,vmax=5)

surf3d(lat_grid,lon_grid,ssha_grid_s3,'Sentinel-3 6-month regional averages over Gulf of Mexico')
surf3d(lat_grid,lon_grid,ssha_gridA,'Jason-3 6-month regional averages over Gulf of Mexico')
'''
###### Ocean surface statistics
LEG = 'ssha [cm]'
frac = np.append(mn_time_b,2021.5)#np.arange(2020,2021.1,0.1)
Nfrac = np.size(frac)
sp=0
if np.size(FNc2)!=0:
    proj=180.
    sp = Nfrac-1
    if sp%2==0:
        ax_sp = [sp/2,2]
        fs = [14,22]
    elif sp%3==0:
        ax_sp = [sp/3,3]
        fs=[12,18]
    fig=plt.figure(figsize=(int(fs[0]),int(fs[1])))
    plt.subplots_adjust(hspace=0.4)
    plt.suptitle('ICESat-2 ('+str(fp_b)+' m  footprint) with CryoSat2. '+REG,fontsize=20)
    for ii in np.arange(Nfrac-1):
        iyrf_c2 = np.where((yrfrac_c2>=frac[ii])&(yrfrac_c2<frac[ii+1]))[0]
        iyrf_b = np.where((yrfrac_b>=frac[ii])&(yrfrac_b<frac[ii+1]))[0]
        TIT=str(np.round(frac[ii],2))#'ICESat-2 ('+str(fp_b)+' m  footprint) with CryoSat2. \n '+REG+' '+str(np.round(frac[ii],1))
        ax=plt.subplot(int(ax_sp[0]),int(ax_sp[1]),ii+1, projection=ccrs.EckertIV(central_longitude=proj))
        if np.size(iyrf_c2)>0:   
            pbil.groundtracks_multi_subplot(fig,ax,lon_b[iyrf_b],lat_b[iyrf_b],(ssha_b[iyrf_b])*100.,TIT,LEG,
                    cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                    lon2=lon_tg2,lat2=lat_tg2,gt2=ss_tg2*100,
                    lon3=lon_c2[iyrf_c2],lat3=lat_c2[iyrf_c2],gt3=(ssha_c2[iyrf_c2])*100.,ss=20)
        else:
            pbil.groundtracks_multi_subplot(fig,ax,lon_b[iyrf_b],lat_b[iyrf_b],(ssha_b[iyrf_b])*100.,TIT,LEG,
                    cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                    lon2=lon_tg2,lat2=lat_tg2,gt2=ss_tg2*100,ss=20)
            pbil.groundtracks_multi_subplot(fig,ax,lon_b[iyrf_b],lat_b[iyrf_b],(ssha_fft_b[iyrf_b])*100.,TIT,LEG,
                    cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                    lon2=lon_tg2,lat2=lat_tg2,gt2=ss_tg2*100,ss=20)
       
            pbil.groundtracks_multi(lon_c2[iyrf_c2],lat_c2[iyrf_c2],(ssha_c2[iyrf_c2])*100.,'CryoSat-2. '+REG+' '+str(yrs_mm),LEG,
                            cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',s1=100)
    iyrf_b = np.where((yrfrac_b>=frac[ii])&(yrfrac_b<frac[ii+1]))[0]
    if np.size(iyrf_b)>0:
        pbil.groundtracks_multi(lon_b[iyrf_b],lat_b[iyrf_b],(ssha_b[iyrf_b])*100.,'ICESat-2 ('+str(fp_b)+' m  footprint) with TG. '+REG+' '+str(yrs_mm),LEG,
                            cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1')
'''
'''
if np.size(mn_time_tg)!= np.shape(mn_time_tg)[0]:
    tg_bias_b = np.empty(np.shape(mn_time_tg)[1])*np.nan
    tg_bias_alt = np.empty(np.shape(mn_time_tg)[1])*np.nan
    mTGii = np.empty(np.shape(mn_time_tg)[1])*np.nan
    plt.figure(figsize=(16,12))
    for ii in np.arange(np.shape(FNtg)[0]):
        plt.subplot(rr,cc,ii+1)
        TGii=np.array(mn_ssha_tg[:,ii])
        mTGii[ii] = np.nanmean(TGii)
        tg_bias_b[ii] = np.nanmean(mn_ssha_b[:,ii])-np.nanmean(mn_ssha_tg[:,ii])+mTGii[ii]
        rms_b = np.round(np.sqrt(np.nanmean(((mn_ssha_b[:,ii]-mn_ssha_tg[:,ii])-(tg_bias_b[ii]-mTGii[ii]))**2))*100.,3)
        print('tg_bias_b: '+str(np.size(tg_bias_b[ii])))
        plt.plot(mn_time_tg[:,ii],(TGii-mTGii[ii])*100.,'.-',label='TG',color=clr[0])
        shftb=np.array(mn_ssha_b[:,ii]-tg_bias_b[ii])*100.
        plt.plot(mn_time_b,shftb,'.-',label='IS2. RMS [IS2-TG] = '+str(rms_b)+' cm',color=clr[1])
        #plt.plot(mn_time_10,np.array(mn_ssha_10)*100.,'-',label='IS2 (10 m)',color='red')
        if np.size(FNj3)!=0:
            tg_bias_alt[ii] = np.nanmean(mn_ssha_alt)-np.nanmean(mn_ssha_tg[:,ii])+mTGii[ii]
            print('tg_bias_alt: '+str(np.size(tg_bias_alt[ii])))
            rms_alt = np.round(np.sqrt(np.nanmean(((mn_ssha_alt-mn_ssha_tg[:,ii])-(tg_bias_alt[ii]-mTGii[ii]))**2))*100,3)
            shftA=np.array(mn_ssha_alt-tg_bias_alt[ii])*100.
            plt.plot(mn_time_alt,shftA,'.-',label='J3. RMS [J3-TG] = '+str(rms_alt)+' cm',color=clr[2])
        plt.legend()
        plt.grid()
        plt.xlabel('time [days since 1985]')
        plt.ylabel('monthly SSHA means [cm]')
        plt.title('UHSLC ID: '+str(uhslc_id[ii]))
        plt.ylim(-40,40)
        
else:
    plt.figure()
    itg = np.where((mn_time_tg>=np.nanmin(mn_time_b))&(mn_time_tg<=np.nanmax(mn_time_b)))[0]
    #tg_bias = np.nanmean(mn_ssha_tg[itg])-np.nanmean(mn_ssha_b)
    tg_bias_b = np.nanmean(mn_ssha_b)-np.nanmean(mn_ssha_tg[itg])
    rms_b = np.round(np.sqrt(np.nanmean((mn_ssha_b-tg_bias_b-mn_ssha_tg)**2))*100.,2)
    plt.plot(mn_time_tg[itg],np.array(mn_ssha_tg[itg])*100.,'.-',label='TG',color=clr[0])
    plt.plot(mn_time_b,np.array(mn_ssha_b-tg_bias_b)*100.,'.-',label='IS2. RMS [IS2-TG] = '+str(rms_b)+' cm',color=clr[1])
    #plt.plot(mn_time_10,np.array(mn_ssha_10)*100.,'.',label='IS2 (10 m)',color='red')
    if np.size(FNj3)!=0:
        tg_bias_alt = np.nanmean(mn_ssha_alt)-np.nanmean(mn_ssha_tg[itg])
        rms_alt = np.round(np.sqrt(np.nanmean((mn_ssha_alt-tg_bias_alt-mn_ssha_tg)**2))*100,2)
        plt.plot(mn_time_alt,np.array(mn_ssha_alt-tg_bias_alt)*100.,'.-',label='J3. RMS [J3-TG] = '+str(rms_alt)+' cm',color=clr[2])
    plt.legend()
    plt.grid()
    plt.xlabel('time [days since 1985]')
    plt.ylabel('monthly SSHA means [cm]')
    plt.title('UHSLC ID: '+str(uhslc_id))    
'''