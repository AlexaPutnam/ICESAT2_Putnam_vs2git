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
def find_averages(binz,x,y,z=[],zconst=[],imin=0,MED=False):
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

###################################
########### INPUT ###########
###################################
# Region/Time
REG = 'newengland'#'north_atlantic'#'ittoqqortoormiit' #'newengland','hawaii','antarctica', 'japan'
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
# Jason-3
if np.size(FNj3)!=0:
    ssha_alt,lat_alt,lon_alt,days_since_1985_alt,ymdhmsA,tsA,swh_alt,yrfracA = lTG.pull_altimetry(FNj3)
    mn_ssha_alt,vr_ssha_alt,mn_time_alt = lTG.month_2_month(ymdhmsA,ssha_alt,yrs_mm,IS2=False)
    mn_swh_alt,vr_swh_alt,mn_time_alt = lTG.month_2_month(ymdhmsA,swh_alt,yrs_mm,IS2=False)
    mn_ts_alt,mn_ymdhms_alt,mn_yrfrac_alt = lTG.tide_days_1985_to_TS(mn_time_alt)

# CryoSat-2
if np.size(FNc2)!=0:
    ssha_c2,lat_c2,lon_c2,days_since_1985_c2,ymdhms_c2,tsA,swh_c2,yrfrac_c2 = lTG.pull_altimetry(FNc2)
    mn_ssha_c2,vr_ssha_v2,mn_time_c2 = lTG.month_2_month(ymdhms_c2,ssha_c2,yrs_mm,IS2=False)
    mn_swh_c2,vr_swh_c2,mn_time_c2 = lTG.month_2_month(ymdhms_c2,swh_c2,yrs_mm,IS2=False)
    mn_ts_c2,mn_ymdhms_c2,mn_yrfrac_c2 = lTG.tide_days_1985_to_TS(mn_time_c2)

# Sentinel-3
if np.size(FNs3)!=0:
    ssha_s3,lat_s3,lon_s3,days_since_1985_s3,ymdhms_s3,tsA,swh_s3,yrfrac_s3 = lTG.pull_altimetry(FNs3)
    mn_ssha_s3,vr_ssha_v2,mn_time_s3 = lTG.month_2_month(ymdhms_s3,ssha_s3,yrs_mm,IS2=False)
    mn_swh_s3,vr_swh_s3,mn_time_s3 = lTG.month_2_month(ymdhms_s3,swh_s3,yrs_mm,IS2=False)
    mn_ts_s3,mn_ymdhms_s3,mn_yrfrac_s3 = lTG.tide_days_1985_to_TS(mn_time_s3)

# IceSat2
if SEG2==True:
    ## 2-m segment 
    ssha_a,ssha_fft_a,swell_hf_a,swell_lf_a,swell_a,lat_a,lon_a,days_since_1985_a,ymdhmsI_a,tsI_a,beam_a,swh_a,swh66_a,N_a,slope_a,skew_a,yrfrac_a,wl_seg_a,wsteep_seg_a,ip_lf_a,ip_hf_a,ip_a,OT_a = lTG.pull_icesat(FNis2,SEG=2)
    iS_a = np.where((lat_a>=41.55)&(lat_a<=41.7)&(lon_a>=-71)&(lon_a<=-70.5))[0]
    swh_aadj = 0.056 + 7.83*(swh_a/4.0)
    #mn_ssha_a,vr_ssha_a,mn_time_a = lTG.month_2_month(ymdhmsI_a,ssha_a,yrs_mm)
    #mn_swh_a,vr_swh_a,mn_time_a = lTG.month_2_month(ymdhmsI_a,swh_a,yrs_mm)
    #mn_swh_aadj,vr_swh_aadj,mn_time_a = lTG.month_2_month(ymdhmsI_a,swh_aadj,yrs_mm)

## 100-m segment
ssha_b,ssha_fft_b,swell_hf_b,swell_lf_b,swell_b,lat_b,lon_b,days_since_1985_b,ymdhmsI_b,tsI_b,beam_b,swh_b,swh66_b,N_b,slope_b,skew_b,yrfrac_b,wl_seg_b,wsteep_seg_b,ip_lf_b,ip_hf_b,ip_b,OT_b = lTG.pull_icesat(FNis2,SEG=fp_b)
swh_badj = 0.523 + 5.393*(swh_b/4.0)
#wgt_b,dist_b = lTG.select_region(lat_b,lon_b,lat_tg[0],lon_tg[0],dlat=1,dlon=1)
#mn_ssha_b,vr_ssha_b,mn_time_b = lTG.month_2_month(ymdhmsI_b,ssha_b,yrs_mm)#,LATLON = np.vstack((lat_b,lon_b)).T) #ymdhmsI_b[idx_b],ssha_b[idx_b],days_since_1985_b[idx_b]
#mn_ssha_fft_b,vr_ssha_fft_b,mn_time_fft_b = lTG.month_2_month(ymdhmsI_b,ssha_fft_b,yrs_mm)#
#mn_swh_b,vr_swh_b,mn_time_b = lTG.month_2_month(ymdhmsI_b,swh_b,yrs_mm,d100=True)
#mn_swh_badj,vr_swh_badj,mn_time_b = lTG.month_2_month(ymdhmsI_b,swh_badj,yrs_mm)
#mn_slope_b,vr_slope_b,mn_time_b = lTG.month_2_month(ymdhmsI_b,slope_b,yrs_mm)
## 2000-m segment
ssha_c,ssha_fft_c,swell_hf_c,swell_lf_c,swell_c,lat_c,lon_c,days_since_1985_c,ymdhmsI_c,tsI_c,beam_c,swh_c,swh66_c,N_c,slope_c,skew_c,yrfrac_c,wl_seg_c,wsteep_seg_c,ip_lf_c,ip_hf_c,ip_c,OT_c = lTG.pull_icesat(FNis2,SEG=fp_c)
#mn_swh_c,vr_swh_c,mn_time_c = lTG.month_2_month(ymdhmsI_c,swh_c,yrs_mm)

        
#############
#############
## Available Beams
#############
#############
plt.figure()
plt.plot(days_since_1985_b,beam_b,'.')
plt.xlabel('time')
plt.ylabel('beams')
plt.grid()

#############
#############
## Morrison improvement figure
#############
#############
BEAM = 2
ist=500000#swell=500000, seas=300000+600
ien=ist+800#2000
#icut,dist_a = dist_sv2pt(lat_a,lon_a,beam_a,lat_a[ist],lon_a[ist],maxKM=2000,beam=[])#np.where((~np.isnan(ssha_a)))[0][ist:ien]
icut =np.where((~np.isnan(ssha_a))&(beam_a==BEAM))[0][ist:ien]
dist_a=(days_since_1985_a[icut]-days_since_1985_a[icut][0])*(24*60*60)*7000.
mm=2
lw=3
pmax = 2000

def spectral_ssha(x,y):
    #x1,y=t_a,ssha_a[icut]
    ce = np.polyfit(x, y, 1)[::-1]
    fity = ce[0]+ce[1]*x
    fitAno0,fA = fft2signal(x,y,Nf=66)
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
    #plt.axvline(x=66)
    plt.xlim(0,250)
    plt.grid()
    #plt.axvline(x=66,color='red')
    plt.show()

t_a = (days_since_1985_a[icut]-days_since_1985_a[icut][0])*(24*60*60)
spectral_ssha(t_a,ssha_a[icut])


plt.figure(figsize=(14,14))
plt.subplots_adjust(top=0.9,hspace=0.4,wspace=0.4)
plt.suptitle('ICESat-2 (gt2r) ocean profile')
plt.subplot(321)
plt.title('2 m SSHA means \n std(ssha) = '+str(np.round(np.nanstd(ssha_a[icut]),4)))
plt.plot(dist_a,ssha_a[icut],'-',label='ssha')
plt.xlabel('distance [m]')
plt.ylabel('ssha [m]')
plt.legend()
plt.grid()
plt.ylim(-mm,mm)
plt.xlim(0,pmax)
plt.subplot(322)
plt.title('2 m SSHA means with LFSW')
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
plt.subplot(323)
plt.title('2 m SSHA means-LFSW \n std(ssha-LFSW) = '+str(np.round(np.nanstd(ssha_a[icut]-swell_a[icut]),4)))
plt.plot(dist_a,ssha_a[icut]-swell_a[icut],'-',label='ssha-SW')
plt.xlabel('distance [m]')
plt.ylabel('ssha [m]')
plt.legend()
plt.grid()
plt.ylim(-mm,mm)
plt.xlim(0,pmax)
plt.subplot(324)
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

plt.figure(figsize=(16,14))
plt.subplots_adjust(top=0.9,hspace=0.4,wspace=0.4)
plt.suptitle('ICESat-2 (gt2r) ocean profile')
plt.subplot(321)
plt.title('2 m SSHA means \n std(ssha) = '+str(np.round(np.nanstd(ssha_a[icut]),4)))
plt.plot(dist_a,ssha_a[icut],'-',label='ssha')
plt.xlabel('distance [m]')
plt.ylabel('ssha [m]')
plt.legend()
plt.grid()
plt.ylim(-mm,mm)
plt.xlim(0,pmax)
plt.subplot(322)
plt.title('2 m SSHA means with LFSW')
plt.plot(dist_a,ssha_a[icut],'-',label='ssha')
plt.plot(dist_a,swell_lf_a[icut],'-',label='LFSW',linewidth=lw)
imm=np.where(ip_lf_a[icut]==1)[0]
plt.plot(dist_a[imm],swell_lf_a[icut][imm],'o',label='max/min',color='red')
plt.xlabel('distance [m]')
plt.ylabel('ssha [m]')
plt.legend()
plt.grid()
plt.ylim(-mm,mm)
plt.xlim(0,pmax)
plt.subplot(323)
plt.title('2 m SSHA means-LFSW \n std(ssha-LFSW) = '+str(np.round(np.nanstd(ssha_a[icut]-swell_lf_a[icut]),4)))
fitAno0_2,fA_2 = fft2signal(dist_a/(7000.),ssha_a[icut]-swell_lf_a[icut],Nf=90)
plt.plot(dist_a,ssha_a[icut]-swell_lf_a[icut],'-',label='ssha-LFSW')
plt.xlabel('distance [m]')
plt.ylabel('ssha [m]')
plt.legend()
plt.grid()
plt.ylim(-mm,mm)
plt.xlim(0,pmax)
plt.subplot(324)
plt.title('2 m SSHA means-LFSW fit with HFSW')
fitAno0_2,fA_2 = fft2signal(dist_a/(7000.),ssha_a[icut]-swell_lf_a[icut],Nf=90)
plt.plot(dist_a,ssha_a[icut]-swell_lf_a[icut],'-',label='ssha-LFSW')
plt.plot(dist_a,swell_hf_a[icut],'-',label='HFSW',linewidth=lw)
imm = np.where(ip_hf_a[icut]==1)[0]
plt.plot(dist_a[imm],swell_hf_a[icut][imm],'o',label='max/min',color='red')
#plt.plot(dist_a,fitAno0_2,'-',label='hf -new',linewidth=lw)
#plt.plot(dist_a,swell_lf_a[icut]+swell_hf_a[icut],'-',label='lf+hf')
plt.xlabel('distance [m]')
plt.ylabel('ssha [m]')
plt.legend()
plt.grid()
plt.ylim(-mm,mm)
plt.xlim(0,pmax)
plt.subplot(325)
plt.title('2 m means-(LFSW+HFSW) fits  \n std(ssha-(LFSW+HFSW))) = '+str(np.round(np.nanstd(ssha_fft_a[icut]),4)))
plt.plot(dist_a,ssha_fft_a[icut],'-',label='ssha-(LFSW+HFSW)')
plt.xlabel('distance [m]')
plt.ylabel('ssha [m]')
plt.legend()
plt.grid()
plt.ylim(-mm,mm)
plt.xlim(0,pmax)
plt.subplot(326)
plt.title('point distributions')
binz = np.arange(-2,2.05,0.05)
from scipy import stats
h1=ssha_a[icut]
pdf1 = stats.norm.pdf(x = binz, loc=np.nanmean(h1), scale=np.nanstd(h1))
plt.plot(binz,pdf1,label='ssha')
h2=ssha_a[icut]-swell_lf_a[icut]
pdf2 = stats.norm.pdf(x = binz, loc=np.nanmean(h2), scale=np.nanstd(h2))
plt.plot(binz,pdf2,label='ssha-LFSW')
h3=ssha_fft_a[icut]
pdf3 = stats.norm.pdf(x = binz, loc=np.nanmean(h3), scale=np.nanstd(h3))
plt.plot(binz,pdf3,label='ssha-(LFSW+HFSW)')
plt.xlabel('ssha [m]')
plt.ylabel('frequency')
plt.legend()
plt.grid()

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
icut_b1,max_dayb = pull_day(ymdhmsI_b,y=[],m=[],d=[],hr=[],mi=[],se=[],elem=0) 
icut_b =np.intersect1d(np.where((beam_b==BEAM)&(~np.isnan(ssha_b)))[0],icut_b1)
dist_b=(days_since_1985_b[icut_b]-days_since_1985_b[icut_b][0])*(24*60*60)*7000.
icut_c1,max_dayc = pull_day(ymdhmsI_c,y=[],m=[],d=[],hr=[],mi=[],se=[],elem=max_dayb) 
icut_c =np.intersect1d(np.where((beam_c==BEAM)&(~np.isnan(ssha_c)))[0],icut_c1)
dist_c=(days_since_1985_c[icut_c]-days_since_1985_c[icut_c][0])*(24*60*60)*7000.
icut_a1,max_daya = pull_day(ymdhmsI_a,y=[],m=[],d=[],hr=[],mi=[],se=[],elem=max_dayb) 
icut_a =np.intersect1d(np.where((beam_a==BEAM)&(~np.isnan(ssha_a)))[0],icut_a1)
dist_a=(days_since_1985_a[icut_a]-days_since_1985_a[icut_a][0])*(24*60*60)*7000.


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

pmax=np.nanmax(dist_c)/1000.
plt.figure(figsize=(16,14))
plt.subplots_adjust(top=0.9,hspace=0.4,wspace=0.4)
plt.suptitle('Improvements made by removing wave signals for 2 m, 100 m and 2 km segments')
plt.subplot(321)
plt.title('2 m SSHA means')
plt.plot(dist_a/1000.,ssha_a[icut_a],'-',label='ssha')
plt.plot(dist_a/1000.,ssha_fft_a[icut_a],'-',label='ssha-(LFSW+HFSW)')
plt.xlabel('distance [km]')
plt.ylabel('ssha [m]')
plt.legend()
plt.grid()
plt.ylim(-mm,mm)
plt.xlim(0,pmax)
plt.subplot(323)
plt.title('100 m SSHA means')
plt.plot(dist_b/1000.,ssha_b[icut_b],'-',label='ssha')
plt.plot(dist_b/1000.,ssha_fft_b[icut_b],'-',label='ssha-(LFSW+HFSW)')
plt.xlabel('distance [km]')
plt.ylabel('ssha [m]')
plt.legend()
plt.grid()
plt.ylim(-mm,mm)
plt.xlim(0,pmax)
plt.subplot(325)
plt.title('2 km SSHA means')
plt.plot(dist_c/1000.,ssha_c[icut_c],'-',label='ssha')
plt.plot(dist_c/1000.,ssha_fft_c[icut_c],'-',label='ssha-(LFSW+HFSW)')
plt.xlabel('distance [km]')
plt.ylabel('ssha [m]')
plt.legend()
plt.grid()
plt.ylim(-mm,mm)
plt.xlim(0,pmax)
plt.subplot(322)
sda = str(np.round(np.nanstd(ssha_fft_a-ssha_a)*100.,3))
plt.title('2 m: original SSHA vs. SSHA w/o wave signal \n $\sigma_{diff}$ = '+sda+' cm')
binz = np.arange(-400,400+db,db)
plt.plot(beb[:-1],hista_norm,label='original',linewidth=6)
plt.plot(beb[:-1],histaF_norm,label='wave signal removed',linewidth=3)
plt.xlabel('ssha [cm]')
plt.ylabel('frequency')
plt.legend()
plt.grid()
plt.subplot(324)
sdb = str(np.round(np.nanstd(ssha_fft_b-ssha_b)*100.,3))
plt.title('100 m: original SSHA vs. SSHA w/o wave signal \n $\sigma_{diff}$ = '+sdb+' cm')
binz = np.arange(-300,300+db,db)
plt.plot(beb[:-1],histb_norm,label='original',linewidth=6)
plt.plot(beb[:-1],histbF_norm,label='wave signal removed',linewidth=3)
plt.xlabel('ssha [cm]')
plt.ylabel('frequency')
plt.legend()
plt.grid()
plt.subplot(326)
sdc = str(np.round(np.nanstd(ssha_fft_c-ssha_c)*100.,3))
plt.title('2 km: original SSHA vs. SSHA w/o wave signal \n $\sigma_{diff}$ = '+sdc+' cm')
binz = np.arange(-50,50+db,db)
plt.plot(beb[:-1],histc_norm,label='original',linewidth=6)
plt.plot(beb[:-1],histcF_norm,label='wave signal removed',linewidth=3)
plt.xlabel('ssha [cm]')
plt.ylabel('frequency')
plt.legend()
plt.grid()

pmax=10
plt.figure(figsize=(7,11))
plt.subplots_adjust(top=0.85,hspace=0.4,wspace=0.4)
plt.suptitle('validation: removing wave signal improves \n SSHA estimates of short segments')
ssha_fft_b_c = subseg_to_seg(ssha_fft_b[icut_b],days_since_1985_b[icut_b],days_since_1985_c[icut_c])
ssha_b_c = subseg_to_seg(ssha_b[icut_b],days_since_1985_b[icut_b],days_since_1985_c[icut_c])
plt.subplot(211)
plt.title('2km ATLCU vs. subsets')
plt.plot(dist_c[1:-1]/1000.,(ssha_c[icut_c[1:-1]])*100.,'-',label='2km ATLCU')
plt.plot(dist_c[1:-1]/1000.,(ssha_b_c[1:-1])*100.,'-',label='2km subset from 100 m (wave signal)')
plt.plot(dist_c[1:-1]/1000.,(ssha_fft_b_c[1:-1])*100.,'-',label='2km subset from 100 m (no wave signal)')
plt.xlabel('distance [km]')
plt.ylabel('ssha [cm]')
plt.legend()
plt.grid()
plt.subplot(212)
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


plt.figure()
plt.grid()
db = 0.05
wl_bins = np.arange(0,4+db,db)
wl = (wl_seg_b)
inn = np.where((~np.isnan(wl))&(~np.isinf(wl)))[0]
ce =  np.polyfit(wl[inn],ssha_b[inn],1)[::-1]
fit = ce[0]+(ce[1]*wl_bins)#+(ce[2]*swh_bins**2)+(ce[3]*swh_bins**3)
ce_fft =  np.polyfit(wl[inn],ssha_fft_b[inn],1)[::-1]
fit_fft = ce_fft[0]+(ce_fft[1]*wl_bins)
plt.plot(wl[wl<100],ssha_b[wl<100],'',label='C')
plt.plot(wl[wl<100],ssha_fft_b[wl<100],',',label='C fft')
plt.plot(wl_bins,fit,color='black')
plt.plot(wl_bins,fit_fft,color='black')
plt.xlabel('Wavelength [m]')
plt.ylabel('SSHA [m]')
plt.title('SSHA vs. Wavelength')
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
        plt.xlim(2020.43,2020.45)#(2020,2021)
        plt.ylim(-130,130)
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


TF=True
if TF==True:
    AVG = 'median'
else:
    AVG = 'mean'
dyr = np.round(4/365.25,5)
imin=15
yrf_bins = np.arange(2021,2022+dyr,dyr)
ssha_yrf_b,N_yrf_mn2_b = find_averages(yrf_bins,yrfrac_b,ssha_fft_b,imin=imin,MED=TF)
ssha_yrf_c,N_yrf_mn2_c = find_averages(yrf_bins,yrfrac_c,ssha_fft_c,imin=imin,MED=TF)
ssha_ws_yrf_b,N_ws_yrf_mn2_b = find_averages(yrf_bins,yrfrac_b,ssha_b,imin=imin,MED=TF)
ssha_ws_yrf_c,N_ws_yrf_mn2_c = find_averages(yrf_bins,yrfrac_c,ssha_c,imin=imin,MED=TF)
ssha_yrf_j3,N_yrf_j3 = find_averages(yrf_bins,yrfracA,ssha_alt,imin=imin,MED=TF)
ssha_yrf_s3,N_yrf_s3 = find_averages(yrf_bins,yrfrac_s3,ssha_s3,imin=imin,MED=TF)

swh_yrf_b,N_yrf_mn2_b = find_averages(yrf_bins,yrfrac_b,swh_b,imin=imin,MED=TF)
swh_yrf_c,N_yrf_mn2_c = find_averages(yrf_bins,yrfrac_c,swh_c,imin=imin,MED=TF)

swh66_yrf_b,N66_yrf_mn2_b = find_averages(yrf_bins,yrfrac_b,swh66_b,imin=imin,MED=TF)
swh66_yrf_c,N66_yrf_mn2_c = find_averages(yrf_bins,yrfrac_c,swh66_c,imin=imin,MED=TF)

swh_yrf_j3,N_yrf_j3 = find_averages(yrf_bins,yrfracA,swh_alt,imin=imin,MED=TF)
swh_yrf_s3,N_yrf_s3 = find_averages(yrf_bins,yrfrac_s3,swh_s3,imin=imin,MED=TF)
innB = np.where((~np.isnan(ssha_yrf_c))&(~np.isnan(ssha_yrf_j3))&(~np.isnan(ssha_yrf_s3)))[0]
mrk='.-'
plt.figure(figsize=(10,15))
plt.subplots_adjust(hspace=0.4,wspace=0.4)
plt.subplot(311)
plt.plot(yrf_bins[innB],ssha_yrf_b[innB],mrk,label='IS2 with WS (100 m)',color='green')
plt.plot(yrf_bins[innB],ssha_ws_yrf_b[innB],':',label='IS2 with WS (100 m)',color='green')
plt.plot(yrf_bins[innB],ssha_yrf_c[innB],mrk,label='IS2 (2 km)',color='red')
plt.plot(yrf_bins[innB],ssha_ws_yrf_c[innB],':',label='IS2 with WS (2 km)',color='red')
plt.plot(yrf_bins[innB],ssha_yrf_j3[innB],mrk,label='J3',color='gray')
plt.plot(yrf_bins[innB],ssha_yrf_s3[innB],mrk,label='S3',color='black')
plt.legend()
plt.xlabel('year fraction')
plt.ylabel('SSHA [m]')
plt.grid()
plt.title(AVG+' SSHA  vs. year fraction')
plt.subplot(312)
plt.plot(yrf_bins[innB],swh_yrf_b[innB],mrk,label='100 m (LFSW & HFSW)',color='red')
plt.plot(yrf_bins[innB],swh66_yrf_b[innB]*1.86,mrk,label='100 m (SW)',color='green')
plt.plot(yrf_bins[innB],swh_yrf_j3[innB],mrk,label='J3',color='gray')
plt.plot(yrf_bins[innB],swh_yrf_s3[innB],mrk,label='S3',color='black')
plt.legend()
plt.xlabel('year fraction')
plt.ylabel('SWH [m]')
plt.grid()
plt.title(AVG+' SWH  vs. year fraction')
plt.subplot(313)
plt.plot(yrf_bins[innB],swh_yrf_c[innB],mrk,label='2 km (LFSW & HFSW)',color='red')
plt.plot(yrf_bins[innB],swh66_yrf_c[innB]*1.86,mrk,label='2 km (SW)',color='green')
plt.plot(yrf_bins[innB],swh_yrf_j3[innB],mrk,label='J3',color='gray')
plt.plot(yrf_bins[innB],swh_yrf_s3[innB],mrk,label='S3',color='black')
plt.legend()
plt.xlabel('year fraction')
plt.ylabel('SWH [m]')
plt.grid()
plt.title(AVG+' SWH  vs. year fraction')



Nbins = np.size(swh_bins)
dswh = np.diff(swh_bins)[0]/2.
dyr = np.round(30/365.25,4)
imin=0
yrf_bins = np.arange(2021,2022+dyr,dyr)
mn_i2b_bins,N_mn_i2b_bins = find_averages(swh_bins,swh_b,ssha_fft_b,imin=imin)#,z=yrfrac_b,zconst=yrf_bins,imin=imin)
mn_i2c_bins,N_mn_i2c_bins = find_averages(swh_bins,swh_c,ssha_fft_c,imin=imin)#,z=yrfrac_c,zconst=yrf_bins,imin=imin)
mn_j3_bins,N_mn_j3_bins = find_averages(swh_bins,swh_alt,ssha_alt,imin=imin)#,z=yrfracA,zconst=yrf_bins,imin=imin)
mn_s3_bins,N_mn_s3_bins = find_averages(swh_bins,swh_s3,ssha_s3,imin=imin)#,z=yrfrac_s3,zconst=yrf_bins,imin=imin)

plt.figure(figsize=(8,6))
plt.plot(swh_bins,mn_i2b_bins,'-',label='is2 (100 m)')
plt.plot(swh_bins,mn_i2c_bins,'-',label='is2 (2 km))')
plt.plot(swh_bins,mn_j3_bins,'-',label='j3')
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




########### BEAMS ###########
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



dyr = 0.1
yrf_bins = np.arange(2021,2022+dyr,dyr)
ssha_yrf_mn2_c,N_yrf_mn2_c = find_averages(yrf_bins,yrfrac_c,ssha_fft_c,z=swh_c,zconst=[0,2],imin=0)
ssha_yrf_mn4_c,N_yrf_mn4_c = find_averages(yrf_bins,yrfrac_c,ssha_fft_c,z=swh_c,zconst=[2,4],imin=0)
ssha_yrf_mn6_c,N_yrf_mn6_c = find_averages(yrf_bins,yrfrac_c,ssha_fft_c,z=swh_c,zconst=[4,6],imin=0)
plt.figure()
plt.plot(yrf_bins,ssha_yrf_mn2_c,label='0-2 m SWH')
plt.plot(yrf_bins,ssha_yrf_mn4_c,label='2-4 m SWH')
plt.plot(yrf_bins,ssha_yrf_mn6_c,label='4-6 m SWH')
plt.legend()
plt.xlabel('year fraction')
plt.ylabel('SSHA [m]')
plt.grid()
plt.title('ICESat-2 mean SSHA (by SWH) vs. year fraction')




plt.figure()
plt.plot(swh_c,ssha_fft_c,'.')
plt.plot(swh_alt,ssha_alt,'.')

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


plt.figure(figsize=(8,6))
plt.plot(swh_b,np.array(ssha_fft_b)*100.,',',label='IS2 fft '+str(fp_b)+' m',color=clr[1])
plt.plot(swh_c,np.array(ssha_fft_c)*100.,'.',label='IS2 fft '+str(fp_c)+' m',color=clr[4])
if np.size(FNj3)!=0:
    plt.plot(swh_alt,np.array(ssha_alt)*100.,'.',label='J3',color=clr[2])
if np.size(FNc2)!=0:
    plt.plot(swh_c2,np.array(ssha_c2)*100.,'.',label='C2',color=clr[3])
if np.size(FNs3)!=0:
    plt.plot(swh_s3,np.array(ssha_s3)*100.,'.',label='S3',color=clr[6])
plt.legend()
plt.grid()
plt.xlabel('SWH [m]')
plt.ylabel('SSHAs [cm]')
plt.title(REG) 

plt.figure(figsize=(8,6))
plt.scatter(np.abs(slope_b),swh_b,c=ssha_b,cmap='jet')
plt.legend()
plt.grid()
cb = plt.colorbar()
cb.set_label('SSHA [cm]')
plt.xlabel('slope [m/s]')
plt.ylabel('SWH [m]')
plt.title(REG) 

plt.figure(figsize=(8,6))
plt.scatter(swh_c,ssha_c,c=slope_c,cmap='jet')
plt.legend()
plt.grid()
cb = plt.colorbar()
cb.set_label('slope [m/s]')
plt.xlabel('swh [m]')
plt.ylabel('SSHA [cm]')
plt.title(REG) 







# TREND?
#ce_mnb = np.polyfit(mn_time_b, mn_ssha_b, 1)[::-1]
ce_b = np.polyfit(days_since_1985_b,ssha_b, 1)[::-1]


########### GRIDDED AVERAGES ###########
mm = 50
cmap = 'RdYlGn_r' #'coolwarm'
lat_grid,lon_grid,time_grid,ssha_grid3d = lTG.month_2_month_grid(ymdhmsI_b,ssha_b,yrs_mm,lat_b,lon_b,dm=3)
lat_grid,lon_grid,ssha_grid = lTG.gridded(ssha_b*100,lat_b,lon_b,lat_minmax=[np.nanmin(lat_b),np.nanmax(lat_b)],lon_minmax=[np.nanmin(lon_b),np.nanmax(lon_b)])
lat_grid,lon_grid,ssha_fft_grid = lTG.gridded(ssha_fft_b*100,lat_b,lon_b,lat_minmax=[np.nanmin(lat_b),np.nanmax(lat_b)],lon_minmax=[np.nanmin(lon_b),np.nanmax(lon_b)])
if np.size(FNj3)!=0:
    lat_gridA,lon_gridA,ssha_gridA = lTG.gridded(ssha_alt*100,lat_alt,lon_alt,lat_minmax=[np.nanmin(lat_b),np.nanmax(lat_b)],lon_minmax=[np.nanmin(lon_b),np.nanmax(lon_b)])
    pbil.groundtracks_contour(np.unique(lon_gridA),np.unique(lat_gridA),ssha_gridA,'Jason-3 gridded. '+REG+' '+str(yrs_mm),'ssha [cm]',
                      cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                      LEV=np.arange(-0.5,0.55,0.05)*100.)
if np.size(FNc2)!=0:
    lat_gridC,lon_gridC,ssha_gridC = lTG.gridded(ssha_c2*100,lat_c2,lon_c2,lat_minmax=[np.nanmin(lat_b),np.nanmax(lat_b)],lon_minmax=[np.nanmin(lon_b),np.nanmax(lon_b)])
    pbil.groundtracks_contour(np.unique(lon_gridC),np.unique(lat_gridC),ssha_gridC,'CryoSat-2 gridded. '+REG+' '+str(yrs_mm),'ssha [cm]',
                      cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                      LEV=np.arange(-0.5,0.55,0.05)*100.)
pbil.groundtracks_contour(np.unique(lon_grid),np.unique(lat_grid),ssha_grid,'ICESat-2 gridded product ('+str(fp_b)+' m  footprint). '+REG+' '+str(yrs_mm),'ssha [cm]',
                    cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',LEV=np.arange(-0.5,0.55,0.05)*100.)
pbil.groundtracks_contour(np.unique(lon_grid),np.unique(lat_grid),ssha_fft_grid,'ICESat-2 gridded product ('+str(fp_b)+' m  footprint fft). '+REG+' '+str(yrs_mm),'ssha [cm]',
                    cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',LEV=np.arange(-0.5,0.55,0.05)*100.)
MONTH_GRID = False
if MONTH_GRID == True:
    for ii in np.arange(np.shape(ssha_grid3d)[2]):
        pbil.groundtracks_contour(np.unique(lon_grid),np.unique(lat_grid),ssha_grid3d[:,:,ii],str(time_grid[ii])+' | ICESat-2 gridded product ('+str(fp_b)+' m  footprint). '+REG+' '+str(yrs_mm),'ssha [cm]',
                          cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',LEV=np.arange(-0.5,0.55,0.05)*100.)
########### SPATIAL PLOTS ###########
mm=50
cmap='coolwarm'#'RdYlGn_r'#'coolwarm'
s1=6
if np.size(FNj3)!=0:
    '''
    LEG = 'ssha [cm] \n gray outline = Jason-3'
    pbil.groundtracks_multi(lon_b,lat_b,(ssha_b)*100.,'ICESat-2 ('+str(fp_b)+' m  footprint) with Jason-3. '+REG+' '+str(yrs_mm),LEG,
                       cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                       lon3=lon_alt,lat3=lat_alt,gt3=(ssha_alt)*100.)
    pbil.groundtracks_multi(lon_b,lat_b,(ssha_fft_b)*100.,'ICESat-2 ('+str(fp_b)+' m  footprint fft) with Jason-3. '+REG+' '+str(yrs_mm),LEG,
                       cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                       lon3=lon_alt,lat3=lat_alt,gt3=(ssha_alt)*100.)
    '''
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

pbil.groundtracks_multi(lon_b,lat_b,(ssha_b)*100.,'ICESat-2 ('+str(fp_b)+' m  footprint). '+REG+' '+str(yrs_mm),LEG,
                    cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',s1=s1)
pbil.groundtracks_multi(lon_b,lat_b,(ssha_fft_b)*100.,'ICESat-2 ('+str(fp_b)+' m  footprint) wave signals removed. '+REG+' '+str(yrs_mm),LEG,
                    cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',s1=s1)

'''
if SEG2==True:
    LEG = 'ssha [cm]'
    pbil.groundtracks_multi(lon_a,lat_a,(ssha_a)*100.,'ICESat-2 ('+str(fp_a)+' m  footprint) with TG and ALT. '+REG+' '+str(yrs_mm),LEG,
                       cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1')
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
'''       
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
