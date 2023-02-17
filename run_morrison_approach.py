#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:15:05 2022

@author: alexaputnam
"""


import numpy as np
from matplotlib import pyplot as plt
from datetime import date, timedelta, datetime
from netCDF4 import Dataset
import sys
import time
import pandas as pd
import netCDF4
import scipy

import lib_read_TG as lTG
import lib_regions as lreg
sys.path.append("/Users/alexaputnam/necessary_functions/")
import plt_bilinear as pbil

LOCDIR = '/Users/alexaputnam/ICESat2/'

'''
ds = np.load('h5_files/ATL03_20210806092915_06681207_005_01_filtered_on_cuyutlan.npy',allow_pickle='TRUE').item()
beams = ds.keys()
'''
def swell_categories():
    period = [13,14,17,20,25]


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

def dist_sv2pt(latSV,lonSV,beamSV,latPt,lonPt,maxKM=2000,beam=[]):
    xA,yA,zA = lla2ecef(latSV,lonSV)
    xP,yP,zP = lla2ecef(latPt,lonPt)
    dist = dist_func(xA,yA,zA,xP,yP,zP)
    if np.size(beam)!=0:
        if np.size(maxKM)==1:
            i20 = np.where((dist<=maxKM)&(beamSV==beam))[0]
        else:
            i20 = np.where((dist>=maxKM[0])&(dist<=maxKM[1])&(beamSV==beam))[0]
    else:
        i20 = np.where(dist<=maxKM)[0]
    return i20,dist

def swh_13(y):
    isrt_fitAno0 = np.argsort(np.abs(y)) 
    srt_fitAno0 = np.abs(y)[isrt_fitAno0][::-1]*2
    N = np.size(srt_fitAno0)
    N13 = int((N/3.0))
    SWH = np.round(np.nanmean(srt_fitAno0[:N13]),4)
    return SWH,srt_fitAno0

def find_ip(x):
    # requies a smoothed timeseries
    bx = np.copy(np.diff(x))
    bx[bx<0]=-1.0
    bx[bx>=0]=1.0
    bx2 = np.empty(np.shape(bx))*np.nan
    bx2[1:]=np.diff(bx)
    ip = np.where((np.round(np.diff(x),3)==0)|(np.abs(bx2)==2))[0]
    return ip

def swh_ip(x,dist,y):
    # requies:
    #### a smoothed timeseries, y
    #### time,x, in seconds
    ip = find_ip(y)
    WH = np.diff(y[ip])
    isrt = np.argsort(np.abs(WH))[::-1]
    Nwh = int(np.round(np.size(WH)/3.0))
    swh = np.round(np.nanmean(np.abs(WH)[isrt[:Nwh+1]]),4)
    Ts = np.round(np.nanmean(np.diff(dist[ip][::2])),3) #np.round(np.nanmean(np.diff(x[ip][::2]))*7000,3) #sv travels at 7000 m/s
    return swh,Ts,ip

def sinfit(x,y,numharm=10,ddeg=2):
    #import ccgfilt
    # x,y = distA2[i20A2],ssha_a[i20A2]
    numpoly=2
    M = np.size(x)
    H = np.ones((M,numharm+numpoly))
    H[:,1] = x
    #H[:,2] = x**2
    iis,iic=2,2
    for ii in np.arange(numharm):
        if ii%2==1:
            H[:,ii+numpoly] = np.sin(iis*np.pi*x)
            iis+=ddeg
        if ii%2==0:
            H[:,ii+numpoly] = np.cos(iic*np.pi*x)
            iic+=ddeg
    ce = np.linalg.inv(H.T.dot(H)).dot(H.T.dot(y))
    fit = ce[0]+ce[1]*x #+ ce[1]*x**2 + ce[2]*x**3
    iis,iic=2,2
    for ii in np.arange(numharm):
        if ii%2==1:
            fit = fit+ce[ii+numpoly]*np.sin(iis*np.pi*x)
            iis+=ddeg
        if ii%2==0:
            fit = fit+ce[ii+numpoly]*np.cos(iic*np.pi*x)
            iic+=ddeg
    return ce,fit

def fft2signal(x,y,Nf=30,FreqMax=False):
    #x,y,Nf = (days_since_1985_a[i20A2]-days_since_1985_a[i20A2][0])*(24*60*60),ssha_a[i20A2],[1,66]
    # https://scipy-lectures.org/intro/scipy/auto_examples/plot_fftpack.html
    ce = np.polyfit(x, y, 1)[::-1]
    fity = ce[0]+ce[1]*x
    # The FFT of the signal
    sig=y-fity#ssha_a[i20A2]
    time_vec = x# seconds
    dt = np.diff(time_vec)
    time_step = np.nanmedian(dt)
    sig_fft = scipy.fftpack.fft(sig)
    # And the power (sig_fft is of complex dtype)
    power = np.abs(sig_fft)**2
    # The corresponding frequencies (cycles/second)
    sample_freq = scipy.fftpack.fftfreq(sig.size, d=time_step)
    # Find the peak frequency: we can focus on only the positive frequencies
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]
    # remove all the high frequencies
    idx = np.argsort(np.abs(sample_freq))
    high_freq_fft = sig_fft.copy()
    if FreqMax==False:
        high_freq_fft[idx[Nf+1:]] = 0#high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
    else:
        idxF = np.where(np.abs(sample_freq)>Nf)[0]
        high_freq_fft[idxF]=0
    filtered_sig = scipy.fftpack.ifft(high_freq_fft)
    #ceB,filtered_sig_no0 = sinfit(x,filtered_sig,numharm=10)
    # dominant frequencies
    if FreqMax==False:
        new_freq_arr = sample_freq[idx[:Nf+1]]#np.copy(sample_freq[np.abs(sample_freq) < peak_freq])
    else:
        idxF2 = np.where(np.abs(sample_freq)<=Nf)[0]
        new_freq_arr = sample_freq[idxF2]
    #print(new_freq_arr)
    #print(high_freq_fft[idx[:Nf[1]+1]])
    '''
    plt.figure()
    plt.subplot(211)
    plt.plot(x,y)
    plt.plot(x,filtered_sig.real)
    plt.subplot(212)
    trange = np.linspace(0, 1.0/time_step, sig.size)
    igt0 = np.where((sample_freq > 0))[0]
    igt0_66 = np.where((sample_freq > 0))[0]
    plt.plot(sample_freq,power)
    plt.plot(sample_freq[idx[:Nf[1]+1]],power[idx[:Nf[1]+1]],'.')
    plt.ylabel('power spectrum')
    plt.xlabel('frequency (Hz)')
    plt.xlim(-100,100)
    '''
    return filtered_sig.real,new_freq_arr


def pull_icesat(FN,SEG=100,pth=LOCDIR+'tide_gauge_match/'):
    ds2 = np.load(pth+FN)#'reg_atl03_lat_41_lon_n73_newengland_2018_12_to_2019_12.npz'
    kys = list(ds2.keys())
    print(kys)
    swell_hf = []
    swell_lf = []
    if SEG==100:
        ATCH = ''
    elif SEG==2:
        ATCH = 'S'
    elif SEG==2000:
        ATCH = 'M'
    ssha_seg = ds2['ssha'+ATCH]#ds2['ssha'+ATCH]# ds2['ssha'+ATCH+'_md'] #
    time_seg = ds2['time'+ATCH]
    lon_segi = ds2['lon'+ATCH]
    lat_segi = ds2['lat'+ATCH]
    #ssha5k = ds2['sshaM']
    #sd = np.nanstd(ssha5k)
    #mn = np.nanmean(ssha5k)
    days_since_1985_seg = time_seg/86400.
    #idx_valid,xmin_200,xmax_200=atl03_regional(days_since_1985_seg,ssha_seg,Z=3.0)
    zval =  2
    llmm = [40.9,41.2,-71.2,-70.6]
    #idx_valid = np.where((lat_segi>=llmm[0])&(lat_segi<=llmm[1])&(lon_segi>=llmm[2])&(lon_segi<=llmm[3]))[0]#np.arange(np.shape(lat_segi)[0])##np.where((ssha_seg>=mn-(zval*sd))&(ssha_seg<=mn+(zval*sd)))[0]
    ssha_seg = ssha_seg
    time_seg = time_seg
    days_since_1985_seg = time_seg/86400.
    lon_seg = ds2['lon'+ATCH]
    lat_seg = ds2['lat'+ATCH]
    skew_seg = ds2['skew'+ATCH]
    tsI_seg,ymdhmsI_seg,yrfrac_seg = lTG.tide_days_1985_to_TS(days_since_1985_seg)
    beam_seg = ds2['beam'+ATCH]
    if ATCH=='S':
        swh_seg = ds2['swh'+ATCH]
    else:
        #swh_seg = ds2['swh_hf'+ATCH]#
        swh_seg = 4.0*np.sqrt(ds2['var'+ATCH]) #ds2['swh'+ATCH][idx_valid]
    N_seg = ds2['N'+ATCH]
    if SEG!=2:
        slope_seg = ds2['slope'+ATCH]
    else:
        slope_seg = np.nan
    return ssha_seg,lat_seg,lon_seg,days_since_1985_seg,ymdhmsI_seg,tsI_seg,beam_seg,swh_seg,N_seg,slope_seg,skew_seg,yrfrac_seg


FNis2= 'reg_atl03_lat_19_lon_n105_cuyutlan_segs_2_100_2000_2021_08_to_2021_08.npz'
ssha_a,lat_a,lon_a,days_since_1985_a,ymdhmsI_a,tsI_a,beam_a,swh_a,N_a,slope_a,skew_a,yrfrac_a = pull_icesat(FNis2,SEG=2,pth=LOCDIR+'h5_files/')
ssha_b,lat_b,lon_b,days_since_1985_b,ymdhmsI_b,tsI_b,beam_b,swh_b,N_b,slope_b,skew_b,yrfrac_b = pull_icesat(FNis2,SEG=100,pth=LOCDIR+'h5_files/')
ssha_c,lat_c,lon_c,days_since_1985_c,ymdhmsI_c,tsI_c,beam_c,swh_c,N_c,slope_c,skew_c,yrfrac_c = pull_icesat(FNis2,SEG=2000,pth=LOCDIR+'h5_files/')

SV_vel = 7000. # ICESat-2 vel = 7km/s
off_space = 90
dT_bin = np.round(off_space/SV_vel,4) # segment_footprint/7000.

beam=2

landpt1 = [18.9284951,-104.0905072]#[18.9437831,-104.1191201]
landpt2 = [18.9284951,-104.0905072]
landpt3 = [18.9284951,-104.0905072]#[18.9104661,-104.0605142]

maxKM=[0,2000]#30000]#30000]
i20A1,distA1 = dist_sv2pt(lat_a,lon_a,beam_a,landpt1[0],landpt1[1],maxKM=maxKM,beam=1)
i20B1,distB1 = dist_sv2pt(lat_b,lon_b,beam_b,landpt1[0],landpt1[1],maxKM=maxKM,beam=1)
i20C1,distC1 = dist_sv2pt(lat_c,lon_c,beam_c,landpt1[0],landpt1[1],maxKM=maxKM,beam=1)
i20A10,distA10 = dist_sv2pt(lat_a,lon_a,beam_a,landpt1[0],landpt1[1],maxKM=maxKM,beam=10)
i20B10,distB10 = dist_sv2pt(lat_b,lon_b,beam_b,landpt1[0],landpt1[1],maxKM=maxKM,beam=10)
i20C10,distC10 = dist_sv2pt(lat_c,lon_c,beam_c,landpt1[0],landpt1[1],maxKM=maxKM,beam=10)

i20A2,distA2 = dist_sv2pt(lat_a,lon_a,beam_a,landpt2[0],landpt2[1],maxKM=maxKM,beam=2)
i20B2,distB2 = dist_sv2pt(lat_b,lon_b,beam_b,landpt2[0],landpt2[1],maxKM=maxKM,beam=2)
i20C2,distC2 = dist_sv2pt(lat_c,lon_c,beam_c,landpt2[0],landpt2[1],maxKM=maxKM,beam=2)
i20A20,distA20 = dist_sv2pt(lat_a,lon_a,beam_a,landpt2[0],landpt2[1],maxKM=maxKM,beam=20)
i20B20,distB20 = dist_sv2pt(lat_b,lon_b,beam_b,landpt2[0],landpt2[1],maxKM=maxKM,beam=20)
i20C20,distC20 = dist_sv2pt(lat_c,lon_c,beam_c,landpt2[0],landpt2[1],maxKM=maxKM,beam=20)

i20A3,distA3 = dist_sv2pt(lat_a,lon_a,beam_a,landpt3[0],landpt3[1],maxKM=maxKM,beam=3)
i20B3,distB3 = dist_sv2pt(lat_b,lon_b,beam_b,landpt3[0],landpt3[1],maxKM=maxKM,beam=3)
i20C3,distC3 = dist_sv2pt(lat_c,lon_c,beam_c,landpt3[0],landpt3[1],maxKM=maxKM,beam=3)
i20A30,distA30 = dist_sv2pt(lat_a,lon_a,beam_a,landpt3[0],landpt3[1],maxKM=maxKM,beam=30)
i20B30,distB30 = dist_sv2pt(lat_b,lon_b,beam_b,landpt3[0],landpt3[1],maxKM=maxKM,beam=30)
i20C30,distC30 = dist_sv2pt(lat_c,lon_c,beam_c,landpt3[0],landpt3[1],maxKM=maxKM,beam=30)


i20A2_a,dist_a = dist_sv2pt(lat_a,lon_a,beam_a,lat_a[0],lon_a[0],maxKM=100000000,beam=2)

def spectral_ssha(x,y):
    #x1,y=t_a,ssha_a[icut]
    ce = np.polyfit(x, y, 1)[::-1]
    fity = ce[0]+ce[1]*x
    fitAno0,fA = fft2signal(x,y,Nf=110,FreqMax=True)
    lim=2
    plt.figure(figsize=(20,5))
    plt.subplots_adjust(top=0.9,hspace=0.4,wspace=0.4)
    plt.suptitle('Swell and Sea (Cuyutlan, Mexico)')
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
    f, Pxx_den = signal.periodogram(y, fs)#,scaling='spectrum')
    #plt.semilogy(f[1:], Pxx_den[1:],'-')
    plt.plot(f[1:], Pxx_den[1:],color='black')
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
    

spectral_ssha((days_since_1985_a[i20A2]-days_since_1985_a[i20A2][0])*(24*60*60),ssha_a[i20A2])



def window_smooth(y,w=50):
    if w%2!=0:
        w=w+1
    N = np.size(y)
    sy = np.empty(N)*np.nan
    idx = np.arange(w/2,N-w/2)
    for ii in idx.astype(int):
        sy[ii] = np.nanmean(y[int(ii-w/2):int(ii+w/2)])
    return sy



def plot_dist_ssha(x,dist,y,XLAB='distance to coast [m]',Nf1=30,Nf2=70,Nf3=90,w=50):
    #x,dist,y,Nf1,Nf2,NF3,XLAB=(days_since_1985_a[i20A2]-days_since_1985_a[i20A2][0])*(24*60*60),distA2[i20A2],ssha_a[i20A2],30,70,66,'distance [m]'
    
    ce = np.polyfit(x, y, 1)[::-1]
    fity = ce[0]+ce[1]*x
    NFS = np.arange(0,1001,1)
    nN = np.size(NFS)
    resNF = np.empty(nN)*np.nan
    fA_size = np.empty(nN)*np.nan
    for ii in np.arange(nN):
        fitAno0,fA = fft2signal(x,y,Nf=NFS[ii],FreqMax=True)
        resNF[ii] = np.nanstd(y-fity-fitAno0)
        fA_size[ii] = np.size(fA)
    knee = 60
    fitAno0_12,fA_12 = fft2signal(x,y,Nf=110,FreqMax=True)#60)
    SWH_swell_12,srt_fitAno0_12 = swh_13(fitAno0_12)

    dist_seg = np.arange(dist[0],dist[-1],2000)
    max_freq = np.empty(np.size(dist_seg)-1)*np.nan
    resNF_all = np.empty((nN,np.size(dist_seg)-1))*np.nan
    i0 = np.where(NFS==110)[0]
    for ii in np.arange(np.size(dist_seg)-1):
        idist = np.where((dist>=dist_seg[ii])&((dist<dist_seg[ii+1])))[0]
        fitAno0,fA = fft2signal(x[idist],y[idist],Nf=knee)
        max_freq[ii] = np.nanmax(np.abs(fA))
        for jj in np.arange(nN):
            ce = np.polyfit(x[idist], y[idist], 1)[::-1]
            fity = ce[0]+ce[1]*x[idist]
            fitAno0,fA = fft2signal(x[idist],y[idist],Nf=NFS[jj],FreqMax=True)
            resNF_all[jj,ii] = np.nanstd(y[idist]-fitAno0-fity)

    plt.figure(figsize=(8,10))
    plt.subplots_adjust(top=0.8,hspace=0.5)
    for ii in np.arange(np.size(dist_seg)-1):
            if ii>np.size(dist_seg)/2:
                mrk=':'
            else:
                mrk='-'
            plt.plot(NFS,resNF_all[:,ii],mrk,label=str(np.round(dist_seg[ii+1]/1000))+' km FC')
    #plt.plot(NFS,np.nanmean(resNF_all,axis=1),'-',label='mean',color='black',linewidth=5)
    plt.grid()
    plt.xlabel('Frequency Limit used for Fit',fontsize=16)
    plt.ylabel('Standard Deviation of the SSHA - wave signal',fontsize=16)
    #plt.axvline(x=110,color='red')
    plt.plot(NFS[i0],np.nanmean(resNF_all,axis=1)[i0],'o',color='black',markersize=10)
    #plt.axhline(y=resNF[NFS==knee][0],color='red')
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    #plt.title('$\sigma_{(ssha-wave signal)}$ vs. frequency limit for multiple 2 km segments \n at different distances from coast (FC)')
    
    plt.figure(figsize=(12,6))
    plt.suptitle('Find the lowest frequency that best represents the wave signal using increments of 2 km segments from coast into open ocean \n use that frequency to define frequency band')
    plt.subplots_adjust(top=0.8,hspace=0.5)
    plt.subplot(111)
    plt.plot(np.round(dist_seg[1:]/1000),max_freq,'o')
    plt.grid()
    plt.xlabel('Segment distance from coast [km FC]')
    plt.ylabel('Max top 60 frequency')
    plt.title('mean = '+str(np.round(np.nanmean(max_freq))))


    plt.figure(figsize=(14,8))
    x1,x2=400,500
    alp=0.2
    plt.subplots_adjust(top=0.85,hspace=0.5)
    plt.suptitle('Improving short segment ICESat-2 SSHA estimates by removing wave signals')
    ax=plt.subplot(221)
    plt.title('2km SSHA segment with wave signals')
    plt.plot(dist,y,label='ssha (2m)')
    plt.legend()
    plt.grid()
    plt.ylim(-0.5,1.5)
    ax.axvspan(x1,x2, alpha=alp, color='red')
    plt.xlabel('distance [m]')
    plt.ylabel('SSHA [m]')
    ax=plt.subplot(222)
    plt.title('100 m SSHA segment with wave signals')
    plt.plot(dist,y,label='ssha (2m)')
    plt.legend()
    plt.grid()
    plt.ylim(-0.5,1.5)
    ax.axvspan(x1,x2, alpha=alp, color='red')
    plt.xlabel('distance [m]')
    plt.ylabel('SSHA [m]')
    plt.xlim(x1,x2)
    ax=plt.subplot(223)
    plt.title('2km SSHA segment without wave signals')
    plt.plot(dist,y-fitAno0_12,label='ssha (2m)')
    plt.legend()
    plt.grid()
    plt.ylim(-0.5,1.5)
    ax.axvspan(x1,x2, alpha=alp, color='red')
    plt.xlabel('distance [m]')
    plt.ylabel('SSHA [m]')
    ax=plt.subplot(224)
    plt.title('100 m SSHA segment without wave signals')
    plt.plot(dist,y-fitAno0_12,label='ssha (2m)')
    plt.legend()
    plt.grid()
    plt.ylim(-0.5,1.5)
    ax.axvspan(x1,x2, alpha=alp, color='red')
    plt.xlabel('distance [m]')
    plt.ylabel('SSHA [m]')
    plt.xlim(x1,x2)


    plt.figure()
    ax=plt.subplot(111)
    plt.title('ICESat-2 2 km SSHA segment')
    plt.plot(dist,y,label='ssha (2m)')
    plt.legend()
    plt.grid()
    plt.ylim(-0.5,1.5)
    #ax.axvspan(x1,x2, alpha=alp, color='red')
    plt.xlabel('distance [m]')
    plt.ylabel('SSHA [m]')


    plt.figure(figsize=(18,5))
    plt.subplots_adjust(top=0.80,hspace=0.5)
    plt.suptitle('Steps to remove wave signals from ICESat-2 SSHA estimates')
    plt.subplot(131)
    plt.title('(1) Fit line to 2km SSHA segment')
    plt.plot(dist,y,label='ssha (2m)')
    ce = np.polyfit(x, y, 1)[::-1]
    fity = ce[0]+ce[1]*x
    plt.plot(dist,fity,label='ssha trend',linewidth=4,color='tab:green')
    plt.legend()
    plt.grid()
    plt.ylim(-0.5,1.5)
    plt.subplot(132)
    res0 = np.nanstd(y)
    RES0=np.str(np.round(res0,4))
    plt.title('(2) Estimate wave signal using FFT and detrended SSHA \n residuals: std(ssha) = '+RES0+' m')
    plt.plot(dist,y,label='ssha (2m)')
    plt.plot(dist,fitAno0_12,label='wave signal')# ('+str(knee)+' freqs)',color='tab:orange')
    plt.legend()
    plt.grid()
    plt.xlabel(XLAB)
    plt.ylabel('SSHA [m')
    plt.ylim(-0.5,1.5)
    ######
    plt.subplot(133)
    res1 = np.nanstd(y-fitAno0_12)
    RES1=np.str(np.round(res1,4))
    plt.title('(3) Remove wave signal from SSHA measurements \n residuals: std(ssha - wave signal) = '+RES1+' m')
    plt.plot(dist, y-fitAno0_12,label='ssha (2m) - wave signal')
    plt.plot(dist,fity,label='ssha trend from (1)',linewidth=4,color='tab:green')
    #plt.plot(x, fitA-fitAno0,label='lf (1st CE) - lf')
    plt.legend()
    plt.grid()
    plt.xlabel(XLAB)
    plt.ylabel('SSHA [m')
    plt.ylim(-0.5,1.5)

    ll,ll2=1,1.5
    FS=16
    plt.figure()
    plt.plot(dist,y)
    ce = np.polyfit(x, y, 1)[::-1]
    fity = ce[0]+ce[1]*x
    plt.plot(dist,fity,label='ssha trend',linewidth=4,color='tab:green')
    plt.xlabel('distance within segment [m]',fontsize=FS)
    plt.ylabel('SSHA [m]',fontsize=FS)
    plt.ylim(-ll,ll2)
    plt.axhline(y=0,color='black')
    plt.legend(fontsize=FS)
    plt.tick_params(axis='both', which='major', labelsize=FS)

    plt.figure()
    plt.plot(dist,y-fity)
    plt.xlabel('distance within segment [m]',fontsize=FS)
    plt.ylabel('detrended SSHA [m]',fontsize=FS)
    plt.ylim(-ll,ll2)
    plt.axhline(y=0,color='black')
    plt.tick_params(axis='both', which='major', labelsize=FS)

    plt.figure()
    ce = np.polyfit(x, y, 1)[::-1]
    fity = ce[0]+ce[1]*x
    plt.plot(dist,fitAno0_12,color='tab:orange')
    plt.xlabel('distance within segment [m]',fontsize=FS)
    plt.ylabel('wave signal [m]',fontsize=FS)
    plt.ylim(-ll,ll2)
    plt.axhline(y=0,color='black')
    plt.tick_params(axis='both', which='major', labelsize=FS)

    plt.figure()
    plt.plot(dist, y-fitAno0_12)
    plt.plot(dist,fity,label='ssha trend',linewidth=4,color='tab:green')
    plt.xlabel('distance within segment [m]',fontsize=FS)
    plt.ylabel('SSHA - wave signal [m]',fontsize=FS)
    plt.ylim(-ll,ll2)
    plt.axhline(y=0,color='black')
    plt.legend(fontsize=FS)
    plt.tick_params(axis='both', which='major', labelsize=FS)


    fitAno0,fA = fft2signal(x,y,Nf=Nf1,FreqMax=True)#30)
    fitAno0_2,fA_2 = fft2signal(x,y-fitAno0,Nf=Nf2,FreqMax=True)#60)
    fitAno0_3,fA_3 = fft2signal(x,y-fitAno0-fitAno0_2,Nf=Nf3,FreqMax=True)#60)
    
    SWH_swell,srt_fitAno0 = swh_13(fitAno0)
    SWH_grav,srt_fitAno0_2 = swh_13(fitAno0_2)
    SWH_cap,srt_fitAno0_3 = swh_13(fitAno0_3)

    SWH_swell2,Ts_swell2,ip_swell = swh_ip(x,dist,fitAno0)
    SWH_swell2_2,Ts_swell2_2,ip_swell_2 = swh_ip(x,dist,fitAno0_2)
    SWH_swell2_12,Ts_swell2_12,ip_swell_12 = swh_ip(x,dist,fitAno0_12)
    SWH_swell2_12,Ts_swell2_12,ip_swell_12 = swh_ip(x,dist,fitAno0+fitAno0_2)
    SWH_swell2_3,Ts_swell2_3,ip_swell_3 = swh_ip(x,dist,fitAno0_3)
    
    mm=1
    MX=[1000,1100]
    plt.figure(figsize=(8,12))
    plt.subplots_adjust(top=0.85,hspace=0.5)
    plt.suptitle('Wave parameters. \n total SWH = '+str(SWH_swell2+SWH_swell2_2)+' m')
    '''
    plt.suptitle('LF SWH (1/3)): '+str(SWH_swell)+' \n HF SWH (1/3)): '+str(SWH_grav)+' \n LF SWH (measured): '+
    str(SWH_swell2)+' \n HF SWH (measured)): '+str(SWH_swell2_2)+' \n LF WL: '+
    str(Ts_swell2)+' \n HF WL: '+str(Ts_swell2_2))
    '''
    plt.subplot(311)
    '''
    plt.title('LFSW SWH (1/3)): '+str(SWH_swell)+' \n LFSW SWH (measured): '+
    str(SWH_swell2)+' \n LFSW WL: '+str(Ts_swell2))'''
    plt.title(' LFSW SWH: '+str(SWH_swell2)+' m \n LFSW WL: '+str(Ts_swell2)+' m')
    plt.plot(dist, fitAno0,label='LFSW')
    plt.plot(dist[ip_swell], fitAno0[ip_swell],'o',label='max/min')
    plt.legend()
    plt.grid()
    plt.xlabel(XLAB)
    plt.ylabel('SSHA [m')
    plt.ylim(-mm,mm)
    plt.xlim(MX[0],MX[1])
    plt.subplot(312)
    '''
    plt.title(' \n HFSW SWH (1/3)): '+str(SWH_grav)+' \n HFSW SWH (measured)): '+str(SWH_swell2_2)+
    ' \n HFSW WL: '+str(Ts_swell2_2))'''
    plt.title('MFSW SWH: '+str(SWH_swell2_2)+' m \n mFSW WL: '+str(Ts_swell2_2)+' m')
    plt.plot(dist, fitAno0_2,label='MFSW')
    plt.plot(dist[ip_swell_2], fitAno0_2[ip_swell_2],'o',label='max/min')
    plt.legend()
    plt.grid()
    plt.xlabel(XLAB)
    plt.ylabel('SSHA [m')
    plt.ylim(-mm,mm)
    plt.xlim(MX[0],MX[1])
    plt.subplot(313)
    plt.title('1 iteration SW SWH: '+str(SWH_swell2_12)+' m \n SW WL: '+str(Ts_swell2_12)+' m')
    plt.plot(dist, fitAno0_12,label='HFSW')
    plt.plot(dist[ip_swell_12], fitAno0_12[ip_swell_12],'o',label='max/min')
    plt.legend()
    plt.grid()
    plt.xlabel(XLAB)
    plt.ylabel('SSHA [m')
    plt.ylim(-mm,mm)
    plt.xlim(MX[0],MX[1])
    '''
    plt.subplot(313)
    plt.title('HFSW SWH: '+str(SWH_swell2_3)+' m \n HFSW WL: '+str(Ts_swell2_3)+' m')
    plt.plot(dist, fitAno0_3,label='HFSW')
    plt.plot(dist[ip_swell_3], fitAno0_3[ip_swell_3],'.',label='max/min')
    plt.legend()
    plt.grid()
    plt.xlabel(XLAB)
    plt.ylabel('SSHA [m')
    plt.ylim(-mm,mm)
    plt.xlim(MX[0],MX[1])
    '''
    '''
    plt.subplot(313)
    plt.plot(srt_fitAno0,'.',label='swell')
    plt.plot(srt_fitAno0_2,'.',label='gravity')
    plt.grid()
    plt.legend()
    '''
    '''
    mm=1
    plt.figure(figsize=(12,8))
    plt.subplots_adjust(hspace=0.4,wspace=0.4)
    plt.suptitle('Coastal SSHA measurements and wave fits for Cuyutlan, Mexico (suggested by Jamie Morrison)')
    plt.subplot(221)
    res0 = np.nanstd(y)
    RES0=np.str(np.round(res0,4))
    plt.title('residuals: std(2-m ssha) = '+RES0+' m')
    plt.plot(dist,y,label='2-m ssha')
    #plt.plot(x, fitA,label='LFSW (with first requency)')
    plt.plot(dist, fitAno0,label='LFSW')
    #plt.plot(x, fitAno0+fitAno0_2,label='fit no0 + fit2 no0')
    plt.legend()
    plt.grid()
    plt.xlabel(XLAB)
    plt.ylabel('SSHA [m')
    plt.ylim(-mm,mm)
    plt.xlim(MX[0],MX[1])
    ######
    plt.subplot(222)
    res1 = np.nanstd(y-fitAno0)
    RES1=np.str(np.round(res1,4))
    plt.title('residuals: std(2-m ssha - LFSW) = '+RES1+' m')
    plt.plot(dist, y-fitAno0,label='2-m ssha - LFSW')
    #plt.plot(x, fitA-fitAno0,label='lf (1st CE) - lf')
    plt.legend()
    plt.grid()
    plt.xlabel(XLAB)
    plt.ylabel('SSHA [m')
    plt.ylim(-mm,mm)
    plt.xlim(MX[0],MX[1])
    ######
    plt.subplot(223)
    plt.plot(dist,y-fitAno0,label='2-m ssha - LFSW')
    #plt.plot(x, fitA_2,label='HFSW (with first requency)')
    plt.plot(dist, fitAno0_2,label='HFSW')
    plt.legend()
    plt.grid()
    plt.xlabel(XLAB)
    plt.ylabel('SSHA [m')
    plt.ylim(-mm,mm)
    plt.xlim(MX[0],MX[1])
    #######
    plt.subplot(224)
    res3 = np.nanstd((y-fitAno0)-fitAno0_2)
    RES3=np.str(np.round(res3,4))
    plt.title('residuals: std(2-m ssha - (LFSW+HFSW)) = '+RES3+' m')
    plt.plot(dist, (y-fitAno0)-fitAno0_2,label='2-m ssha - (LFSW+HFSW)')
    #plt.plot(x, fitA_2-fitAno0_2,label='hf (1st CE) - hf') 
    plt.legend()
    plt.grid()
    plt.xlabel(XLAB)
    plt.ylabel('SSHA [m')
    plt.ylim(-mm,mm)
    plt.xlim(MX[0],MX[1])
    #plt.xlim(0,0.25/2)
    ######



    '''
    plt.figure(figsize=(14,8))
    plt.subplots_adjust(hspace=0.4,wspace=0.4)
    plt.suptitle('Coastal SSHA measurements and wave fits for Cuyutlan, Mexico (suggested by Jamie Morrison)')
    plt.subplot(321)
    res0 = np.nanstd(y)
    RES0=np.str(np.round(res0,4))
    plt.title('residuals: std(2-m ssha) = '+RES0+' m')
    plt.plot(dist,y,label='2-m ssha')
    #plt.plot(x, fitA,label='LFSW (with first requency)')
    plt.plot(dist, fitAno0,label='LFSW')
    #plt.plot(x, fitAno0+fitAno0_2,label='fit no0 + fit2 no0')
    plt.legend()
    plt.grid()
    plt.xlabel(XLAB)
    plt.ylabel('SSHA [m')
    plt.ylim(-mm,mm)
    plt.xlim(MX[0],MX[1])
    ######
    plt.subplot(322)
    res1 = np.nanstd(y-fitAno0)
    RES1=np.str(np.round(res1,4))
    plt.title('residuals: std(2-m ssha - LFSW) = '+RES1+' m')
    plt.plot(dist, y-fitAno0,label='2-m ssha - LFSW')
    #plt.plot(x, fitA-fitAno0,label='lf (1st CE) - lf')
    plt.legend()
    plt.grid()
    plt.xlabel(XLAB)
    plt.ylabel('SSHA [m')
    plt.ylim(-mm,mm)
    plt.xlim(MX[0],MX[1])
    ######
    plt.subplot(323)
    plt.plot(dist,y-fitAno0,label='2-m ssha - LFSW')
    #plt.plot(x, fitA_2,label='HFSW (with first requency)')
    plt.plot(dist, fitAno0_2,label='HFSW')
    plt.legend()
    plt.grid()
    plt.xlabel(XLAB)
    plt.ylabel('SSHA [m')
    plt.ylim(-mm,mm)
    plt.xlim(MX[0],MX[1])
    #######
    plt.subplot(324)
    res3 = np.nanstd((y-fitAno0)-fitAno0_2)
    RES3=np.str(np.round(res3,4))
    plt.title('residuals: std(2-m ssha - (LFSW+HFSW)) = '+RES3+' m')
    plt.plot(dist, (y-fitAno0)-fitAno0_2,label='2-m ssha - (LFSW+HFSW)')
    #plt.plot(x, fitA_2-fitAno0_2,label='hf (1st CE) - hf') 
    plt.legend()
    plt.grid()
    plt.xlabel(XLAB)
    plt.ylabel('SSHA [m')
    plt.ylim(-mm,mm)
    plt.xlim(MX[0],MX[1])
    #plt.xlim(0,0.25/2)
    ######
    plt.subplot(325)
    plt.plot(dist,y-fitAno0-fitAno0_2,label='2-m ssha - LFSW - HFSW')
    #plt.plot(x, fitA_2,label='HFSW (with first requency)')
    plt.plot(dist, fitAno0_3,label='VHFSW')
    plt.legend()
    plt.grid()
    plt.xlabel(XLAB)
    plt.ylabel('SSHA [m')
    plt.ylim(-mm,mm)
    plt.xlim(MX[0],MX[1])
    #######
    plt.subplot(326)
    res4 = np.nanstd((y-fitAno0)-fitAno0_2-fitAno0_3)
    RES4=np.str(np.round(res4,4))
    plt.title('residuals: std(2-m ssha - (LFSW+MFSW+HFSW)) = '+RES4+' m')
    plt.plot(dist, (y-fitAno0)-fitAno0_2-fitAno0_3,label='2-m ssha - (LFSW+MFSW+HFSW)')
    #plt.plot(x, fitA_2-fitAno0_2,label='hf (1st CE) - hf')
    plt.legend()
    plt.grid()
    plt.xlabel(XLAB)
    plt.ylabel('SSHA [m')
    plt.ylim(-mm,mm)
    plt.xlim(MX[0],MX[1])

plot_dist_ssha((days_since_1985_a[i20A2]-days_since_1985_a[i20A2][0])*(24*60*60),distA2[i20A2],ssha_a[i20A2],Nf1=100,Nf2=150,Nf3=110)#distA2[i20A2],ssha_a[i20A2]


plt.figure()
plt.plot(days_since_1985_b[beam_b==30],ssha_b[beam_b==30],label='gt3l')
plt.plot(days_since_1985_b[beam_b==20],ssha_b[beam_b==20],label='gt2l')
plt.plot(days_since_1985_b[beam_b==10],ssha_b[beam_b==10],label='gt1l')
plt.plot(days_since_1985_b[beam_b==3],ssha_b[beam_b==3],label='gt3r')
plt.plot(days_since_1985_b[beam_b==2],ssha_b[beam_b==2],label='gt2r')
plt.plot(days_since_1985_b[beam_b==1],ssha_b[beam_b==1],label='gt1r')
plt.legend()
plt.grid()
plt.xlabel('time')
plt.ylabel('SSHA [m')

plt.figure()
plt.plot(distA2[i20A2],ssha_a[i20A2],'-')
plt.plot(distB2[i20B2],ssha_b[i20B2],'-')
plt.plot(distC2[i20C2],ssha_c[i20C2],'-')
plt.xlim(0,maxKM)
plt.grid()

plt.figure()
#plt.plot(distA2[i20A2],ssha_a[i20A2],'-',label='2a')
plt.plot(distB2[i20B2],ssha_b[i20B2],'-',label='2b')
plt.plot(distC2[i20C2],ssha_c[i20C2],'-',label='2c')
#plt.plot(distA20[i20A20],ssha_a[i20A20],'-',label='20a')
plt.plot(distB20[i20B20],ssha_b[i20B20],'-',label='20b')
plt.plot(distC20[i20C20],ssha_c[i20C20],'-',label='20c')
plt.xlim(0,maxKM)
plt.grid()

dt1=np.nanmean(days_since_1985_b[i20B1])-np.nanmean(days_since_1985_b[i20B10])

dt3=np.nanmean(days_since_1985_b[i20B1])-np.nanmean(days_since_1985_b[i20B30])

#import lib_col_xov as lcx

i20B_2_20,distB_2_20 = dist_sv2pt(lat_b[beam_b==20],lon_b[beam_b==20],beam_b,lat_b[beam_b==2][0],lon_b[beam_b==2][0],maxKM=maxKM,beam=[])
dt2=(days_since_1985_b[beam_b==20][distB_2_20==np.nanmin(distB_2_20)])-(days_since_1985_b[beam_b==2][0])
plt.figure()
#plt.plot(days_since_1985_a[i20A2],ssha_a[i20A2],'-',label='2a')
plt.plot(days_since_1985_b[i20B2],ssha_b[i20B2],'-',label='2b')
plt.plot(days_since_1985_c[i20C2],ssha_c[i20C2],'-',label='2c')
#plt.plot(days_since_1985_a[i20A20],ssha_a[i20A20],'-',label='20a')
plt.plot(days_since_1985_b[i20B20]-dt2,ssha_b[i20B20],'-',label='20b')
plt.plot(days_since_1985_c[i20C20]-dt2,ssha_c[i20C20],'-',label='20c')
#plt.xlim(0,maxKM)
plt.grid()



'''
### Create files to plot on Google to find coast
root_grp = Dataset('h5_files/cuyutlan_lat_lon.nc', 'w', format='NETCDF4')
root_grp.description = 'This file contains lat/lon for h5 ICESat2 files'
root_grp.history = "Author: Alexa Angelina Putnam Shilleh. Institute: University of Colorado Boulder" + time.ctime(time.time())
dim = np.shape(lat_c)[0]
root_grp.createDimension('x', dim)

x1a = root_grp.createVariable('lat', 'f8', ('x',))
x1a[:] = lat_c
x1b = root_grp.createVariable('lon', 'f8', ('x',))
x1b[:] = lon_c
x1c = root_grp.createVariable('ssha', 'f8', ('x',))
x1c[:] = ssha_c
root_grp.close()


precip_nc_file = r'h5_files/cuyutlan_lat_lon.nc'
nc = netCDF4.Dataset(precip_nc_file, mode='r')
cols = list(nc.variables.keys())
list_nc = []
for c in cols:
    list_nc.append(list(nc.variables[c][:]))
df_nc = pd.DataFrame(list_nc)
df_nc = df_nc.T
df_nc.columns = cols
df_nc.to_csv('h5_files/cuyutlan_lat_lon.csv', index = False)
'''


'''

plt.figure()
plt.plot(days_since_1985_b,beam_b,'.')

from scipy.optimize import curve_fit 
# https://stackoverflow.com/questions/32590720/create-scipy-curve-fitting-definitions-for-fourier-series-dynamically
def fourier(x, *a):
    ret = a[0]#(a[0] * np.cos(np.pi * x))+(a[0] * np.sin(np.pi * x))
    for deg in range(0, len(a)):
        ret += (a[deg] * np.cos((deg+1) * np.pi * x))+(a[deg] * np.sin((deg+1) * np.pi * x))
    return ret

mm=1
poptB, pcovB = curve_fit(fourier, distB2[i20B2],ssha_b[i20B2], [1.0] * 8)
plt.figure()
plt.subplot(211)
plt.plot(distB2[i20B2],ssha_b[i20B2],label='gt2r')
plt.plot(distB2[i20B2], fourier(distB2[i20B2], *poptB))
#plt.plot(distB2[i20B2], fourier(distB2[i20B2], popt[0], popt[1], popt[2]))
plt.legend()
plt.grid()
plt.xlabel('distance [m]')
plt.ylabel('SSHA [m')
plt.ylim(-mm,mm)
plt.subplot(212)
plt.plot(distB2[i20B2], ssha_b[i20B2]-fourier(distB2[i20B2], *poptB))
#plt.plot(distB2[i20B2], fourier(distB2[i20B2], popt[0], popt[1], popt[2]))
plt.legend()
plt.grid()
plt.xlabel('distance [m]')
plt.ylabel('SSHA [m')
plt.ylim(-mm,mm)

def sinfit(x,y,numharm=10,ddeg=2):
    #import ccgfilt
    # x,y = distA2[i20A2],ssha_a[i20A2]
    numpoly=2
    M = np.size(x)
    H = np.ones((M,numharm+numpoly))
    H[:,1] = x
    #H[:,2] = x**2
    iis,iic=2,2
    for ii in np.arange(numharm):
        if ii%2==1:
            H[:,ii+numpoly] = np.sin(iis*np.pi*x)
            iis+=ddeg
        if ii%2==0:
            H[:,ii+numpoly] = np.cos(iic*np.pi*x)
            iic+=ddeg
    ce = np.linalg.inv(H.T.dot(H)).dot(H.T.dot(y))
    fit = ce[0]+ce[1]*x #+ ce[1]*x**2 + ce[2]*x**3
    iis,iic=2,2
    for ii in np.arange(numharm):
        if ii%2==1:
            fit = fit+ce[ii+numpoly]*np.sin(iis*np.pi*x)
            iis+=ddeg
        if ii%2==0:
            fit = fit+ce[ii+numpoly]*np.cos(iic*np.pi*x)
            iic+=ddeg
    return ce,fit
    #fit = ccgfilt.fitFunc(ce, x, numpoly, numharm)


    
ce,fit = sinfit(distA2[i20A2],ssha_a[i20A2],numharm=300,ddeg=2)

ceB,fitB = sinfit(distB2[i20B2],ssha_b[i20B2],numharm=10)


'''