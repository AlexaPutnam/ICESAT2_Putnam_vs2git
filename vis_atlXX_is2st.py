#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-03-02

@author: alexaputnam
"""
import lib_read_TG as lTG
import lib_atlXX as lXX

import time
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import rasterio as rio
from rasterio.plot import show
#sys.path.append("/Users/alexaputnam/necessary_functions/")
#import plt_bilinear as pbil





def tide_scale(LOC):
    if LOC == 'skagway':
        #f1 = 'elfin_cove_9452634_july2023_tidesandcurrents_noaa.txt'
        f1 = 'port_alexander_9451054_aug2023_tidesandcurrents_noaa_1m.txt'
        f2 = 'juneau_9452210_aug2023_tidesandcurrents_noaa_1m.txt'
        f3 = 'skagway_9452400_aug2023_tidesandcurrents_noaa_1m.txt'
        Nm = ['Port Alexander','Juneau','Skagway']#'Elfin Cove',
        dist = [0,163,252]
        dyrf_min=2
    elif LOC == 'juneau':
        #f1 = 'elfin_cove_9452634_july2023_tidesandcurrents_noaa.txt'
        f1 = 'port_alexander_9451054_aug2023_tidesandcurrents_noaa_1m.txt'
        f2 = 'juneau_9452210_aug2023_tidesandcurrents_noaa_1m.txt'
        Nm = ['Port Alexander','Juneau']#'Elfin Cove',
        dist = [0,163]
        dyrf_min=2
    elif LOC == 'skagway2':
        f1 = 'elfin_cove_9452634_july2023_tidesandcurrents_noaa_1m.txt'
        f2 = 'juneau_9452210_aug2023_tidesandcurrents_noaa_1m.txt'
        f3 = 'skagway_9452400_aug2023_tidesandcurrents_noaa_1m.txt'
        Nm = ['Elfin Cove','Juneau','Skagway']#'Elfin Cove',
        dist = [0,163,252]
        dyrf_min=2
    elif LOC == 'juneau2':
        f1 = 'elfin_cove_9452634_july2023_tidesandcurrents_noaa_1m.txt'
        f2 = 'juneau_9452210_aug2023_tidesandcurrents_noaa_1m.txt'
        Nm = ['Elfin Cove','Juneau']#'Elfin Cove',
        dist = [0,163]
        dyrf_min=2
    elif LOC=='anchorage':
        f1 = 'seldovia_9455500_aug2023_tidesandcurrents_noaa.txt'
        f2 = 'nikiski_9455760_aug2023_tidesandcurrents_noaa.txt'
        f3 = 'anchorage_9455920_aug2023_tidesandcurrents_noaa.txt'
        Nm = ['Seldovia','Nikiski','Anchorage']
        dist = [0,106,166]
        pts = [[59.66309, -150.10212],[61.2968, -151.82455]]
        dyrf_min=60#15
    elif LOC == 'seattle':
        f1 = 'neah_bay_9443090_aug2023_tidesandcurrents_noaa.txt'
        f2 = 'port_townsend_9444900_aug2023_tidesandcurrents_noaa.txt'
        f3 = 'seattle_9447130_aug2023_tidesandcurrents_noaa.txt'
        Nm = ['Neah Bay','Port Townsend','Seattle']
        dist=[0,87,126]
        dyrf_min=15
    elif LOC == 'vancouver':
        f1 = 'neah_bay_9443090_aug2023_tidesandcurrents_noaa.txt'
        f2 = 'port_angeles_9444090_aug2023_tidesandcurrents_noaa.txt'
        f3 = 'vancouver_9440083_aug2023_tidesandcurrents_noaa.txt'
        Nm = ['Neah Bay','Port Angeles','Vancouver']
        dist=[0,55,125]
        dyrf_min=15
    
    latL,lonL,ldist,area,xbath,fitY,dXdY=path_up_fjord(LOC) 
    TG = ['seattle','port_town','skagway','juneau','skagway2','juneau2','anchorage','nikiski']
    delay_tg = np.asarray([240,180,18,14,18,14,300,135])
    amp_tg = np.asarray([0.76,0.24,1.06,0.9,1.06,0.91,2.21,0.55])
    slope_tg = np.asarray([-0.05,-0.9,1.7,2.58,1.1,-1.24,0.24,0.24])
    mlonD_tg = np.asarray([36,84,18,23,22,24,60,64])
    totD_tg = np.asarray([208,68,362,260,264,187,240,75])
    N = np.shape(Nm)[0]
    yrf1,ssh1 = lXX.tide_pull(f1)
    yrf2,ssh2 = lXX.tide_pull(f2)
    if N>=3:
        yrf3,ssh3 = lXX.tide_pull(f3)
    if N>=4:
        yrf4,ssh4 = lXX.tide_pull(f4)

    dyrf = dyrf_min/(60.*24.*365.25)
    t_bin = np.arange(np.nanmin(yrf1),np.nanmax(yrf1)+dyrf,dyrf)
    ssh1B = lXX.data2bin(t_bin,yrf1,ssh1)
    ssh1B=ssh1B-np.nanmean(ssh1B)
    ssh2B = lXX.data2bin(t_bin,yrf2,ssh2)
    ssh2B=ssh2B-np.nanmean(ssh2B)
    if N>=3:
        ssh3B = lXX.data2bin(t_bin,yrf3,ssh3)
        ssh3B=ssh3B-np.nanmean(ssh3B)
    if N>=4:
        ssh4B = lXX.data2bin(t_bin,yrf4,ssh4)

    dTphase12,Ap12 = lXX.find_phase_diff(t_bin,ssh1B,ssh2B)
    DT12 = np.round(dTphase12*(365.25*24.*60.),2)
    A12 = np.round(Ap12+1,2)
    if N>=3:
        dTphase13,Ap13 = lXX.find_phase_diff(t_bin,ssh1B,ssh3B)
        DT13 = np.round(dTphase13*(365.25*24.*60.),2)
        A13 = np.round(Ap13+1,2)
    if N==4:
        dTphase14,Ap14 = lXX.find_phase_diff(t_bin,ssh1B,ssh4B)
        DT14 = np.round(dTphase14*(365.25*24.*60.),2)
        A14 = np.round(Ap14,2)

    dxlim = 3/(365.25)

    binz = np.arange(-10,10.1,0.1)
    dssb2B = ssh2B[int(DT12/dyrf_min):]-ssh1B[int(DT12/dyrf_min):]
    dssb3B = ssh3B[int(DT13/dyrf_min):]-ssh1B[int(DT13/dyrf_min):]

    dssb2B_cor = ssh2B[int(DT12/dyrf_min):]-ssh1B[:-int(DT12/dyrf_min)]
    dssb3B_cor = ssh3B[int(DT13/dyrf_min):]-ssh1B[:-int(DT13/dyrf_min)]

    A_test = np.arange(1,3.1,0.1)
    Nt = np.size(A_test)
    Amp_est12 = np.empty(Nt)*np.nan
    Amp_est13 = np.empty(Nt)*np.nan
    for ii in np.arange(Nt):
        Amp_est12[ii] = np.nanstd(ssh2B[int(DT12/dyrf_min):]-(ssh1B[:-int(DT12/dyrf_min)]*A_test[ii]))
        Amp_est13[ii] = np.nanstd(ssh3B[int(DT13/dyrf_min):]-(ssh1B[:-int(DT13/dyrf_min)]*A_test[ii]))
    A_12_cor = np.round(A_test[Amp_est12==np.nanmin(Amp_est12)],1)
    A_13_cor = np.round(A_test[Amp_est13==np.nanmin(Amp_est13)],1)

    dssb2B_corA = ssh2B[int(DT12/dyrf_min):]-(ssh1B[:-int(DT12/dyrf_min)]*A_12_cor)
    dssb3B_corA = ssh3B[int(DT13/dyrf_min):]-(ssh1B[:-int(DT13/dyrf_min)]*A_13_cor)

    Sdssb2B = str(np.round(np.nanstd(dssb2B),2))
    Sdssb3B = str(np.round(np.nanstd(dssb3B),2))
    Sdssb2B_cor = str(np.round(np.nanstd(dssb2B_cor),2))
    Sdssb3B_cor = str(np.round(np.nanstd(dssb3B_cor),2))
    Sdssb2B_corA = str(np.round(np.nanstd(dssb2B_corA),2))
    Sdssb3B_corA = str(np.round(np.nanstd(dssb3B_corA),2))

    
    plt.figure(figsize=(10,10))
    plt.suptitle('Histogram of the difference between lower estuary ocean tide (measured, delayed, delayed & scaled) \n and tide measurements from a location within the estuary')#('Estimating ocean tide within an estuary using ocean tide corrections from the mouth (lower) of the estuary')
    plt.subplot(211)
    plt.title('Lower vs. Middle')
    plt.hist(dssb2B,bins=binz,label='middle - lower, $\sigma$ = '+Sdssb2B+' m')
    plt.hist(dssb2B_cor,bins=binz,label='middle - lower [$\Delta$], $\sigma$ = '+Sdssb2B_cor+' m')
    plt.hist(dssb2B_corA,bins=binz,label='middle - lower [$\Delta$ & A], $\sigma$ = '+Sdssb2B_corA+' m')
    plt.legend()
    plt.grid()
    plt.xlabel('Water level [m]')
    plt.ylabel('Occurrences')
    plt.subplot(212)
    plt.title('Lower vs. Upper')
    plt.hist(dssb3B,bins=binz,label='upper - lower, $\sigma$ = '+Sdssb3B+' m')
    plt.hist(dssb3B_cor,bins=binz,label='upper - lower [$\Delta$], $\sigma$ = '+Sdssb3B_cor+' m')
    plt.hist(dssb3B_corA,bins=binz,label='upper - lower [$\Delta$ & A], $\sigma$ = '+Sdssb3B_corA+' m')
    plt.legend()
    plt.grid()
    plt.xlabel('Water level [m]')
    plt.ylabel('Occurrences')

    plt.figure(figsize=(7,6))
    mk = '.-'
    plt.plot((t_bin-t_bin[0])*365.25*24,ssh1B,mk,label='lower estuary ('+Nm[0]+')',color='black')
    plt.plot((t_bin-t_bin[0])*365.25*24,ssh2B,mk,label='middle estuary ('+Nm[1]+')',color='tab:orange')
    if N>=3:
        plt.plot((t_bin-t_bin[0])*365.25*24,ssh3B,mk,label='upper estuary ('+Nm[2]+')',color='tab:green')
    if N>=4:
        plt.plot((t_bin-t_bin[0])*365.25*24,ssh4B,mk,label=Nm[3])
    plt.legend(bbox_to_anchor =(0.5,-0.32), loc='lower center')
    mn = np.nanmean(yrf1)
    plt.xlim(0,48)
    plt.grid()
    plt.xlabel('Hours since August 1st, 2023 00:00')
    plt.ylabel('Water level [m]')
    plt.title('15 min tide gauge height averages from \n NOAA\'s verified water level heights (6 min averages)')

    plt.figure(figsize=(7,6))
    plt.title('Apply time delay to shift tides from the lower estuary \n to match a point within the estuary')
    plt.plot((t_bin[int(DT12/dyrf_min):]-t_bin[int(DT12/dyrf_min):][0])*365.25*24,ssh2B[int(DT12/dyrf_min):],'.-',label='middle estuary',color='tab:orange')
    plt.plot((t_bin[:-int(DT12/dyrf_min)]-t_bin[:-int(DT12/dyrf_min)][0])*365.25*24,ssh1B[:-int(DT12/dyrf_min)],'--',label='lower estuary delayed to match middle estuary (dT = '+str(DT12)+' min)',color='saddlebrown')
    plt.xlim(0,48)
    if N>=3:
        plt.plot((t_bin[int(DT13/dyrf_min):]-t_bin[int(DT12/dyrf_min):][0])*365.25*24,ssh3B[int(DT13/dyrf_min):],'.-',label='upper estuary',color='tab:green')
        plt.plot((t_bin[int(DT13/dyrf_min):]-t_bin[int(DT12/dyrf_min):][0])*365.25*24,ssh1B[:-int(DT13/dyrf_min)],'--',label='lower estuary delayed to match upper estuary (dT = '+str(DT13)+' min)',color='darkgreen')
    if N>=4:
        plt.plot(t_bin,ssh4B,mk,label=Nm[3]+' dT = '+str(DT14)+' min, A='+str(A14)+' m')
    plt.legend(bbox_to_anchor =(0.5,-0.32), loc='lower center')
    plt.xlim(0,48)
    plt.grid()
    plt.xlabel('Hours since August 1st, 2023 00:00')
    plt.ylabel('Water level [m]')

    plt.figure(figsize=(7,6))
    plt.title('Apply time delay and scale tides from the lower estuary \n to match a point within the estuary')
    plt.plot((t_bin[int(DT12/dyrf_min):]-t_bin[int(DT12/dyrf_min):][0])*365.25*24,ssh2B[int(DT12/dyrf_min):],'.-',label='middle estuary',color='tab:orange')
    plt.plot((t_bin[:-int(DT12/dyrf_min)]-t_bin[:-int(DT12/dyrf_min)][0])*365.25*24,ssh1B[:-int(DT12/dyrf_min)]*A_12_cor,'--',label='lower estuary delayed and scaled to match middle estuary (dT = '+str(DT12)+' min, A='+str(A_12_cor[0])+')',color='saddlebrown')
    plt.xlim(0,48)
    if N>=3:
        plt.plot((t_bin[int(DT13/dyrf_min):]-t_bin[int(DT12/dyrf_min):][0])*365.25*24,ssh3B[int(DT13/dyrf_min):],'.-',label='upper estuary',color='tab:green')
        plt.plot((t_bin[int(DT13/dyrf_min):]-t_bin[int(DT12/dyrf_min):][0])*365.25*24,ssh1B[:-int(DT13/dyrf_min)]*A_13_cor,'--',label='lower estuary delayed and scaled to match upper estuary (dT = '+str(DT13)+' min, A='+str(A_13_cor[0])+')',color='darkgreen')
    if N>=4:
        plt.plot(t_bin,ssh4B,mk,label=Nm[3]+' dT = '+str(DT14)+' min, A='+str(A14)+' m')
    plt.legend(bbox_to_anchor =(0.5,-0.32), loc='lower center')
    plt.xlim(0,48)
    plt.grid()
    plt.xlabel('Hours since August 1st, 2023 00:00')
    plt.ylabel('Water level [m]')



def pull_info(NAME):
    # tg_id,fn,sonel_file = pull_info(NAME)
    fn = []
    sonel_file = []
    tg_id = 999999
    LLMM_tides=[]
    LAT_tide,LON_tide=[],[]
    UHSLC=[]
    LON_tg,LAT_tg=[],[]
    if NAME=='ALERT':
        tg_id = 1110
    elif NAME == 'COCOS':
        tg_id = 983
    elif NAME == 'NY-ALESUND':
        fn='reg_atl03_lat_79_lon_10_alesund_segs_2_2018_10_to_2023_06.npz'
        sonel_file = '/Users/alexaputnam/ICESat2/Rel_Grid/dNABG_10338M008_NGL14.neu'
        sonel_file2 = '/Users/alexaputnam/ICESat2/Rel_Grid/dNYA2_10317M008_NGL14.neu'
        tg_id = 1421
    elif NAME == 'NARVIK':
        tg_id = 312
    elif NAME == 'KABELVAG': # ocean outlet to NARVIK
        tg_id = 45
    elif NAME == 'TROMSO':
        tg_id = 680
    elif NAME == 'RORVIK':
        fn = '' 
        ds_uh = 'd803_uhslc_fd_rorvik.nc'
        tg_id = 1241
    elif NAME == 'HEIMSJOEN':
        tg_id = 313
    elif NAME == 'TRONDHEIM 2':
        tg_id = 1748
    elif NAME == 'OSLO':
        fn = 'reg_atl03_lat_60_lon_11_oslo_segs_2_2018_10_to_2023_06.npz'
        tg_id = 62
        LAT_tide,LON_tide = 59.15821, 10.65545
    elif NAME == 'OSCARSBORG':
        fn = 'reg_atl03_lat_60_lon_11_oslo_segs_2_2018_10_to_2023_06.npz'
        tg_id = 33
    elif NAME == 'UDDEVALLA': #0.35 m bias?
        fn = 'reg_atl03_lat_58_lon_12_uddevalla_segs_2_2018_10_to_2023_06.npz'
        tg_id = 2360
    elif NAME == 'STENUNGSUND': 
        tg_id = 2112
    elif NAME == 'STOCKHOLM': 
        fn = 'reg_atl03_lat_59_lon_18_stockholm_segs_2_2018_10_to_2023_06.npz'
        tg_id = 78
    elif NAME=='OULU / ULEABORG':
        tg_id = 79
    elif NAME == 'SKAGWAY': 
        fn = 'reg_atl03_lat_59_lon_n135_skagway2_segs_2_2018_10_to_2023_06.npz'#['reg_atl03_lat_59_lon_n135_skagway_segs_2_2018_10_to_2023_06.npz','reg_atl03_lat_59_lon_n135_skagway2_segs_2_2018_10_to_2023_06.npz','reg_atl03_lat_58_lon_n134_juneau_segs_2_2018_10_to_2023_06.npz']
        tg_id = 495
        LLMM_tides= [55.86734,59.86166,-138.77894,-131.75632] #[56.02069,56.0975,-134.59519,-134.35351]
        LAT_tide,LON_tide=58.13187,-134.97286
    elif NAME == 'JUNEAU': # Down stream from Skagway
        fn = 'reg_atl03_lat_58_lon_n134_juneau_segs_2_2018_10_to_2023_06.npz'#'reg_atl03_lat_58_lon_n135_juneau2_segs_2_2018_10_to_2023_06.npz'#'reg_atl03_lat_58_lon_n134_juneau_segs_2_2018_10_to_2023_06.npz'#'reg_atl03_lat_58_lon_n135_juneau_segs_2_2018_10_to_2023_06.npz'
        sonel_file = 'dJNU1_49519S001_NGL14.neu'
        tg_id = 405
        LAT_tide,LON_tide = 56.0975, -134.59519
        LAT_tg,LON_tg=58.26965, -134.36227
    elif NAME == 'PORT ALEXANDER': # ocean outlet to Juneau
        tg_id = 2299
    elif NAME == 'SITKA': # ocean outlet to Juneau
        tg_id = 426
    elif NAME == 'PUERTO SOBERANIA': # ocean outlet to Juneau
        tg_id = 1603
    elif NAME == 'KARASUK NORTH': # ocean outlet to Juneau
        tg_id = 999999
        fn='reg_atl03_lat_65_lon_n51_karasuk_north_segs_2_2018_10_to_2023_06.npz'
        LAT_tide,LON_tide=64.172, -51.72
        UHSLC= 'd820_uhslc_fd_nuuk.nc'
    elif NAME=='ANCHORAGE':
        tg_id = 1070
        fn = 'reg_atl03_lat_61_lon_n151_anchorage_segs_2_2018_10_to_2023_06.npz'
        LAT_tide,LON_tide=59.05913, -152.51886
        LAT_tg,LON_tg=61.19656,-150.18214
    elif NAME=='NIKISKI':
        tg_id = 1350
        fn = 'reg_atl03_lat_60_lon_n152_nikiski_segs_2_2018_10_to_2023_06.npz'
        LAT_tide,LON_tide=59.05913, -152.51886
        LAT_tg,LON_tg=60.72969, -151.57304
    elif NAME=='SEATTLE':
        tg_id = 127
        fn = 'reg_atl03_lat_48_lon_n123_seattle_segs_2_2018_10_to_2023_06.npz'
        LAT_tide,LON_tide=48.28068, -123.19239
        LAT_tg,LON_tg=47.60695, -122.44271
    elif NAME=='CAMP KANGIUSAQ':
        tg_id = 999999
        fn = 'reg_atl03_lat_65_lon_n51_camp_kangiusaq_segs_2_2018_10_to_2023_06.npz'
        LAT_tide,LON_tide=64.172, -51.72
        UHSLC= 'd820_uhslc_fd_nuuk.nc'
    elif NAME=='QEQQATA SERMERSOOQ':
        tg_id = 999999
        fn = 'reg_atl03_lat_65_lon_n51_qeqqata_sermersooq_fjord_segs_2_2018_10_to_2023_06.npz'
        LAT_tide,LON_tide=64.172, -51.72
    elif NAME == 'SEWARD': # ocean outlet to Juneau
        tg_id = 999999
        LAT_tide,LON_tide=59.86622,-149.45412
        UHSLC= 'h560_uhlslc_fd_seward.nc'
    elif NAME == 'KETCHIKAN': # ocean outlet to Juneau
        tg_id = 999999
        LAT_tide,LON_tide=54.78708,-131.39953
        UHSLC= 'h571_uhlslc_fd_ketchikan.nc'
    elif NAME == 'QAQORTOQ': # ocean outlet to Juneau
        tg_id = 999999
        LAT_tide,LON_tide=60.57956, -46.33445
        UHSLC= 'h299_uhslc_fd_qaqortoq.nc'
    return tg_id,fn,sonel_file,LLMM_tides,LON_tide,LAT_tide,LON_tg,LAT_tg,UHSLC


def path_up_fjord(LOC):
    # latL,lonL,ldist,area,xbath,fitY,dXdY=path_up_fjord(LOC)
    if LOC=='anchorage':
        lin_pts = [[59.4715, -152.52996],[60.72969, -151.57304],[61.19656, -150.18214]]
    elif LOC=='skagway':
        lin_pts = [[56.23156, -134.49638],[58.00802, -134.85185],[59.29455, -135.39574],[59.45, -135.32666]]
    elif LOC=='juneau':
        lin_pts = [[56.23156, -134.49638],[56.88579, -134.49887],[57.53271, -133.59796],[58.29833, -134.41166]]
    elif LOC=='skagway2':
        lin_pts = [[58.19499, -136.40159],[58.32479, -135.78202],[58.00802, -134.85185],[59.45, -135.32666]]
    elif LOC=='juneau2':
        lin_pts = [[58.19499, -136.40159],[58.32479, -135.78202],[58.00802, -134.85185],[58.43645, -135.01574],[58.29833, -134.41166]]
    elif LOC == 'seattle':
        lin_pts = [[48.42097, -124.60755],[48.18527, -122.78244],[47.87385, -122.44408],[47.60695, -122.44271]]
    elif LOC == 'vancouver':
        lin_pts = [[48.42097, -124.60755],[48.18527, -122.78244],[48.86333, -122.75666]]
    N = np.shape(lin_pts)[0]
    latL = []
    lonL = []
    xdist = []
    xbath = []
    ldist = []
    ident = []
    xdist_seg = np.empty(N-1)
    ldist_seg = np.empty(N-1)
    slope_seg = np.empty(N-1)
    DD=0
    for ii in np.arange(N-1):
        MB,lat_line,lon_line,Xdist,Xbath,Ldist = lXX.mean_depth_line(LOC,lin_pts[ii],lin_pts[ii+1])
        xdist_seg[ii] = np.nanmean(Xdist)
        ldist_seg[ii] = np.nanmean(Ldist+DD)
        TT,slope_seg[ii] = np.polyfit(Ldist[~np.isnan(Xbath)]+DD, Xbath[~np.isnan(Xbath)],1)[::-1]
        latL = np.hstack((latL,lat_line))
        lonL = np.hstack((lonL,lon_line))
        xdist = np.hstack((xdist,Xdist))
        xbath = np.hstack((xbath,Xbath))
        ldist = np.hstack((ldist,Ldist+DD))
        ident = np.hstack((ident,np.ones(np.size(Xbath))*ii))
        DD = ldist[-1]
    #ldist = np.abs(lXX.geo2dist(latL,lonL,latL[0],lonL[0])/1000.0)
    Ydist = np.empty(np.size(xdist))*np.nan
    Ydist[:-1] = np.diff(np.abs(lXX.geo2dist(latL,np.ones(np.size(latL))*lonL[0],latL[0],lonL[0])/1000.0))
    Ydist[-1] = Ydist[-2]
    area = Ydist*xdist #km^2
    yI,dXdY = np.polyfit(ldist[~np.isnan(xbath)], xbath[~np.isnan(xbath)],1)[::-1]
    fitY = yI+(dXdY*ldist)
    plt.figure()
    plt.title(LOC+': slope = '+str(np.round(dXdY,2))+' m/km')
    for ii in np.arange(N-1):
        plt.plot(ldist[ident==ii],xbath[ident==ii],'.',label='slope =  '+str(np.round(slope_seg[ii],3)))
    plt.plot(ldist,fitY,'-',color='red')
    plt.legend()

    plt.figure()
    plt.title(LOC+': mean longitudinal distance = '+str(np.round(np.nanmean(xdist),2))+' km \n total distance = '+str(np.round(np.nanmax(ldist),2))+' km')
    for ii in np.arange(N-1):
        plt.plot(ldist[ident==ii],xdist[ident==ii],'.',label='lon dist =  '+str(np.round(xdist_seg[ii],3))+', total dist =  '+str(np.round(ldist_seg[ii],3)))
    plt.legend()
    return latL,lonL,ldist,area,xbath,fitY,dXdY

def find_delay_crunch(h2,utc2,t2,LON_tide,LAT_tide,VR_MAX,N_MIN,min_min=0,max_min=60,dmin=5):
    '''
    max_min=60
    dmin=5
    '''
    # Test various delays
    aT_delay = np.arange(min_min,max_min+dmin,dmin)
    Ntd = np.size(aT_delay)
    mn_Tdelay = np.empty(Ntd)*np.nan
    cc_Tdelay = np.empty(Ntd)*np.nan
    for ii in np.arange(Ntd):
        sT = time.time()
        h2m,h2mti,tide_totaltt = correct_is2_ssh(h2,utc2,LON_tide,LAT_tide,t_delay_min=aT_delay[ii],A=1)
        h2mt = lXX.filter_is2_ssh(h2mti,t2,MODE=False,VR_MAX=VR_MAX,N_MIN=N_MIN)
        ikp = np.where(~np.isnan(h2mt))[0]
        if np.size(ikp)>0:
            #t_bins,mn_h2mt,vr_hrmt = lXX.t_mean(t2[~np.isnan(h2mt)],h2mt[~np.isnan(h2mt)],dt_days=dt_days,N_MIN=5)
            mn_Tdelay[ii] = np.nanmean(h2mt)
            cc_Tdelay[ii] = np.nanvar(h2mt)#np.corrcoef(h2t[~np.isnan(h2t)])[1,0]
            print('time for '+str(aT_delay[ii])+' min delay: '+str(np.round((time.time()-sT)/60.0,2))+' min')
        else:
            mn_Tdelay[ii] = 99999
            cc_Tdelay[ii] = 99999
    iTd = np.where(cc_Tdelay==np.nanmin(cc_Tdelay))[0]
    if np.size(iTd)!=0:
        if aT_delay[iTd][0]==99999:
            T_delay_hr=99999
        else:
            T_delay_hr = aT_delay[iTd][0]
    else:
        if aT_delay[iTd]==99999:
            T_delay_hr=99999
        else:
            T_delay_hr=aT_delay[iTd]
    #'KARASUK NORTH' tide delay = 10 min (16.666 hours
    return aT_delay,cc_Tdelay,mn_Tdelay,T_delay_hr

def run_delay_analysis(h2,utc2,t2,LON_tide,LAT_tide,min_min=0,max_min=300,dmin=60,VR_MAX=2,N_MIN=50):
    '''
    h2,utc2,t2,LON_tide,LAT_tide=ssh13[icrt],utc13[icrt],t13[icrt],LON_tide,LAT_tide
    min_min=0
    max_min=30
    dmin=5
    VR_MAX=2
    N_MIN=50
 
    t_bins,mn_h13,vr_h13 = lXX.t_mean(t13[~np.isnan(ssh13_new)],ssh13[~np.isnan(ssh13_new)],dt_days=dt_days,N_MIN=5)
    '''
    # Estimate delay
    aT_delay,cc_Tdelay,mn_Tdelay,T_delay_hr=find_delay_crunch(h2,utc2,t2,LON_tide,LAT_tide,VR_MAX,N_MIN,min_min=min_min,max_min=max_min,dmin=dmin)
    if np.sum(cc_Tdelay)!=99999.*np.size(cc_Tdelay):
        plt.figure()
        plt.plot(aT_delay,cc_Tdelay,'o-')
        plt.xlabel('delay [min]')
        plt.ylabel('SSH variance [m$^2$]')
        plt.title('Approximate time delay ($\delta$T) for '+NAME+' \n $\delta$T = '+str(aT_delay[cc_Tdelay==np.nanmin(cc_Tdelay)][0])+' min')
        plt.legend()
        plt.grid()

        scl = np.arange(1,3.02,0.02)
        Ns = np.size(scl)
        vr_scl = np.empty(Ns)*np.nan
        delay_sev = aT_delay[cc_Tdelay==np.nanmin(cc_Tdelay)][0]
        if np.size(delay_sev)>1:
            delay_sev = delay_sev[0]
        h2m,h2mtii,tide_total = correct_is2_ssh(h2,utc2,LON_tide,LAT_tide,t_delay_min=delay_sev,A=1)
        h2mti = lXX.filter_is2_ssh(h2mtii,t2,VR_MAX=VR_MAX,N_MIN=N_MIN)
        h2mi = np.copy(h2m)
        h2mi[np.isnan(h2mti)]=np.nan
        for ii in np.arange(Ns):
            h2mt = h2mi-(tide_total*scl[ii])
            #mn = np.nanmean(h2mt)
            #sd = np.nanstd(h2mt)
            #ikp = np.where((h2mt>=mn-2*sd)&(h2mt<=mn+2*sd))[0]
            vr_scl[ii] = np.nanvar(h2mt)
        amp_sev = scl[vr_scl==np.nanmin(vr_scl)][0]
        plt.figure()
        plt.plot(scl,vr_scl,'o-')
        plt.xlabel('scale (tide x A)')
        plt.ylabel('SSH variance [m$^2$]')
        plt.title('Approximate scale factor (A) for '+NAME+' \n A = '+str(np.round(scl[vr_scl==np.nanmin(vr_scl)][0],2)))
        plt.legend()
        plt.grid()
    else:
        delay_sev,amp_sev=np.nan,np.nan
    return delay_sev,amp_sev





        

def test_other_atlXX(dt_days):
    # ATL07 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    f07 = 'reg_atl07_lat_64_lon_n52_nuuk_2018_10_to_2023_06.npz'
    ds07 = np.load(f07,allow_pickle='TRUE')
    print(ds07.files)
    i07 = np.where((np.abs(ds07['h'])<200))[0]#&(ds07['lat']>=LLMM2[0])&(ds07['lat']<=LLMM2[1])&(ds07['lon']>=LLMM2[2])&(ds07['lon']<=LLMM2[3]))[0]
    t07=lXX.utc2yrfrac(ds07['time_utc'][i07])
    lat07=ds07['lat'][i07]
    lon07=ds07['lon'][i07]
    ssh07=ds07['h'][i07]
    geo07=ds07['geoid'][i07]
    mss07=ds07['mss'][i07]
    dynib07=ds07['dynib'][i07]
    dac07=ds07['dac'][i07]
    ft07=ds07['flag_type'][i07]
    t_bins,mn_h07,vr_h07 = lXX.t_mean(t07,ssh07,dt_days=dt_days,N_MIN=5)

    # ATL19 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ds19 = np.load('reg_atl19_lat_65_lon_n51_karasuk_north_2018_10_to_2023_06.npz',allow_pickle='TRUE')
    print(ds19.files)
    lat19=ds19['lat']
    lon19=ds19['lon']
    dot19=ds19['dot_dfw'][0,:,:]
    dot19[np.abs(dot19)>1000]=np.nan
    geo19=ds19['geoid_dfw'][0,:,:]
    geo19[np.abs(geo19)>1000]=np.nan
    ssh19i=dot19+geo19
    ssh19=ssh19i[~np.isnan(ssh19i)]
    t19= lXX.utc2yrfrac(ds19['time_utc_beg'])
    h19 = np.nanmean(ssh19i,axis=1)
    LLMM = [64.05,65.2,-52.9,-49]


def plot_image(lonIS,latIS,hIS,lon19,lat19,h19,lonA,latA,hA,vmin=27,vmax=30):
    # https://medium.com/@bertrandlobo/accessing-and-plotting-satellite-imagery-part-1-dcc81b467cdf
    from glob import glob
    import rasterio as rio
    from rasterio.plot import show
    import os
    pthEO = '/Users/alexaputnam/ICESat2/Fjord/EO_sentinel2/EO_Browser_images-3/'
    fnEO = 'nuuk_1_20210927_EO_s2_Highlight_Optimized_Natural_Color.tiff'
    tst = "/Users/alexaputnam/ICESat2/Fjord/EO_sentinel2/EO_Browser_images-3/2021-09-27-00/00_2021-09-27-23/59_Sentinel-2_L2A_B8A_(Raw).tiff"
    src = rio.open(pthEO+fnEO)
    fig, ax = plt.subplots(figsize=(18,10))
    show(src,ax=ax,title=NAME+' from ICESat-2 (2018/10 - 2023/06) plotted over Sentinel-2A image with ATL19 grid points (red outline)',cmap='viridis')
    c=ax.scatter(lonIS,latIS,c=hIS,s=1,cmap='viridis',vmin=vmin,vmax=vmax)
    ax.scatter(lon19,lat19,c=h19,s=50,cmap='viridis',vmin=vmin,vmax=vmax,marker='s')#,edgecolors='red')
    ax.scatter(lonA,latA,c=hA,s=1,cmap='viridis',vmin=vmin,vmax=vmax)
    fig.colorbar(c,ax=ax,location='bottom',label='sea surface height [m]',shrink=0.7)
    ax.set_xlabel('longitude [deg]')
    ax.set_ylabel('latitude [deg]')
    fig.tight_layout()
    plt.show()


def NAME_info(NAME):
    # fnEO,f12,f13,LLMM2,tide_delay,tide_scale,tide_delay_func = NAME_info(NAME)
    pthEO = '/Users/alexaputnam/ICESat2/Fjord/EO_sentinel2/'
    pthNOAA=[]
    tide_delay_func=[]
    if NAME=='ANCHORAGE':
        f12 = 'reg_atl12_lat_61_lon_n151_anchorage_2018_10_to_2023_06.npz'
        f13 = 'reg_atl13_lat_59_lon_n153_cook_inlet_2018_10_to_2023_06.npz'#'reg_atl13_lat_61_lon_n151_anchorage_2018_10_to_2023_06.npz'
        fnEO = 'anchorage_2023-09-04-23_59_Sentinel-2_L2A_True_color.tiff'#'anchorage_20221125_Sentinel-1_AWS-IW-VVVH_VH_-_decibel_gamma0_-_radiometric_terrain_corrected.tiff'#
        pthNOAA = '/Users/alexaputnam/ICESat2/Fjord/anchorage/'
        LLMM2 = [-90,90,-180,180]#[60.97673,61.23156,-150.77667,-149.8291]
        tide_delay,tide_scale = [],1.0#280,2.0
        tide_delay_func=np.asarray([-9660.97,   162.  ])
    elif NAME =='CAMP KANGIUSAQ':
        f12 = 'reg_atl12_lat_64_lon_n52_nuuk_2018_10_to_2023_06.npz'
        f13 = 'reg_atl13_lat_64_lon_n52_nuuk_2018_10_to_2023_06.npz'
        fnEO = 'EO_Browser_images-3/nuuk_1_20210927_EO_s2_Highlight_Optimized_Natural_Color.tiff'
        LLMM2 = [64.61945,64.79659,-50.94267,-50.12276]
        tide_delay,tide_scale = 12,1.2
    elif NAME =='KARASUK NORTH':
        f12 = 'reg_atl12_lat_64_lon_n52_nuuk_2018_10_to_2023_06.npz'
        f13 = 'reg_atl13_lat_64_lon_n52_nuuk_2018_10_to_2023_06.npz'
        fnEO = 'EO_Browser_images-3/nuuk_1_20210927_EO_s2_Highlight_Optimized_Natural_Color.tiff'
        LLMM2 = [64.55528,64.60598,-51.42739,-50.91996]
        tide_delay,tide_scale = 18,1.1
    elif NAME =='NUUK':
        f12 = 'reg_atl12_lat_64_lon_n52_nuuk_2018_10_to_2023_06.npz'
        f13 = 'reg_atl13_lat_64_lon_n52_nuuk_2018_10_to_2023_06.npz'
        fnEO = 'EO_Browser_images-3/nuuk_1_20210927_EO_s2_Highlight_Optimized_Natural_Color.tiff'
        LLMM2 = [-90,90,-180,180]#[64.55528,64.79659,-51.42739,-50.12276]
        tide_delay,tide_scale = 18,1.1
    elif NAME=='SEWARD':
        f12 = 'reg_atl12_lat_60_lon_n150_seward_2018_10_to_2023_06.npz'
        f13 = 'reg_atl13_lat_60_lon_n150_seward_2018_10_to_2023_06.npz'
        fnEO = 'seward_2022-11-25-23_59_Sentinel-2_L2A_True_color.tiff'
        LLMM2 = [-90,90,-180,180]#[59.82005,60.14654,-149.6768,-148.97937]
        tide_delay,tide_scale = 285,1.0
    elif NAME=='KETCHIKAN':
        f12 = 'reg_atl12_lat_54_lon_n133_ketchikan_2018_10_to_2023_06.npz'
        f13 = 'reg_atl13_lat_54_lon_n133_ketchikan_2018_10_to_2023_06.npz'
        fnEO='ketchikan_2022-12-21-23_59_Sentinel-2_L2A_True_color.tiff'
        dl=0.5
        LLMM2 = [55.30523-dl,55.35269+dl,-131.70415-dl,-131.54587+dl]
        tide_delay,tide_scale = 14,1
    elif NAME=='QAQORTOQ':
        f12 = 'reg_atl12_lat_61_lon_n46_qaqortoq_2018_10_to_2023_06.npz'
        f13 = 'reg_atl13_lat_61_lon_n46_qaqortoq_2018_10_to_2023_06.npz'
        fnEO='qaqortoq_2022-11-12-23_59_Sentinel-2_L2A_True_color.tiff'
        dl=0.5
        LLMM2 = [-90,90,-180,180]#[55.30523-dl,55.35269+dl,-131.70415-dl,-131.54587+dl]
        tide_delay,tide_scale = 0,1
    elif NAME=='VANCOUVER ISLAND':
        f12 = 'reg_atl12_lat_47_lon_n127_vancouver_island_2018_10_to_2023_06.npz'
        f13 = 'reg_atl13_lat_47_lon_n127_vancouver_island_2018_10_to_2023_06.npz'
        fnEO='vancouver_2023-09-20-23_59_Sentinel-2_L2A_B8A_(Raw).tiff'
        dl=0.5
        LLMM2 = [-90,90,-180,180]#[55.30523-dl,55.35269+dl,-131.70415-dl,-131.54587+dl]
        tide_delay,tide_scale = 0,1
    else:
        f12 = 'reg_atl12_lat_61_lon_n46_qaqortoq_2018_10_to_2023_06.npz'
        f13 = 'reg_atl13_lat_61_lon_n46_qaqortoq_2018_10_to_2023_06.npz'
        fnEO = 'Sentinel-2_L2A_True_color_2019_08_24.tiff'
        LLMM2 = [60.97673,61.23156,-150.77667,-149.8291]
        tide_delay,tide_scale = 280,2.0
    return pthEO+fnEO,f12,f13,LLMM2,tide_delay,tide_scale,pthNOAA,tide_delay_func

def ana_atl12_data(NAME,VR_MAX,N_MIN,LAT_tide,LON_tide,dt_days=1):
    #VR_MAX,N_MIN,dt_days=5,1,1 
    # ATL12 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fnEO,f12,f13,LLMM2,tide_delay,tide_scale,pthNOAA,tide_delay_func = NAME_info(NAME)
    # Pull data
    ds12 = np.load(f12,allow_pickle='TRUE')
    print(ds12.files)
    h_abs=np.abs(ds12['ssh'])
    i12 = np.where((h_abs<200)&(ds12['lat']>=LLMM2[0])&(ds12['lat']<=LLMM2[1])&(ds12['lon']>=LLMM2[2])&(ds12['lon']<=LLMM2[3]))[0]
    utc12 = ds12['time_utc'][i12]
    t12=lXX.utc2yrfrac(utc12)
    lat12=ds12['lat'][i12]
    lon12=ds12['lon'][i12]
    ssh12=ds12['ssh'][i12]
    dac12=ds12['dac'][i12]
    geo12=ds12['geoid'][i12]
    geoF2M12=ds12['geoid_free2mean'][i12]
    geoF2M12[np.abs(geoF2M12)>100]=0
    earF2M12=ds12['tide_earth_free2mean'][i12]
    earF2M12[np.abs(earF2M12)>100]=0
    tE12=ds12['tide_equilibrium'][i12]
    tE12[np.abs(tE12)>100]=0
    tL12=ds12['tide_load'][i12]
    tL12[np.abs(tL12)>100]=0
    tO12=ds12['tide_ocean'][i12]+tL12+tE12
    tO12[np.abs(tO12)>100]=0
    ssh12 = ssh12+tO12+geoF2M12+earF2M12+dac12
    # Filter and tide
    ssh012,sshM12,sshDS12,sshT12,tideM12,tideDS12,tideT12,mss12_c15,mss12_d21,t_frac_moments12 = tide_atlXX(utc12,t12,lat12,lon12,ssh12,dt_days,tide_delay,tide_scale,LAT_tide,LON_tide,VR_MAX,N_MIN,'ATL12 '+NAME,tide_delay_func=tide_delay_func)
    return utc12,t12,lat12,lon12,ssh12,ssh012,sshM12,sshDS12,sshT12,tideM12,tideDS12,tideT12,mss12_c15,mss12_d21,t_frac_moments12


def ana_atl13_data(NAME,VR_MAX,N_MIN,LAT_tide,LON_tide,dt_days=1):
    #VR_MAX,N_MIN,dt_days=2,50,1
    # ATL13 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fnEO,f12,f13,LLMM2,tide_delay,tide_scale,pthNOAA,tide_delay_func = NAME_info(NAME)
    # Pull data
    ds13 = np.load(f13,allow_pickle='TRUE')
    print(ds13.files)
    #&(ds13['flag_pod']==0), (qf_cloud==0)& (qf_water==3)
    #water_type = ds13['water_type'] # waterbody type: 1=Lake, 2=Known Reservoir, 3=(Reserved for future use),4=Ephemeral Water, 5=River, 6=Estuary or Bay,7=Coastal Water
    #qf_frac = ds13['flag_frac']# 20% min --> quality flags aren't good in these regions
    qf_pod = ds13['flag_pod']# pod ppd flag: 0 =nominal
    #qf_cloud = ds13['flag_cloud'] #0 =nominal
    #qf_water = ds13['flag_water']# ATL09 flag: 0=ice free water, 1=snow and ice free land, 2=snow, 3=ice
    h_abs = np.abs(ds13['ht_water_surf'])
    i13 = np.where((qf_pod==0)&(h_abs<200)&(ds13['lat']>=LLMM2[0])&(ds13['lat']<=LLMM2[1])&(ds13['lon']>=LLMM2[2])&(ds13['lon']<=LLMM2[3]))[0]#(qf_pod==0)&(h_abs<200))[0]#np.where((np.abs(ds13['ht_water_surf'])<200))[0]
    utc13 = ds13['time_utc']
    t13=lXX.utc2yrfrac(utc13)[i13]
    lat13=ds13['lat'][i13]
    lon13=ds13['lon'][i13]
    #dac13=ds13['dac'][i13]
    #dac13[np.abs(dac13)>1000]=np.nan
    #tL13=ds13['tide_load'][i13]
    #tL13[np.abs(tL13)>100]=0
    ssh13=ds13['ht_water_surf'][i13]#-dac13#+tL13
    geo13=ds13['geoid'][i13]
    dem13=ds13['dem'][i13]
    tO13=ds13['tide_ocean'][i13]#+tL13
    tO13[np.abs(tO13)>100]=0    
    limLL = []
    if NAME=='ANCHORAGE':
        OTf14 = ds13['tide_ocean_f14'][i13]+ds13['tide_load_f14'][i13]
    elif NAME=='VANCOUVER ISLAND':
        OTf14 = np.zeros(np.size(i13))
    elif NAME=='NUUK':
        OTf14 = np.zeros(np.size(i13))
    # Filter and tide
    ssh013,sshM13,sshDS13,sshT13,tideM13,tideDS13,tideT13,mss13_c15,mss13_d21,t_frac_moments13 = tide_atlXX(utc13,t13,lat13,lon13,ssh13,OTf14,dt_days,tide_delay,tide_scale,LAT_tide,LON_tide,VR_MAX,N_MIN,'ATL13 '+NAME,tide_delay_func=tide_delay_func)
    ssha_Gi = (ssh13-tO13)-mss13_d21
    ssha_Gi[np.abs(ssha_Gi)>2]=np.nan
    ssha_G = lXX.filter_is2_ssh(ssha_Gi,t13,MODE=False,VR_MAX=VR_MAX,N_MIN=N_MIN)
    sshG13 = ssha_G+mss13_d21
    return utc13,t13,lat13,lon13,ssh13,ssh013,sshM13,sshDS13,sshT13,sshG13,tideM13,tideDS13,tideT13,tO13,mss13_c15,mss13_d21,t_frac_moments13


def misclassification(NAME,VR_MAX,N_MIN,LAT_tide,LON_tide,dt_days=1):
    #VR_MAX,N_MIN,dt_days=2,50,1
    # ATL13 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fnEO,f12,f13,LLMM2,tide_delay,tide_scale,pthNOAA,tide_delay_func = NAME_info(NAME)
    # Pull data
    ds13 = np.load(f13,allow_pickle='TRUE')
    print(ds13.files)
    #&(ds13['flag_pod']==0), (qf_cloud==0)& (qf_water==3)
    water_type = ds13['water_type'] # waterbody type: 1=Lake, 2=Known Reservoir, 3=(Reserved for future use),4=Ephemeral Water, 5=River, 6=Estuary or Bay,7=Coastal Water
    qf_frac = ds13['flag_frac']# 20% min --> quality flags aren't good in these regions
    qf_pod = ds13['flag_pod']# pod ppd flag: 0 =nominal
    qf_cloud = ds13['flag_cloud'] #0 =nominal
    qf_water = ds13['flag_water']# ATL09 flag: 0=ice free water, 1=snow and ice free land, 2=snow, 3=ice
    h_abs = np.abs(ds13['ht_water_surf'])

    CLASS = 'Ice free water'
    if CLASS=='Lakes':
        cls = 1
    elif CLASS=='Known Reservoir':
        cls = 2
    elif CLASS=='Reserved':
        cls = 3
    elif CLASS=='Ephemeral Water':
        cls = 4
    elif CLASS=='River':
        cls = 5
    elif CLASS=='Estuary or Bay':
        cls = 6
    elif CLASS=='Coastal Water':
        cls = 7
    elif CLASS=='Ice free water':
        cls = 0
    elif CLASS=='Snow and ice free land':
        cls = 1
    elif CLASS=='Snow':
        cls = 2
    elif CLASS=='Ice':
        cls = 3
    i13_ice = np.where((qf_pod==0)&(qf_water==3)&(ds13['lat']>=LLMM2[0])&(ds13['lat']<=LLMM2[1])&(ds13['lon']>=LLMM2[2])&(ds13['lon']<=LLMM2[3]))[0]
    i13_snw = np.where((qf_pod==0)&(qf_water==2)&(ds13['lat']>=LLMM2[0])&(ds13['lat']<=LLMM2[1])&(ds13['lon']>=LLMM2[2])&(ds13['lon']<=LLMM2[3]))[0]
    i13_lnd = np.where((qf_pod==0)&(qf_water==1)&(ds13['lat']>=LLMM2[0])&(ds13['lat']<=LLMM2[1])&(ds13['lon']>=LLMM2[2])&(ds13['lon']<=LLMM2[3]))[0]
    i13_wtr = np.where((qf_pod==0)&(qf_water==0)&(ds13['lat']>=LLMM2[0])&(ds13['lat']<=LLMM2[1])&(ds13['lon']>=LLMM2[2])&(ds13['lon']<=LLMM2[3]))[0]
    
    i13 = np.where((qf_pod==0)&(qf_water==cls)&(ds13['lat']>=LLMM2[0])&(ds13['lat']<=LLMM2[1])&(ds13['lon']>=LLMM2[2])&(ds13['lon']<=LLMM2[3]))[0]#np.where((qf_pod==0)&(h_abs<200)&(ds13['lat']>=LLMM2[0])&(ds13['lat']<=LLMM2[1])&(ds13['lon']>=LLMM2[2])&(ds13['lon']<=LLMM2[3]))[0]#(qf_pod==0)&(h_abs<200))[0]#np.where((np.abs(ds13['ht_water_surf'])<200))[0]
    utc13 = ds13['time_utc']
    lat13=ds13['lat'][i13]
    lon13=ds13['lon'][i13]
    tO13=ds13['tide_ocean'][i13]#+tL13
    tO13[np.abs(tO13)>100]=0    
    limLL = []
    if NAME=='ANCHORAGE':
        TIT='Cook Inlet, Alaska'
        fnEO='anchorage_2023-09-04-23_59_Sentinel-2_L2A_True_color.tiff'
    elif NAME=='VANCOUVER ISLAND':
        fnEO='vancouver_2023-09-15-23_59_Sentinel-2_L2A_True_color.tiff'#'vancouver_2023-09-20-23_59_Sentinel-2_L2A_B8A_(Raw).tiff'
        limLL=[47.5,50,-126.5,-123.4]
    elif NAME=='NUUK':
        TIT='Nuup Kangerlua Fjord, Greenland'
        fnEO = 'Sentinel-2_L2A_True_color_2019_08_24.tiff'

    pthEO = '/Users/alexaputnam/ICESat2/Fjord/EO_sentinel2/'
    src2 = rio.open(pthEO+fnEO)
    fig, ax = plt.subplots(figsize=(10,10))
    fs=16
    #plt.title('Nuup Kangerlua Fjord, Greenland',fontsize=fs)
    #plt.title('Vancouver Island, Canada \n Classification: '+CLASS,fontsize=fs)
    plt.title(TIT+' \n Water flag: '+CLASS,fontsize=fs)
    show(src2,ax=ax,cmap='gray')
    ax.scatter(lon13,lat13,s=5,c='lime',label='ICESat-2 ATL13')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    if np.size(limLL)!=0:
        plt.ylim(limLL[0],limLL[1])
        plt.xlim(limLL[2],limLL[3])
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(10,10))
    fs=16
    #plt.title('Nuup Kangerlua Fjord, Greenland',fontsize=fs)
    #plt.title('Vancouver Island, Canada \n Classification: '+CLASS,fontsize=fs)
    plt.title(TIT+' \n Water flag: '+CLASS,fontsize=fs)
    show(src2,ax=ax,cmap='gray')
    ax.scatter(ds13['lon'][i13_wtr],ds13['lat'][i13_wtr],s=5,c='blue',label='Ice free water')
    ax.scatter(ds13['lon'][i13_lnd],ds13['lat'][i13_lnd],s=5,c='red',label='Snow and ice free land',alpha=0.2)
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    if np.size(limLL)!=0:
        plt.ylim(limLL[0],limLL[1])
        plt.xlim(limLL[2],limLL[3])
    plt.legend()
    plt.show()

    LIM=[0,200] #[20,50] #180
    plt.figure(figsize=(10,10))
    plt.suptitle(TIT+' \n Water flag',fontsize=fs)
    plt.subplot(221)
    plt.title('Ice')
    plt.plot(lXX.utc2yrfrac(utc13[i13_ice]),ds13['ht_water_surf'][i13_ice],'.',label='Ice')
    plt.grid()
    plt.xlabel('Year fraction')
    plt.ylabel('Sea surface height [m]')
    plt.ylim(LIM[0],LIM[1])
    plt.subplot(222)
    plt.title('Snow')
    plt.plot(lXX.utc2yrfrac(utc13[i13_snw]),ds13['ht_water_surf'][i13_snw],'.',label='Snow')
    plt.grid()
    plt.xlabel('Year fraction')
    plt.ylabel('Sea surface height [m]')
    plt.ylim(LIM[0],LIM[1])
    plt.subplot(223)
    plt.title('Snow and ice free land')
    plt.plot(lXX.utc2yrfrac(utc13[i13_lnd]),ds13['ht_water_surf'][i13_lnd],'.',label='Snow and ice free land')
    plt.grid()
    plt.xlabel('Year fraction')
    plt.ylabel('Sea surface height [m]')
    plt.ylim(LIM[0],LIM[1])
    plt.subplot(224)
    plt.title('Ice free water')
    plt.plot(lXX.utc2yrfrac(utc13[i13_wtr]),ds13['ht_water_surf'][i13_wtr],'.',label='Ice free water')
    plt.grid()
    plt.ylim(LIM[0],LIM[1])
    plt.xlabel('Year fraction')
    plt.ylabel('Sea surface height [m]')

def tide_atlXX(utc,tm,lat,lon,ssh,OTf14,dt_days,tide_delay,tide_scale,LAT_tide,LON_tide,VR_MAX,N_MIN,TIT,tide_delay_func=[]):
    #utc,tm,lat,lon,ssh,dt_days,tide_delay,tide_scale,VR_MAX,N_MIN,TIT=utc13,t13,lat13,lon13,ssh13,dt_days,tide_delay,tide_scale,VR_MAX,N_MIN,'ATL13 '+NAME
    # utc,tm,lat,lon,ssh,dt_days,tide_delay,tide_scale = utc12,t12,lat12,lon12,ssh12,dt_days,tide_delay,tide_scale
    # ssh012,sshM12,sshDS12,sshT12,tideM12,tideDS12,tideT12,mss12_c15,mss12_d21 = tide_atlXX(utc12,t12,lat12,lon12,ssh12,dt_days,tide_delay,tide_scale)
    # Estimate tides
    dist13 = np.abs(lXX.geo2dist(lat,lon,LAT_tide,LON_tide))/1000.
    inn = np.where(~np.isnan(ssh))[0]
    # Find MSS from CNES/CLS 2015 and DTU 2021
    mss_c15 = np.empty(np.size(lat))*np.nan
    mss_d21 = np.empty(np.size(lat))*np.nan
    mss_c15[inn] = lXX.mss_model(lat[inn],lon[inn],MODEL='cnes15')
    mss_d21[inn] = lXX.mss_model(lat[inn],lon[inn],MODEL='dtu21')
    ### Ocean tide at mouth
    tideM = np.empty(np.size(ssh))*np.nan
    tideM[inn] = lXX.tide_estimation(utc[inn],ssh[inn],LON_tide,LAT_tide,t_delay_min=0)
    ### Ocean tide delayed and scaled
    tideDS = np.empty(np.size(ssh))*np.nan
    if np.size(tide_delay_func)==0:
        tideDS[inn],T_moments,tide_moments,h_moments = lXX.tide_estimation(utc[inn],ssh[inn],LON_tide,LAT_tide,t_delay_min=tide_delay,Return_moments=True,A=tide_scale)
    else:
        PP=0
        T_moments=[]
        arr_dT = np.round(tide_delay_func[0]+(tide_delay_func[1]*lat[inn]))
        if 'ANCHORAGE' in TIT:
            rarr_dT = np.asarray([0,  42.55, 117.55, 192.55, 267.55])#np.asarray([0.0,39.0,114.0,189.0,246.0])#np.asarray([0.0,39.0,120.0,201.0,282.0])#fit_coeff= array([-8942.45,   150.  ])np.asarray([  0, 0,  60, 120, 180, 270, 330])#n
            rarr_A = np.asarray([1.15,1.25,1.35,1.5,1.6])#np.asarray([1.1,1.1,1.5,1.6])
            lat_crtG = np.asarray([59.4,59.9,60.4,60.9,61.4])# fit_coeff = np.asarray([-12.64,   0.23])
            dist_crtG = np.asarray([  0,  50, 100, 150, 200, 250, 300])
        elif 'NUUK' in TIT:
                dcG = 0.3
                lat_crtG = np.arange(np.nanmin(lat[~np.isnan(ssh)]),np.nanmax(lat[~np.isnan(ssh)])+dcG,dcG)
                lon_crtG = np.arange(np.nanmin(lon[~np.isnan(ssh)]),np.nanmax(lon[~np.isnan(ssh)])+dcG,dcG)
                mlat_crtG,mlon_crtG = np.meshgrid(lat_crtG,lon_crtG)
                flat_crtG,flon_crtG = mlat_crtG.flatten('F'),mlon_crtG.flatten('F')
                NcrtG = np.size(flat_crtG)

        else:
            rarr_dT = np.asarray([0.0,39.0,120.0,201.0,282.0,300.0])#np.arange(0,330,10)#np.unique(arr_dT)
            rarr_A = np.ones(np.size(rarr_dT))
            lat_crtG = np.asarray([59.4,59.9,60.4,60.9,61.4,70])
        dlat = np.abs(np.diff(lat_crtG)[0])
        dl = np.abs(np.diff(dist_crtG)[0])
        for ii in np.arange(np.size(rarr_dT)):
            if 'ANCHORAGE' in TIT:
                idt = np.where((lat>=lat_crtG[ii]-dlat/2)&(lat<lat_crtG[ii]+dlat/2))[0]#np.where((arr_dT>=rarr_dT[ii]-dlat/2)&(arr_dT<rarr_dT[ii]+dlat/2))[0]np.where((dist13>=dist_crtG[ii]-(dl/2.0))&(dist13<dist_crtG[ii]+(dl/2.0)))[0]#n
            else:
                idt = np.where((lat>=flat_crtG[ii]-dcG/2.0)&(lat<flat_crtG[ii]+dcG/2.0)&(lon>=flon_crtG[ii]-dcG/2.0)&(lon<flon_crtG[ii]+dcG/2.0))[0]
            tDS,T_momentsi,tide_moments,h_moments = lXX.tide_estimation(utc[inn][idt],ssh[inn][idt],LON_tide,LAT_tide,t_delay_min=rarr_dT[ii],Return_moments=True,A=rarr_A[ii])
            T_moments=np.unique(np.hstack((T_moments,T_momentsi)))
            tideDS[inn[idt]] = np.copy(tDS)
            PP+=1
            if PP%5==0:
                print(np.str(np.round(PP/(np.size(rarr_dT)*1.0),3)*100)+' perc done')
            #tideDS[inn],T_moments,tide_moments,h_moments = lXX.tide_estimation(utc[inn],ssh[inn],LON_tide,LAT_tide,t_delay_min=tide_delay,Return_moments=True,A=tide_scale)
    t_frac_moments =lXX.utc2yrfrac(T_moments)
    #yr13,mnt13,dy13,hr13,mi13,date13 = lXX.convert_partial_year(t_frac_moments)
    ### Ocean tide within fjord (if available)
    if 'ANCHORAGE' in TIT:
        #import ana_fes2014_tide_reg_pyTMD as atide 
        #time_SF = atide.utc2utc_stamp(utc[inn])
        #tide_ocean = atide.ocean_tide_replacement(lon[inn],lat[inn],time_SF,LOAD=False,method='spline')
        #tide_load = atide.ocean_tide_replacement(lon[inn],lat[inn],time_SF,LOAD=True,method='spline')
        tideT = OTf14
        #tideT[inn] = (tide_ocean+tide_load).squeeze()#lXX.tide_estimation(utc[inn],ssh[inn],lat[inn],lon[inn],LON_tg,LAT_tg,t_delay_min=0)
        ## delayed and scaled tide
        ssha_Ti = (ssh)-(tideT)-mss_d21
        ssha_Ti[np.abs(ssha_Ti)>2]=np.nan
        ssha_T = lXX.filter_is2_ssh(ssha_Ti,tm,MODE=False,VR_MAX=VR_MAX,N_MIN=N_MIN)
        sshT = ssha_T+mss_d21
        '''
        # Temporal means
        t_bins,mn_tideM,vr_tideM = lXX.t_mean(tm,tideM,dt_days=dt_days,N_MIN=5)
        t_bins,mn_tideDS,vr_tideDS = lXX.t_mean(tm,tideDS,dt_days=dt_days,N_MIN=5)
        t_bins,mn_tideT_13,vr_tideT_13 = lXX.t_mean(tm,tideT,dt_days=dt_days,N_MIN=5)

        plt.figure()
        plt.title(TIT)
        plt.plot(t_bins[~np.isnan(mn_tideM)],mn_tideM[~np.isnan(mn_tideM)],label='Geocentric ocean tide at fjord mouth')
        plt.plot(t_bins[~np.isnan(mn_tideDS)],mn_tideDS[~np.isnan(mn_tideDS)],label='Geocentric ocean tide delayed and scaled')
        plt.plot(t_bins[~np.isnan(mn_tideT_13)],mn_tideT_13[~np.isnan(mn_tideT_13)],label='Geocentric ocean tide within fjord')
        plt.legend()
        '''
    else:
        sshT,tideT = np.empty(np.size(tm))*np.nan,np.empty(np.size(tm))*np.nan
    
    # Filter and correct SSH
    ## no tide correction
    ssha_0i = (ssh)-mss_d21
    ssha_0i[np.abs(ssha_0i)>2]=np.nan
    ssha_0 = lXX.filter_is2_ssh(ssha_0i,tm,MODE=False,VR_MAX=VR_MAX,N_MIN=N_MIN)
    ssh0 = ssha_0+mss_d21
    ## tide at mouth
    ssha_Mi = (ssh)-(tideM)-mss_d21
    ssha_Mi[np.abs(ssha_Mi)>2]=np.nan
    ssha_M = lXX.filter_is2_ssh(ssha_Mi,tm,MODE=False,VR_MAX=VR_MAX,N_MIN=N_MIN)
    sshM = ssha_M+mss_d21
    ## delayed and scaled tide
    ssha_DSi = (ssh)-(tideDS)-mss_d21
    ssha_DSi[np.abs(ssha_DSi)>2]=np.nan
    ssha_DS = lXX.filter_is2_ssh(ssha_DSi,tm,MODE=False,VR_MAX=VR_MAX,N_MIN=N_MIN)
    sshDS = ssha_DS+mss_d21
    return ssh0,sshM,sshDS,sshT,tideM,tideDS,tideT,mss_c15,mss_d21,t_frac_moments

def ana_atlXX_nuuk():
    NAME = 'NUUK'
    LAT_tide,LON_tide=64.172, -51.72
    fnEO,f12,f13,LLMM2,tide_delay,tide_scale,pthNOAA,tide_delay_func = NAME_info(NAME)
    src = rio.open(fnEO)
    utc13,t13,lat13,lon13,ssh13,ssh013,sshM13,sshDS13,sshT13,sshG13,tideM13,tideDS13,tideT13,tideG13,mss13_c15,mss13_d21,t_frac_moments13 = ana_atl13_data(NAME,2,50,LAT_tide,LON_tide,dt_days=1)
    yr13,mnt13,dy13,hr13,mi13,date13 = lXX.convert_partial_year(t_frac_moments13)
    Nd = np.shape(date13)[0]
    file_list = []
    inan = np.where((lon13<-52))[0]
    ssh13[inan]=np.nan
    sshDS13[inan]=np.nan

    # Create new MSS models
    inn13 = np.where(~np.isnan(sshDS13))[0]
    lat13_d21,lon13_d21,mss_d21_model = lXX.mss_model(lat13[inn13],lon13[inn13],MODEL='dtu21',RETURN_COORD=True,mss_in=[],TP2WGS=True,IS2DATE=False)
    lat13_d21,lon13_d21,mss_d21_ds13,mss_var_ds13,N_d21_ds13 = lXX.mss_model(lat13[inn13],lon13[inn13],MODEL='dtu21',RETURN_COORD=True,mss_in=sshDS13[inn13],TP2WGS=True,IS2DATE=False,tm=t13[inn13])
    lat13_d21,lon13_d21,mss_d21_013,mss_var_013,N_d21_013 = lXX.mss_model(lat13[inn13],lon13[inn13],MODEL='dtu21',RETURN_COORD=True,mss_in=ssh13[inn13],TP2WGS=True,IS2DATE=False,tm=t13[inn13])
    lat13_d21,lon13_d21,mss_d21_M13,mss_var_M13,N_d21_M13 = lXX.mss_model(lat13[inn13],lon13[inn13],MODEL='dtu21',RETURN_COORD=True,mss_in=sshM13[inn13],TP2WGS=True,IS2DATE=False,tm=t13[inn13])
    lat13_d21,lon13_d21,mss_d21_T13,mss_var_T13,N_d21_T13 = lXX.mss_model(lat13[inn13],lon13[inn13],MODEL='dtu21',RETURN_COORD=True,mss_in=sshT13[inn13],TP2WGS=True,IS2DATE=False,tm=t13[inn13])

    pthEO = '/Users/alexaputnam/ICESat2/Fjord/EO_sentinel2/'
    fnEO = 'Sentinel-2_L2A_True_color_2019_08_24.tiff'
    src2 = rio.open(pthEO+fnEO)

    fnEO3 = 'EO_Browser_images-5/2023-09-17-23_59_Sentinel-2_L2A_False_color.tiff'#'EO_Browser_images-3/nuuk_1_20210927_EO_s2_Highlight_Optimized_Natural_Color.tiff'#'Sentinel-2_L2A_True_color_2019_08_24.tiff'
    src3 = rio.open(pthEO+fnEO3)


    tJ,latJ,lonJ,hJ=lXX.radar_alt('j2',LLMM=[-90,90,-180,180])
    tC,latC,lonC,hC=lXX.radar_alt('c2',LLMM=[-90,90,-180,180])
    tS,latS,lonS,hS=lXX.radar_alt('3a',LLMM=[-90,90,-180,180])
    iJ = np.where((latJ>=np.nanmin(lat13))&(latJ<=np.nanmax(lat13))&(lonJ>=-53.4))[0]#np.where((latJ>=np.nanmin(lat13))&(latJ<=np.nanmax(lat13))&(lonJ>=np.nanmin(lon13))&(lonJ<=np.nanmax(lon13)))[0]
    iC = np.where((latC>=np.nanmin(lat13))&(latC<=np.nanmax(lat13))&(lonC>=-53.4))[0]
    iS = np.where((latS>=np.nanmin(lat13))&(latS<=np.nanmax(lat13))&(lonS>=-53.4))[0]
    lon_coast,lat_coast = lXX.coastline()
    iCL = np.where((latJ>=np.nanmin(lat13))&(latJ<=np.nanmax(lat13))&(lonJ>=np.nanmin(lonJ))&(lonJ<=np.nanmax(lon13)))[0]
    
    fig, ax = plt.subplots(figsize=(10,10))
    fs=16
    TIT='Radar altimetry vs. ICESat-2 groundtrack'
    #plt.suptitle('Nuup Kangerlua Fjord, Greenland',fontsize=20)
    plt.title(TIT,fontsize=fs)
    show(src3,ax=ax)
    #plt.plot(lon_coast[iCL],lat_coast[iCL],color='black',linewidth=4)
    ax.scatter(lon13[~np.isnan(sshDS13)],lat13[~np.isnan(sshDS13)],s=5,c='lime',label='ICESat-2')
    plt.scatter(lonJ[iJ],latJ[iJ],s=15,c='red',label='T/P, J1, J2, J3 & S6-MF')
    plt.scatter(lonS[iS],latS[iS],s=15,c='blue',label='Sentinel-3A')
    #plt.scatter(lonC[iC],latC[iC],s=15,c='orange',label='Cryosat-2')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),ncol=4,fontsize=fs)
    plt.xlim(-52.5,-51.25)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    #fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(12,6))
    fs=16
    TIT='Varicance reduction of mean sea level ($\sigma^2_{OTC}$ - $\sigma^2_{No OTC}$) \n mean($\Delta \sigma^2$) = '+str(np.round(np.nanmean(mss_var_ds13-mss_var_013),2))+ ' m$^2$'
    plt.suptitle('Nuup Kangerlua Fjord, Greenland',fontsize=20)
    plt.title(TIT,fontsize=fs)
    show(src2,ax=ax,cmap='binary_r')
    c=ax.scatter(lon13_d21,lat13_d21,c=mss_var_M13-mss_var_013,s=20,cmap='bwr',vmin=-1,vmax=1)
    plt.plot(LON_tide,LAT_tide,'v',markersize=30,color='indianred',mec= "yellow")
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='$\Delta$ variance of sea surface height [m$^2$]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    #fig.tight_layout()
    plt.show()

    ### New MSS (atl13)
    fig, ax = plt.subplots(figsize=(12,6))
    fs=16
    mss_d21_model_nan = np.copy(mss_d21_model)
    mss_d21_model_nan[np.isnan(mss_d21_013)]=np.nan
    TIT='DTU21 mean sea surface (MSS) \n $\mu \pm \sigma$ = '+str(np.round(np.nanmean(mss_d21_model_nan),2))+ ' $\pm$ '+str(np.round(np.nanstd(mss_d21_model_nan),2))+ ' m'
    plt.suptitle('Nuup Kangerlua Fjord, Greenland',fontsize=20)
    plt.title(TIT,fontsize=fs)
    show(src,ax=ax,cmap='binary_r')
    c=ax.scatter(lon13_d21,lat13_d21,c=mss_d21_model_nan,s=20,cmap='viridis',vmin=27,vmax=33)
    plt.plot(LON_tide,LAT_tide,'v',markersize=30,color='red')
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='Mean sea surface height [m]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(12,6))
    fs=16
    TIT='Mean sea level (no ocean tide correction applied) \n $\mu \pm \sigma$ = '+str(np.round(np.nanmean(mss_d21_013),2))+ ' $\pm$ '+str(np.round(np.nanstd(mss_d21_013),2))+ ' m'
    plt.suptitle('Nuup Kangerlua Fjord, Greenland',fontsize=20)
    plt.title(TIT,fontsize=fs)
    show(src,ax=ax,cmap='binary_r')
    c=ax.scatter(lon13_d21,lat13_d21,c=mss_d21_013,s=20,cmap='viridis',vmin=27,vmax=33)
    plt.plot(LON_tide,LAT_tide,'v',markersize=30,color='red')
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='Mean sea surface height [m]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(12,6))
    fs=16
    TIT='Mean sea level (ocean tide correction applied) \n $\mu \pm \sigma$ = '+str(np.round(np.nanmean(mss_d21_M13),2))+ ' $\pm$ '+str(np.round(np.nanstd(mss_d21_M13),2))+ ' m'
    plt.suptitle('Nuup Kangerlua Fjord, Greenland',fontsize=20)
    plt.title(TIT,fontsize=fs)
    show(src,ax=ax,cmap='binary_r')
    c=ax.scatter(lon13_d21,lat13_d21,c=mss_d21_M13,s=20,cmap='viridis',vmin=27,vmax=33)
    plt.plot(LON_tide,LAT_tide,'v',markersize=30,color='red')
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='Mean sea surface height [m]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(12,6))
    fs=16
    TIT='Mean sea level (FES2014 OTC) \n $\mu \pm \sigma$ = '+str(np.round(np.nanmean(mss_d21_T13),2))+ ' $\pm$ '+str(np.round(np.nanstd(mss_d21_T13),2))+ ' m'
    plt.suptitle('Nuup Kangerlua Fjord, Greenland',fontsize=20)
    plt.title(TIT,fontsize=fs)
    show(src,ax=ax,cmap='binary_r')
    c=ax.scatter(lon13_d21,lat13_d21,c=mss_d21_T13,s=20,cmap='viridis',vmin=27,vmax=33)
    plt.plot(LON_tide,LAT_tide,'v',markersize=30,color='red')
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='Mean sea surface height [m]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    fig.tight_layout()
    plt.show()

    ############### Mean SSH (DTU21 vs. X)
    fig, ax = plt.subplots(figsize=(12,6))
    fs=16
    TIT='Mean sea level difference, SSH(no OTC) - DTU21, \n $\mu \pm \sigma$ = '+str(np.round(np.nanmean(mss_d21_013-mss_d21_model),2))+ ' $\pm$ '+str(np.round(np.nanstd(mss_d21_013-mss_d21_model),2))+ ' m'
    plt.suptitle('Nuup Kangerlua Fjord, Greenland',fontsize=20)
    plt.title(TIT,fontsize=fs)
    show(src,ax=ax,cmap='binary_r')
    c=ax.scatter(lon13_d21,lat13_d21,c=mss_d21_013-mss_d21_model,s=20,cmap='Spectral_r',vmin=-2,vmax=2)
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='$\Delta$ mean sea surface height [m]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(12,6))
    fs=16
    TIT='Mean sea level difference, SSH(estimated OTC) - DTU21, \n $\mu \pm \sigma$ = '+str(np.round(np.nanmean(mss_d21_M13-mss_d21_model),2))+ ' $\pm$ '+str(np.round(np.nanstd(mss_d21_M13-mss_d21_model),2))+ ' m'
    plt.suptitle('Nuup Kangerlua Fjord, Greenland',fontsize=20)
    plt.title(TIT,fontsize=fs)
    show(src,ax=ax,cmap='binary_r')
    c=ax.scatter(lon13_d21,lat13_d21,c=mss_d21_M13-mss_d21_model,s=20,cmap='Spectral_r',vmin=-2,vmax=2)
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='$\Delta$ mean sea surface height [m]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    fig.tight_layout()
    plt.show()
    ############### END



    dcG = 0.3
    lon_crtG = np.arange(-52,-49.5+dcG,dcG)
    NcrtG = np.size(lon_crtG)
    '''
    # dTG=array([ 0.,  0.,  0.,  0.,  5.,  5.,  0.,  0.,  5., nan,  0.,  0., 15.,
        0., 15.,  5.,  0.,  0.,  0.,  5., 10.,  0.,  0.,  5., 15.,  5.,
        0.,  0.,  0., nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
       nan])
    AG = array([1.14, 1.1 , 1.06, 1.  , 1.14, 1.08, 1.04, 1.02, 1.  ,  nan, 1.1 ,
       2.26, 1.14, 1.08, 1.1 , 1.26, 1.08, 1.2 , 1.28, 1.16, 1.  , 1.9 ,
       1.  , 1.04, 1.  , 1.22, 1.14, 1.16, 1.26,  nan,  nan,  nan,  nan,
        nan,  nan,  nan,  nan,  nan,  nan,  nan])
    lat_crtG = np.arange(np.nanmin(lat13[~np.isnan(ssh13)]),np.nanmax(lat13[~np.isnan(ssh13)])+dcG,dcG)
    lon_crtG = np.arange(np.nanmin(lon13[~np.isnan(ssh13)]),np.nanmax(lon13[~np.isnan(ssh13)])+dcG,dcG)
    mlat_crtG,mlon_crtG = np.meshgrid(lat_crtG,lon_crtG)
    flat_crtG,flon_crtG = mlat_crtG.flatten('F'),mlon_crtG.flatten('F')
    NcrtG = np.size(flat_crtG)
    '''

    fig, ax = plt.subplots(figsize=(14,8))
    show(src,ax=ax,title=NAME+': mean sea level (ocean tide correction applied) plotted over Sentinel-2A \n $\sigma$ = '+str(np.round(np.nanstd(mss_d21_ds13),2))+ 'm',cmap='viridis')
    c=ax.scatter(lon13_d21,lat13_d21,c=mss_d21_013,s=20,cmap='viridis',label='atl13 (tide correction)',vmin=26,vmax=32)
    #plt.scatter(lon13,lat13,c=ssh13,s=1,cmap='viridis',vmin=0,vmax=14)
    plt.plot(LON_tide,LAT_tide,'v',markersize=30,color='red')
    '''
    for ii in np.arange(np.size(lat_crtG)):
        plt.axhline(y=lat_crtG[ii],color='red')
    '''
    for ii in np.arange(np.size(lon_crtG)):
        plt.axvline(x=lon_crtG[ii],color='red')
    ax.legend()
    fig.colorbar(c,ax=ax,location='bottom',label='sea surface height [m]',shrink=0.7)
    ax.set_xlabel('longitude [deg]')
    ax.set_ylabel('latitude [deg]')
    fig.tight_layout()
    plt.show()
    
    ##########

    
    dTG,AG = np.empty(NcrtG)*np.nan,np.empty(NcrtG)*np.nan
    szG=np.empty(NcrtG)*np.nan
    for ii in np.arange(7):
        icrt = np.where((lon13>=lon_crtG[ii]-dcG/2.0)&(lon13<lon_crtG[ii]+dcG/2.0))[0]#np.where((lat13>=flat_crtG[ii]-dcG/2.0)&(lat13<flat_crtG[ii]+dcG/2.0)&(lon13>=flon_crtG[ii]-dcG/2.0)&(lon13<flon_crtG[ii]+dcG/2.0))[0]
        szG[ii]=np.size(icrt)
        if szG[ii]>=10:
            dTGii,AGii=run_delay_analysis(ssh13[icrt],utc13[icrt],t13[icrt],LON_tide,LAT_tide,min_min=0,max_min=30,dmin=5,VR_MAX=3,N_MIN=50) #,max_min=310,dmin=30,VR_MAX=1,N_MIN=50) 
            if ~np.isnan(dTGii):
                dTG[ii],AG[ii]=dTGii,AGii
    
    CE_dTG = np.round(np.polyfit(lon_crtG,dTG, 1)[::-1],2)
    dTG_fit = CE_dTG[0]+(CE_dTG[1]*lon_crtG)

    CE_AG = np.round(np.polyfit(lon_crtG,AG, 1)[::-1],2)
    AG_fit = CE_AG[0]+(CE_AG[1]*lon_crtG)

    plt.figure(figsize=(12,6))
    plt.suptitle('Ocean tide time delay and amplitude scale estimates \n as a funciton of latitude')
    plt.subplot(121)
    plt.title('Time delay ($\Delta$)')# = '+str(CE_dTG[0])+' + '+str(CE_dTG[1])+' * latitude')
    plt.plot(lon_crtG,dTG,'o',color='tab:blue',markersize=10,label='estimated $\Delta T$')
    plt.plot(lon_crtG,dTG_fit,'--',color='black',markersize=10,label='linear fit')
    plt.ylabel('Latitude [deg]')
    plt.xlabel('Time delay, $\Delta T$ [min]')
    plt.grid()
    plt.legend()
    for ii in np.arange(NcrtG):
        plt.axvline(x=lon_crtG[ii],color='red')
    plt.subplot(122)
    plt.title('Amplitude scale (A)')# = '+str(CE_AG[0])+' + '+str(CE_AG[1])+' * latitude')
    plt.plot(lon_crtG,AG,'o',color='tab:blue',markersize=10,label='amplitude scale')
    plt.plot(lon_crtG,AG,'--',color='black',markersize=10,label='linear fit')
    plt.ylabel('Latitude [deg]')
    plt.xlabel('Amplitude scale, A [-]')
    plt.grid()
    plt.legend()
    for ii in np.arange(NcrtG):
        plt.axvline(x=lon_crtG[ii],color='red')
    




def ana_atlXX_anchorage():
    '''
    #Test
    from datetime import timedelta, datetime
    yf = 2023+((dt.datetime(2023,1,1,0,0)-dt.datetime(2023,1,1,0,0)).total_seconds()/(365.25*24*60*60)) 
    year = int(yf)
    d = timedelta(days=(yf - year)*365.25)
    day_one = datetime(year,1,1)
    date = d + day_one
    print(date)
    '''

    NAME = 'ANCHORAGE'
    fnEO,f12,f13,LLMM2,tide_delay,tide_scale,pthNOAA,tide_delay_func = NAME_info(NAME)
    src = rio.open(fnEO)
    utc13,t13,lat13,lon13,ssh13,ssh013,sshM13,sshDS13,sshT13,sshG13,tideM13,tideDS13,tideT13,tideG13,mss13_c15,mss13_d21,t_frac_moments13 = ana_atl13_data(NAME,2,50,LAT_tide,LON_tide,dt_days=1)
    yr13,mnt13,dy13,hr13,mi13,date13 = lXX.convert_partial_year(t_frac_moments13)
    Nd = np.shape(date13)[0]
    inan = np.where((lat13<59.9)&(lon13>-151.9))[0]
    ssh13[inan]=np.nan
    sshDS13[inan]=np.nan
    #anchorage = 9455920
    #nikiski = 9455760
    #seldovia = 9455500
    '''
    aSTAT = ['anchorage','nikiski','seldovia','ninilchik','anchor_point','kalgan_island','north_foreland','point_possession','fire_island','port_mackenzie','chinulna_point','cape_kasilof']
    for STAT in aSTAT:
        if STAT=='anchorage':
            SID = str(9455920)
        elif STAT=='nikiski':
            SID = str(9455760)
        elif STAT=='seldovia':
            SID = str(9455500)
        elif STAT=='ninilchik':
            SID = str(9455653)
        elif STAT=='anchor_point':
            SID = str(9455606)
        elif STAT=='kalgan_island':
            SID = str(9455732)
        elif STAT=='north_foreland':
            SID = str(9455869)
        elif STAT=='point_possession':
            SID = str(9455866)
        elif STAT=='fire_island':
            SID = str(9455912)
        elif STAT=='port_mackenzie':
            SID = str(9455934)
        elif STAT=='chinulna_point':
            SID = str(9455735)
        elif STAT=='cape_kasilof':
            SID = str(9455711)
        for ii in np.arange(Nd):
            yr = "{0:0=4d}".format(int(yr13[ii]))#f"{int(yr13[ii]):04}" #"{0:0=4d}".format(yr13[ii])
            mn = "{0:0=2d}".format(int(mnt13[ii]))
            dy = "{0:0=2d}".format(int(dy13[ii]))
            hr = "{0:0=2d}".format(int(hr13[ii]))
            #dy2 = "{0:0=2d}".format(int(dy13[ii]+1))
            """
            if int(hr13[ii])==24:
                dy2,hr2 = "{0:0=2d}".format(int(dy13[ii]+1)),"{0:0=2d}".format(1)
            else:
                dy2,hr2 = "{0:0=2d}".format(int(dy13[ii])),"{0:0=2d}".format(int(hr13[ii]+1))
            """
            # fli = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date="+yr+mn+dy+" "+hr+":00&end_date="+yr+mn+dy2+" "+hr+":00&station="+SID+"&product=water_level&datum=MSL&time_zone=gmt&interval=6&units=metric&application=DataAPI_Sample&format=csv"
            fli = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date="+yr+mn+dy+" "+hr+":00&range=2&station="+SID+"&product=predictions&datum=MSL&time_zone=gmt&interval=1&units=metric&format=csv"
            r = requests.get(fli, allow_redirects=True)
            open('cook_inlet/'+STAT+'/pred_'+STAT+'_'+SID+'_'+yr+mn+dy+'_'+hr+'00.csv', 'wb').write(r.content)
    '''
    '''
    dTG,AG = np.empty(NcrtG)*np.nan,np.empty(NcrtG)*np.nan
    for ii in np.arange(NcrtG):
        icrt = np.where((dist13>=dist_crtG[ii]-(dl/2.0))&(dist13<dist_crtG[ii]+(dl/2.0)))[0]
        #icrt = np.where((lat13>=lat_crtG[ii]-dcG/2.0)&(lat13<lat_crtG[ii]+dcG/2.0))[0]#&(lon13>=flon_crtG[ii]-dcG/2.0)&(lon13<flon_crtG[ii]+dcG/2.0))[0]
        if np.size(icrt)>=10:
            dTG[ii],AG[ii]=run_delay_analysis(ssh13[icrt],utc13[icrt],t13[icrt],LON_tide,LAT_tide,min_min=0,max_min=310,dmin=30,VR_MAX=2,N_MIN=50) 

    
    CE_dTG = np.round(np.polyfit(dist_crtG,dTG, 1)[::-1],2)
    dTG_fit = CE_dTG[0]+(CE_dTG[1]*dist_crtG)

    CE_AG = np.round(np.polyfit(dist_crtG,AG, 1)[::-1],2)
    AG_fit = CE_AG[0]+(CE_AG[1]*dist_crtG)

    #CE_AG = np.round(np.polyfit(lat_crtG[np.asaAG[np.asarray([0,1,3,4])]rray([0,1,3,4])],, 1)[::-1],2)
    #AG_fit = CE_AG[0]+(CE_AG[1]*lat_crtG)
    
    plt.figure(figsize=(12,6))
    plt.suptitle('Ocean tide time delay and amplitude scale estimates \n as a funciton of latitude')
    plt.subplot(121)
    plt.title('Time delay ($\Delta$)')# = '+str(CE_dTG[0])+' + '+str(CE_dTG[1])+' * latitude')
    plt.plot(dTG,dist_crtG,'o',color='tab:blue',markersize=10,label='estimated $\Delta T$')
    plt.plot(dTG_fit,dist_crtG,'--',color='black',markersize=10,label='linear fit')
    plt.ylabel('Latitude [deg]')
    plt.xlabel('Time delay, $\Delta T$ [min]')
    plt.grid()
    plt.legend()
    for ii in np.arange(NcrtG):
        plt.axhline(y=dist_crtG[ii],color='red')
    plt.subplot(122)
    plt.title('Amplitude scale (A)')# = '+str(CE_AG[0])+' + '+str(CE_AG[1])+' * latitude')
    plt.plot(AG,dist_crtG,'o',color='tab:blue',markersize=10,label='amplitude scale')
    plt.plot(AG_fit,dist_crtG,'--',color='black',markersize=10,label='linear fit')
    plt.ylabel('Latitude [deg]')
    plt.xlabel('Amplitude scale, A [-]')
    plt.grid()
    plt.legend()
    for ii in np.arange(NcrtG):
        plt.axhline(y=dist_crtG[ii],color='red')
    '''

    ######################## START IMG
    dcG = 0.5
    lat_crtG = np.arange(np.nanmin(lat13),np.nanmax(lat13)+dcG,dcG)
    #dl=50
    #dist_crtG = np.arange(0,300+dl,dl)
    NcrtG = np.size(lat_crtG)
    dist13 = np.abs(lXX.geo2dist(lat13,lon13,LAT_tide,LON_tide))/1000.
    fig, ax = plt.subplots(figsize=(14,8)) #KEEEP
    show(src,ax=ax)
    fs=15
    plt.title('Cook Inlet: tide gauge locations ',fontsize=20)
    c=ax.scatter(lon13,lat13,c=ssh13,s=20,cmap='viridis',label='atl13 (tide correction)',vmin=4,vmax=14)
    plt.plot(-151.72,59.44, 'v',markersize=30,color='black',label='Seldovia (lower)',mec= "red")
    plt.plot(-151.39666,60.68333, 'v',markersize=30,color='orange',label='Nikiski (middle)',mec= "red")
    plt.plot(-149.88999,61.23833, 'v',markersize=30,color='green',label='Anchorage (upper)',mec= "red")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),ncol=4,fontsize=fs)
    #plt.plot(LON_tide,LAT_tide,'v',markersize=30,color='red',label='Sample point')
    for ii in np.arange(NcrtG):
        plt.axhline(y=lat_crtG[ii],color='red')
    ax.set_xlabel('longitude [deg]',fontsize=fs)
    ax.set_ylabel('latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    fig.tight_layout()
    plt.show()
    ######################## END IMG

    # Create new MSS models
    inn13 = np.where(~np.isnan(sshDS13))[0]
    lat13_d21,lon13_d21,mss_d21_model = lXX.mss_model(lat13[inn13],lon13[inn13],MODEL='dtu21',RETURN_COORD=True,mss_in=[],TP2WGS=True,IS2DATE=False)
    lat13_d21,lon13_d21,mss_d21_ds13,mss_var_ds13,N_d21_ds13 = lXX.mss_model(lat13[inn13],lon13[inn13],MODEL='dtu21',RETURN_COORD=True,mss_in=sshDS13[inn13],TP2WGS=True,IS2DATE=False,tm=t13[inn13])
    lat13_d21,lon13_d21,mss_d21_ds013,mss_var_ds013,N_d21_ds013 = lXX.mss_model(lat13[inn13],lon13[inn13],MODEL='dtu21',RETURN_COORD=True,mss_in=ssh13[inn13],TP2WGS=True,IS2DATE=False,tm=t13[inn13])
    lat13_d21,lon13_d21,mss_d21_dsT13,mss_var_dsT13,N_d21_ds013 = lXX.mss_model(lat13[inn13],lon13[inn13],MODEL='dtu21',RETURN_COORD=True,mss_in=sshT13[inn13],TP2WGS=True,IS2DATE=False,tm=t13[inn13])

    #############################   MSS
    ### New MSS (atl13)
    fig, ax = plt.subplots(figsize=(12,8))
    fs=16
    TIT='Mean sea level (ocean tide correction applied) \n $\mu \pm \sigma$ = '+str(np.round(np.nanmean(mss_d21_ds13),2))+ ' $\pm$ '+str(np.round(np.nanstd(mss_d21_ds13),2))+ ' m'
    plt.suptitle('Cook Inlet',fontsize=20)
    plt.title(TIT,fontsize=fs)
    show(src,ax=ax,cmap='binary_r')
    c=ax.scatter(lon13_d21,lat13_d21,c=mss_d21_ds13,s=20,cmap='viridis',vmin=3,vmax=13)
    plt.plot(LON_tide,LAT_tide,'v',markersize=30,color='red')
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='Mean sea surface height [m]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    fig.tight_layout()
    plt.show()


    ### New MSS (atl13 no OTC)
    fig, ax = plt.subplots(figsize=(12,8))
    fs=16
    TIT='Mean sea level (NO ocean tide correction applied) \n $\mu \pm \sigma$ = '+str(np.round(np.nanmean(mss_d21_ds013),2))+ ' $\pm$ '+str(np.round(np.nanstd(mss_d21_ds013),2))+ ' m'
    plt.suptitle('Cook Inlet',fontsize=20)
    plt.title(TIT,fontsize=fs)
    show(src,ax=ax,cmap='binary_r')
    c=ax.scatter(lon13_d21,lat13_d21,c=mss_d21_ds013,s=20,cmap='viridis',vmin=3,vmax=13)
    plt.plot(LON_tide,LAT_tide,'v',markersize=30,color='red')
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='Mean sea surface height [m]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    fig.tight_layout()
    plt.show()


    #############################   MSS Variance
    ### New MSS (atl13)
    fig, ax = plt.subplots(figsize=(12,8))
    fs=16
    TIT='Varicance reduction of mean sea level ($\sigma^2_{OTC}$ - $\sigma^2_{No OTC}$) \n mean($\Delta \sigma^2$) = '+str(np.round(np.nanmean(mss_var_ds13-mss_var_ds013),2))+ ' m$^2$'
    plt.suptitle('Cook Inlet',fontsize=20)
    plt.title(TIT,fontsize=fs)
    show(src,ax=ax,cmap='binary_r')
    c=ax.scatter(lon13_d21,lat13_d21,c=mss_var_ds13-mss_var_ds013,s=20,cmap='seismic',vmin=-5,vmax=5)
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='$\Delta$ variance of sea surface height [m$^2$]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    fig.tight_layout()
    plt.show()

    ############### Mean SSH (FES vs. X)
    fig, ax = plt.subplots(figsize=(12,8))
    fs=16
    TIT='Mean sea level difference, SSH(no OTC)- SSH(FES2014), \n $\mu \pm \sigma$ = '+str(np.round(np.nanmean(mss_d21_ds013-mss_d21_dsT13),2))+ ' $\pm$ '+str(np.round(np.nanstd(mss_d21_ds013-mss_d21_dsT13),2))+ ' m'
    plt.suptitle('Cook Inlet',fontsize=20)
    plt.title(TIT,fontsize=fs)
    show(src,ax=ax,cmap='binary_r')
    c=ax.scatter(lon13_d21,lat13_d21,c=mss_d21_ds013-mss_d21_dsT13,s=20,cmap='Spectral_r',vmin=-2,vmax=2)
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='$\Delta$ mean sea surface height [m]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(12,8))
    fs=16
    TIT='Mean sea level difference, SSH(estimated OTC)- SSH(FES2014), \n $\mu \pm \sigma$ = '+str(np.round(np.nanmean(mss_d21_ds13-mss_d21_dsT13),2))+ ' $\pm$ '+str(np.round(np.nanstd(mss_d21_ds13-mss_d21_dsT13),2))+ ' m'
    plt.suptitle('Cook Inlet',fontsize=20)
    plt.title(TIT,fontsize=fs)
    show(src,ax=ax,cmap='binary_r')
    c=ax.scatter(lon13_d21,lat13_d21,c=mss_d21_ds13-mss_d21_dsT13,s=20,cmap='Spectral_r',vmin=-2,vmax=2)
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='$\Delta$ mean sea surface height [m]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    fig.tight_layout()
    plt.show()
    ############### END


    ############### Mean SSH (DTU21 vs. X)
    fig, ax = plt.subplots(figsize=(12,8))
    fs=16
    TIT='Mean sea level difference, SSH(no OTC) - DTU21, \n $\mu \pm \sigma$ = '+str(np.round(np.nanmean(mss_d21_ds013-mss_d21_model),2))+ ' $\pm$ '+str(np.round(np.nanstd(mss_d21_ds013-mss_d21_model),2))+ ' m'
    plt.suptitle('Cook Inlet',fontsize=20)
    plt.title(TIT,fontsize=fs)
    show(src,ax=ax,cmap='binary_r')
    c=ax.scatter(lon13_d21,lat13_d21,c=mss_d21_ds013-mss_d21_model,s=20,cmap='Spectral_r',vmin=-2,vmax=2)
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='$\Delta$ mean sea surface height [m]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(12,8))
    fs=16
    TIT='Mean sea level difference, SSH(estimated OTC) - DTU21, \n $\mu \pm \sigma$ = '+str(np.round(np.nanmean(mss_d21_ds13-mss_d21_model),2))+ ' $\pm$ '+str(np.round(np.nanstd(mss_d21_ds13-mss_d21_model),2))+ ' m'
    plt.suptitle('Cook Inlet',fontsize=20)
    plt.title(TIT,fontsize=fs)
    show(src,ax=ax,cmap='binary_r')
    c=ax.scatter(lon13_d21,lat13_d21,c=mss_d21_ds13-mss_d21_model,s=20,cmap='Spectral_r',vmin=-2,vmax=2)
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='$\Delta$ mean sea surface height [m]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    fig.tight_layout()
    plt.show()
    ############### END



def pull_tg_anchorage():
    LL_anc = [61.23833, -149.88999]
    LL_sel = [59.44, -151.72]
    LL_nik = [60.68333, -151.39666]
    LL_nin = [60.05394, -151.68288]
    LL_ap = [59.77233, -151.87253]
    LL_ki = [60.51239, -151.9533]
    LL_nf = [61.04136, -151.17504]
    LL_pp = [61.0387, -150.40394]
    LL_fi = [61.17438, -150.20962]
    LL_pm = [61.26858, -149.91848]
    LL_cp = [60.505, -151.28641]
    LL_ck = [60.33885, -151.38494]
    yf_tg_anc,H_tg_anc,sig_tg_anc,dt_tg_anc = lXX.pull_csv('/Users/alexaputnam/ICESat2/Fjord/cook_inlet/anchorage/')
    yf_tg_sel,H_tg_sel,sig_tg_sel,dt_tg_sel = lXX.pull_csv('/Users/alexaputnam/ICESat2/Fjord/cook_inlet/seldovia/')
    yf_tg_nik,H_tg_nik,sig_tg_nik,dt_tg_nik = lXX.pull_csv('/Users/alexaputnam/ICESat2/Fjord/cook_inlet/nikiski/')
    
    yf_tg_nin,H_tg_nin,sig_tg_nin,dt_tg_nin = lXX.pull_csv('/Users/alexaputnam/ICESat2/Fjord/cook_inlet/ninilchik/')
    yf_tg_ap,H_tg_ap,sig_tg_ap,dt_tg_ap = lXX.pull_csv('/Users/alexaputnam/ICESat2/Fjord/cook_inlet/anchor_point/')
    yf_tg_ki,H_tg_ki,sig_tg_ki,dt_tg_ki = lXX.pull_csv('/Users/alexaputnam/ICESat2/Fjord/cook_inlet/kalgan_island/')
    yf_tg_nf,H_tg_nf,sig_tg_nf,dt_tg_nf = lXX.pull_csv('/Users/alexaputnam/ICESat2/Fjord/cook_inlet/north_foreland/')
    yf_tg_pp,H_tg_pp,sig_tg_pp,dt_tg_pp = lXX.pull_csv('/Users/alexaputnam/ICESat2/Fjord/cook_inlet/point_possession/')
    yf_tg_fi,H_tg_fi,sig_tg_fi,dt_tg_fi = lXX.pull_csv('/Users/alexaputnam/ICESat2/Fjord/cook_inlet/fire_island/')
    yf_tg_pm,H_tg_pm,sig_tg_pm,dt_tg_pm = lXX.pull_csv('/Users/alexaputnam/ICESat2/Fjord/cook_inlet/port_mackenzie/')
    yf_tg_cp,H_tg_cp,sig_tg_cp,dt_tg_cp = lXX.pull_csv('/Users/alexaputnam/ICESat2/Fjord/cook_inlet/chinulna_point/')
    yf_tg_ck,H_tg_ck,sig_tg_ck,dt_tg_ck = lXX.pull_csv('/Users/alexaputnam/ICESat2/Fjord/cook_inlet/cape_kasilof/')




def ana_vancouver_island_coast():
    ATLX=12
    if ATLX==12:
        fn = 'reg_atl12_lat_47_lon_n127_vancouver_island_2018_10_to_2023_06.npz'
        ds12 = np.load(fn,allow_pickle='TRUE')
        h_abs=np.abs(ds12['ssh'])
        i12 = np.where((h_abs<200))[0]
        ds12 = np.load(fn,allow_pickle='TRUE')
        print(ds12.files)
        utc12 = ds12['time_utc'][i12]
        t12=lXX.utc2yrfrac(ds12['time_utc'][i12])
        lat12=ds12['lat'][i12]
        lon12=ds12['lon'][i12]
        dac12=ds12['dac'][i12]
        ssh12g=ds12['ssh'][i12]
        geo12=ds12['geoid'][i12]
        tL12=ds12['tide_load'][i12]
        tL12[np.abs(tL12)>100]=0
        tO12=ds12['tide_ocean'][i12]+tL12
        tO12[np.abs(tO12)>100]=0

        tL12f=ds12['tide_load_f14'][i12]
        tL12f[np.abs(tL12f)>100]=0
        tO12f=ds12['tide_ocean_f14'][i12]+tL12f
        tO12f[np.abs(tO12f)>100]=0

        ssh12 = ssh12g+tO12
        ssh12f = ssh12-tO12f
        ssh12f_noDAC = ssh12-tO12f+dac12
        #Tmi = T_moments[np.asarray([6,26,27,36,52,60,68,69,129,131,132,137,262,264,301])]
    else:
        fn = 'reg_atl13_lat_47_lon_n127_vancouver_island_2018_10_to_2023_06.npz'
        ds13 = np.load(fn,allow_pickle='TRUE')
        print(ds13.files)
        #&(ds13['flag_pod']==0), (qf_cloud==0)& (qf_water==3)
        water_type = ds13['water_type'] # waterbody type: 1=Lake, 2=Known Reservoir, 3=(Reserved for future use),4=Ephemeral Water, 5=River, 6=Estuary or Bay,7=Coastal Water
        qf_frac = ds13['flag_frac']# 20% min --> quality flags aren't good in these regions
        qf_pod = ds13['flag_pod']# pod ppd flag: 0 =nominal
        qf_cloud = ds13['flag_cloud'] #0 =nominal
        qf_water = ds13['flag_water']# ATL09 flag: 0=ice free water, 1=snow and ice free land, 2=snow, 3=ice
        h_abs = np.abs(ds13['ht_water_surf'])
        
        i13 = np.where((qf_pod==0)&(h_abs<200))[0]#(qf_pod==0)&(h_abs<200))[0]#np.where((np.abs(ds13['ht_water_surf'])<200))[0]
        utc12 = ds13['time_utc'][i13]
        t12=lXX.utc2yrfrac(utc12)
        lat12=ds13['lat'][i13]
        lon12=ds13['lon'][i13]
        dac12=ds13['dac'][i13]
        dac12[np.abs(dac12)>1000]=np.nan
        ssh12=ds13['ht_water_surf'][i13]#+tL13
        geo12=ds13['geoid'][i13]
        dem12=ds13['dem'][i13]
        tO12=ds13['tide_ocean'][i13]#+tL13
        tO12[np.abs(tO12)>100]=0
        tL13 = np.zeros(np.size(tO12))

        tL12f=ds13['tide_load_f14'][i13]
        tL12f[np.abs(tL12f)>100]=0
        tO12f=ds13['tide_ocean_f14'][i13]+tL12f
        tO12f[np.abs(tO12f)>100]=0

        ssh12g = ssh12-tO12
        ssh12f = ssh12-tO12f
    mss_c15 = lXX.mss_model(lat12,lon12,MODEL='cnes15')
    mss_d21 = lXX.mss_model(lat12,lon12,MODEL='dtu21')
    ssh12nan = np.copy(ssh12)
    ssh12nan[np.isnan(ssh12f)]=np.nan
    fn = 'vancouver_2023-09-20-23_59_Sentinel-2_L2A_B8A_(Raw).tiff'#'vancouver_2023-09-20-23_59_Sentinel-2_L2A_Highlight_Optimized_Natural_Color.tiff'#'vancouver_2023-09-20-23_59_Sentinel-2_L2A_False_color.tiff' #'vancouver_pass_2023-09-20-23_59_Sentinel-2_L2A_Highlight_Optimized_Natural_Color.tiff'
    src2=rio.open('/Users/alexaputnam/ICESat2/Fjord/EO_sentinel2/'+fn)

    LLMM_pass = [48.9,49,-125.5,-125]
    idP = np.where((lat12>LLMM_pass[0])&(lat12<LLMM_pass[1])&(lon12>LLMM_pass[2])&(lon12<LLMM_pass[3])&(t12>2022.6)&(t12>2022.8))[0]
    inn = np.where(~np.isnan(ssh12f)&~np.isnan(ssh12g))[0]
    tt,T_moments,tide_moments,h_moments = lXX.tide_estimation(utc12[inn],ssh12f[inn],LON_tide,LAT_tide,t_delay_min=0,Return_moments=True)
    LL_tofino = [49.15, -125.91666]
    dist12 = np.abs(lXX.geo2dist(lat12,lon12,LL_tofino[0],LL_tofino[1]))/1000.
    Tmi = T_moments[np.asarray([6,26,27,36,52,60,68,69,129,131,132,137,262,264,301])] #[6,26,27,36,52,60,68,129,131,132,137,169,184,188,255,262,264,301]
    Tmi_yf=lXX.utc2yrfrac(np.asarray(Tmi))
    Nt = np.shape(Tmi_yf)[0]
    idP = []
    for ii in np.arange(Nt):
        idP = np.hstack((idP,np.where(np.abs(utc12-Tmi[ii])<60)[0]))
    idP=idP.astype(int)

    '''
    for ii in np.arange(0,50): #np.asarray([6,26,27,36,52,132,137,188,255]):#
        Tmi = T_moments[ii]
        idP = np.where(np.abs(utc12-Tmi)<60)[0]
        if np.size(idP)>150:
            fig, ax = plt.subplots(figsize=(14,8))
            show(src2,ax=ax,title='Vancouver '+str(ii),cmap='viridis')
            c=ax.scatter(lon12[idP],lat12[idP],c=(ssh12g[idP]),s=20,cmap='Spectral_r',label='atl13 (tide correction)',vmin=-24,vmax=-16)
            ax.legend()
            fig.colorbar(c,ax=ax,location='bottom',label='standard deviation of sea surface height [m]',shrink=0.7)
            ax.set_xlabel('longitude [deg]')
            ax.set_ylabel('latitude [deg]')
            fig.tight_layout()
            plt.show()
    '''

    fig, ax = plt.subplots(figsize=(10,10))
    fs=16
    plt.suptitle('Vancouver Island',fontsize=20)
    plt.title('SSH(FES14) - SSH(GOT4.8) \n '+str(Nt)+' ICESat-2 (ATL12) passes',fontsize=fs)
    show(src2,ax=ax,cmap='binary_r')
    c=ax.scatter(lon12[idP],lat12[idP],c=ssh12f[idP],s=20,cmap='viridis',label='atl12 SSH(FES14)',vmin=-24,vmax=-16)
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    plt.xticks(np.asarray([-127,-126.5,-126,-125.5,-125,-124.5]))
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='Sea surface height [cm]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    fig.tight_layout()
    plt.show()

    ilon = np.where(lon12<=-124.15)[0]
    lat12_d21,lon12_d21,mss_d21_ds12,mss_var_d21_ds12,N_d21_ds12 = lXX.mss_model(lat12[ilon],lon12[ilon],MODEL='dtu21',RETURN_COORD=True,mss_in=ssh12nan[ilon],TP2WGS=True,IS2DATE=False,tm=t12)
    lat12_d21,lon12_d21,mss_d21_ds12g,mss_var_d21_ds12g,N_d21_ds12g = lXX.mss_model(lat12[ilon],lon12[ilon],MODEL='dtu21',RETURN_COORD=True,mss_in=ssh12g[ilon],TP2WGS=True,IS2DATE=False,tm=t12)
    lat12_d21,lon12_d21,mss_d21_ds12f,mss_var_d21_ds12f,N_d21_ds12f = lXX.mss_model(lat12[ilon],lon12[ilon],MODEL='dtu21',RETURN_COORD=True,mss_in=ssh12f[ilon],TP2WGS=True,IS2DATE=False,tm=t12)
    lat12_d21,lon12_d21,mss_d21_ds12f_noDAC,mss_var_d21_ds12f_noDAC,N_d21_ds12f_noDAC = lXX.mss_model(lat12[ilon],lon12[ilon],MODEL='dtu21',RETURN_COORD=True,mss_in=ssh12f_noDAC[ilon],TP2WGS=True,IS2DATE=False,tm=t12)
    lat12_d21,lon12_d21,mss_d21_dSSH,mss_var_d21_dSSH,N_d21_dSSH = lXX.mss_model(lat12[ilon],lon12[ilon],MODEL='dtu21',RETURN_COORD=True,mss_in=(ssh12f-ssh12g)[ilon],TP2WGS=True,IS2DATE=False,tm=t12)
    ######################## START IMG
    fig, ax = plt.subplots(figsize=(10,10))
    fs=16
    plt.suptitle('Vancouver Island',fontsize=20)
    plt.title('Standard deviation of gridded sea surface height (FES14 correction applied) \n ICESat-2 (ATL12) years 2018 - 2023',fontsize=fs)
    show(src2,ax=ax,cmap='binary_r')
    c=ax.scatter(lon12_d21,lat12_d21,c=np.sqrt(mss_var_d21_ds12f),s=10,cmap='viridis',vmin=0,vmax=1)
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    plt.xticks(np.asarray([-127,-126.5,-126,-125.5,-125,-124.5]))
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='Standard deviaiton of sea surface height [m]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10,10))
    fs=16
    plt.suptitle('Vancouver Island',fontsize=20)
    plt.title('Standard deviation of gridded sea surface height (no OTC applied) \n ICESat-2 (ATL12) years 2018 - 2023',fontsize=fs)
    show(src2,ax=ax,cmap='binary_r')
    c=ax.scatter(lon12_d21,lat12_d21,c=np.sqrt(mss_var_d21_ds12),s=10,cmap='viridis',vmin=0,vmax=1)
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    plt.xticks(np.asarray([-127,-126.5,-126,-125.5,-125,-124.5]))
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='Standard deviaiton of sea surface height [m]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10,10))
    fs=16
    plt.suptitle('Vancouver Island',fontsize=20)
    show(src2,ax=ax,cmap='binary_r')
    c=ax.scatter(lon12[idP],lat12[idP],c=(((ssh12f-mss_d21)-(ssh12f-mss_c15))*100.)[idP],s=20,cmap='jet_r',vmin=-7,vmax=7)
    dsig = str(np.round(np.nanmean((((ssh12f-mss_d21)-(ssh12f-mss_c15))*100.)[idP]),2))+' $\pm$ '+str(np.round(np.nanstd((((ssh12f-mss_d21)-(ssh12f-mss_c15))*100.)[idP]),2))
    plt.title('SSHA(DTU21) - SSHA(CNES/CLS2015) \n $\mu_{\Delta SSH} \pm \sigma_{\Delta SSH}$ = '+dsig+' cm \n '+str(Nt)+' ICESat-2 (ATL12) passes',fontsize=fs)
    cbar=fig.colorbar(c,ax=ax,location='bottom',shrink=0.7,extend='both')
    ax.set_xlabel('Longitude [deg]',fontsize=fs)
    ax.set_ylabel('Latitude [deg]',fontsize=fs)
    plt.xticks(np.asarray([-127,-126.5,-126,-125.5,-125,-124.5]))
    ax.tick_params(axis='both', which='major', labelsize=fs)
    cbar.set_label(label='$\Delta$ Sea surface height anomaly [cm]', size='large', weight='bold')
    cbar.ax.tick_params(labelsize=fs)
    #plt.plot(LL_tofino[1],LL_tofino[0],'v',markersize=20,color='blue')
    fig.tight_layout()
    plt.show()



    f1 = 'coast/h541_bamfield_uhslc_fd.nc'
    f2 = 'coast/h542_tofino_uhslc_fd.nc'
    f3 = 'coast/h558_neah_bay_uhslc_fd.nc'
    h_UH1,lat_UH1,lon_UH1,days_since_1985_UH1,ymdhms_UH1,uhslc_id1,t_UH1 = lTG.pull_hawaii_tg([f1])
    h_UH2,lat_UH2,lon_UH2,days_since_2985_UH2,ymdhms_UH2,uhslc_id2,t_UH2 = lTG.pull_hawaii_tg([f2])
    h_UH3,lat_UH3,lon_UH3,days_since_3985_UH3,ymdhms_UH3,uhslc_id3,t_UH3 = lTG.pull_hawaii_tg([f3])


#---------------------------------------------------------------------------------------
### Data (external)
'''
sonel GPS timeseries: https://www.sonel.org/-GPS-.html
MIDAS readme: http://geodesy.unr.edu/velocities/midas.readme.txt
'''
"""
---------------------------------------------------------------------------------------
#START
---------------------------------------------------------------------------------------
"""
# UHSLC: 'SEWARD', 'KETCHIKAN', 'QAQORTOQ'
NAME = 'ANCHORAGE'# 'CAMP KANGIUSAQ'#'SKAGWAY'#'KARASUK NORTH'#'SKAGWAY'#'JUNEAU'#''OSLO'#'STENUNGSUND'#'UDDEVALLA'#'STOCKHOLM'#'ALERT'#'COCOS'
working_dir = '/Users/alexaputnam/ICESat2/VLM/'
tg_id,fn,sonel_file,LLMM_tides,LON_tide,LAT_tide,LON_tg,LAT_tg,UHSLC = pull_info(NAME)
if tg_id!=999999:
    t_tg,sl_tg,yfr_sonel,h_sonel,t_gps_str,vlm_gps,vlm_Hi = lXX.external_data(tg_id,NAME)
if np.size(UHSLC)!=0:
    h_UH,lat_UH,lon_UH,days_since_1985_UH,ymdhms_UH,uhslc_id,t_UH = lTG.pull_hawaii_tg([UHSLC])
    plt.figure()
    plt.title(NAME+' UHSLC TG')
    plt.plot(t_UH,h_UH)
    plt.xlabel('time [year]')
    plt.ylabel('ssh [m]')
    print(t_UH)

#ana_atlXX(NAME)

# Real Time Water Levels Data - 9414290 San Francisco, CA - Today. 
"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date=20181104 13:00&end_date=20181104 14:00&station=9455920&product=water_level&datum=STND&time_zone=gmt&units=metric&application=DataAPI_Sample&format=csv"

