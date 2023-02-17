#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:15:05 2022

@author: alexaputnam
"""


import numpy as np
from matplotlib import pyplot as plt
import sys

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

########### INPUT ###########
# Region/Time
REG = 'ittoqqortoormiit' #'norway'#'japan' #'newengland','hawaii','antarctica', 'japan'
yrs_mm = np.arange(2019,2022)
fp = 100
lat_min,lat_max,lon_min,lon_max = lreg.regions(REG)
FNtg,FNalt,FNis2=lTG.file4_is2_alt_tg(REG)
# Tide gauge file


########### DATA ###########
clr = ['black','tab:orange','tab:green','tab:purple','tab:red','tab:blue','tab:olive']
# Tide Gauge
sl_tg,lat_tg,lon_tg,days_since_1985_tg,ymdhms_tg,uhslc_id = lTG.pull_hawaii_tg(FNtg)
if np.shape(sl_tg)[0]!= np.size(sl_tg):
    Ntg=np.shape(sl_tg)[1]
    count_mn_tg = np.empty(Ntg)*np.nan
    for ii in np.arange(Ntg):
        mn_ssha_tg,mn_time_tg = lTG.month_2_month(ymdhms_tg[:,:,ii],sl_tg[:,ii],days_since_1985_tg[:,ii],yrs_mm,IS2=False)
        count_mn_tg[ii] = np.size(mn_ssha_tg)
else:
    mn_ssha_tg,mn_time_tg = lTG.month_2_month(ymdhms_tg,sl_tg,days_since_1985_tg,yrs_mm,IS2=False)
if np.shape(sl_tg)[0]!= np.size(sl_tg):
    count_mn_tg = count_mn_tg.astype(int)
    mn_ssha_tg,mn_time_tg = np.empty((np.nanmax(count_mn_tg),Ntg))*np.nan,np.empty((np.nanmax(count_mn_tg),Ntg))*np.nan
    for ii in np.arange(Ntg):
        mn_ssha_tg[:,ii],mn_time_tg[:,ii] = lTG.month_2_month(ymdhms_tg[:,:,ii],sl_tg[:,ii],days_since_1985_tg[:,ii],yrs_mm,IS2=False)
#mn_ts_tg,mn_ymdhms_tg = tide_days_1985_to_TS(mn_time_tg)

# Jason-3
if np.size(FNalt)!=0:
    ssha_alt,lat_alt,lon_alt,days_since_1985_alt,ymdhmsA,tsA,swh_alt = lTG.pull_altimetry(FNalt)
    mn_ssha_alt,mn_time_alt = lTG.month_2_month(ymdhmsA,ssha_alt,days_since_1985_alt,yrs_mm,IS2=False)
    mn_swh_alt,mn_time_alt = lTG.month_2_month(ymdhmsA,swh_alt,days_since_1985_alt,yrs_mm,IS2=False)
    mn_ts_alt,mn_ymdhms_alt = lTG.tide_days_1985_to_TS(mn_time_alt)

# IceSat2
## 10-m segment 
#ssha_10,lat_10,lon_10,days_since_1985_10,ymdhmsI_10,tsI_10,beam_10,swh_10,N_10 = lTG.pull_icesat(FNis2,SEG=10)
#mn_ssha_10,mn_time_10 = lTG.month_2_month(ymdhmsI_10,ssha_10,days_since_1985_10,yrs_mm)
## 200-m segment
ssha_200,lat_200,lon_200,days_since_1985_200,ymdhmsI_200,tsI_200,beam_200,swh_200,N_200 = lTG.pull_icesat(FNis2,SEG=fp)
ssha_5000,lat_5000,lon_5000,days_since_1985_5000,ymdhmsI_5000,tsI_5000,beam_5000,swh_5000,N_5000 = lTG.pull_icesat(FNis2,SEG=2000)
plt.figure()
binz = np.arange(-1,1.1,0.1)
#hist10,be10=np.histogram(ssha_10,bins=binz)
hist200,be200=np.histogram(ssha_200,bins=binz)
hist5k,be5k=np.histogram(ssha_5000,bins=binz)
#plt.plot(be10[:-1],hist10/np.max(hist10),label='10')
plt.plot(be200[:-1],hist200/np.max(hist200),label='200')
plt.plot(be5k[:-1],hist5k/np.max(hist5k),label='5000')
plt.legend()
print('size 200: '+str(np.size(ssha_200)))


if np.size(FNtg)==1:
    #### REGIONAL FILTER CAN CAUSE ISSUES WITH 'WHILE' STATEMENT
    wgt_200,dist_200 = lTG.select_region(lat_tg,lon_tg,lat_200,lon_200)
    mn_ssha_200,mn_time_200 = lTG.month_2_month(ymdhmsI_200,ssha_200,days_since_1985_200,yrs_mm,wgt=wgt_200)#,LATLON = np.vstack((lat_200,lon_200)).T) #ymdhmsI_200[idx_200],ssha_200[idx_200],days_since_1985_200[idx_200]

else:
    mn_ssha_200 = np.empty(np.shape(mn_ssha_tg))*np.nan
    for ii in np.arange(np.shape(mn_ssha_tg)[1]):
        wgt_200,dist_200 = lTG.select_region(lat_tg[ii],lon_tg[ii],lat_200,lon_200)
        plt.figure()
        plt.plot(dist_200,wgt_200,'.',color='blue')
        plt.plot(dist_200[wgt_200==0],wgt_200[wgt_200==0],'.',color='blue')
        plt.xlabel('IS2 distance from TG [km]')
        plt.ylabel('weights')
        plt.title('UHSLC ID: '+str(uhslc_id[ii]))
        mn_ssha_200[:,ii],mn_time_200 = lTG.month_2_month(ymdhmsI_200,ssha_200,days_since_1985_200,yrs_mm,wgt=wgt_200)#,LATLON = np.vstack((lat_200,lon_200)).T) #ymdhmsI_200[idx_200],ssha_200[idx_200],days_since_1985_200[idx_200]



########### BEAMS ###########
binz = np.arange(-1,1.1,0.1)
binz0,count0,pdf0,cdf0 = lTG.hist_cdf(ssha_200,bins=binz)
binz1,count1,pdf1,cdf1 = lTG.hist_cdf(ssha_200[beam_200==1],bins=binz)
binz2,count2,pdf2,cdf2 = lTG.hist_cdf(ssha_200[beam_200==2],bins=binz)
binz3,count3,pdf3,cdf3 = lTG.hist_cdf(ssha_200[beam_200==3],bins=binz)
binz10,count10,pdf10,cdf10 = lTG.hist_cdf(ssha_200[beam_200==10],bins=binz)
binz20,count20,pdf20,cdf20 = lTG.hist_cdf(ssha_200[beam_200==20],bins=binz)
binz30,count30,pdf30,cdf30 = lTG.hist_cdf(ssha_200[beam_200==30],bins=binz)


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

if np.size(mn_time_tg)!= np.shape(mn_time_tg)[0]:
    tg_bias_200 = np.empty(np.shape(mn_time_tg)[1])*np.nan
    tg_bias_alt = np.empty(np.shape(mn_time_tg)[1])*np.nan
    mTGii = np.empty(np.shape(mn_time_tg)[1])*np.nan
    plt.figure(figsize=(16,12))
    for ii in np.arange(np.shape(FNtg)[0]):
        plt.subplot(rr,cc,ii+1)
        TGii=np.array(mn_ssha_tg[:,ii])
        mTGii[ii] = np.nanmean(TGii)
        tg_bias_200[ii] = np.nanmean(mn_ssha_200[:,ii])-np.nanmean(mn_ssha_tg[:,ii])+mTGii[ii]
        rms_200 = np.round(np.sqrt(np.nanmean(((mn_ssha_200[:,ii]-mn_ssha_tg[:,ii])-(tg_bias_200[ii]-mTGii[ii]))**2))*100.,3)
        print('tg_bias_200: '+str(np.size(tg_bias_200[ii])))
        plt.plot(mn_time_tg[:,ii],(TGii-mTGii[ii])*100.,'.-',label='TG',color=clr[0])
        shft200=np.array(mn_ssha_200[:,ii]-tg_bias_200[ii])*100.
        plt.plot(mn_time_200,shft200,'.-',label='IS2. RMS [IS2-TG] = '+str(rms_200)+' cm',color=clr[1])
        #plt.plot(mn_time_10,np.array(mn_ssha_10)*100.,'-',label='IS2 (10 m)',color='red')
        if np.size(FNalt)!=0:
            tg_bias_alt[ii] = np.nanmean(mn_ssha_alt)-np.nanmean(mn_ssha_tg[:,ii])+mTGii[ii]
            print('tg_bias_alt: '+str(np.size(tg_bias_alt[ii])))
            rms_alt = np.round(np.sqrt(np.nanmean(((mn_ssha_alt-mn_ssha_tg[:,ii])-(tg_bias_alt[ii]-mTGii[ii]))**2))*100,3)
            shftA=np.array(mn_ssha_alt-tg_bias_alt[ii])*100.
            plt.plot(mn_time_alt,shftA,'.-',label='J3. RMS [J3-TG] = '+str(rms_alt)+' cm',color=clr[2])
        plt.legend()
        plt.grid()
        plt.xlabel('year fraction')
        plt.ylabel('monthly SSHA means [cm]')
        plt.title('UHSLC ID: '+str(uhslc_id[ii]))
        plt.ylim(-40,40)
        
        
else:
    plt.figure(figsize=(8,6))
    itg = np.where((mn_time_tg>=np.nanmin(mn_time_200))&(mn_time_tg<=np.nanmax(mn_time_200)))[0]
    #tg_bias = np.nanmean(mn_ssha_tg[itg])-np.nanmean(mn_ssha_200)
    mTGii = np.nanmean(mn_ssha_tg[itg])
    tg_bias_200 = (np.nanmean(mn_ssha_200)-np.nanmean(mn_ssha_tg[itg]))+mTGii
    rms_200 = np.round(np.sqrt(np.nanmean(((mn_ssha_200[itg]-tg_bias_200)-(mn_ssha_tg[itg]-mTGii))**2))*100.,2)
    plt.plot(mn_time_tg[itg],np.array(mn_ssha_tg[itg]-mTGii)*100.,'.-',label='TG',color=clr[0])
    plt.plot(mn_time_200,np.array(mn_ssha_200-tg_bias_200)*100.,'.-',label='IS2. RMS [IS2-TG] = '+str(rms_200)+' cm',color=clr[1])
    #plt.plot(mn_time_10,np.array(mn_ssha_10)*100.,'.',label='IS2 (10 m)',color='red')
    if np.size(FNalt)!=0:
        tg_bias_alt = (np.nanmean(mn_ssha_alt)-np.nanmean(mn_ssha_tg[itg]))+mTGii
        rms_alt = np.round(np.sqrt(np.nanmean(((mn_ssha_alt[itg]-tg_bias_alt)-(mn_ssha_tg[itg]-mTGii))**2))*100,2)
        plt.plot(mn_time_alt,np.array(mn_ssha_alt-tg_bias_alt)*100.,'.-',label='J3. RMS [J3-TG] = '+str(rms_alt)+' cm',color=clr[2])
    plt.legend()
    plt.grid()
    plt.xlabel('year fraction')
    plt.ylabel('monthly SSHA means [cm]')
    plt.title('UHSLC ID: '+str(uhslc_id))    

    plt.figure(figsize=(5,3))
    itg = np.where((mn_time_tg>=np.nanmin(mn_time_200))&(mn_time_tg<=np.nanmax(mn_time_200)))[0]
    #tg_bias = np.nanmean(mn_ssha_tg[itg])-np.nanmean(mn_ssha_200)
    mTGii = np.nanmean(mn_ssha_tg[itg])
    tg_bias_200 = (np.nanmean(mn_ssha_200)-np.nanmean(mn_ssha_tg[itg]))+mTGii
    rms_200 = np.round(np.sqrt(np.nanmean(((mn_ssha_200[itg]-tg_bias_200)-(mn_ssha_tg[itg]-mTGii))**2))*100.,2)
    plt.plot(mn_time_tg[itg],np.array(mn_ssha_tg[itg]-mTGii)*100.,'.-',label='tide gauge',color='tab:orange')
    plt.plot(mn_time_200,np.array(mn_ssha_200-tg_bias_200)*100.,'.-',label='ICESat-2',color='tab:green')
    #plt.plot(mn_time_10,np.array(mn_ssha_10)*100.,'.',label='IS2 (10 m)',color='red')
    if np.size(FNalt)!=0:
        tg_bias_alt = (np.nanmean(mn_ssha_alt)-np.nanmean(mn_ssha_tg[itg]))+mTGii
        rms_alt = np.round(np.sqrt(np.nanmean(((mn_ssha_alt[itg]-tg_bias_alt)-(mn_ssha_tg[itg]-mTGii))**2))*100,2)
        plt.plot(mn_time_alt,np.array(mn_ssha_alt-tg_bias_alt)*100.,'.-',label='Jason-3',color='tab:purple')
    plt.legend()
    plt.grid()
    plt.xlabel('year fraction')
    plt.ylabel('monthly SSHA means [cm]')
    plt.title('Sea surface height anomaly comparison \n ICESat-2 vs. Jason-3 vs. tide gauge (UHSLC ID: '+str(int(uhslc_id[0]))+')')   
    
    plt.figure()
    plt.plot(dist_200,wgt_200,'.',color='blue')
    plt.plot(dist_200[wgt_200==0],wgt_200[wgt_200==0],'.',color='blue')
    plt.xlabel('IS2 distance from TG [km]')
    plt.ylabel('weights')
    plt.title('UHSLC ID: '+str(uhslc_id))  
    plt.grid()
    plt.xlim(0,50)


    plt.figure(figsize=(8,6))
    plt.plot(mn_time_200,np.array(mn_ssha_200)*100.,'.-',label='IS2. RMS [IS2-TG] = '+str(rms_200)+' cm',color=clr[1])
    if np.size(FNalt)!=0:
        plt.plot(mn_time_alt,np.array(mn_ssha_alt)*100.,'.-',label='J3. RMS [J3-TG] = '+str(rms_alt)+' cm',color=clr[2])
    plt.legend()
    plt.grid()
    plt.xlabel('year fraction')
    plt.ylabel('monthly SSHA means [cm]')
    plt.title('UHSLC ID: '+str(uhslc_id))  
    
    
    # TREND?
    ce_mn200 = np.polyfit(mn_time_200, mn_ssha_200, 1)[::-1]
    ce_200 = np.polyfit(days_since_1985_200,ssha_200, 1)[::-1]

########### GRIDDED AVERAGES ###########
mm = 50
cmap = 'RdYlGn_r' #'coolwarm'
lat_grid,lon_grid,ssha_grid = lTG.gridded(ssha_200*100,lat_200,lon_200,lat_minmax=[np.nanmin(lat_200),np.nanmax(lat_200)],lon_minmax=[np.nanmin(lon_200),np.nanmax(lon_200)])
pbil.groundtracks_contour(np.unique(lon_grid),np.unique(lat_grid),ssha_grid,'ICESat-2 gridded product ('+str(fp)+' m  footprint). '+REG+' '+str(yrs_mm),'ssha [cm]',
                  cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                  lon_fix=lon_tg,lat_fix=lat_tg,gt_fix=(np.nanmean(mn_ssha_tg,axis=0)-np.nanmean(mn_ssha_tg-mn_ssha_200,axis=0))*100.,
                  LEV=np.arange(-0.5,0.55,0.05)*100.)
pbil.groundtracks_contour(np.unique(lon_grid),np.unique(lat_grid),ssha_grid,'Gridded ICESat-2: '+str(fp)+' m  footprint averged within 0.1$^o$ lat/lon bins \n New England (2019-2021) \n black dot = tide gauge location','ssha [cm]',
                cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                lon_fix=lon_tg,lat_fix=lat_tg,gt_fix=[],
                LEV=np.arange(-0.5,0.55,0.05)*100.) #lon_fix2=np.unique(lon_gridA),lat_fix2= np.unique(lat_gridA),gt_fix2=ssha_gridA*100,

if np.size(FNalt)!=0:
    lat_gridA,lon_gridA,ssha_gridA = lTG.gridded(ssha_alt*100,lat_alt,lon_alt,lat_minmax=[np.nanmin(lat_200),np.nanmax(lat_200)],lon_minmax=[np.nanmin(lon_200),np.nanmax(lon_200)])
    ialt_10days = np.where((days_since_1985_alt-days_since_1985_alt[0])<10)[0]
    pbil.groundtracks_contour(np.unique(lon_grid),np.unique(lat_grid),ssha_grid,'Gridded ICESat-2: '+str(fp)+' m  footprint averged within 0.1$^o$ lat/lon bins \n New England (2019-2021) \n black dot = tide gauge location (UHSLC ID: '+str(int(uhslc_id[0]))+') \n grey dots = Jason-3 groundtrack','ssha [cm]',
                      cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                      lon_fix=lon_tg,lat_fix=lat_tg,gt_fix=[],
                      lon_fix2=lon_alt[ialt_10days],lat_fix2=lat_alt[ialt_10days],gt_fix2=(ssha_alt[ialt_10days])*100.,
                      LEV=np.arange(-0.5,0.55,0.05)*100.) #lon_fix2=np.unique(lon_gridA),lat_fix2= np.unique(lat_gridA),gt_fix2=ssha_gridA*100,
    pbil.groundtracks_contour(np.unique(lon_grid),np.unique(lat_grid),ssha_grid,'gridded ICESat-2 ('+str(fp)+' m  footprint). '+REG+' '+str(yrs_mm),'ssha [cm]',
                      cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                      lon_fix=lon_tg,lat_fix=lat_tg,gt_fix=(np.nanmean(mn_ssha_tg,axis=0)-np.nanmean(mn_ssha_tg-mn_ssha_200,axis=0))*100.,
                      lon_fix2=np.unique(lon_gridA),lat_fix2= np.unique(lat_gridA),gt_fix2=ssha_gridA*100,
                      LEV=np.arange(-0.5,0.55,0.05)*100.) #
    

########### SPATIAL PLOTS ###########
if np.size(FNtg)==1:
    LEG='ssha [cm] \n black outline = UHSLC ID '+str(uhslc_id)+'\n gray outline = Jason-3'
else:
    LEG='ssha [cm] \n black outline = UHSLC TGs \n gray outline = Jason-3'
mm=50

if np.size(mn_time_tg)!= np.shape(mn_time_tg)[0]:
    if np.size(FNalt)!=0:
        pbil.groundtracks_multi(lon_200,lat_200,(ssha_200)*100.,'ICESat-2 ('+str(fp)+' m  footprint) with TG and ALT. '+REG+' '+str(yrs_mm),LEG,
                           cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                           lon2=lon_tg,lat2=lat_tg,gt2=(np.nanmean(mn_ssha_tg,axis=0)-np.nanmean(mn_ssha_tg-mn_ssha_200,axis=0))*100.,
                           lon3=lon_alt,lat3=lat_alt,gt3=(ssha_alt)*100.)
    else:
        pbil.groundtracks_multi(lon_200,lat_200,ssha_200*100.,'ICESat-2 ('+str(fp)+' m  footprint) with TG and ALT. '+REG+' '+str(yrs_mm),LEG,
                           cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                           lon2=lon_tg,lat2=lat_tg,gt2=(np.nanmean(mn_ssha_tg,axis=0)-np.nanmean(mn_ssha_tg-mn_ssha_200,axis=0))*100.)
else:
    if np.size(FNalt)!=0:
        pbil.groundtracks_multi(lon_200,lat_200,(ssha_200)*100.,'ICESat-2 ('+str(fp)+' m  footprint) with TG and ALT. '+REG+' '+str(yrs_mm),LEG,
                           cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                           lon2=lon_tg,lat2=lat_tg,gt2=(np.nanmean(mn_ssha_tg)-np.nanmean(mn_ssha_tg-mn_ssha_200))*100.,
                           lon3=lon_alt,lat3=lat_alt,gt3=(ssha_alt)*100.)
    else:
        pbil.groundtracks_multi(lon_200,lat_200,ssha_200*100.,'ICESat-2 ('+str(fp)+' m  footprint) with TG and ALT. '+REG+' '+str(yrs_mm),LEG,
                           cm=cmap,vmin=-mm,vmax=mm,FN=[],proj=180.,fc='0.1',
                           lon2=lon_tg,lat2=lat_tg,gt2=(np.nanmean(mn_ssha_tg)-np.nanmean(mn_ssha_tg-mn_ssha_200))*100.)


'''
if np.size(mn_time_tg)!= np.shape(mn_time_tg)[0]:
    tg_bias_200 = np.empty(np.shape(mn_time_tg)[1])*np.nan
    tg_bias_alt = np.empty(np.shape(mn_time_tg)[1])*np.nan
    mTGii = np.empty(np.shape(mn_time_tg)[1])*np.nan
    plt.figure(figsize=(16,12))
    for ii in np.arange(np.shape(FNtg)[0]):
        plt.subplot(rr,cc,ii+1)
        TGii=np.array(mn_ssha_tg[:,ii])
        mTGii[ii] = np.nanmean(TGii)
        tg_bias_200[ii] = np.nanmean(mn_ssha_200[:,ii])-np.nanmean(mn_ssha_tg[:,ii])+mTGii[ii]
        rms_200 = np.round(np.sqrt(np.nanmean(((mn_ssha_200[:,ii]-mn_ssha_tg[:,ii])-(tg_bias_200[ii]-mTGii[ii]))**2))*100.,3)
        print('tg_bias_200: '+str(np.size(tg_bias_200[ii])))
        plt.plot(mn_time_tg[:,ii],(TGii-mTGii[ii])*100.,'.-',label='TG',color=clr[0])
        shft200=np.array(mn_ssha_200[:,ii]-tg_bias_200[ii])*100.
        plt.plot(mn_time_200,shft200,'.-',label='IS2. RMS [IS2-TG] = '+str(rms_200)+' cm',color=clr[1])
        #plt.plot(mn_time_10,np.array(mn_ssha_10)*100.,'-',label='IS2 (10 m)',color='red')
        if np.size(FNalt)!=0:
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
    itg = np.where((mn_time_tg>=np.nanmin(mn_time_200))&(mn_time_tg<=np.nanmax(mn_time_200)))[0]
    #tg_bias = np.nanmean(mn_ssha_tg[itg])-np.nanmean(mn_ssha_200)
    tg_bias_200 = np.nanmean(mn_ssha_200)-np.nanmean(mn_ssha_tg[itg])
    rms_200 = np.round(np.sqrt(np.nanmean((mn_ssha_200-tg_bias_200-mn_ssha_tg)**2))*100.,2)
    plt.plot(mn_time_tg[itg],np.array(mn_ssha_tg[itg])*100.,'.-',label='TG',color=clr[0])
    plt.plot(mn_time_200,np.array(mn_ssha_200-tg_bias_200)*100.,'.-',label='IS2. RMS [IS2-TG] = '+str(rms_200)+' cm',color=clr[1])
    #plt.plot(mn_time_10,np.array(mn_ssha_10)*100.,'.',label='IS2 (10 m)',color='red')
    if np.size(FNalt)!=0:
        tg_bias_alt = np.nanmean(mn_ssha_alt)-np.nanmean(mn_ssha_tg[itg])
        rms_alt = np.round(np.sqrt(np.nanmean((mn_ssha_alt-tg_bias_alt-mn_ssha_tg)**2))*100,2)
        plt.plot(mn_time_alt,np.array(mn_ssha_alt-tg_bias_alt)*100.,'.-',label='J3. RMS [J3-TG] = '+str(rms_alt)+' cm',color=clr[2])
    plt.legend()
    plt.grid()
    plt.xlabel('time [days since 1985]')
    plt.ylabel('monthly SSHA means [cm]')
    plt.title('UHSLC ID: '+str(uhslc_id))    
'''
