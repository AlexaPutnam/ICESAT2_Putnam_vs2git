#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:33:45 2022

@author: alexaputnam
"""

# TIDE GAUGE DATA:
# http://uhslc.soest.hawaii.edu/data/?rq#uh233

def regions(REG):
    #Bounding box: LR = lower right [lat min,lon max], UL = upper left [lat max,lon min]
    if REG == 'lameshur': # Lameshur, USVI
        LR = [18.292905, -64.717636]
        UL = [18.321084, -64.731934]
    elif REG == 'kodiak': # Kodiak, AK
        LR = [57.693376, -152.488525]
        UL = [57.741260, -152.570486]
    elif REG == 'wake': # Wake Island, USA
        LR = [19.209380, 166.704773]
        UL = [19.377347, 166.533330]
    elif REG == 'boston': # Boston, MA, USA 
        LR = [42.302781, -70.946857]
        UL = [42.414756, -71.125643]
    elif REG == 'woodshole': # Woods Hole, MA, USA
        LR = [41.501698, -70.652774]
        UL = [41.537043, -70.699122]
    elif REG == 'naples': # Naples, FL, USA
        LR = [26.095475, -81.796599]
        UL = [26.168217, -82.097349]
    elif REG == 'newfoundland': #Newfoundland, CAN (cry2ice)
        LR = [41.6591649,-51.3518667]
        UL = [50.2158105,-58.5058748]
    elif REG =='alaska': # South Alaska 
        LR = [49.75745, -127.08233]
        UL = [61.75568, 175.43719]
    elif REG =='gulf_of_alaska': # Gulf of Alaska
        LR = [53.99036, -130.3635]
        UL = [61.54012, -155.19254]
    elif REG =='newengland': #CT, RI, NY, MA
        LR = [40.89754, -69.67782]
        UL = [41.76374, -72.56722]
    elif REG =='lagos': #CT, RI, NY, MA
        LR = [6.23, 4.0] #6,4
        UL = [6.7, 3.0]
    elif REG =='mumbai': #CT, RI, NY, MA
        LR = [18, 73.5]
        UL = [20,71.5]
    elif REG =='hawaii': #Nawiliwili, Honolulu, Kahului, Kawaihae, Hilo
        LR = [19.20, -154.37]
        UL = [22.73,-159.66]
    elif REG == 'antarctica':
        LR = [-64.27812, -55.04667]
        UL = [-61.36965, -61.50573]
    elif REG == 'greenland':
        LR = [60.01806, -45.46835]
        UL = [61.03663, -47.6657]
    elif REG == 'norway':
        LR = [69.28673, 16.25908]
        UL = [69.35002, 16.05309]
    elif REG == 'japan':
        LR = [29.45117, 131.89127]
        UL = [32.9623, 129.47403]
    elif REG == 'french_antarctic':
        LR = [-49.96584, 71.21502]
        UL = [-48.6714, 69.61139]
    elif REG =='ittoqqortoormiit_winter':
        LR = [69.19208, -19.77286]
        UL = [71.88207, -26.27677]
    elif REG =='ittoqqortoormiit_summer':
        LR = [69.19208, -19.77286]
        UL = [71.88207, -26.27677]
    elif REG =='ittoqqortoormiit':
        LR = [69.19208, -19.77286]
        UL = [71.88207, -26.27677]
    elif REG =='north_atlantic':
        LR = [64.65765, -3.32294]
        UL = [66.0591, -7.83891]
    elif REG=='gom':
        LR = [23,-83]
        UL = [28,-88]
    elif REG=='benghazi':
        LR = [31.56024, 20.45346]
        UL = [32.73259, 19.09114]
    elif REG=='hunga_tonga':
        LR = [-21.62482, -174.47484]
        UL = [-19.56916, -176.08076]
    elif REG == 'bowman_island':
        LR = [-66.3306, 103.98283]
        UL = [-64.60664, 101.74793]
    elif REG == 'carolina':
        LR = [33.12028, -77.51624]
        UL = [34.71912, -79.09827]
    '''
    elif REG == '':
        LR = 
        UL = 
    '''
    lat_min = LR[0]
    lat_max = UL[0]
    lon_min = UL[1]
    lon_max = LR[1]
    return lat_min,lat_max,lon_min,lon_max
