#!/usr/bin/env python
'''
The following functions have been copied from pyTMD and edited specifically for FES2014 and to speed things up a bit.
Editing was carried out by Alexa Putnam from the University of Colorado Boulder.
'''
import numpy as np
import datetime
import os
import netCDF4
import scipy
from astropy.time import Time

import lib_fes2014_tide_pyTMD as ltide


#### pyTMD functions
def constants(ellipsoid='WGS84', units='MKS'):
    # https://github.com/tsutterley/pyTMD/blob/main/pyTMD/constants.py
    if ellipsoid=='WGS84':
        a = 6378137.0# [m] semimajor axis of the ellipsoid
        f = 1.0/298.257223563# flattening of the ellipsoid
    elif ellipsoid=='TOPEX':
        a = 6378136.3# [m] semimajor axis of the ellipsoid
        f = 1.0/298.257# flattening of the ellipsoid
        GM = 3.986004415e14# [m^3/s^2]
    # universal gravitational constant [N*m^2/kg^2]
    G = 6.67430e-11
    # convert units to CGS
    if units == 'CGS':
        a *= 100.0
        GM *= 1e6
        G *= 1000.0 # [dyn*cm^2/g^2]
    return a,f

def to_cartesian(lon, lat, h=0.0):
    """
    https://github.com/tsutterley/pyTMD/blob/main/pyTMD/spatial.py
    Converts geodetic coordinates to Cartesian coordinates
    """
    a_axis,flat=constants(ellipsoid='WGS84', units='MKS')
    # verify axes
    lon = np.atleast_1d(lon)
    lat = np.atleast_1d(lat)
    # fix coordinates to be 0:360
    lon[lon < 0] += 360.0
    # Linear eccentricity and first numerical eccentricity
    lin_ecc = np.sqrt((2.0*flat - flat**2)*a_axis**2)
    ecc1 = lin_ecc/a_axis
    # convert from geodetic latitude to geocentric latitude
    dtr = np.pi/180.0
    # geodetic latitude in radians
    latitude_geodetic_rad = lat*dtr
    # prime vertical radius of curvature
    N = a_axis/np.sqrt(1.0 - ecc1**2.0*np.sin(latitude_geodetic_rad)**2.0)
    # calculate X, Y and Z from geodetic latitude and longitude
    X = (N + h) * np.cos(latitude_geodetic_rad) * np.cos(lon*dtr)
    Y = (N + h) * np.cos(latitude_geodetic_rad) * np.sin(lon*dtr)
    Z = (N * (1.0 - ecc1**2.0) + h) * np.sin(latitude_geodetic_rad)
    # return the cartesian coordinates
    return (X,Y,Z)

# PURPOSE: Extend a longitude array
def spline(ilon, ilat, idata, lon, lat,
           fill_value=None,
           dtype=np.float64,
           reducer=np.ceil,
           **kwargs):
    """
    https://github.com/tsutterley/pyTMD/blob/main/pyTMD/interpolate.py
    `Bivariate spline interpolation
    <https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.interpolate.RectBivariateSpline.html>`_
    of input data to output coordinates
    Parameters
    """
    # set default keyword arguments
    kwargs.setdefault('kx', 1)
    kwargs.setdefault('ky', 1)
    # verify that input data is masked array
    if not isinstance(idata, np.ma.MaskedArray):
        idata = np.ma.array(idata)
        idata.mask = np.zeros_like(idata, dtype=bool)
    # interpolate gridded data values to data
    npts = len(lon)
    # allocate to output interpolated data array
    data = np.ma.zeros((npts), dtype=dtype, fill_value=fill_value)
    data.mask = np.ones((npts), dtype=bool)
    # construct splines for input data and mask
    if np.iscomplexobj(idata):
        s1 = scipy.interpolate.RectBivariateSpline(ilon, ilat,
            idata.data.real.T, **kwargs)
        s2 = scipy.interpolate.RectBivariateSpline(ilon, ilat,
            idata.data.imag.T, **kwargs)
        s3 = scipy.interpolate.RectBivariateSpline(ilon, ilat,
            idata.mask.T, **kwargs)
        # evaluate the spline at input coordinates
        data.data.real[:] = s1.ev(lon, lat)
        data.data.imag[:] = s2.ev(lon, lat)
        data.mask[:] = reducer(s3.ev(lon, lat)).astype(bool)
    else:
        s1 = scipy.interpolate.RectBivariateSpline(ilon, ilat,
            idata.data.T, **kwargs)
        s2 = scipy.interpolate.RectBivariateSpline(ilon, ilat,
            idata.mask.T, **kwargs)
        # evaluate the spline at input coordinates
        data.data[:] = s1.ev(lon, lat).astype(dtype)
        data.mask[:] = reducer(s2.ev(lon, lat)).astype(bool)
    # return interpolated values
    return data

def regulargrid(ilon, ilat, idata, lon, lat,
                fill_value=None,
                dtype=np.float64,
                reducer=np.ceil,
                **kwargs):
    """
    https://github.com/tsutterley/pyTMD/blob/main/pyTMD/interpolate.py
    `Regular grid interpolation
    <https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.interpolate.RegularGridInterpolator.html>`_
    of input data to output coordinates
    method: str, default 'linear'
        Method of interpolation
            - ``'linear'``
            - ``'nearest'``
            - ``'slinear'``
            - ``'cubic'``
            - ``'quintic'``    """
    # set default keyword arguments
    kwargs.setdefault('bounds_error', False)
    kwargs.setdefault('method', 'linear')
    # verify that input data is masked array
    if not isinstance(idata, np.ma.MaskedArray):
        idata = np.ma.array(idata)
        idata.mask = np.zeros_like(idata, dtype=bool)
    # interpolate gridded data values to data
    npts = len(lon)
    # allocate to output interpolated data array
    data = np.ma.zeros((npts), dtype=dtype, fill_value=fill_value)
    data.mask = np.ones((npts), dtype=bool)
    # use scipy regular grid to interpolate values for a given method
    r1 = scipy.interpolate.RegularGridInterpolator((ilat, ilon),
        idata.data, fill_value=fill_value, **kwargs)
    r2 = scipy.interpolate.RegularGridInterpolator((ilat, ilon),
        idata.mask, fill_value=1, **kwargs)
    # evaluate the interpolator at input coordinates
    data.data[:] = r1.__call__(np.c_[lat, lon])
    data.mask[:] = reducer(r2.__call__(np.c_[lat, lon])).astype(bool)
    # return interpolated values
    return data

# PURPOSE: Nearest-neighbor extrapolation of valid data to output data
def extrapolate(ilon, ilat, idata, lon, lat,
                fill_value=np.nan,
                dtype=np.float64,
                cutoff=np.inf,
                EPSG='4326'):
    """
    https://github.com/tsutterley/pyTMD/blob/main/pyTMD/interpolate.py
    Nearest-neighbor (`NN`) extrapolation of valid model data using `kd-trees
    <https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.spatial.cKDTree.html>`_
    """
    # verify output dimensions
    lon = np.atleast_1d(lon)
    lat = np.atleast_1d(lat)
    # extrapolate valid data values to data
    npts = len(lon)
    # return none if no invalid points
    if (npts == 0):
        return

    # allocate to output extrapolate data array
    data = np.ma.zeros((npts), dtype=dtype, fill_value=fill_value)
    data.mask = np.ones((npts), dtype=bool)
    # initially set all data to fill value
    data.data[:] = idata.fill_value

    # create combined valid mask
    valid_mask = (~idata.mask) & np.isfinite(idata.data)
    # reduce to model points within bounds of input points
    valid_bounds = np.ones_like(idata.mask, dtype=bool)

    # calculate coordinates for nearest-neighbors
    if (EPSG == '4326'):
        # global or regional equirectangular model
        # calculate meshgrid of model coordinates
        gridlon, gridlat = np.meshgrid(ilon, ilat)
        # ellipsoidal major axis in kilometers
        a_axis = 6378.137
        # calculate Cartesian coordinates of input grid
        gridx, gridy, gridz = to_cartesian(
            gridlon, gridlat, a_axis=a_axis)
        # calculate Cartesian coordinates of output coordinates
        xs, ys, zs = to_cartesian(
            lon, lat, a_axis=a_axis)
        # range of output points in cartesian coordinates
        xmin, xmax = (np.min(xs), np.max(xs))
        ymin, ymax = (np.min(ys), np.max(ys))
        zmin, zmax = (np.min(zs), np.max(zs))
        # reduce to model points within bounds of input points
        valid_bounds = np.ones_like(idata.mask, dtype=bool)
        valid_bounds &= (gridx >= (xmin - 2.0*cutoff))
        valid_bounds &= (gridx <= (xmax + 2.0*cutoff))
        valid_bounds &= (gridy >= (ymin - 2.0*cutoff))
        valid_bounds &= (gridy <= (ymax + 2.0*cutoff))
        valid_bounds &= (gridz >= (zmin - 2.0*cutoff))
        valid_bounds &= (gridz <= (zmax + 2.0*cutoff))
        # check if there are any valid points within the input bounds
        if not np.any(valid_mask & valid_bounds):
            # return filled masked array
            return data
        # find where input grid is valid and close to output points
        indy, indx = np.nonzero(valid_mask & valid_bounds)
        # create KD-tree of valid points
        tree = scipy.spatial.cKDTree(np.c_[gridx[indy, indx],
            gridy[indy, indx], gridz[indy, indx]])
        # flattened valid data array
        flattened = idata.data[indy, indx]
        # output coordinates
        points = np.c_[xs, ys, zs]
    else:
        # projected model
        # calculate meshgrid of model coordinates
        gridx, gridy = np.meshgrid(ilon, ilat)
        # range of output points
        xmin, xmax = (np.min(lon), np.max(lon))
        ymin, ymax = (np.min(lat), np.max(lat))
        # reduce to model points within bounds of input points
        valid_bounds = np.ones_like(idata.mask, dtype=bool)
        valid_bounds &= (gridx >= (xmin - 2.0*cutoff))
        valid_bounds &= (gridx <= (xmax + 2.0*cutoff))
        valid_bounds &= (gridy >= (ymin - 2.0*cutoff))
        valid_bounds &= (gridy <= (ymax + 2.0*cutoff))
        # check if there are any valid points within the input bounds
        if not np.any(valid_mask & valid_bounds):
            # return filled masked array
            return data
        # find where input grid is valid and close to output points
        indy, indx = np.nonzero(valid_mask & valid_bounds)
        # flattened model coordinates
        tree = scipy.spatial.cKDTree(np.c_[gridx[indy, indx],
            gridy[indy, indx]])
        # flattened valid data array
        flattened = idata.data[indy, indx]
        # output coordinates
        points = np.c_[lon, lat]

    # query output data points and find nearest neighbor within cutoff
    dd, ii = tree.query(points, k=1, distance_upper_bound=cutoff)
    # spatially extrapolate using nearest neighbors
    if np.any(np.isfinite(dd)):
        ind, = np.nonzero(np.isfinite(dd))
        data.data[ind] = flattened[ii[ind]]
        data.mask[ind] = False
    # return extrapolated values
    return data


def bilinear(ilon, ilat, idata, lon, lat,
             fill_value=np.nan,
             dtype=np.float64):
    """
    https://github.com/tsutterley/pyTMD/blob/main/pyTMD/interpolate.py
    Bilinear interpolation of input data to output coordinates
    """
    # verify that input data is masked array
    if not isinstance(idata, np.ma.MaskedArray):
        idata = np.ma.array(idata)
        idata.mask = np.zeros_like(idata, dtype=bool)
    # find valid points (within bounds)
    valid, = np.nonzero((lon >= ilon.min()) & (lon <= ilon.max()) &
        (lat > ilat.min()) & (lat < ilat.max()))
    # interpolate gridded data values to data
    npts = len(lon)
    # allocate to output interpolated data array
    data = np.ma.zeros((npts), dtype=dtype, fill_value=fill_value)
    data.mask = np.ones((npts), dtype=bool)
    # initially set all data to fill value
    data.data[:] = data.fill_value
    # for each valid point
    for i in valid:
        # calculating the indices for the original grid
        ix, = np.nonzero((ilon[0:-1] <= lon[i]) & (ilon[1:] > lon[i]))
        iy, = np.nonzero((ilat[0:-1] <= lat[i]) & (ilat[1:] > lat[i]))
        # corner data values for adjacent grid cells
        IM = np.ma.zeros((4), fill_value=fill_value, dtype=dtype)
        IM.mask = np.ones((4), dtype=bool)
        # corner weight values for adjacent grid cells
        WM = np.zeros((4))
        # build data and weight arrays
        for j,XI,YI in zip([0,1,2,3],[ix,ix+1,ix,ix+1],[iy,iy,iy+1,iy+1]):
            IM.data[j], = idata.data[YI,XI].astype(dtype)
            IM.mask[j], = idata.mask[YI,XI]
            WM[3-j], = np.abs(lon[i]-ilon[XI])*np.abs(lat[i]-ilat[YI])
        # if on corner value: use exact
        if (np.isclose(lat[i],ilat[iy]) & np.isclose(lon[i],ilon[ix])):
            data.data[i] = idata.data[iy,ix].astype(dtype)
            data.mask[i] = idata.mask[iy,ix]
        elif (np.isclose(lat[i],ilat[iy+1]) & np.isclose(lon[i],ilon[ix])):
            data.data[i] = idata.data[iy+1,ix].astype(dtype)
            data.mask[i] = idata.mask[iy+1,ix]
        elif (np.isclose(lat[i],ilat[iy]) & np.isclose(lon[i],ilon[ix+1])):
            data.data[i] = idata.data[iy,ix+1].astype(dtype)
            data.mask[i] = idata.mask[iy,ix+1]
        elif (np.isclose(lat[i],ilat[iy+1]) & np.isclose(lon[i],ilon[ix+1])):
            data.data[i] = idata.data[iy+1,ix+1].astype(dtype)
            data.mask[i] = idata.mask[iy+1,ix+1]
        elif np.any(np.isfinite(IM) & (~IM.mask)):
            # find valid indices for data summation and weight matrix
            ii, = np.nonzero(np.isfinite(IM) & (~IM.mask))
            # calculate interpolated value for i
            data.data[i] = np.sum(WM[ii]*IM[ii])/np.sum(WM[ii])
            data.mask[i] = np.all(IM.mask[ii])
    # return interpolated values
    return data

def extend_array(input_array, step_size):
    """
    location: https://github.com/tsutterley/pyTMD/blob/main/pyTMD/io/FES.py
    Extends a longitude array
    """
    n = len(input_array)
    temp = np.zeros((n+2), dtype=input_array.dtype)
    # extended array [x-1,x0,...,xN,xN+1]
    temp[0] = input_array[0] - step_size
    temp[1:-1] = input_array[:]
    temp[-1] = input_array[-1] + step_size
    return temp

# PURPOSE: Extend a global matrix
def extend_matrix(input_matrix):
    """
    location: https://github.com/tsutterley/pyTMD/blob/main/pyTMD/io/FES.py
    Extends a global matrix
    """
    ny, nx = np.shape(input_matrix)
    temp = np.ma.zeros((ny,nx+2), dtype=input_matrix.dtype)
    temp[:,0] = input_matrix[:,-1]
    temp[:,1:-1] = input_matrix[:,:]
    temp[:,-1] = input_matrix[:,0]
    return temp

def reduce_model(lon,lat,ilon,ilat):
    #idx_lat,idx_lon,idx = reduce_model(lon,lat,ilon,ilat)
    buff = 5
    llmm = [np.nanmin(ilat),np.nanmax(ilat),np.nanmin(ilon),np.nanmax(ilon)]
    idx_lat = np.where((lat>=llmm[0]-buff)&(lat<=llmm[1]+buff))[0]
    idx_lon = np.where((lon>=llmm[2]-buff)&(lon<=llmm[3]+buff))[0]

    if np.nanmax(idx_lat)<np.size(lat) and np.nanmax(idx_lon)<np.size(lon):
        idx = [np.nanmin(idx_lat),np.nanmax(idx_lat)+1,np.nanmin(idx_lon),np.nanmax(idx_lon)+1]
    elif np.nanmax(idx_lat)<np.size(lat) and np.nanmax(idx_lon)==np.size(lon):
        idx = [np.nanmin(idx_lat),np.nanmax(idx_lat)+1,np.nanmin(idx_lon),np.nanmax(idx_lon)]
    elif np.nanmax(idx_lat)==np.size(lat) and np.nanmax(idx_lon)<np.size(lon):
        idx = [np.nanmin(idx_lat),np.nanmax(idx_lat),np.nanmin(idx_lon),np.nanmax(idx_lon)+1]
    elif np.nanmax(idx_lat)==np.size(lat) and np.nanmax(idx_lon)==np.size(lon):
        idx = [np.nanmin(idx_lat),np.nanmax(idx_lat),np.nanmin(idx_lon),np.nanmax(idx_lon)]
    return idx_lat,idx_lon,idx

def read_netcdf_file(input_file,ilon,ilat,type):
    """
    location: https://github.com/tsutterley/pyTMD/blob/main/pyTMD/io/FES.py
    Read FES (Finite Element Solution) tide model netCDF4 file
    """
    # read the netcdf format tide elevation file
    fileID = netCDF4.Dataset(os.path.expanduser(input_file), 'r')
    # variable dimensions for each model
    lon = fileID.variables['lon'][:]
    lat = fileID.variables['lat'][:]
    ilon=adj_lon(lon,ilon)
    # grid step size of tide model
    dlon = lon[1] - lon[0]
    # amplitude and phase components for each type
    if (type == 'z'):
        amp_key = 'amplitude'
        phase_key = 'phase'
    elif (type == 'u'):
        amp_key = 'Ua'
        phase_key = 'Ug'
    elif (type == 'v'):
        amp_key = 'Va'
        phase_key = 'Vg'
    # reduce grid size to speed up function
    idx_lat,idx_lon,idx = reduce_model(lon,lat,ilon,ilat)
    lon = np.copy(lon[idx_lon])
    lat = np.copy(lat[idx_lat])
    # get amplitude and phase components
    amp = fileID.variables[amp_key][:][idx[0]:idx[1],idx[2]:idx[3]]
    ph = fileID.variables[phase_key][:][idx[0]:idx[1],idx[2]:idx[3]]
    # close the file
    fileID.close()
    # calculate complex form of constituent oscillation
    mask = (amp.data == amp.fill_value) | \
        (ph.data == ph.fill_value) | \
        np.isnan(amp.data) | np.isnan(ph.data)
    hc = np.ma.array(amp*np.exp(-1j*ph*np.pi/180.0), mask=mask,
        fill_value=np.ma.default_fill_value(np.dtype(complex)))
    # return output variables
    if np.size(lat)!=np.shape(hc)[0]:
        raise('reduced section does not match along latitude')
    if np.size(lon)!=np.shape(hc)[1]:
        raise('reduced section does not match along latitude')
    return hc, lon, lat, ilon, dlon

def adj_lon(lon_model,lon_data):
    # adjust longitudinal convention of input latitude and longitude
    # to fit tide model convention
    if (np.min(lon_data) < 0.0) & (np.max(lon_model) > 180.0):
        # input points convention (-180:180)
        # tide model convention (0:360)
        lon_data[lon_data<0.0] += 360.0
    elif (np.max(lon_data) > 180.0) & (np.min(lon_model) < 0.0):
        # input points convention (0:360)
        # tide model convention (-180:180)
        lon_data[lon_data>180.0] -= 360.0
    return lon_data

def extract_constants(ilon,ilat,method,model_files=None,extrapolate=False,cutoff=10.0,scale=1.0/100.0,type='z'):
    """
    HEAVY EDITING TO THIS FUNCTION TO SPEED THINGS UP (10X FASTER NOW)
    total time for 35759 points: 6.3 min using edited model
    location: https://github.com/tsutterley/pyTMD/blob/main/pyTMD/io/FES.py
    Reads files for a FES ascii or netCDF4 tidal model
    Makes initial calculations to run the tide program
    Spatially interpolates tidal constituents to input coordinates
    """
    # adjust dimensions of input coordinates to be iterable
    ilon = np.atleast_1d(np.copy(ilon))
    ilat = np.atleast_1d(np.copy(ilat))
    # number of points
    npts = len(ilon)
    # number of constituents
    nc = len(model_files)

    # amplitude and phase
    amplitude = np.ma.zeros((npts,nc))
    amplitude.mask = np.zeros((npts,nc),dtype=bool)
    ph = np.ma.zeros((npts,nc))
    ph.mask = np.zeros((npts,nc),dtype=bool)
    # read and interpolate each constituent
    for i, fi in enumerate(model_files):
        # check that model file is accessible
        if not os.access(os.path.expanduser(fi), os.F_OK):
            raise FileNotFoundError(os.path.expanduser(fi))
        # read constituent from elevation file
        # FES netCDF4 constituent files for 'FES2012','FES2014','EOT20' only #!!!
        hc,lon,lat,ilon,dlon = read_netcdf_file(os.path.expanduser(fi),ilon,ilat,type=type)
        # replace original values with extend arrays/matrices
        if np.isclose(lon[-1] - lon[0], 360.0 - dlon):
            lon = extend_array(lon, dlon)
            hc = extend_matrix(hc)
        # determine if any input points are outside of the model bounds
        invalid = (ilon < lon.min()) | (ilon > lon.max()) | \
                  (ilat < lat.min()) | (ilat > lat.max())
        # interpolate amplitude and phase of the constituent
        if (method == 'bilinear'):
            # replace invalid values with nan
            hc.data[hc.mask] = np.nan
            # use quick bilinear to interpolate values
            hci = bilinear(lon, lat, hc, ilon, ilat,
                dtype=hc.dtype)
            # replace nan values with fill_value
            hci.mask[:] |= np.isnan(hci.data)
            hci.data[hci.mask] = hci.fill_value
        elif (method == 'spline'):
            # interpolate complex form of the constituent
            # use scipy splines to interpolate values
            hci = spline(lon, lat, hc, ilon, ilat,
                dtype=hc.dtype,
                reducer=np.ceil,
                kx=1, ky=1)
            # replace invalid values with fill_value
            hci.data[hci.mask] = hci.fill_value
        else:
            # interpolate complex form of the constituent
            # use scipy regular grid to interpolate values
            hci = regulargrid(lon, lat, hc, ilon, ilat,
                dtype=hc.dtype,
                method=method,
                reducer=np.ceil,
                bounds_error=False)
            # replace invalid values with fill_value
            hci.mask[:] |= (hci.data == hci.fill_value)
            hci.data[hci.mask] = hci.fill_value
        # extrapolate data using nearest-neighbors
        if extrapolate and np.any(hci.mask):
            # find invalid data points
            inv, = np.nonzero(hci.mask)
            # replace invalid values with nan
            hc.data[hc.mask] = np.nan
            # extrapolate points within cutoff of valid model points
            hci[inv] = extrapolate(lon, lat, hc,
                ilon[inv], ilat[inv], dtype=hc.dtype,
                cutoff=cutoff)
        # convert amplitude from input units to meters
        amplitude.data[:,i] = np.abs(hci.data)*scale
        amplitude.mask[:,i] = np.copy(hci.mask)
        # phase of the constituent in radians
        ph.data[:,i] = np.arctan2(-np.imag(hci.data),np.real(hci.data))
        ph.mask[:,i] = np.copy(hci.mask)
        # update mask to invalidate points outside model domain
        amplitude.mask[:,i] |= invalid
        ph.mask[:,i] |= invalid

    # convert phase to degrees
    phase = ph*180.0/np.pi
    phase.data[phase.data < 0] += 360.0
    # replace data for invalid mask values
    amplitude.data[amplitude.mask] = amplitude.fill_value
    phase.data[phase.mask] = phase.fill_value
    # typically returns the interpolated values: (amplitude, phase)
    cph = -1j*phase*np.pi/180.0
    hc = amplitude*np.exp(cph)
    return hc # (amplitude, phase)


# PURPOSE: calculate the delta time from calendar date
# http://scienceworld.wolfram.com/astronomy/JulianDate.html

def convert_calendar_dates(year, month, day, hour=0.0, minute=0.0, second=0.0,
    epoch=(1992, 1, 1, 0, 0, 0), scale=1.0):
    """
    https://github.com/tsutterley/pyTMD/blob/main/pyTMD/time.py
    Calculate the time in time units (days) since ``epoch`` from calendar dates
    """
    # calculate date in Modified Julian Days (MJD) from calendar date
    # MJD: days since November 17, 1858 (1858-11-17T00:00:00)
    MJD = 367.0*year - np.floor(7.0*(year + np.floor((month+9.0)/12.0))/4.0) - \
        np.floor(3.0*(np.floor((year + (month - 9.0)/7.0)/100.0) + 1.0)/4.0) + \
        np.floor(275.0*month/9.0) + day + hour/24.0 + minute/1440.0 + \
        second/86400.0 + 1721028.5 - 2400000.5
    epoch1 = datetime.datetime(1858, 11, 17, 0, 0, 0)
    epoch2 = datetime.datetime(*epoch)
    delta_time_epochs = (epoch2 - epoch1).total_seconds()
    # return the date in days since epoch
    return scale*np.array(MJD - delta_time_epochs/86400.0, dtype=np.float64) 


def time_series(t, hc, constituents, deltat=0.0, corrections='OTIS'):
    """
    # https://github.com/tsutterley/pyTMD/blob/main/pyTMD/predict_tidal_ts.py
    # https://github.com/tsutterley/pyTMD/blob/main/pyTMD/predict.py
    Predict tidal time series at a single location using harmonic constants
    Parameters
    ----------
    t: float
        days relative to 1992-01-01T00:00:00
    hc: complex
        harmonic constant vector
    constituents: list
        tidal constituent IDs
    deltat: float, default 0.0
        time correction for converting to Ephemeris Time (days)
    corrections: str, default ''
        use nodal corrections from OTIS/ATLAS or GOT/FES models
    Returns
    -------
    ht: float
        tidal time series reconstructed using the nodal corrections
    References
    ----------
    .. [1] Egbert and Erofeeva, "Efficient Inverse Modeling of Barotropic
        Ocean Tides," *Journal of Atmospheric and Oceanic Technology*,
        19(2), 183--204, (2002).
        `doi: 10.1175/1520-0426(2002)019<0183:EIMOBO>2.0.CO;2`__
    .. __: https://doi.org/10.1175/1520-0426(2002)019<0183:EIMOBO>2.0.CO;2
    """
    nt = len(t)
    # load the nodal corrections
    # convert time to Modified Julian Days (MJD)
    pu,pf,G = ltide.load_nodal_corrections(t + 48622.0, constituents,
        deltat=deltat, corrections=corrections)
    # allocate for output time series
    ht = np.ma.zeros((nt))
    ht.mask = np.zeros((nt),dtype=bool)
    # for each constituent
    for k,c in enumerate(constituents):
        th = G[:,k]*np.pi/180.0 + pu[:,k]
        # sum over all tides at location
        ht.data[:] += pf[:,k]*hc.real[0,k]*np.cos(th) - \
            pf[:,k]*hc.imag[0,k]*np.sin(th)
        ht.mask[:] |= np.any(hc.real.mask[0,k] | hc.imag.mask[0,k])
    # return the tidal time series
    return ht

# PURPOSE: infer the minor corrections from the major constituents


def pull_model(LOAD):
    # model_dir,MOD_NAME,LOAD=model_truth,'FES2014',False
    NC={}
    m_pth = '/Users/alexaputnam/ICESat2/fes2014/'
    NC['type'] = 'z'
    NC['pth'] = m_pth
    NC['format'] = 'FES'
    mylist = ['2n2.nc','eps2.nc','j1.nc','k1.nc',
                'k2.nc','l2.nc','la2.nc','m2.nc','m3.nc','m4.nc',
                'm6.nc','m8.nc','mf.nc','mks2.nc','mm.nc',
                'mn4.nc','ms4.nc','msf.nc','msqm.nc','mtm.nc',
                'mu2.nc','n2.nc','n4.nc','nu2.nc','o1.nc','p1.nc',
                'q1.nc','r2.nc','s1.nc','s2.nc','s4.nc','sa.nc',
                'ssa.nc','t2.nc']
    NC['model_files'] = mylist
    NC['constituents'] = ['2n2','eps2','j1','k1','k2','l2',
                'lambda2','m2','m3','m4','m6','m8','mf','mks2','mm',
                'mn4','ms4','msf','msqm','mtm','mu2','n2','n4','nu2',
                'o1','p1','q1','r2','s1','s2','s4','sa','ssa','t2']
    NC['scale'] = 1.0/100.0
    NC['version'] = 'FES2014'
    if LOAD == False:
        NC['model_directory'] = m_pth+'/ocean_tide/'
    else:
        NC['model_directory'] = m_pth+'/load_tide/'
    newlist = [NC['model_directory'] + s for s in mylist]
    NC['model_file'] = newlist
    return NC
    

def dt_utc_tt(utc_time):
    N = np.shape(utc_time)[0]
    dt = np.empty(N)*np.nan
    tide_time = np.empty(N)*np.nan
    for ii in np.arange(N):
        strAP = utc_time[ii].strftime('%Y-%m-%dT%H:%M:%S.%f')
        #strAP2 = '1992-01-01T00:00:00'#utc_time[ii].strftime('%Y-%m-%dT00:00:00')
        t_utc = Time(strAP, format='isot', scale='utc')
        #t_utc2 = Time(strAP2, format='isot', scale='utc')
        t_tt = t_utc.tt
        #t_tt2 = t_utc2.tt
        dt[ii]=(t_tt.decimalyear-t_utc.decimalyear)*365.2422#*(86400)
        #dt2 = np.abs(t_tt.decimalyear-t_tt2.decimalyear)-np.abs(t_utc.decimalyear-t_utc2.decimalyear)#(t_tt2.decimalyear-t_utc2.decimalyear)*365.2422#*(86400)#
        Y,M,D = int(utc_time[ii].strftime('%Y')),int(utc_time[ii].strftime('%m')),int(utc_time[ii].strftime('%d'))
        Hr,Mi,Se = int(utc_time[ii].strftime('%H')),int(utc_time[ii].strftime('%M')),float(utc_time[ii].strftime('%S.%f'))
        tide_time[ii] = convert_calendar_dates(Y,M,D, hour=Hr, minute=Mi, second=Se)                                  
    return dt,tide_time


def ocean_tide_replacement(lon,lat,utc_time,LOAD,method='spline'):
    # https://github.com/tsutterley/pyTMD/blob/main/pyTMD/model.py   class model(pyTMD.io.model)
    NC = pull_model(LOAD)
    constituents = NC['constituents']
    time_datetime = np.asarray(list(map(datetime.datetime.fromisoformat,utc_time)))
    unique_date_list = np.unique([a.date() for a in time_datetime])
    tide_heights = np.empty(len(lon),dtype=np.float32)
    for unique_date in unique_date_list: #i.e. 2022-03-14
        idx_unique_date = np.asarray([a.date() == unique_date for a in time_datetime])
        time_unique_date = time_datetime[idx_unique_date]
        lon_unique_date = lon[idx_unique_date]
        lat_unique_date = lat[idx_unique_date]
        DELTAT,tide_time = dt_utc_tt(time_unique_date)
        hc = extract_constants(np.atleast_1d(lon_unique_date),np.atleast_1d(lat_unique_date),method, model_files=NC['model_file'])
        # https://astronomy.stackexchange.com/questions/47712/conversion-of-seconds-since-j2000-epoch-terrestrial-time-to-utc-time
        tmp_tide_heights = np.empty(len(lon_unique_date))
        for ii in range(len(lon_unique_date)):
            if np.any(hc[ii].mask) == True:
                tmp_tide_heights[ii] = np.nan
            else:
                # https://github.com/tsutterley/pyTMD/blob/main/pyTMD/predict_tidal_ts.py  pyTMD.predict.time_series(*args, **kwargs)
                TIDE = time_series(np.atleast_1d(tide_time[ii]),np.ma.array(data=[hc.data[ii]],mask=[hc.mask[ii]]),constituents,deltat=DELTAT[ii],corrections=NC['format'])#(np.atleast_1d(tide_time[idx_time[ii]]),np.ma.array(data=[hc.data[ii]],mask=[hc.mask[ii]]),constituents,deltat=DELTAT[idx_time[ii]],corrections=NC['format'])#
                # https://github.com/tsutterley/pyTMD/blob/main/pyTMD/infer_minor_corrections.py  pyTMD.predict.infer_minor(t, zmajor, constituents, **kwargs)
                MINOR = ltide.infer_minor(np.atleast_1d(tide_time[ii]),np.ma.array(data=[hc.data[ii]],mask=[hc.mask[ii]]),constituents,deltat=DELTAT[ii],corrections=NC['format'])#(np.atleast_1d(tide_time[idx_time[ii]]),np.ma.array(data=[hc.data[ii]],mask=[hc.mask[ii]]),constituents,deltat=DELTAT[idx_time[ii]],corrections=NC['format'])
                TIDE.data[:] += MINOR.data[:]
                tmp_tide_heights[ii] = TIDE.data
        tide_heights[idx_unique_date] = tmp_tide_heights
    return tide_heights,DELTAT,unique_date_list



def utc2utc_stamp(time_utc):
    gps2utc2 = (datetime.datetime(1985, 1, 1,0,0,0)-datetime.datetime(1980, 1, 6,0,0,0)).total_seconds()
    gps_time = time_utc+gps2utc2+18
    t0 = datetime.datetime(1980,1,6,0,0,0,0)
    leap_seconds = -18 #applicable to everything after 2017-01-01, UTC is currently 18 s behind GPS
    dt = (gps_time + leap_seconds) * datetime.timedelta(seconds=1)
    utc_time = t0+dt
    utc_time_str = np.asarray([str(x) for x in utc_time])
    return utc_time_str

'''
# test
fn = '/Users/alexaputnam/ICESat2/ana_fes2014/reg_atl03_lat_41_lon_n73_newengland_segs_2_100_2000_2020_12_to_2021_03.npz'
#d3s = np.load(f3s,allow_pickle='TRUE', encoding='bytes').item()
#beams3s = d3s.keys()
d3 = np.load(fn)
ssha_fft100 = d3['ssha_fft']
time100 = d3['time'] # from time_utc_mean_cu100
lon100 = d3['lon']
lat100= d3['lat']
dem100= d3['dem']
ot100= d3['ocean_tide']
time_utc = utc2utc_stamp(time100) # datetime.datetime.strptime(time_utc2[0], '%Y-%m-%d %H:%M:%S.%f')
N = 10#np.size(lon100)
t1 = time.time()
f14_ocean,DELTAT,unique_date_list = ocean_tide_replacement(lon100[:N],lat100[:N],time_utc[:N],LOAD=False,method='spline')
f14_load,DELTAT,unique_date_list = ocean_tide_replacement(lon100[:N],lat100[:N],time_utc[:N],LOAD=True,method='spline')
f14_geo_ocean = f14_ocean+f14_load
print('total time for '+str(N)+' points: '+str(np.round(time.time()-t1)/60)+' min using edited model')

import ocean_tide as ot
t1 = time.time()
f14_12_truth,DELTAT_truth = ot.ocean_tide_replacement(lon100[:N],lat100[:N],time_utc[:N])
print('total time for '+str(N)+' points: '+str(np.round(time.time()-t1)/60)+' min using truth model')
'''