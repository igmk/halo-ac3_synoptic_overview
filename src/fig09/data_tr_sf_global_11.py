from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import cartopy.crs as ccrs
#from mpl_toolkits.basemap import Basemap
#import nbtools.map

#from nbtools import Lambert
from matplotlib.colors import ListedColormap
#from colormaps import parula
# import pandas as pn
import netCDF4
import socket
import math as m
from numpy import dtype
#from sympy import *
import time,calendar,datetime
#import scipy.integrate as integrate


#ACLOUD

#campaign
data_halo_11 = netCDF4.Dataset("/net/secaire/mlauer/data/ERA5/HALO/climatology/data_HALO_11.nc","r")
data_halo = netCDF4.Dataset("/net/secaire/mlauer/data/ERA5/HALO/climatology/data_HALO.nc","r")

#calculate rainfall from sf and tp
tp_11 = data_halo_11.variables['tp'][:,::-1,:][:,:,:]*1000.
sf_11 = data_halo_11.variables['sf'][:,::-1,:][:,:,:]*1000.
sf_11 = np.where(sf_11 > tp_11, tp_11, sf_11)
tr_11 = tp_11 - sf_11

sic = data_halo.variables['siconc'][:,::-1,:][:,:,:]
tp = data_halo.variables['tp'][:,::-1,:][:,:,:]*1000.
sf = data_halo.variables['sf'][:,::-1,:][:,:,:]*1000.
sf = np.where(sf > tp, tp, sf)
tr = tp - sf


latitude = data_halo.variables['latitude'][::-1][:]
longitude = data_halo.variables['longitude'][:][:]

sic_start = sic[9288,:,:]
sic_end = sic[9503,:,:]
tp_campaign = tp[9288::,:,:]
sf_campaign = sf[9288::,:,:]
tr_campaign = tr[9288::,:,:]

tp_11_campaign = tp_11[1032::,:,:]
sf_11_campaign = sf_11[1032::,:,:]
tr_11_campaign = tr_11[1032::,:,:]

#daily averaged precipitation rate for
#only campaign fraction_campaign
#for the climatology (1979 - 2022) regarding the 11 - 20 March 
#anomaly: campaign - climatology
#deviation: campaign from climatology (how much was tp/sf/tr higher/lower during the campaign compared to climatology)
#these steps are done for tp, sf, and tr
tp_fraction_campaign = (np.nansum(tp_campaign,axis=0) + np.nansum(tp_11_campaign,axis=0))/(len(tp_campaign) + len(tp_11_campaign))
tp_fraction_full = (np.nansum(tp,axis=0) + np.nansum(tp_11,axis=0))/(len(tp) + len(tp_11))
tp_anomaly = np.array(tp_fraction_campaign) - np.array(tp_fraction_full)

tp_deviation = np.where(tp_anomaly < 0, (np.array(tp_fraction_full)/np.array(tp_fraction_campaign)),np.array(tp_fraction_campaign)/np.array(tp_fraction_full))

sf_fraction_campaign = (np.nansum(sf_campaign,axis=0) + np.nansum(sf_11_campaign,axis=0))/(len(sf_campaign) + len(sf_11_campaign))
sf_fraction_full = (np.nansum(sf,axis=0) + np.nansum(sf_11,axis=0))/(len(sf) + len(sf_11))
sf_anomaly = np.array(sf_fraction_campaign) - np.array(sf_fraction_full)

sf_deviation = np.where(sf_anomaly < 0, (np.array(sf_fraction_full)/np.array(sf_fraction_campaign)),np.array(sf_fraction_campaign)/np.array(sf_fraction_full))

tr_fraction_campaign = (np.nansum(tr_campaign,axis=0) + np.nansum(tr_11_campaign,axis=0))/(len(tr_campaign) + len(tr_11_campaign))
tr_fraction_full = (np.nansum(tr,axis=0) + np.nansum(tr_11,axis=0))/(len(tr) + len(tr_11))
tr_anomaly = np.array(tr_fraction_campaign) - np.array(tr_fraction_full)

tr_deviation = np.where(tr_anomaly < 0, (np.array(tr_fraction_full)/np.array(tr_fraction_campaign)),np.array(tr_fraction_campaign)/np.array(tr_fraction_full))

#safe data for plotting
ncout = netCDF4.Dataset('data_tr_sf_global_11.nc','w', format="NETCDF4")
    
# define axis size
#ncout.createDimension('time', None)  # unlimited  
nlat = len(latitude)
nlon = len(longitude)
    
ncout.createDimension('lat', nlat)
ncout.createDimension('lon', nlon)

# create variable array
tp_campaign = ncout.createVariable('tp_campagin', dtype('double').char,('lat', 'lon'))
tp_full = ncout.createVariable('tp_full', dtype('double').char,('lat', 'lon'))
tp_ano = ncout.createVariable('tp_anomaly', dtype('double').char,('lat', 'lon'))
tp_dev = ncout.createVariable('tp_deviation', dtype('double').char,('lat', 'lon'))
    
sf_campaign = ncout.createVariable('sf_campagin', dtype('double').char,('lat', 'lon'))
sf_full = ncout.createVariable('sf_full', dtype('double').char,('lat', 'lon'))
sf_ano = ncout.createVariable('sf_anomaly', dtype('double').char,('lat', 'lon'))
sf_dev = ncout.createVariable('sf_deviation', dtype('double').char,('lat', 'lon'))

tr_campaign = ncout.createVariable('tr_campagin', dtype('double').char,('lat', 'lon'))
tr_full = ncout.createVariable('tr_full', dtype('double').char,('lat', 'lon'))
tr_ano = ncout.createVariable('tr_anomaly', dtype('double').char,('lat', 'lon'))
tr_dev = ncout.createVariable('tr_deviation', dtype('double').char,('lat', 'lon'))

siconc_start = ncout.createVariable('sic_start', dtype('double').char,('lat', 'lon'))
siconc_end = ncout.createVariable('sic_end', dtype('double').char,('lat', 'lon'))

tp_campaign[:,:] = tp_fraction_campaign[:,:]
tp_full[:,:] = tp_fraction_full[:,:]
tp_ano[:,:] = tp_anomaly[:,:]
tp_dev[:,:] = tp_deviation[:,:]

sf_campaign[:,:] = sf_fraction_campaign[:,:]
sf_full[:,:] = sf_fraction_full[:,:]
sf_ano[:,:] = sf_anomaly[:,:]
sf_dev[:,:] = sf_deviation[:,:]

tr_campaign[:,:] = tr_fraction_campaign[:,:]
tr_full[:,:] = tr_fraction_full[:,:]
tr_ano[:,:] = tr_anomaly[:,:]
tr_dev[:,:] = tr_deviation[:,:]

siconc_start[:,:] = sic_start[:,:]
siconc_end[:,:] = sic_end[:,:]
ncout.close()
exit()


