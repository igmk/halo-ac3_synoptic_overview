from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import cartopy.crs as ccrs
from mpl_toolkits.basemap import Basemap
#import nbtools.map

from nbtools import Lambert
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
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
data_halo = netCDF4.Dataset("/net/secaire/mlauer/data/ERA5/HALO/climatology/southern_area/data_tr_sf_global_11.nc","r")


latitude = np.arange(60,89.5,0.25)
longitude = np.arange(-70,70+0.25,0.25)



tp_fraction_campaign = data_halo.variables['tp_campagin'][:,:]
tp_fraction_full = data_halo.variables['tp_full'][:,:]
tp_anomaly = data_halo.variables['tp_anomaly'][:,:]
tp_deviation = data_halo.variables['tp_deviation'][:,:]

tr_fraction_campaign = data_halo.variables['tr_campagin'][:,:]
tr_fraction_full = data_halo.variables['tr_full'][:,:]
tr_anomaly = data_halo.variables['tr_anomaly'][:,:]
tr_deviation = data_halo.variables['tr_deviation'][:,:]

sf_fraction_campaign = data_halo.variables['sf_campagin'][:,:]
sf_fraction_full = data_halo.variables['sf_full'][:,:]
sf_anomaly = data_halo.variables['sf_anomaly'][:,:]
sf_deviation = data_halo.variables['sf_deviation'][:,:]


tp_fraction_campaign = np.where(tp_fraction_campaign == 0, np.nan, tp_fraction_campaign)
tp_fraction_full = np.where(tp_fraction_full == 0, np.nan, tp_fraction_full)

sf_fraction_campaign = np.where(sf_fraction_campaign == 0, np.nan, sf_fraction_campaign)
sf_fraction_full = np.where(sf_fraction_full == 0, np.nan, sf_fraction_full)

tr_fraction_campaign = np.where(tr_fraction_campaign == 0, np.nan, tr_fraction_campaign)
tr_fraction_full = np.where(tr_fraction_full == 0, np.nan, tr_fraction_full)

sic_start = data_halo.variables['sic_start'][:,:]
sic_end = data_halo.variables['sic_end'][:,:]

fig, axes = plt.subplots(3,3,figsize=(9,8))

colors = ['#E3E3E3','#41C8E5','#8eb5fa','#0850cd','#053588','#021a44']

my_cmap = LinearSegmentedColormap.from_list("my_cmap", colors)
#ACLOUD
#AR
x = plt.subplot(331)
 
lon1 = -50#np.min(lon)
lon2 = 50#np.max(lon)
lat1 = 60
lat2 = 89.75
lamb = Lambert(lon1, lon2, lat1, lat2)


lamb.bmap.drawcoastlines(color='k', linewidth=0.5)
lamb.bmap.drawparallels(np.arange(lat1, lat2+10, 10), labels=[0, 0, 0, 0])
lamb.bmap.drawmeridians(np.arange(lon1, lon2+20, 20), labels=[0, 0, 0, 0])


x,y = np.meshgrid(longitude,latitude)
X,Y = lamb.bmap(x,y)
lamb.make_mask()

tp_fraction_campaign = np.where(tp_fraction_campaign == 0, np.nan, tp_fraction_campaign)
tp_fraction_full = np.where(tp_fraction_full == 0, np.nan, tp_fraction_full)
sf_fraction_campaign = np.where(sf_fraction_campaign == 0, np.nan, sf_fraction_campaign)
sf_fraction_full = np.where(sf_fraction_full == 0, np.nan, sf_fraction_full)
tr_fraction_campaign = np.where(tr_fraction_campaign == 0, np.nan, tr_fraction_campaign)
tr_fraction_full = np.where(tr_fraction_full == 0, np.nan, tr_fraction_full)

im=lamb.bmap.contourf(X, Y, tp_fraction_campaign, np.arange(0,1.1+0.01,0.01),cmap='ocean_r')
lamb.bmap.contour(X,Y,sic_start,levels=[15],colors='grey',linestyles='dotted',linewidths=0.5)
lamb.bmap.contour(X,Y,sic_end,levels=[15],colors='grey',linestyles='solid',linewidths=0.5)
lamb.add_lc_labels(spacelon=20, spacelat=10, fontsize=8)

plt.figtext(0.1,0.87, "(a)", ha="center", va="top", fontsize=12)
plt.figtext(0.4,0.87, "(b)", ha="center", va="top", fontsize=12)
plt.figtext(0.7,0.87, "(c)", ha="center", va="top", fontsize=12)
plt.figtext(0.1,0.6, "(d)", ha="center", va="top", fontsize=12)
plt.figtext(0.4,0.6, "(e)", ha="center", va="top", fontsize=12)
plt.figtext(0.7,0.6, "(f)", ha="center", va="top", fontsize=12)
plt.figtext(0.1,0.34, "(g)", ha="center", va="top", fontsize=12)
plt.figtext(0.4,0.34, "(h)", ha="center", va="top", fontsize=12)
plt.figtext(0.7,0.34, "(i)", ha="center", va="top", fontsize=12)
#cyclone

x = plt.subplot(332)
     		# matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

lon1 = -50#np.min(lon)
lon2 = 50#np.max(lon)
lat1 = 60
lat2 = 89.75
lamb = Lambert(lon1, lon2, lat1, lat2)


lamb.bmap.drawcoastlines(color='k', linewidth=0.5)
lamb.bmap.drawparallels(np.arange(lat1, lat2+10, 10), labels=[0, 0, 0, 0])
lamb.bmap.drawmeridians(np.arange(lon1, lon2+20, 20), labels=[0, 0, 0, 0])


x,y = np.meshgrid(longitude,latitude)
X,Y = lamb.bmap(x,y)
lamb.make_mask()

im=lamb.bmap.contourf(X, Y, tp_fraction_full, np.arange(0,1.1+0.01,0.01),cmap='ocean_r')
lamb.bmap.contour(X,Y,sic_start,levels=[15],colors='grey',linestyles='dotted',linewidths=0.5)
lamb.bmap.contour(X,Y,sic_end,levels=[15],colors='grey',linestyles='solid',linewidths=0.5)
lamb.add_lc_labels(spacelon=20, spacelat=10, fontsize=8)
plt.title("total precipitation", fontsize=12, y = 1.25,fontweight="bold")


fig.subplots_adjust(wspace=0.04, hspace = 0.4)
cbar_ax = fig.add_axes([0.05, 0.085, 0.5, 0.05])
#fig.colorbar(im, cax=cbar_ax)
cb=plt.colorbar(im,ticks=[0,0.2,0.4,0.6,0.8,1],cax=cbar_ax,orientation="horizontal")#,pad=0.1, aspect=30)
cb.set_label('Hourly averaged precipitation (mm h$^{-1}$)', fontsize=12)
cb.ax.set_xticklabels(['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=12.5)

x = plt.subplot(333)
     		# matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

lon1 = -50#np.min(lon)
lon2 = 50#np.max(lon)
lat1 = 60
lat2 = 89.75
lamb = Lambert(lon1, lon2, lat1, lat2)


lamb.bmap.drawcoastlines(color='k', linewidth=0.5)
lamb.bmap.drawparallels(np.arange(lat1, lat2+10, 10), labels=[0, 0, 0, 0])
lamb.bmap.drawmeridians(np.arange(lon1, lon2+20, 20), labels=[0, 0, 0, 0])


x,y = np.meshgrid(longitude,latitude)
X,Y = lamb.bmap(x,y)
lamb.make_mask()

tp_deviation = np.where(tp_deviation < -7, -7, tp_deviation)
im_ano=lamb.bmap.contourf(X, Y, tp_deviation, np.arange(-7,7+0.01,0.01),cmap='bwr', extend='min')
lamb.bmap.contour(X,Y,sic_start,levels=[15],colors='grey',linestyles='dotted',linewidths=0.5)
lamb.bmap.contour(X,Y,sic_end,levels=[15],colors='grey',linestyles='solid',linewidths=0.5)
lamb.add_lc_labels(spacelon=20, spacelat=10, fontsize=8)
#plt.title("ACLOUD", fontsize=12, y = 1.325,fontweight="bold")


cbar_ax = fig.add_axes([0.95, 0.695, 0.015, 0.2])
#fig.colorbar(im, cax=cbar_ax)
cb=plt.colorbar(im_ano,ticks=[-6,-4,-2,0,2,4,6],cax=cbar_ax,orientation="vertical")#,pad=0.1, aspect=30)
cb.set_label('Deviation', fontsize=12)
#cb.ax.set_xticklabels(['-6','-4','-2','0','2','4','6'],fontsize=12.5)


x = plt.subplot(334)
     		# matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

lon1 = -50#np.min(lon)
lon2 = 50#np.max(lon)
lat1 = 60
lat2 = 89.75
lamb = Lambert(lon1, lon2, lat1, lat2)


lamb.bmap.drawcoastlines(color='k', linewidth=0.5)
lamb.bmap.drawparallels(np.arange(lat1, lat2+10, 10), labels=[0, 0, 0, 0])
lamb.bmap.drawmeridians(np.arange(lon1, lon2+20, 20), labels=[0, 0, 0, 0])


x,y = np.meshgrid(longitude,latitude)
X,Y = lamb.bmap(x,y)
lamb.make_mask()

im=lamb.bmap.contourf(X, Y, sf_fraction_campaign, np.arange(0,1.1+0.01,0.01),cmap='ocean_r')
lamb.bmap.contour(X,Y,sic_start,levels=[15],colors='grey',linestyles='dotted',linewidths=0.5)
lamb.bmap.contour(X,Y,sic_end,levels=[15],colors='grey',linestyles='solid',linewidths=0.5)
lamb.add_lc_labels(spacelon=20, spacelat=10, fontsize=8)
#plt.title("ACLOUD", fontsize=12, y = 1.325,fontweight="bold")

#cyclone

x = plt.subplot(335)
     		# matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

lon1 = -50#np.min(lon)
lon2 = 50#np.max(lon)
lat1 = 60
lat2 = 89.75
lamb = Lambert(lon1, lon2, lat1, lat2)


lamb.bmap.drawcoastlines(color='k', linewidth=0.5)
lamb.bmap.drawparallels(np.arange(lat1, lat2+10, 10), labels=[0, 0, 0, 0])
lamb.bmap.drawmeridians(np.arange(lon1, lon2+20, 20), labels=[0, 0, 0, 0])


x,y = np.meshgrid(longitude,latitude)
X,Y = lamb.bmap(x,y)
lamb.make_mask()

im=lamb.bmap.contourf(X, Y, sf_fraction_full, np.arange(0,1.1+0.01,0.01),cmap='ocean_r')
lamb.bmap.contour(X,Y,sic_start,levels=[15],colors='grey',linestyles='dotted',linewidths=0.5)
lamb.bmap.contour(X,Y,sic_end,levels=[15],colors='grey',linestyles='solid',linewidths=0.5)
lamb.add_lc_labels(spacelon=20, spacelat=10, fontsize=8)
plt.title("snowfall", fontsize=12, y = 1.25,fontweight="bold")


fig.subplots_adjust(wspace=0.04, hspace = 0.4)
cbar_ax = fig.add_axes([0.0275, 0.085, 0.5, 0.05])
#fig.colorbar(im, cax=cbar_ax)
cb=plt.colorbar(im,ticks=[0,0.2,0.4,0.6,0.8,1],cax=cbar_ax,orientation="horizontal")#,pad=0.1, aspect=30)
cb.set_label('Hourly averaged precipitation (mm h$^{-1}$)', fontsize=12)
cb.ax.set_xticklabels(['0.0','0.2','0.4','0.6','0.8','1.0'],fontsize=12.5)

x = plt.subplot(336)
     		# matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

lon1 = -50#np.min(lon)
lon2 = 50#np.max(lon)
lat1 = 60
lat2 = 89.75
lamb = Lambert(lon1, lon2, lat1, lat2)


lamb.bmap.drawcoastlines(color='k', linewidth=0.5)
lamb.bmap.drawparallels(np.arange(lat1, lat2+10, 10), labels=[0, 0, 0, 0])
lamb.bmap.drawmeridians(np.arange(lon1, lon2+20, 20), labels=[0, 0, 0, 0])


x,y = np.meshgrid(longitude,latitude)
X,Y = lamb.bmap(x,y)
lamb.make_mask()

sf_deviation = np.where(sf_deviation < -7, -7, sf_deviation)
im_ano=lamb.bmap.contourf(X, Y, sf_deviation, np.arange(-7,7+0.01,0.01),cmap='bwr', extend='min')
#im_ano.cmap.set_under('#021a44')

lamb.bmap.contour(X,Y,sic_start,levels=[15],colors='grey',linestyles='dotted',linewidths=0.5)
lamb.bmap.contour(X,Y,sic_end,levels=[15],colors='grey',linestyles='solid',linewidths=0.5)
#cs=lamb.bmap.contour(X, Y, sf_deviation, np.arange(0.1,6,1),cmap='Greys', linewidths=0, alpha =0)
#plt.clabel(cs, inline=True, fmt='%1.0f', fontsize=5.5, colors='gray')
lamb.add_lc_labels(spacelon=20, spacelat=10, fontsize=8)
#plt.title("ACLOUD", fontsize=12, y = 1.325,fontweight="bold")


cbar_ax = fig.add_axes([0.95, 0.415, 0.015, 0.2])
#fig.colorbar(im, cax=cbar_ax)
cb=plt.colorbar(im_ano,ticks=[-6,-4,-2,0,2,4,6],cax=cbar_ax,orientation="vertical")#,pad=0.1, aspect=30)
cb.set_label('Deviation', fontsize=12)
#cb.ax.set_xticklabels(['-6','-4','-2','0','2','4','6'],fontsize=12.5)


x = plt.subplot(337)
     		# matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

lon1 = -50#np.min(lon)
lon2 = 50#np.max(lon)
lat1 = 60
lat2 = 89.75
lamb = Lambert(lon1, lon2, lat1, lat2)


lamb.bmap.drawcoastlines(color='k', linewidth=0.5)
lamb.bmap.drawparallels(np.arange(lat1, lat2+10, 10), labels=[0, 0, 0, 0])
lamb.bmap.drawmeridians(np.arange(lon1, lon2+20, 20), labels=[0, 0, 0, 0])


x,y = np.meshgrid(longitude,latitude)
X,Y = lamb.bmap(x,y)
lamb.make_mask()

im=lamb.bmap.contourf(X, Y, tr_fraction_campaign, np.arange(0,1.1+0.01,0.01),cmap='ocean_r')
lamb.bmap.contour(X,Y,sic_start,levels=[15],colors='grey',linestyles='dotted',linewidths=0.5)
lamb.bmap.contour(X,Y,sic_end,levels=[15],colors='grey',linestyles='solid',linewidths=0.5)
lamb.add_lc_labels(spacelon=20, spacelat=10, fontsize=8)
#plt.title("ACLOUD", fontsize=12, y = 1.325,fontweight="bold")

#cyclone

x = plt.subplot(338)
     		# matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

lon1 = -50#np.min(lon)
lon2 = 50#np.max(lon)
lat1 = 60
lat2 = 89.75
lamb = Lambert(lon1, lon2, lat1, lat2)


lamb.bmap.drawcoastlines(color='k', linewidth=0.5)
lamb.bmap.drawparallels(np.arange(lat1, lat2+10, 10), labels=[0, 0, 0, 0])
lamb.bmap.drawmeridians(np.arange(lon1, lon2+20, 20), labels=[0, 0, 0, 0])


x,y = np.meshgrid(longitude,latitude)
X,Y = lamb.bmap(x,y)
lamb.make_mask()

im=lamb.bmap.contourf(X, Y, tr_fraction_full, np.arange(0,1.1+0.01,0.01),cmap='ocean_r')
lamb.bmap.contour(X,Y,sic_start,levels=[15],colors='grey',linestyles='dotted',linewidths=0.5)
lamb.bmap.contour(X,Y,sic_end,levels=[15],colors='grey',linestyles='solid',linewidths=0.5)
lamb.add_lc_labels(spacelon=20, spacelat=10, fontsize=8)
plt.title("rainfall", fontsize=12, y = 1.25,fontweight="bold")


cbar_ax = fig.add_axes([0.1, 0.085, 0.5, 0.015])
#fig.colorbar(im, cax=cbar_ax)
cb=plt.colorbar(im,ticks=[0,0.2,0.4,0.6,0.8,1.0],cax=cbar_ax,orientation="horizontal")#,pad=0.1, aspect=30)
cb.set_label('Hourly averaged precipitation (mm h$^{-1}$)', fontsize=12)
#cb.ax.set_xticklabels(['0.00','0.15','0.30','0.45','0.60','0.75'],fontsize=12.5)

x = plt.subplot(339)
     		# matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

lon1 = -50#np.min(lon)
lon2 = 50#np.max(lon)
lat1 = 60
lat2 = 89.75
lamb = Lambert(lon1, lon2, lat1, lat2)


lamb.bmap.drawcoastlines(color='k', linewidth=0.5)
lamb.bmap.drawparallels(np.arange(lat1, lat2+10, 10), labels=[0, 0, 0, 0])
lamb.bmap.drawmeridians(np.arange(lon1, lon2+20, 20), labels=[0, 0, 0, 0])


x,y = np.meshgrid(longitude,latitude)
X,Y = lamb.bmap(x,y)
lamb.make_mask()

tr_deviation = np.where(tr_deviation < -36, -36, tr_deviation)
im_ano=lamb.bmap.contourf(X, Y, tr_deviation, np.arange(-36,36+0.01,0.01),cmap='bwr', extend='min')
lamb.bmap.contour(X,Y,sic_start,levels=[15],colors='grey',linestyles='dotted',linewidths=0.5)
lamb.bmap.contour(X,Y,sic_end,levels=[15],colors='grey',linestyles='solid',linewidths=0.5)
#cs=lamb.bmap.contour(X, Y, tr_deviation, np.arange(0.1,36,6),cmap='Greys', linewidths=0.6, alpha =0.8)
#plt.clabel(cs, inline=True, fmt='%1.0f', fontsize=5.5, colors='gray')
lamb.add_lc_labels(spacelon=20, spacelat=10, fontsize=8)
#plt.title("ACLOUD", fontsize=12, y = 1.325,fontweight="bold")


cbar_ax = fig.add_axes([0.95, 0.125, 0.015, 0.2])
#fig.colorbar(im, cax=cbar_ax)
cb=plt.colorbar(im_ano,ticks=[-36,-24,-12,0,12,24,36],cax=cbar_ax,orientation="vertical")#,pad=0.1, aspect=30)
cb.set_label('Deviation', fontsize=12)
#cb.ax.set_xticklabels(['-36','-24','-12','0','12','24','36'],fontsize=12.5)

fig.subplots_adjust(hspace=0.15, wspace=0.35)

plt.savefig('tr_sf_global_paper.png',bbox_inches='tight',dpi=400)
plt.close("all")
    	




