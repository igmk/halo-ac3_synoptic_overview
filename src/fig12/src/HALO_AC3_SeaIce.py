#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy

from datetime import datetime, timedelta
import cmocean
import pandas as pd
import matplotlib.pyplot as plt 
import os
import glob
import xarray as xr
from pyproj import Proj,CRS #for projection/coordinate reference systems
import sys
from netCDF4 import Dataset



def plot_map_HALO(date1, date2,extent=[-18,23,66,90], product="merged", boxes="no", savefig="no", outpath="/home/jrueckert/Output/HALO/results/", specific_filename=""):
    """plot_map_HALO(date1, date2, extent, product="merged", boxes="no", savefig="no", outpath="/home/jrueckert/Output/HALO/results/", specific_filename="")
    plots Sea ice concentration maps for the HALO AC3 campaign
    input:
    date1: first day, date2: second day, sea ice concentration is averaged over these time period (mean)
    product: Sea ice concentration product, choose between 'merged'(ASI-MODIS product) and 'asi'
    boxes: if true, draw areas that are used in HALO AC3 paper
    savefig: if yes, saves the figure as png
    outpath: where to save the figure
    specific_filename: name extension of saved figure
    output: 
    mean sic, lon, lat
"""
    if date2 < date1:
         raise Exception(f"second date must be after first date") 
    matplotlib.rcParams['xtick.labelsize'] = 18 ##to adjust colorbar ticklabels
    print(date1)
    if product=="merged":
        path = f'/ssmi/www/htdocs/data/modis_amsr2/netcdf/Arctic/{date1:%Y}/' #insert the path to your file here, e.g. '/home/username/Data/sea_ice_concentration/'
        filename = f'sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_{date1:%Y%m%d}.nc' #name of the file, if you use a different version of the data make sure to adapt the filename accordingly
        file = os.path.join(path, filename) #joined path and filename
        ds = xr.open_dataset(file) #open the file
        sic = [ds.variables['sic_merged']]
    elif product=="asi":
        path = f'/ssmi/www/htdocs/data/amsr2/asi_daygrid_swath/n6250/netcdf/{date1:%Y}/' #insert the path to your file here, e.g. '/home/username/Data/sea_ice_concentration/'
        filename = f'asi-AMSR2-n6250-{date1:%Y%m%d}-v5.4.nc' #name of the file, if you use a different version of the data make sure to adapt the filename accordingly
        file = os.path.join(path, filename) #joined path and filename
        ds = xr.open_dataset(file) #open the file
        sic = [ds.variables['z']]
    x = ds.variables['x']
    y = ds.variables['y']
   
    projection = ds.variables["polar_stereographic"] #To convert from x and y to lat/lon we need the information about the grid contained in this variable
    m=Proj(projection.attrs["spatial_ref"]) #define a function for conversion that gets the information from the string contained in projection
    xx,yy = np.meshgrid(x.data,y.data) #create a meshgrid from x and y to use in function m()
    lon, lat = m(xx,yy, inverse=True)
    ds.close()
    for dates in pd.date_range(date1+ timedelta(1), date2):
        print(dates)
        if product=="merged":
 #       path = f'/ssmi/www/htdocs/data/amsr2/asi_daygrid_swath/n6250/netcdf/{dates:%Y}/' #insert the path to your file here, e.g. '/home/username/Data/sea_ice_concentration/'
  #      filename = f'asi-AMSR2-n6250-{dates:%Y%m%d}-v5.4.nc' #name of the file, if you use a different version of the data make sure to adapt the filename accordingly
            path = f'/ssmi/www/htdocs/data/modis_amsr2/netcdf/Arctic/{dates:%Y}/' #insert the path to your file here, e.g. '/home/username/Data/sea_ice_concentration/'
            filename = f'sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_{dates:%Y%m%d}.nc'
            file = os.path.join(path, filename) #joined path and filename
            ds = xr.open_dataset(file) #open the file
            sic.append(ds.variables['sic_merged'])
            ds.close()
        elif product=="asi":
            filename = f'asi-AMSR2-n6250-{dates:%Y%m%d}-v5.4.nc' #name of the file, if you use a different version of the data make sure to adapt the filename accordingly
            path = f'/ssmi/www/htdocs/data/amsr2/asi_daygrid_swath/n6250/netcdf/{dates:%Y}/' #insert the path to your file here, e.g. '/home/username/Data/sea_ice_concentration/'
            file = os.path.join(path, filename) #joined path and filename
            ds = xr.open_dataset(file) #open the file
            sic.append(ds.variables['z'])
            ds.close()
    if product=="merged":
        sic = np.flipud(np.mean(sic, axis=0)) ##needs to flipped, see Valentins mail from January 25 2022
      #  sic = (np.mean(sic, axis=0)) 
    elif product =="asi":
        sic = (np.mean(sic, axis=0)) 
##########PLOTTING
    fig = plt.figure(figsize=(12,12), dpi=300)
    cmap=cm.get_cmap("cmo.ice").copy()
    cmap.set_bad("lightgrey") #to mask out invalid values

    npstere = ccrs.NorthPolarStereo(central_longitude=0, true_scale_latitude=70) #projection that is used for the map (NorthPolarStereo)
    ax = fig.add_subplot(1,1,1,projection=npstere)
    ax.set_extent(extent, ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, dms=True,x_inline=False, y_inline=True,linestyle = '--')
    gl.xlabel_style = {'size': 18, 'color': 'black'}
    gl.ylabel_style = {'size': 18, 'color': 'black'}
  #  gl.xlocator = mticker.FixedLocator([ -60, -30, 0, 30, 60])
    gl.xlabels_right=True
    #ax.set_global()
    ax.coastlines(resolution='10m', color = 'black', zorder=2)
    ax.add_feature(cfeature.LAND, facecolor="darkgrey", zorder=1) #to shade the land 
    #ax.set_extent(extent, crs=ccrs.PlateCarree())
    if product=="merged":
        ax.set_title(f"Mean MODIS-AMSR2 sea ice concentration  for {date1:%d %b}-{date2:%d %b, %Y}", fontsize=18)
    elif product=="asi":
        ax.set_title(f"Mean ASI-AMSR2 sea ice concentration  for {date1:%d %b}-{date2:%d %b, %Y}", fontsize=18)
    pcm=ax.pcolormesh(lon,lat,sic[1:,1 :],cmap=cmap,shading="flat",transform=ccrs.PlateCarree(), vmin=0, vmax=100, zorder=0) #last row and column need to be dropped in order to use shading flat
    cbar = plt.colorbar(pcm, pad = 0.1, shrink =0.6,orientation="horizontal").set_label( label='SIC (%)', size=18)#
   # cbar.ax.tick_params(labelsize='large')
    if boxes == "true":
        ###needs lines to follow latitudes:
        
       # lon_southern = [0,23,23,0,0]
        lon_southern = np.concatenate([np.linspace(0,23,23), np.linspace(23,0,23),[0]])
      #  lat_southern  = [70.6,70.6,75,75, 70.6]
        lat_southern  = np.concatenate([np.ones(23)*70.6, np.ones(23)*75,[70.6]])
        ax.plot(lon_southern, lat_southern,color="orange", linestyle='-',transform=ccrs.PlateCarree(), label="", zorder=3) #Geodetic not used to reprocude plots from pthers
        #lon_central1 = [-9.0,30.0,30.0,-15,-9]
       # lon_central2= [-15,-54,-54,-9]
        lon_central1  = np.concatenate([[-9],np.linspace(-9,30,39), np.linspace(30,-9,39)])
        lon_central2= np.concatenate([np.linspace(-9,-54,49), np.linspace(-54,-9,49)])#,[-9]])
       # lat_central1  = [81.5,81.5, 89.3,89.3,81.5]
        lat_central1 = np.concatenate([[84.5],np.ones(39)*81.5, np.ones(39)*89.3])#[81.5]])
       # lat_central2  = [89.3,89.3,84.5,84.5]
        lat_central2 = np.concatenate([np.ones(49)*89.3, np.ones(49)*84.5])#,[89.3]])

        ax.plot(lon_central1, lat_central1,color="orange", linestyle='-',transform=ccrs.PlateCarree(), label="", zorder=3)
        ax.plot(lon_central2, lat_central2,color="orange", linestyle='-',transform=ccrs.PlateCarree(), label="", zorder=3)
        #lat_Fram = [75,75,81.5,81.5,75]
        lat_Fram = np.concatenate([np.ones(35)*75, np.ones(35)*81.5,[75]])
        #lon_Fram = [-9,16,16,-9,-9]
        lon_Fram  = np.concatenate([np.linspace(-9,16,35), np.linspace(16,-9,35),[-9]])
        ax.plot(lon_Fram, lat_Fram,color="orange", linestyle='-',transform=ccrs.PlateCarree(), label="", zorder=3)
        
    if savefig=="yes":
        filename=f"mean_SIC_{date1:%Y%m%d}_{date2:%Y%m%d}_{product}_{specific_filename}.png"
        f = outpath+filename
        plt.savefig(f, dpi=300, bbox_inches="tight")
      #  filename=f"mean_SIC_{date1:%Y%m%d}_{date2:%Y%m%d}_{product}_{specific_filename}.pdf"
     #   f = outpath+filename
      #  plt.savefig(f, dpi=300, bbox_inches="tight")

    plt.show()
    return sic, lon, lat


#######################

def read_osi_halo(date, area="southern"):
    """read_osi_halo(date, area="southern")
    reads OSI-SAF climate record data for given date and specified area
    input: 
    date: date as datetime 
    area: area to be considered, can be 'southern', 'central' or 'Fram'
    sic_colloc: sic as array in area
    lat, lon: corresponding arrays of latitude and longitude,
    date
    mean sic
    median sic
        """
    path_osi=f"/mnt/spaces/IceConcentration/OSI-SAF/climate_records/{date:%Y}/"
    if date.year> 2015:
        filename=f"ice_conc_nh_ease2-250_icdr-v2p0_{date:%Y%m%d}1200.nc"
    elif date.year<= 2015:
        filename=f"ice_conc_nh_ease2-250_cdr-v2p0_{date:%Y%m%d}1200.nc"
        #sic = Dataset(filename)
    ##collocate to lat/lon 
    try:
        file = os.path.join(path_osi, filename) #joined path and filename
        ds = xr.open_dataset(file) #open the file
   
        lon = ds.variables['lon'].data
        lat = ds.variables['lat'].data
        sic = ds.variables['ice_conc'][0].data
        if area =="southern":
            lonmin= 0.0 #E
            latmin =70.6 #N
            lonmax= 23.0 #E
            latmax = 75 #N
            useindex = np.where((lon>lonmin) & (lat>latmin) &(lon<lonmax)&(lat<latmax))
            sic_colloc = sic[useindex]
            sic_colloc[sic_colloc==-32767]=np.nan  ## fill values are replaced by nans
         #   print(np.shape(sic["lat"]))
            lat = lat[useindex]
            lon = lon[useindex]
        elif area =="central":

            lonmin1 = -9.0 #E
            latmin1 = 81.5 #N
            lonmax1 = 30.0 #E
            latmax1 = 89.3 #N
            lonmin2 = -30 #E
            latmin2 = 84.5 #N
            lonmax2 = -9 #E
            latmax2 = 89.3 #N
            useindex = np.where(((lon>lonmin1) & (lat>latmin1) &(lon<lonmax1)&(lat<latmax1)) |((lon>lonmin2) & (lat>latmin2) &(lon<lonmax2)&(lat<latmax2))  )

            sic_colloc = sic[useindex]
            sic_colloc[sic_colloc==-32767]=np.nan  ## fill values are replaced by nans
            lat = lat[useindex]
            lon = lon[useindex]


        elif area =="Fram": 

            lonmin= -9. #E
            latmin =75 #N
            lonmax= 16 #E
            latmax = 81.5 #N
            useindex = np.where((lon>lonmin) & (lat>latmin) &(lon<lonmax)&(lat<latmax))

            sic_colloc = sic[useindex]
            sic_colloc[sic_colloc==-32767]=np.nan  ## fill values are replaced by nans
            lat = lat[useindex]
            lon = lon[useindex]
        ds.close()
        return sic_colloc,lat, lon, date, np.nanmean(sic_colloc), np.nanmedian(sic_colloc)
    except:
        #print("File not available? ", date)
        return np.nan, np.nan, np.nan, date,np.nan, np.nan
    
def statistics_per_day(startdate, enddate, area):
    """statistics_per_day(startdate, enddate, area) 
    function to loop through period given by startdate and enddate for all years between startdate and enddate getting the mean sic for specidfied area and then calculating mean, median, std and quantiles per day for that time period, providing a climatology
    input:
    startdate as datetime
    enddate as datetime
    area: can be 'southern', 'central' or 'Fram'
    output: 
    dictionary with keys given by 'month-day' each value consistings of an array of [mean sic, median sic, std sic, 75% quantile, 25% quantile]"""
    sic_climatology=dict()
    for date in pd.date_range(startdate,  datetime(startdate.year, enddate.month,enddate.day)):
        print(f"{date:%b-%d}")
        sic=[]

        for year in range(startdate.year,enddate.year+1):
            date=datetime(year, date.month, date.day)
            sic_colloc,lat, lon, date, sic_mean, sic_median  = read_osi_halo(date, area=area)
            sic = np.append(sic, sic_mean)
        sic_climatology[f"{date.month}-{date.day}"] =  [np.nanmean(sic), np.nanmedian(sic), np.nanstd(sic), np.nanquantile(sic, 0.75), np.nanquantile(sic, 0.25),np.nanquantile(sic, 0.9), np.nanquantile(sic, 0.1)]
    return sic_climatology
        
def sic_per_year(startdate, enddate, area):
    """sic_per_year(startdate, enddate, area)
    calculates mean sic for given area per year and returns 2 dictionaries with years as keys and values consisting of 2 arrays: mean/median sic concentration and dates 
    input:
    startdate as datetime
    enddate as datetime
    area: can be 'southern', 'central' or 'Fram'
    output:
    sic_mean_per_years: dictionary with keys: years, containing arrays of mean sic and dates 
    sic_median_per_years dictionary with keys: years, containing arrays of median sic and dates """
    sic_mean_per_years=dict()
    sic_median_per_years=dict()

    for year in range(startdate.year,enddate.year+1):
        print(year)
        mean_sic=[]
        median_sic = []
        dates = []
        for date in pd.date_range(datetime(year, startdate.month,startdate.day), datetime(year, enddate.month,enddate.day)):
            sic_colloc,lat, lon, date, sic_mean, sic_median  = read_osi_halo(date, area=area)
            mean_sic = np.append(mean_sic, sic_mean)
            median_sic = np.append(median_sic, sic_median)
            dates = np.append(dates, date)
        mean_sic[mean_sic==-1]=np.nan #no data
        median_sic[median_sic==-1]=np.nan 
        sic_mean_per_years[year] = [mean_sic, dates]
        sic_median_per_years[year] = [median_sic, dates]
    return sic_mean_per_years, sic_median_per_years  
    
def read_osi_climatology(startday, startmonth,startyear, endday, endmonth, endyear):
    """ read_osi_climatology(startday, startmonth,startyear, endday, endmonth, endyear):
    calculates mean sea ice concentration per grid cell from OSI-SAF climate record for a given time period over given years
    input:
    startday: first day of timeperiod
    startmonth: month of first day of timeperiod
    startyear: year to start calculating the climatology from
    endday: last day of timeperiod
    endmonth: month of last day of timeperiod
    endyear: year to end calculating the climatology from
    output:
    sic_climatology: mean sic, 
    lon, lat: corresponding latitude and longitude
    """
   # date = datetime(startyear, startmonth, startday)
  #  path_osi=f"/mnt/raid01/IceConcentration/OSI-SAF/climate_records/{date:%Y}/"
  #  if date.year> 2014:
  #      filename=f"ice_conc_nh_ease2-250_icdr-v2p0_{date:%Y%m%d}1200.nc"
 #   elif date.year< 2015:
  #      filename=f"ice_conc_nh_ease2-250_cdr-v2p0_{date:%Y%m%d}1200.nc"
   # file = os.path.join(path_osi, filename) #joined path and filename
  #  ds = xr.open_dataset(file) #open the file
   

  #  sic = ds.variables['ice_conc'][0].data
 #   sic[sic==-32767]=np.nan  ## fill values are replaced by nans
    sic = [] #make to an array to append to it
  #  ds.close()

    for years in range(startyear, endyear +1):
        path_osi=f"/mnt/spaces/IceConcentration/OSI-SAF/climate_records/{years}/"

        print(years)
        for dates in pd.date_range(datetime(years, startmonth, startday), datetime(years, endmonth, endday)):
           # print(dates)
            if dates.year> 2015:
                filename=f"ice_conc_nh_ease2-250_icdr-v2p0_{dates:%Y%m%d}1200.nc"
            elif dates.year<= 2015:
                filename=f"ice_conc_nh_ease2-250_cdr-v2p0_{dates:%Y%m%d}1200.nc"
            try:

                file = os.path.join(path_osi, filename) #joined path and filename
                ds = xr.open_dataset(file) #open the file
                sic_year = ds.variables['ice_conc'][0].data
                sic_year[sic_year==-32767]=np.nan
                sic.append(sic_year)
                lon = ds.variables['lon'].data
                lat = ds.variables['lat'].data
                ds.close()


            except Exception:
                print("no data for ", dates)

                pass
    sic_climatology = np.nanmean(sic, axis=0)
    return sic_climatology, lon, lat


def read_osi_climatology_uncertainty(startday, startmonth,startyear, endday, endmonth, endyear):
    """ read_osi_climatology_uncertainty(startday, startmonth,startyear, endday, endmonth, endyear):
    calculates uncertainty of sea ice concentration product per grid cell from OSI-SAF climate record for a given time period over given years
    input:
    startday: first day of timeperiod
    startmonth: month of first day of timeperiod
    startyear: year to start calculating the climatology from
    endday: last day of timeperiod
    endmonth: month of last day of timeperiod
    endyear: year to end calculating the climatology from
    output:
    sic_climatology: mean sic uncertainty, 
    lon, lat: corresponding latitude and longitude
    """
    date = datetime(startyear, startmonth, startday)
    path_osi=f"/mnt/spaces/IceConcentration/OSI-SAF/climate_records/{date:%Y}/"
    if date.year> 2015:
        filename=f"ice_conc_nh_ease2-250_icdr-v2p0_{date:%Y%m%d}1200.nc"
    elif date.year<= 2015:
        filename=f"ice_conc_nh_ease2-250_cdr-v2p0_{date:%Y%m%d}1200.nc"
    file = os.path.join(path_osi, filename) #joined path and filename
    ds = xr.open_dataset(file) #open the file
   
    lon = ds.variables['lon'].data
    lat = ds.variables['lat'].data
    sic = ds.variables['total_standard_error'][0].data
    sic[sic==-32767]=np.nan  ## fill values are replaced by nans
    sic = [sic] #make to an array to append to it
    ds.close()
    for years in range(startyear, endyear +1):
        path_osi=f"/mnt/raid01/IceConcentration/OSI-SAF/climate_records/{years}/"

        print(years)
        for dates in pd.date_range(datetime(years, startmonth, startday), datetime(years, endmonth, endday)):
           # print(dates)
            if dates.year> 2015:
                filename=f"ice_conc_nh_ease2-250_icdr-v2p0_{dates:%Y%m%d}1200.nc"
            elif dates.year<= 2015:
                filename=f"ice_conc_nh_ease2-250_cdr-v2p0_{dates:%Y%m%d}1200.nc"
            try:

                file = os.path.join(path_osi, filename) #joined path and filename
                ds = xr.open_dataset(file) #open the file
                sic_year = ds.variables['total_standard_error'][0].data
                sic_year[sic_year==-32767]=np.nan
                sic.append(ds.variables['total_standard_error'][0].data)
                ds.close()

            except Exception:
                print("no data for ", dates)
                pass
    sic_climatology = np.nanmean(sic, axis=0)
    return sic_climatology, lon, lat


def plot_climatology_figure_HALO(sic1,lon1,lat1, sic2, lon2, lat2, extent, title1, title2,sic_climatological_mean, sic_climatological_std,sic_mean_per_years, sic_climatological_25quantile,sic_climatological_75quantile,area, outpath="/home/jrueckert/Output/HALO/results/", filename_out="", boxes="true"):
    matplotlib.rcParams['xtick.labelsize'] = 18
    plt.clf()
    fig= plt.figure(figsize=(20,20), dpi=300)
    fontsizetitle=20
    #plt.suptitle(" ",x=0.5,y=0.9)
    npstere = ccrs.NorthPolarStereo(central_longitude=0, true_scale_latitude=70) #projection that is used for the map (NorthPolarStereo)

    gs = fig.add_gridspec(2,3, hspace=0.1,  height_ratios=[1.5,1],wspace=0.15) #width_ratios= [1,1,1,1],
    ax1 = fig.add_subplot(gs[0,0],projection=npstere)
    ax2 = fig.add_subplot(gs[0,1],projection=npstere)
    ax3 = fig.add_subplot(gs[0,2],projection=npstere)
    axfram = fig.add_subplot(gs[1,:])
        
    cmap=cm.get_cmap("cmo.ice").copy()
    cmap.set_bad("lightgrey") #to mask out invalid values

    ax1.set_extent(extent, ccrs.PlateCarree())
    gl = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=True,linestyle = '--')
    gl.xlabel_style = {'size': 18, 'color': 'black'}
    gl.ylabel_style = {'size': 18, 'color': 'black'}
    #ax.set_global()
    ax1.coastlines(resolution='10m', color = 'black', zorder=2)
    ax1.add_feature(cfeature.LAND, facecolor="darkgrey", zorder=1) #to shade the land 
    ax1.set_title(f'{title1}', fontsize=fontsizetitle)
    pcm=ax1.pcolormesh(lon1,lat1,sic1[1:,1 :],cmap=cmap,shading="flat",transform=ccrs.PlateCarree(), vmin=0, vmax=100)
    
    cbar = plt.colorbar(pcm, ax=ax1,pad = 0.1, shrink =1,orientation="horizontal").set_label( label='SIC (%)', size=18)#
    ax2.set_extent(extent, ccrs.PlateCarree())
    gl = ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=True,linestyle = '--')
    gl.xlabel_style = {'size': 18, 'color': 'black'}
    gl.ylabel_style = {'size': 18, 'color': 'black'}
    #ax.set_global()
    ax2.coastlines(resolution='10m', color = 'black', zorder=2)
    ax2.add_feature(cfeature.LAND, facecolor="darkgrey", zorder=1) #to shade the land 
    ax2.set_title(f'{title2}', fontsize=fontsizetitle)
    pcm=ax2.pcolormesh(lon2,lat2,sic2[1:,1 :],cmap=cmap,shading="flat",transform=ccrs.PlateCarree(), vmin=0, vmax=100)
    
    cbar = plt.colorbar(pcm, ax=ax2, pad = 0.1, shrink =0.9,orientation="horizontal",use_gridspec = True).set_label( label='SIC (%)', size=18)#
    ax3.set_extent(extent, ccrs.PlateCarree())
    gl = ax3.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=True,linestyle = '--')
    gl.xlabel_style = {'size': 18, 'color': 'black'}
    gl.ylabel_style = {'size': 18, 'color': 'black'}
    #ax.set_global()
    ax3.coastlines(resolution='10m', color = 'black', zorder=2)
    ax3.add_feature(cfeature.LAND, facecolor="darkgrey", zorder=1) #to shade the land cv
    ax3.set_title(f'(c) HALO-AC3 year - climatology \n 07 March-12 April', fontsize=fontsizetitle)
    n_bins = 100  # Discretizes the interpolation into bins
    cmap_name = 'cool_warm_mod'
    colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]  # blue -> white -> red
    if boxes == "true":
        ###needs lines to follow latitudes:
        for ax in [ax1,ax2,ax3]:    
               # lon_southern = [0,23,23,0,0]
            lon_southern = np.concatenate([np.linspace(0,23,23), np.linspace(23,0,23),[0]])
          #  lat_southern  = [70.6,70.6,75,75, 70.6]
            lat_southern  = np.concatenate([np.ones(23)*70.6, np.ones(23)*75,[70.6]])
            ax.plot(lon_southern, lat_southern,color="orange", linestyle='-',transform=ccrs.PlateCarree(), label="", zorder=3) #Geodetic not used to reprocude plots from pthers
            #lon_central1 = [-9.0,30.0,30.0,-15,-9]
           # lon_central2= [-15,-54,-54,-9]
            lon_central1  = np.concatenate([[-9],np.linspace(-9,30,39), np.linspace(30,-9,39)])
            lon_central2= np.concatenate([np.linspace(-9,-54,49), np.linspace(-54,-9,49)])#,[-9]])
           # lat_central1  = [81.5,81.5, 89.3,89.3,81.5]
            lat_central1 = np.concatenate([[84.5],np.ones(39)*81.5, np.ones(39)*89.3])#[81.5]])
           # lat_central2  = [89.3,89.3,84.5,84.5]
            lat_central2 = np.concatenate([np.ones(49)*89.3, np.ones(49)*84.5])#,[89.3]])

            ax.plot(lon_central1, lat_central1,color="orange", linestyle='-',transform=ccrs.PlateCarree(), label="", zorder=3)
            ax.plot(lon_central2, lat_central2,color="orange", linestyle='-',transform=ccrs.PlateCarree(), label="", zorder=3)
            #lat_Fram = [75,75,81.5,81.5,75]
            lat_Fram = np.concatenate([np.ones(35)*75, np.ones(35)*81.5,[75]])
            #lon_Fram = [-9,16,16,-9,-9]
            lon_Fram  = np.concatenate([np.linspace(-9,16,35), np.linspace(16,-9,35),[-9]])
            ax.plot(lon_Fram, lat_Fram,color="orange", linestyle='-',transform=ccrs.PlateCarree(), label="", zorder=3)
        

    # Create the colormap
    #cmap_mod = LinearSegmentedColormap.from_list(cmap_name, [(0 ,colors[0]),(2/5, colors[1]), (3/5, colors[1]), (1, colors[2])], N=n_bin)
    cmap_mod = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, [(0 ,colors[0]),(1/3, colors[1]), (2/3, colors[1]), (1, colors[2])], N=n_bins)

    pcm=ax3.pcolormesh(lon2,lat2,sic1[1:,1 :]-sic2[1:,1 :],cmap=cmap_mod,shading="flat",transform=ccrs.PlateCarree(), vmin=-15, vmax=15)
    
    cbar = fig.colorbar(pcm, ax=ax3,pad = 0.1, shrink =0.9,orientation="horizontal",use_gridspec = True).set_label( label='ΔSIC (%)', size=18)#
    
    ### Time series
    dates=pd.date_range(datetime(2022,3,7), datetime(2022,4,12))


    axfram.plot([], [], color="grey", label="1979-2022")

    linewidth=1.5
    for year in range(1979, 2023): ##iterate over dictionary keys
        df = pd.DataFrame(sic_mean_per_years[year][1], columns=['ts'])#timestamp
        df['date'] =  pd.to_datetime(df['ts']) #date to be messed with in order to plot multiple years on one axis (very complicated, probably possible a lot easier to be done)
        df['year'] = df.date.dt.year + (2022-year)
        df["month"] = df.date.dt.month
        df["day"]= df.date.dt.day
        df["date"] = pd.to_datetime( df[['year', 'month', 'day']])
        df["sic"] =sic_mean_per_years[year][0]
        df =df.dropna()
        color= "grey"#-(1/(2022-1977))*year + 2022/(2022-1977)#so that it is a grey scale
        label=""
        if year ==2022:
            linewidth=3
            color="red"
            label="2022"
        axfram.plot(df.date,df.sic, label=label, color=f"{color}",linewidth= linewidth)
  

    axfram.plot(dates,sic_climatological_mean, label=f"climatological mean", color=f"black", ls="-",linewidth= 2.5)
    y =np.array(sic_climatological_mean)
    yerr=np.array(sic_climatological_std)
    axfram.fill_between(dates, y-yerr, y+yerr, color="lightblue", alpha=0.5, label="1σ")
    axfram.fill_between(dates, sic_climatological_25quantile,sic_climatological_75quantile, color="darkblue", alpha=0.5, label="quantiles (25th, 75th)")
    axfram.set_ylim(20,55)


    axfram.set_ylabel("SIC (%)")
    axfram.set_xlabel("Date")

    fontsizey=24
    fontsizex=24
    fontsizetix=22
    fontsizelegend=16


    axfram.set_title(area, size=fontsizetitle, loc="left")
    month_day_fmt = mdates.DateFormatter('%d %b') # "Locale's abbreviated month name. + day of the month"
    axfram.xaxis.set_major_formatter(month_day_fmt)
    plt.legend( loc="upper left", fontsize=fontsizelegend)#bbox_to_anchor=(1.02,3.42),

    for ax in [axfram]:
        plt.setp(ax.yaxis.get_label(), 'size', fontsizey)
        plt.setp(ax.xaxis.get_label(), 'size', fontsizey)

        plt.setp(ax.get_xticklabels(), fontsize=fontsizetix)
        plt.setp(ax.get_yticklabels(), fontsize=fontsizetix)


    
    
    #plt.tight_layout(w_pad=12)
    area = area.replace(" ", "")
    plt.savefig(f"{outpath}comparison_maps_climatology_{area}_{filename_out}.png", bbox_inches="tight", dpi=300)
    plt.show()
    
    
def plot_asi_and_climatology_figure_HALO(asi_sic1, asi_sic2,lon_asi,lat_asi, sic_anomaly, lon_anomaly, lat_anomaly, extent, title1, title2,sic_climatological_mean, sic_climatological_std,sic_mean_per_years, sic_climatological_25quantile,sic_climatological_75quantile,area, outpath="/home/jrueckert/Output/HALO/results/", filename_out="", boxes="true"):
    """plot_asi_and_climatology_figure_HALO(asi_sic1, asi_sic2,lon_asi,lat_asi, sic_anomaly, lon_anomaly, lat_anomaly, extent, title1, title2,sic_climatological_mean, sic_climatological_std,sic_mean_per_years, sic_climatological_25quantile,sic_climatological_75quantile,area, outpath="/home/jrueckert/Output/HALO/results/", filename_out="", boxes="true")
    input:
    asi_sic1: array sea ice concentration for first time period (figure a)
    asi_sic2: array sea ice concentration for second time period (figure b)
    lon_asi, lat_asi: arrays of lon and lat for asi_sic
    sic_anomaly: halo year - climatology sea ice concentration 
     lon_anomaly, lat_anomaly:arrays of lon and lat for anomaly
     extent: extent of map as [minlon, maxlon, minlat, maxlat], e.g. [-18,23.5,66,90]
     title1, title2: title for subplot a and b
     sic_climatological_mean: array with daily mean values for certain time period from climatology
     sic_climatological_std: array with daily std for certain time period of sic from climatology 
     sic_mean_per_years: dictionary, keys are years, values containing daily mean sic for certain time period for each year and corresponding time stamps. 
     sic_climatological_25quantile: array with daily 25th percentile sic value for certain time period of sic from climatology 
     sic_climatological_75quantile: array with daily 75th percentile sic value for certain time period of sic from climatology 
     outpath: where to save the figure
     filename_out: filename extension
     boxes: if "true" draw boxes around defined areas (HALO)
     """
    matplotlib.rcParams['xtick.labelsize'] = 18
    plt.clf()
    fig= plt.figure(figsize=(20,20), dpi=300)
    fontsizetitle=20
    #plt.suptitle(" ",x=0.5,y=0.9)
    npstere = ccrs.NorthPolarStereo(central_longitude=0, true_scale_latitude=70) #projection that is used for the map (NorthPolarStereo)

    gs = fig.add_gridspec(2,3, hspace=0.1,  height_ratios=[1.5,1],wspace=0.15) #width_ratios= [1,1,1,1],
    ax1 = fig.add_subplot(gs[0,0],projection=npstere)
    ax2 = fig.add_subplot(gs[0,1],projection=npstere)
    ax3 = fig.add_subplot(gs[0,2],projection=npstere)
    axfram = fig.add_subplot(gs[1,:])
        
    cmap=cm.get_cmap("cmo.ice").copy()
    cmap.set_bad("lightgrey") #to mask out invalid values

    ax1.set_extent(extent, ccrs.PlateCarree())
    gl = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=True,linestyle = '--')
    gl.xlabel_style = {'size': 18, 'color': 'black'}
    gl.ylabel_style = {'size': 18, 'color': 'black'}
    #ax.set_global()
    ax1.coastlines(resolution='10m', color = 'black', zorder=2)
    ax1.add_feature(cfeature.LAND, facecolor="darkgrey", zorder=1) #to shade the land 
    ax1.set_title(f'{title1}', fontsize=fontsizetitle)
    pcm=ax1.pcolormesh(lon_asi,lat_asi,asi_sic1[1:,1 :],cmap=cmap,shading="flat",transform=ccrs.PlateCarree(), vmin=0, vmax=100)
    
    cbar = plt.colorbar(pcm, ax=ax1,pad = 0.1, shrink =1,orientation="horizontal").set_label( label='SIC (%)', size=18)#
    ax2.set_extent(extent, ccrs.PlateCarree())
    gl = ax2.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=True,linestyle = '--')
    gl.xlabel_style = {'size': 18, 'color': 'black'}
    gl.ylabel_style = {'size': 18, 'color': 'black'}
    #ax.set_global()
    ax2.coastlines(resolution='10m', color = 'black', zorder=2)
    ax2.add_feature(cfeature.LAND, facecolor="darkgrey", zorder=1) #to shade the land 
    ax2.set_title(f'{title2}', fontsize=fontsizetitle)
    pcm=ax2.pcolormesh(lon_asi,lat_asi,asi_sic2[1:,1 :],cmap=cmap,shading="flat",transform=ccrs.PlateCarree(), vmin=0, vmax=100) ##to make pcolormesh work, one column needs to be dropped
    
    cbar = plt.colorbar(pcm, ax=ax2, pad = 0.1, shrink =1,orientation="horizontal",use_gridspec = True).set_label( label='SIC (%)', size=18)#
    ax3.set_extent(extent, ccrs.PlateCarree())
    gl = ax3.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=True,linestyle = '--')
    gl.xlabel_style = {'size': 18, 'color': 'black'}
    gl.ylabel_style = {'size': 18, 'color': 'black'}
    #ax.set_global()
    ax3.coastlines(resolution='10m', color = 'black', zorder=2)
    ax3.add_feature(cfeature.LAND, facecolor="darkgrey", zorder=1) #to shade the land cv
    ax3.set_title(f'(c) HALO-(AC)3 year - climatology 1979–2022 \n 07 March–12 April', fontsize=fontsizetitle)
    n_bins = 100  # Discretizes the interpolation into bins
    cmap_name = 'cool_warm_mod'
    colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]  # blue -> white -> red
    if boxes == "true":
        ###needs lines to follow latitudes:
        for ax in [ax1,ax2,ax3]:    
               # lon_southern = [0,23,23,0,0]
            lon_southern = np.concatenate([np.linspace(0,23,23), np.linspace(23,0,23),[0]])
          #  lat_southern  = [70.6,70.6,75,75, 70.6]
            lat_southern  = np.concatenate([np.ones(23)*70.6, np.ones(23)*75,[70.6]])
            ax.plot(lon_southern, lat_southern,color="orange", linestyle='-',transform=ccrs.PlateCarree(), label="", zorder=3) #Geodetic not used to reprocude plots from pthers
            #lon_central1 = [-9.0,30.0,30.0,-15,-9]
           # lon_central2= [-15,-54,-54,-9]
            lon_central1  = np.concatenate([[-9],np.linspace(-9,30,39), np.linspace(30,-9,39)])
            lon_central2= np.concatenate([np.linspace(-9,-54,49), np.linspace(-54,-9,49)])#,[-9]])
           # lat_central1  = [81.5,81.5, 89.3,89.3,81.5]
            lat_central1 = np.concatenate([[84.5],np.ones(39)*81.5, np.ones(39)*89.3])#[81.5]])
           # lat_central2  = [89.3,89.3,84.5,84.5]
            lat_central2 = np.concatenate([np.ones(49)*89.3, np.ones(49)*84.5])#,[89.3]])

            ax.plot(lon_central1, lat_central1,color="orange", linestyle='-',transform=ccrs.PlateCarree(), label="", zorder=3)
            ax.plot(lon_central2, lat_central2,color="orange", linestyle='-',transform=ccrs.PlateCarree(), label="", zorder=3)
            #lat_Fram = [75,75,81.5,81.5,75]
            lat_Fram = np.concatenate([np.ones(35)*75, np.ones(35)*81.5,[75]])
            #lon_Fram = [-9,16,16,-9,-9]
            lon_Fram  = np.concatenate([np.linspace(-9,16,35), np.linspace(16,-9,35),[-9]])
            ax.plot(lon_Fram, lat_Fram,color="orange", linestyle='-',transform=ccrs.PlateCarree(), label="", zorder=3)
        

    # Create the colormap
    #cmap_mod = LinearSegmentedColormap.from_list(cmap_name, [(0 ,colors[0]),(2/5, colors[1]), (3/5, colors[1]), (1, colors[2])], N=n_bin)
    cmap_mod = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, [(0 ,colors[0]),(1/3, colors[1]), (2/3, colors[1]), (1, colors[2])], N=n_bins)

    pcm=ax3.pcolormesh(lon_anomaly,lat_anomaly,sic_anomaly[1:,1 :],cmap=cmap_mod,shading="flat",transform=ccrs.PlateCarree(), vmin=-15, vmax=15)
    
    cbar = fig.colorbar(pcm, ax=ax3,pad = 0.1, shrink =0.9,orientation="horizontal",use_gridspec = True).set_label( label='ΔSIC (%)', size=18)#
    
    ### Time series
    dates=pd.date_range(datetime(2022,3,7), datetime(2022,4,12))


    axfram.plot([], [], color="grey", label="1979-2022")

    linewidth=1.5
    for year in range(1979, 2023): ##iterate over dictionary keys
        df = pd.DataFrame(sic_mean_per_years[year][1], columns=['ts'])#timestamp
        df['date'] =  pd.to_datetime(df['ts']) #date to be messed with in order to plot multiple years on one axis (very complicated, probably possible a lot easier to be done)
        df['year'] = df.date.dt.year + (2022-year)
        df["month"] = df.date.dt.month
        df["day"]= df.date.dt.day
        df["date"] = pd.to_datetime( df[['year', 'month', 'day']])
        df["sic"] =sic_mean_per_years[year][0]
        df =df.dropna()
        color= "grey"#-(1/(2022-1977))*year + 2022/(2022-1977)#so that it is a grey scale
        label=""
        if year ==2022:
            linewidth=3
            color="red"
            label="2022"
        axfram.plot(df.date,df.sic, label=label, color=f"{color}",linewidth= linewidth)
  

    axfram.plot(dates,sic_climatological_mean, label=f"climatological mean", color=f"black", ls="-",linewidth= 2.5)
    y =np.array(sic_climatological_mean)
    yerr=np.array(sic_climatological_std)
    axfram.fill_between(dates, y-yerr, y+yerr, color="lightblue", alpha=0.5, label="1σ")
    axfram.fill_between(dates, sic_climatological_25quantile,sic_climatological_75quantile, color="darkblue", alpha=0.5, label="quantiles (25th, 75th)")
    axfram.set_ylim(20,55)


    axfram.set_ylabel("SIC (%)")
    axfram.set_xlabel("Date")

    fontsizey=24
    fontsizex=24
    fontsizetix=22
    fontsizelegend=16


    axfram.set_title(area, size=fontsizetitle, loc="left")
    month_day_fmt = mdates.DateFormatter('%d %b') # "Locale's abbreviated month name. + day of the month"
    axfram.xaxis.set_major_formatter(month_day_fmt)
    plt.legend( loc="upper left", fontsize=fontsizelegend)#bbox_to_anchor=(1.02,3.42),

    for ax in [axfram]:
        plt.setp(ax.yaxis.get_label(), 'size', fontsizey)
        plt.setp(ax.xaxis.get_label(), 'size', fontsizey)

        plt.setp(ax.get_xticklabels(), fontsize=fontsizetix)
        plt.setp(ax.get_yticklabels(), fontsize=fontsizetix)


    
    
    #plt.tight_layout(w_pad=12)
    area = area.replace(" ", "")
    plt.savefig(f"{outpath}comparison_maps_climatology_and_ASI_{area}_{filename_out}.png", bbox_inches="tight", dpi=300)
    plt.show()
    
    
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
  
    
def plot_asi_and_climatology_figure_HALO_all(asi_sic1, asi_sic2,asi_sic3, lon_asi,lat_asi, 
                                             sic_anomaly, lon_anomaly, lat_anomaly, extent,
                                             title1, title2,title3, 
                                             sic_climatological_mean, sic_climatological_std,
                                             sic_mean_per_years, sic_climatological_lowquantile,
                                             sic_climatological_highquantile,quantile_label,area, outpath="/home/jrueckert/Output/HALO/results/", 
                                             filename_out="", boxes="true"):
    """plot_asi_and_climatology_figure_HALO(asi_sic1, asi_sic2,lon_asi,lat_asi, sic_anomaly, lon_anomaly, lat_anomaly, extent, title1, title2,sic_climatological_mean, sic_climatological_std,sic_mean_per_years, sic_climatological_25quantile,sic_climatological_75quantile,area, outpath="/home/jrueckert/Output/HALO/results/", filename_out="", boxes="true")
    input:
    asi_sic1: array sea ice concentration for first time period (figure a)
    asi_sic2: array sea ice concentration for second time period (figure b)
     asi_sic3: array sea ice concentration for second time period (figure c)
    lon_asi, lat_asi: arrays of lon and lat for asi_sic
    sic_anomaly: halo year - climatology sea ice concentration 
     lon_anomaly, lat_anomaly:arrays of lon and lat for anomaly
     extent: extent of map as [minlon, maxlon, minlat, maxlat], e.g. [-18,23.5,66,90]
     title1, title2, title3: title for subplot a and b and c
     sic_climatological_mean: array with daily mean values for certain time period from climatology
     sic_climatological_std: array with daily std for certain time period of sic from climatology 
     sic_mean_per_years: dictionary, keys are years, values containing daily mean sic for certain time period for each year and corresponding time stamps. 
     sic_climatological_25quantile: array with daily 25th percentile sic value for certain time period of sic from climatology 
     sic_climatological_75quantile: array with daily 75th percentile sic value for certain time period of sic from climatology 
     outpath: where to save the figure
     filename_out: filename extension
     boxes: if "true" draw boxes around defined areas (HALO)
     """
    matplotlib.rcParams['xtick.labelsize'] = 18
    matplotlib.rcParams['ytick.labelsize'] = 18

    plt.clf()
    
    ###wind barbs, calculate average wind for the three time periods:
    E_DS = xr.open_dataset("/home/jrueckert/Output/HALO/data/ERA5_single_level_10m_wind_20220301-20220430.nc")
    E_DS['u10'] *= 1.943844 ##convert to knots
    E_DS['v10'] *= 1.943844 
    E_DS1 = E_DS.sel(time=slice('2022-03-09', '2022-03-11')).mean('time')
    E_DS2 = E_DS.sel(time=slice('2022-03-14', '2022-03-16')).mean('time')
    E_DS3 = E_DS.sel(time=slice('2022-04-10', '2022-04-12')).mean('time')
    
    fig= plt.figure(figsize=(20,20),dpi=300)# dpi=300)
    fontsizetitle=20
    #plt.suptitle(" ",x=0.5,y=0.9)
    npstere = ccrs.NorthPolarStereo(central_longitude=0, true_scale_latitude=70) #projection that is used for the map (NorthPolarStereo)

    gs = fig.add_gridspec(2,3, hspace=0.1,  height_ratios=[1.5,1],wspace=0.15) #width_ratios= [1,1,1,1],
    ax1 = fig.add_subplot(gs[0,0],projection=npstere)
    ax2 = fig.add_subplot(gs[0,1],projection=npstere)
    ax3 = fig.add_subplot(gs[0,2],projection=npstere)
    axfram = fig.add_subplot(gs[1,:-1])
    axanomaly = fig.add_subplot(gs[1,2],projection=npstere)

        
    cmap=cm.get_cmap("cmo.ice").copy()
    cmap.set_bad("lightgrey") #to mask out invalid values

    ax1.set_extent(extent, ccrs.PlateCarree())
    gl = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=True,linestyle = '--')
    gl.xlabel_style = {'size': 18, 'color': 'black'}
    gl.ylabel_style = {'size': 18, 'color': 'black'}
    gl.top_labels = False
    #ax.set_global()
    ax1.coastlines(resolution='10m', color = 'black', zorder=2)
    ax1.add_feature(cfeature.LAND, facecolor="darkgrey", zorder=1) #to shade the land 
    ax1.set_title(f'{title1}', fontsize=fontsizetitle)
    pcm=ax1.pcolormesh(lon_asi,lat_asi,asi_sic1[1:,1 :],cmap=cmap,shading="flat",transform=ccrs.PlateCarree(), vmin=0, vmax=100,rasterized=True)
    
    cbar = plt.colorbar(pcm, ax=ax1,pad = 0.1, shrink =1,orientation="horizontal").set_label( label='SIC (%)', size=18)#
    
    ax1.barbs(E_DS1.longitude.values, E_DS1.latitude.values, E_DS1.u10.values, E_DS1.v10.values, pivot='middle', length=6, linewidth=2.5, barbcolor=(0,0,0), transform=ccrs.PlateCarree(), regrid_shape=13)
    ax1.barbs(E_DS1.longitude.values, E_DS1.latitude.values, E_DS1.u10.values, E_DS1.v10.values, pivot='middle', length=6, linewidth=0.5, barbcolor=(1,1,1), transform=ccrs.PlateCarree(), regrid_shape=13) 
    
    
    ax2.set_extent(extent, ccrs.PlateCarree())
    gl = ax2.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=True,linestyle = '--')
    gl.xlabel_style = {'size': 18, 'color': 'black'}
    gl.ylabel_style = {'size': 18, 'color': 'black'}
    #ax.set_global()
    ax2.coastlines(resolution='10m', color = 'black', zorder=2)
    ax2.add_feature(cfeature.LAND, facecolor="darkgrey", zorder=1) #to shade the land 
    ax2.set_title(f'{title2}', fontsize=fontsizetitle)
    pcm=ax2.pcolormesh(lon_asi,lat_asi,asi_sic2[1:,1 :],cmap=cmap,shading="flat",transform=ccrs.PlateCarree(), vmin=0, vmax=100,rasterized=True) ##to make pcolormesh work, one column needs to be dropped
    
    cbar = plt.colorbar(pcm, ax=ax2, pad = 0.1, shrink =1,orientation="horizontal",use_gridspec = True).set_label( label='SIC (%)', size=18)#
    ax2.barbs(E_DS2.longitude.values, E_DS2.latitude.values, E_DS2.u10.values, E_DS2.v10.values, pivot='middle', length=6, linewidth=2.5, barbcolor=(0,0,0), transform=ccrs.PlateCarree(), regrid_shape=13)
    ax2.barbs(E_DS2.longitude.values, E_DS2.latitude.values, E_DS2.u10.values, E_DS2.v10.values, pivot='middle', length=6, linewidth=0.5, barbcolor=(1,1,1), transform=ccrs.PlateCarree(), regrid_shape=13) 
    
    ax3.set_extent(extent, ccrs.PlateCarree())
    gl = ax3.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=True,linestyle = '--')
    gl.xlabel_style = {'size': 18, 'color': 'black'}
    gl.ylabel_style = {'size': 18, 'color': 'black'}
    #ax.set_global()
    ax3.coastlines(resolution='10m', color = 'black', zorder=2)
    ax3.add_feature(cfeature.LAND, facecolor="darkgrey", zorder=1) #to shade the land cv
    ax3.set_title(f'{title3}', fontsize=fontsizetitle)
    pcm=ax3.pcolormesh(lon_asi,lat_asi,asi_sic3[1:,1 :],cmap=cmap,shading="flat",transform=ccrs.PlateCarree(), vmin=0, vmax=100,rasterized=True) ##to make pcolormesh work, one column needs to be dropped
    
    cbar = plt.colorbar(pcm, ax=ax3, pad = 0.1, shrink =1,orientation="horizontal",use_gridspec = True).set_label( label='SIC (%)', size=18)#
    ax3.barbs(E_DS3.longitude.values, E_DS3.latitude.values, E_DS3.u10.values, E_DS3.v10.values, pivot='middle', length=6, linewidth=2.5, barbcolor=(0,0,0), transform=ccrs.PlateCarree(), regrid_shape=13)
    ax3.barbs(E_DS3.longitude.values, E_DS3.latitude.values, E_DS3.u10.values, E_DS3.v10.values, pivot='middle', length=6, linewidth=0.5, barbcolor=(1,1,1), transform=ccrs.PlateCarree(), regrid_shape=13) 
    
    axanomaly.set_extent(extent, ccrs.PlateCarree())
    gl = axanomaly.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=True,linestyle = '--')
    gl.xlabel_style = {'size': 18, 'color': 'black'}
    gl.ylabel_style = {'size': 18, 'color': 'black'}
    #ax.set_global()
    axanomaly.coastlines(resolution='10m', color = 'black', zorder=2)
    axanomaly.add_feature(cfeature.LAND, facecolor="darkgrey", zorder=1) #to shade the land cv
    axanomaly.set_title(f'(e) HALO-(AC)$^3$ - climatology \n 07 March–12 April ', fontsize=fontsizetitle, y= 1.05)
    n_bins = 100  # Discretizes the interpolation into bins
    cmap_name = 'cool_warm_mod'
    colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]  # blue -> white -> red
    if boxes == "true":
        ###needs lines to follow latitudes:
        for ax in [ax1,ax2,ax3, axanomaly]:    
            linewidth=1.5
               # lon_southern = [0,23,23,0,0]
            lon_southern = np.concatenate([np.linspace(0,23,23), np.linspace(23,0,23),[0]])
          #  lat_southern  = [70.6,70.6,75,75, 70.6]
            lat_southern  = np.concatenate([np.ones(23)*70.6, np.ones(23)*75,[70.6]])
            ax.plot(lon_southern, lat_southern,color="orange", linestyle='-',linewidth=linewidth,transform=ccrs.PlateCarree(), label="", zorder=3) #Geodetic not used to reprocude plots from pthers
            #lon_central1 = [-9.0,30.0,30.0,-15,-9]
           # lon_central2= [-15,-54,-54,-9]
            lon_central1  = np.concatenate([[-9],np.linspace(-9,30,39), np.linspace(30,-9,39)])
            lon_central2= np.concatenate([np.linspace(-9,-54,49), np.linspace(-54,-9,49)])#,[-9]])
           # lat_central1  = [81.5,81.5, 89.3,89.3,81.5]
            lat_central1 = np.concatenate([[84.5],np.ones(39)*81.5, np.ones(39)*89.3])#[81.5]])
           # lat_central2  = [89.3,89.3,84.5,84.5]
            lat_central2 = np.concatenate([np.ones(49)*89.3, np.ones(49)*84.5])#,[89.3]])
           
            ax.plot(lon_central1, lat_central1,color="orange", linewidth=linewidth,linestyle='-',transform=ccrs.PlateCarree(), label="", zorder=3) #,rasterized=True #rasterized true to have smaller pdf file
            ax.plot(lon_central2, lat_central2,color="orange",linewidth=linewidth, linestyle='-',transform=ccrs.PlateCarree(), label="", zorder=3) #,rasterized=True
            #lat_Fram = [75,75,81.5,81.5,75]
            lat_Fram = np.concatenate([np.ones(35)*75, np.ones(35)*81.5,[75]])
            #lon_Fram = [-9,16,16,-9,-9]
            lon_Fram  = np.concatenate([np.linspace(-9,16,35), np.linspace(16,-9,35),[-9]])
            if ax == axanomaly:
                linewidth= 5
            ax.plot(lon_Fram, lat_Fram,color="orange", linewidth=linewidth, linestyle='-',transform=ccrs.PlateCarree(), label="", zorder=3) #,rasterized=True
        

    # Create the colormap
    #cmap_mod = LinearSegmentedColormap.from_list(cmap_name, [(0 ,colors[0]),(2/5, colors[1]), (3/5, colors[1]), (1, colors[2])], N=n_bin)
    cmap_mod = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, [(0 ,colors[0]),(1/3, colors[1]), (2/3, colors[1]), (1, colors[2])], N=n_bins)

    pcm=axanomaly.pcolormesh(lon_anomaly,lat_anomaly,sic_anomaly[1:,1 :],cmap=cmap_mod,shading="flat",transform=ccrs.PlateCarree(), vmin=-15, vmax=15)
    axins = inset_axes(axanomaly,
                   width="5%",  # width = 5% of parent_bbox width
                   height="100%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1.11, 0, 1, 1),
                   bbox_transform=axanomaly.transAxes,
                   borderpad=0,
                   )
    cbar = fig.colorbar(pcm, cax=axins,pad = 0.1, shrink =0.9,orientation="vertical",use_gridspec = True).set_label( label='ΔSIC (%)', size=18)#
    
    ### Time series
    dates=pd.date_range(datetime(2022,3,7), datetime(2022,4,12))


    for year in range(2022, 2023): ##iterate over dictionary keys
        df = pd.DataFrame(sic_mean_per_years[year][1], columns=['ts'])#timestamp
        df['date'] =  pd.to_datetime(df['ts']) #date to be messed with in order to plot multiple years on one axis (very complicated, probably possible a lot easier to be done)
        df['year'] = df.date.dt.year + (2022-year)
        df["month"] = df.date.dt.month
        df["day"]= df.date.dt.day
        df["date"] = pd.to_datetime( df[['year', 'month', 'day']])
        df["sic"] =sic_mean_per_years[year][0]
        df =df.dropna()
        color= "grey"#-(1/(2022-1977))*year + 2022/(2022-1977)#so that it is a grey scale
        label=""
        if year ==2022:
            linewidth=3
            color="red"
            label="2022"
        axfram.plot(df.date,df.sic, label=label, color=f"{color}",linewidth= linewidth)
  

    axfram.plot(dates,sic_climatological_mean, label=f"climatological mean", color=f"black", ls="-",linewidth= 2.5)
    y =np.array(sic_climatological_mean)
    yerr=np.array(sic_climatological_std)
  #  axfram.fill_between(dates, y-yerr, y+yerr, color="lightblue", alpha=0.5, label="1σ")
    axfram.fill_between(dates, sic_climatological_lowquantile,sic_climatological_highquantile, color="darkblue", alpha=0.25, label=quantile_label)
    axfram.set_ylim(20,55)
    axfram.set_xlim((datetime(2022,3,7), datetime(2022,4,12)))
    axfram.grid(color='grey',linewidth=0.4,alpha=0.5) 

    axfram.set_ylabel("SIC (%)")
    axfram.set_xlabel("Date")

    fontsizey=22
    fontsizex=22
    fontsizetix=18
    fontsizelegend=16


    axfram.set_title(area, size=fontsizetitle, loc="left", y= 0.95)
    month_day_fmt = mdates.DateFormatter('%d %b') # "Locale's abbreviated month name. + day of the month"
    axfram.xaxis.set_major_formatter(month_day_fmt)
    axfram.legend( loc="upper left", fontsize=fontsizelegend)#bbox_to_anchor=(1.02,3.42),

    for ax in [axfram]:
        plt.setp(ax.yaxis.get_label(), 'size', fontsizey)
        plt.setp(ax.xaxis.get_label(), 'size', fontsizey)

        plt.setp(ax.get_xticklabels(), fontsize=fontsizetix)
        plt.setp(ax.get_yticklabels(), fontsize=fontsizetix)


    
    
    #plt.tight_layout(w_pad=12)
    area = area.replace(" ", "")
    plt.savefig(f"{outpath}comparison_maps_climatology_and_ASI_all_{filename_out}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{outpath}comparison_maps_climatology_and_ASI_all_{filename_out}.pdf", bbox_inches="tight", dpi=300)
    plt.show()
    
    
def plot_asi_sic(date, extent=[-18,23.5,66,90], outpath = "/home/jrueckert/Output/HALO/results/ASI/"):
    if date.year <2012:
        path = f'/ssmi/www/htdocs/data/amsre/asi_daygrid_swath/n6250/netcdf/{date:%Y}/' #insert the path to your file here, e.g. '/home/username/Data/sea_ice_concentration/'
        filename = f'asi-n6250-{date:%Y%m%d}-v5.4.nc' #name of the file, if you use a different version of the data make sure to adapt the filename accordingly
        file = os.path.join(path, filename) #joined path and filename
    elif date.year >2012: 
        path = f'/ssmi/www/htdocs/data/amsr2/asi_daygrid_swath/n6250/netcdf/{date:%Y}/' #insert the path to your file here, e.g. '/home/username/Data/sea_ice_concentration/'
        filename = f'asi-AMSR2-n6250-{date:%Y%m%d}-v5.4.nc' #name of the file, if you use a different version of the data make sure to adapt the filename accordingly
        file = os.path.join(path, filename) #joined path and filename
    ds = xr.open_dataset(file) #open the file
    x = ds.variables['x']
    y = ds.variables['y']
    sic = ds.variables['z']
    projection = ds.variables["polar_stereographic"] #To convert from x and y to lat/lon we need the information about the grid contained in this variable
    m=Proj(projection.attrs["spatial_ref"]) #define a function for conversion that gets the information from the string contained in projection
    xx,yy = np.meshgrid(x.data,y.data) #create a meshgrid from x and y to use in function m()
    lon, lat = m(xx,yy, inverse=True)
    
    fig = plt.figure(figsize=(15,15))
    cmap=cm.get_cmap("cmo.ice").copy()
    cmap.set_bad("lightgrey")
    npstere = ccrs.NorthPolarStereo(central_longitude=0, true_scale_latitude=70) #projection that is used for the map (NorthPolarStereo)
    ax = fig.add_subplot(1,1,1,projection=npstere)
    ax.set_extent(extent, ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=True,linestyle = '--')
    gl.xlabel_style = {'size': 18, 'color': 'black'}
    gl.ylabel_style = {'size': 18, 'color': 'black'}
#ax.set_global()
    ax.coastlines(resolution='50m', color = 'black', zorder=2)
    ax.add_feature(cfeature.LAND, facecolor="darkgrey", zorder=1) #to shade the land 
#ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_title(f"Daily ASI sea ice concentration for {date:%Y-%m-%d}", fontsize=18)
    pcm=ax.scatter(lon,lat,c=sic,cmap=cmap,transform=ccrs.PlateCarree(), s=1)

    plt.colorbar(pcm, pad = 0.1, ax = ax, orientation="horizontal").set_label( label='SIC', size=18)#, weight='bold')
    plt.savefig(f"{outpath}asi_sic_{date:%Y%m%d}_HALO_area.png", bbox_inches="tight", dpi=300)

    plt.show()