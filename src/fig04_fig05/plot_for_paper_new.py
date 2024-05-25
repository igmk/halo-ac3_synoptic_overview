import numpy as np
import pandas as pd
import xarray as xr
from pyhdf import SD
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib.dates as dates
import matplotlib.patheffects as mpl_pe
from matplotlib.legend_handler import HandlerBase

    
# functions to calculate mean and standard deviation time series for inhomogeneously shaped fields, weight with latitude
def nanmean2d(inp_arr,selection,lat_arr):
    out_arr = np.empty(len(inp_arr[:,0,0]))
    for i,val in enumerate(inp_arr[:,0,0]):
        inp_arr_i = inp_arr[i,:,:]
        #out_arr[i] = np.nanmean(inp_arr_i[selection]) 
        out_arr[i] = np.average(inp_arr_i[selection], weights=np.cos(lat_arr[selection]/180.*np.pi))
    return out_arr
def nanstd2d(inp_arr,selection,lat_arr):
    out_arr = np.empty(len(inp_arr[:,0,0]))
    for i,val in enumerate(inp_arr[:,0,0]):
        inp_arr_i = inp_arr[i,:,:]
        M = len(np.cos(lat_arr[selection]/180.*np.pi)[np.cos(lat_arr[selection]/180.*np.pi) != 0.].flatten())
        #out_arr[i] = np.nanstd(inp_arr_i[selection])
        out_arr[i] = np.sqrt((np.sum(np.cos(lat_arr[selection]/180.*np.pi)*(inp_arr_i[selection] - np.average(inp_arr_i[selection], weights=np.cos(lat_arr[selection]/180.*np.pi)))**2.))/((M-1)/M*np.sum(np.cos(lat_arr[selection]/180.*np.pi))))
    return out_arr
    

# plotting settings
mpl.use('TkAgg')
mpl.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['figure.dpi'] = 75 #300
mpl.rcParams['font.size'] = 14
cm = 3./4. #1/2.54

# latitude/longitude boundaries of ERA5 data (indicated in file names)
lat_l = '60'
lat_u = '90'
lon_l = '-80'
lon_u = '80'

time = '12:00 UTC'

# read sea ice grid
###################
GeoGrid = SD.SD('/projekt_agmwend/home_rad/Sebastian/Satellite_Icefraction/LongitudeLatitudeGrid-n3125-Arctic3125.hdf')
latgrid = GeoGrid.select('Latitudes')[:,:]
longrid = GeoGrid.select('Longitudes')[:,:]
longrid[longrid > 180] = longrid[longrid > 180]-360.
del(GeoGrid)

# read separate ERA5 data sets for March/April
##############################################

# surface pressure, temperature, wind and integrated water vapour (daily resolution 12 UTC)
ERA_data_surf_March = xr.load_dataset('/home/sbecker/Dokumente/HALO-AC3_Synoptic/datasets/ERA5_surf_T_p_uv_IWV_202203_'+lat_l+'_'+lat_u+'_'+lon_l+'_'+lon_u+'.nc')
ERA_data_surf_April = xr.load_dataset('/home/sbecker/Dokumente/HALO-AC3_Synoptic/datasets/ERA5_surf_T_p_uv_IWV_202204_'+lat_l+'_'+lat_u+'_'+lon_l+'_'+lon_u+'.nc')

# pressure levels: geopotential height (850 hPa, 500 hPa), temperature, relative humidity, specfic water contents (850 hPa) (daily resolution 12 UTC)
ERA_data_850_March = xr.load_dataset('/home/sbecker/Dokumente/HALO-AC3_Synoptic/datasets/ERA5_850_T_h_RH_mixing_202203_'+lat_l+'_'+lat_u+'_'+lon_l+'_'+lon_u+'.nc')
ERA_data_850_April = xr.load_dataset('/home/sbecker/Dokumente/HALO-AC3_Synoptic/datasets/ERA5_850_T_h_RH_mixing_202204_'+lat_l+'_'+lat_u+'_'+lon_l+'_'+lon_u+'.nc')
ERA_data_500_March = xr.load_dataset('/home/sbecker/Dokumente/HALO-AC3_Synoptic/datasets/ERA5_500_h_202203_'+lat_l+'_'+lat_u+'_'+lon_l+'_'+lon_u+'.nc')
ERA_data_500_April = xr.load_dataset('/home/sbecker/Dokumente/HALO-AC3_Synoptic/datasets/ERA5_500_h_202204_'+lat_l+'_'+lat_u+'_'+lon_l+'_'+lon_u+'.nc')

# read time and geographic coordinates
datetime = np.hstack((ERA_data_surf_March['time'].values.astype('datetime64[D]'),ERA_data_surf_April['time'].values.astype('datetime64[D]')))
date = pd.to_datetime(datetime)
latitude = ERA_data_surf_March['latitude'].values
longitude = ERA_data_surf_March['longitude'].values

# read ERA5 land mask
#####################
Land_Mask = xr.load_dataset('/home/sbecker/Dokumente/HALO-AC3_Synoptic/datasets/ERA5_Land-Sea-Mask.nc')
land = (Land_Mask['lsm'].values)[0,:,:]


g_n = 9.80665       # gravity constant
Pres_0 = 1000.
Pres_850 = 850.

R_dry = 287.05      # specific gas constant dry air
R_wv = 461.         # specific gas constant water vapour

cp_dry = 1004.      # specific heat capacity of dry air at constant pressure


# stack March/April data
########################

# pressure level properties
Geop_500 = np.vstack((ERA_data_500_March['z'].values,ERA_data_500_April['z'].values))/g_n/10.           # 500 hPa geopotential height
Geop_850 = np.vstack((ERA_data_850_March['z'].values,ERA_data_850_April['z'].values))/g_n/10.           # 850 hPa geopotenital height (not needed further)
Temp_850 = np.vstack((ERA_data_850_March['t'].values,ERA_data_850_April['t'].values)) - 273.15          # 850 hPa temperature (for Theta_E_850 calculation and for time series)
RH_850 = np.vstack((ERA_data_850_March['r'].values,ERA_data_850_April['r'].values))                     # 850 hPa relative humidity (for Theta_E_850 calculation)
q_lw = np.vstack((ERA_data_850_March['clwc'].values,ERA_data_850_April['clwc'].values))                 # specific liquid water content
q_iw = np.vstack((ERA_data_850_March['ciwc'].values,ERA_data_850_April['ciwc'].values))                 # specific ice water content
q_rw = np.vstack((ERA_data_850_March['crwc'].values,ERA_data_850_April['crwc'].values))                 # specific rain water content
q_sw = np.vstack((ERA_data_850_March['cswc'].values,ERA_data_850_April['cswc'].values))                 # specific snow water content
q_wv = np.vstack((ERA_data_850_March['q'].values,ERA_data_850_April['q'].values))                       # specific humidity
r_wv_850 = q_wv/(1-(q_lw + q_iw + q_rw + q_sw + q_wv))                                                  # water vapour mixing ratio (for Theta_E_850 calculation)
r_tw_850 = (q_lw + q_iw + q_rw + q_sw + q_wv)/(1-(q_lw + q_iw + q_rw + q_sw + q_wv))                    # total water mixing ratio (not needed further)

# surface properties and integrated water vapour (for time series)
Temp_srf = np.vstack((ERA_data_surf_March['t2m'].values,ERA_data_surf_April['t2m'].values)) - 273.15    # 2 m air temperature
Pres_srf = np.vstack((ERA_data_surf_March['msl'].values,ERA_data_surf_April['msl'].values))/100.        # mean sea level pressure
u_srf = np.vstack((ERA_data_surf_March['u10'].values,ERA_data_surf_April['u10'].values))                # 10 m wind (eastward component)
v_srf = np.vstack((ERA_data_surf_March['v10'].values,ERA_data_surf_April['v10'].values))                # 10 m wind (northward component)
IWV = np.vstack((ERA_data_surf_March['tcwv'].values,ERA_data_surf_April['tcwv'].values))                # integrated water vapour


# calculate equivcalent-potential temperature in 850 hPa (Theta_E_850) and surface wind direction (wind_dir_srf) and speed (wind_speed_srf)
###########################################################################################################################################
L_wv = (2500.8 - 2.36*Temp_850 + 0.0016*Temp_850**2. - 0.00006*Temp_850**3.)*1000.
Theta_E_850 = (Temp_850+273.15)*(Pres_0/Pres_850)**(R_dry/cp_dry) *(RH_850/100.)**(-r_wv_850*R_wv/cp_dry) *np.exp((L_wv*r_wv_850)/(cp_dry*(Temp_850+273.15))) - 273.15
#Theta_E_850 = (Temp_850+273.15)*(Pres_0/Pres_850)**(R_dry/cp_dry) *np.exp((L_wv*r_wv_850)/(cp_dry*(Temp_850+273.15))) - 273.15
#Theta_E_850 = (Temp_850+273.15)*(Pres_0/Pres_850)**(R_dry/(cp_dry+r_tw_850*cp_lw)) *RH_850**(-r_wv_850*R_wv/(cp_dry+r_tw_850*cp_lw)) *np.exp((L_wv*r_wv_850)/((cp_dry+r_tw_850*cp_lw)*(Temp_850+273.15))) - 273.15
wind_dir_srf = np.arctan2(u_srf,v_srf)*180./np.pi + 180.
wind_speed_srf = np.sqrt(u_srf**2. + v_srf**2.)

Pres_grad = np.gradient(Pres_srf, axis=0)

lon, lat = np.meshgrid(longitude,latitude)

# 6 exemplary days of synoptic overview maps
Dates = ['2022-03-13','2022-03-15','2022-03-21','2022-03-28','2022-04-01','2022-04-08']
plabel = ['(a)','(b)','(c)','(d)','(e)','(f)']



############
# plot maps
############

# basic map configuration
def prepare_map(ax):
    lon_KIR, lat_KIR = np.meshgrid(20.225,67.856)
    lon_LYR, lat_LYR = np.meshgrid(15.647,78.223)

    map = Basemap(llcrnrlon=-20, llcrnrlat=60, urcrnrlon=76, urcrnrlat=80, lat_0=70, lon_0=10, projection='stere', resolution='l', ax=ax)
    map.drawcoastlines(ax=ax)
    map.drawparallels(np.arange(0.,90.,5.), linewidth=1.2, labels=[1,0,0,0], color='grey', zorder=2, ax=ax)
    map.drawmeridians(np.arange(-180.,180.,15.), linewidth=1.2, labels=[0,0,0,1], color='grey', zorder=2, ax=ax)
    map.scatter(lon_KIR, lat_KIR, color='red', marker='^', s=80, latlon=True, zorder=10)
    map.scatter(lon_LYR, lat_LYR, color='red', marker='^', s=80, latlon=True, zorder=10)
    return map


weights = np.cos(lat*np.pi/180.)

fig, ax = plt.subplots(3,2, figsize=(16*cm,20*cm))
map = Basemap(llcrnrlon=-20, llcrnrlat=60, urcrnrlon=76, urcrnrlat=80, lat_0=70, lon_0=10, projection='stere', resolution='l')

# location of H/L to plot on the panels of the exemplary days
coords_H = {'2022-03-13': [map(40,73.5)], '2022-03-15': [], '2022-03-21': [map(-37,73.8)], '2022-03-28': [], '2022-04-01': [map(-33.5,71.2)], '2022-04-08': [map(-37.5,78.5)]}
#coords_H = {'2022-03-13': [map(40,73.5)], '2022-03-15': [map(25,67)], '2022-03-21': [map(-37,73.8))], '2022-03-28': [map(10,85.2)], '2022-04-01': [map(-33.5,71.2),map(-22,84.3)], '2022-04-08': [map(-37.5,78.5)]}
coords_L = {'2022-03-13': [map(-48,77)], '2022-03-15': [map(-26,75.7)], '2022-03-21': [map(25,77.5)], '2022-03-28': [], '2022-04-01': [map(15,74)], '2022-04-08': []}
#coords_L = {'2022-03-13': [map(-48,77)], '2022-03-15': [map(-26,75.7)], '2022-03-21': [map(25,77.5)], '2022-03-28': [map(34,68.5)], '2022-04-01': [map(15,74)], '2022-04-08': [map(20,63.7)]}


# loop over exemplary days
for t,val in enumerate(Dates):

    # read satellite-derived sea ice concentration
    ##############################################
    sat_file = SD.SD('/projekt_agmwend/home_rad/Sebastian/Satellite_Icefraction/HALO-AC3/asi-AMSR2-n3125-'+Dates[t][0:4]+Dates[t][5:7]+Dates[t][8:10]+'-v5.4.hdf')
    ice_conc_xy = sat_file.select('ASI Ice Concentration')[:,:]
    del(sat_file)
    
    # plot maps of mean sea level pressure (Pres_srf), 500 hPa geopotential (Geop_500), 850 hPa equivalent-potential temperature (Theta_E_850), sea ice concentation (ice_conc_xy) for exemplary day
    map = prepare_map(ax[int(np.floor(t/2.)),int(t-2*np.floor(t/2.))])    
    ax[int(np.floor(t/2.)),int(t-2*np.floor(t/2.))].set_title(plabel[t]+' '+pd.to_datetime(np.datetime64(Dates[t])).strftime('%d %B %Y'))
    if ((Dates[t] == '2022-04-08') | (Dates[t] == '2022-04-09')):
        l10 = map.contour(lon, lat, Pres_srf[date == pd.to_datetime(np.datetime64(Dates[t])),:,:][0,:,:], levels=np.arange(0,1100,2), colors='black', linestyles='-', linewidths=1.5, latlon=True, ax=ax[int(np.floor(t/2.)),int(t-2*np.floor(t/2.))])
        l1 = map.contour(lon, lat, Pres_srf[date == pd.to_datetime(np.datetime64(Dates[t])),:,:][0,:,:], levels=np.arange(0,1100,2), colors='white', linestyles='-', linewidths=1.0, latlon=True, ax=ax[int(np.floor(t/2.)),int(t-2*np.floor(t/2.))])
    else:
        l10 = map.contour(lon, lat, Pres_srf[date == pd.to_datetime(np.datetime64(Dates[t])),:,:][0,:,:], levels=np.arange(0,1100,4), colors='black', linestyles='-', linewidths=2.0, latlon=True, ax=ax[int(np.floor(t/2.)),int(t-2*np.floor(t/2.))])
        l1 = map.contour(lon, lat, Pres_srf[date == pd.to_datetime(np.datetime64(Dates[t])),:,:][0,:,:], levels=np.arange(0,1100,4), colors='white', linestyles='-', linewidths=1.5, latlon=True, ax=ax[int(np.floor(t/2.)),int(t-2*np.floor(t/2.))])
    l2 = map.contour(lon, lat, Geop_500[date == pd.to_datetime(np.datetime64(Dates[t])),:,:][0,:,:], levels=np.arange(0,700,4), colors='black', linestyles='-', linewidths=1., latlon=True, ax=ax[int(np.floor(t/2.)),int(t-2*np.floor(t/2.))])
    f1 = map.contourf(lon, lat, Theta_E_850[date == pd.to_datetime(np.datetime64(Dates[t])),:,:][0,:,:], levels=np.arange(-32,34,2), extend='both', extendfrac='auto', cmap='gist_rainbow_r', latlon=True, ax=ax[int(np.floor(t/2.)),int(t-2*np.floor(t/2.))])
    f11 = map.contour(lon, lat, Theta_E_850[date == pd.to_datetime(np.datetime64(Dates[t])),:,:][0,:,:], levels=np.arange(-32,36,4), colors='grey', linestyles='-', linewidths=0.5, latlon=True, ax=ax[int(np.floor(t/2.)),int(t-2*np.floor(t/2.))])
    c10 = ax[int(np.floor(t/2.)),int(t-2*np.floor(t/2.))].clabel(l10, l10.levels, inline=True, fmt="%4i", fontsize=8, colors='none')
    c1 = ax[int(np.floor(t/2.)),int(t-2*np.floor(t/2.))].clabel(l1, l1.levels, inline=True, fmt="%4i", fontsize=8, colors='white')
    plt.setp(c1, path_effects=[mpl_pe.withStroke(linewidth=1.0, foreground='black')])
    c2 = ax[int(np.floor(t/2.)),int(t-2*np.floor(t/2.))].clabel(l2, l2.levels, inline=True, fmt="%4i", fontsize=8)
    c3 = ax[int(np.floor(t/2.)),int(t-2*np.floor(t/2.))].clabel(f11, f11.levels, inline=True, fmt="%4i", fontsize=8)
    f4 = map.contour(longrid, latgrid, ice_conc_xy, levels=[15], colors='mediumblue', linestyles='-', linewidths=2., latlon=True, ax=ax[int(np.floor(t/2.)),int(t-2*np.floor(t/2.))])
    # add Highs/Lows
    for i,val in enumerate(coords_H[Dates[t]]):
        ax[int(np.floor(t/2.)),int(t-2*np.floor(t/2.))].annotate('H', coords_H[Dates[t]][i], xycoords='data', va='center', ha='center', fontsize=18, fontweight='bold', color='white', path_effects=[mpl_pe.Stroke(linewidth=2.0, foreground='black'), mpl_pe.Normal()])
    for i,val in enumerate(coords_L[Dates[t]]):
        ax[int(np.floor(t/2.)),int(t-2*np.floor(t/2.))].annotate('L', coords_L[Dates[t]][i], xycoords='data', va='center', ha='center', fontsize=18, fontweight='bold', color='white', path_effects=[mpl_pe.Stroke(linewidth=2.0, foreground='black'), mpl_pe.Normal()])

# colorbar and legend entries
cbar = fig.colorbar(f1, ax=ax, orientation='vertical', shrink=0.6, label='850 hPa equivalent-potential temperature (°C)')
ls0 = lines.Line2D([], [], color='black', linewidth=2.0, label='Mean sea level pressure (hPa)')
ls = lines.Line2D([], [], color='white', linewidth=1.0, label='Mean sea level pressure (hPa)')
ls1 = (ls0,ls)
lh = lines.Line2D([], [], color='black', linewidth=1., label='500 hPa geopotential height (gpdm)')
li = lines.Line2D([], [], color='darkblue', linestyle='-', linewidth=2., label='15 % sea ice concentration')
ax[0,0].legend(handles=[ls1,lh], labels=[ls0.get_label(),lh.get_label()], loc='lower left')
ax[0,1].legend(handles=[li], loc='lower left')
    
#plt.savefig('/home/sbecker/Dokumente/HALO-AC3_Synoptic/plots/Map_MSLP_thetaE850_overview3.pdf')
#plt.close()
plt.show()



plot = 'separate' #'together' 'separate' 'alternative'


###################
# plot time series
###################

# select domains
latlon_select = [((lat >= 70.6) & (lat <= 75.) & (lon >= 0.) & (lon <= 23.) & (land == 0.0)),
                 ((lat >= 75.) & (lat <= 81.5) & (lon >= -9.) & (lon <= 16.) & (land == 0.0)),
                 (((lat >= 81.5) & (lat <= 89.3) & (lon >= -9.) & (lon <= 30.)) | ((lat >= 84.5) & (lat <= 89.3) & (lon >= -54.) & (lon < -9.)) & (land == 0.0))]
colors = ['red','green','blue']
labels = ['S','C','N'] 
pos_brbs = [-2.5,0,2.5]
#pos_bars = [-0.3,0,0.3]
#pos_bars = [-7,0,7]
ls = []


# set up plot
panels = 5
rows=panels+1
fig, axs = plt.subplots(rows,1,figsize=(8.3*cm,12*cm), sharex=False, squeeze=False, gridspec_kw={'height_ratios': [0.2, 1,0.5,1,1,1]})
plabel = ['(a)','(b)','(c)','(d)','(e)']
qs = [1,2,3,4,5] #[1,2,3,4] #[1,2,3,4,5]
ps = np.zeros(panels, dtype=int)
qleg, pleg, posleg = 0, 0, 'center'
pos_label = 0.87


# loop over domains
for sel,val in enumerate(latlon_select):

    # caluclate mean and standard deviation for domains
    ###################################################
    
    Temp_850_mean = nanmean2d(Temp_850,latlon_select[sel],lat)
    Temp_srf_mean = nanmean2d(Temp_srf,latlon_select[sel],lat)
    Pres_srf_mean = nanmean2d(Pres_srf,latlon_select[sel],lat)
    wind_dir_srf_mean = nanmean2d(wind_dir_srf,latlon_select[sel],lat)      # not needed further, calculated from u, v
    wind_speed_srf_mean = nanmean2d(wind_speed_srf,latlon_select[sel],lat)  # not needed further, calculated from u, v
    u_srf_mean = nanmean2d(u_srf,latlon_select[sel],lat)
    v_srf_mean = nanmean2d(v_srf,latlon_select[sel],lat)
    IWV_mean = nanmean2d(IWV,latlon_select[sel],lat)
    
    Temp_850_std = nanstd2d(Temp_850,latlon_select[sel],lat)
    Temp_srf_std = nanstd2d(Temp_srf,latlon_select[sel],lat)
    Pres_srf_std = nanstd2d(Pres_srf,latlon_select[sel],lat)
    wind_dir_srf_std = nanstd2d(wind_dir_srf,latlon_select[sel],lat)        # not needed further
    wind_speed_srf_std = nanstd2d(wind_speed_srf,latlon_select[sel],lat)    # not needed further
    IWV_std = nanstd2d(IWV,latlon_select[sel],lat)
    
    # plot time series of mean sea level pressure (Pres_srf), 2 m air temperature (Temp_srf) 805 hPa temperature (Temp_850), wind barbs, integrated water vapour (IWV)
    # plot mean as line and standard deviation as shading at daily resolution
    ##################################################################################################################################################################
    axs[qs[0],ps[0]].plot(date,Pres_srf_mean, color=colors[sel])
    axs[qs[0],ps[0]].fill_between(date, Pres_srf_mean-Pres_srf_std ,Pres_srf_mean+Pres_srf_std, color=colors[sel], edgecolor=None, alpha=0.5)
    axs[qs[0],ps[0]].grid(which='major', axis='y', color='grey', alpha=0.5, linewidth=1)
    axs[qs[2],ps[2]].plot(date,Temp_srf_mean, color=colors[sel])
    axs[qs[2],ps[2]].fill_between(date, Temp_srf_mean-Temp_srf_std ,Temp_srf_mean+Temp_srf_std, color=colors[sel], edgecolor=None, alpha=0.5)
    axs[qs[2],ps[2]].set_yticks([-30,-20,-10,0])
    axs[qs[2],ps[2]].grid(which='major', axis='y', color='grey', alpha=0.5, linewidth=1)
    axs[qs[1],ps[1]].barbs(date,np.arange(len(date))*0.+pos_brbs[sel], u_srf_mean/0.514, v_srf_mean/0.514, barbcolor=colors[sel], flagcolor=colors[sel], pivot='middle', length=5, zorder=10)
    axs[qs[1],ps[1]].set_ylim(-4,6)
    axs[qs[1],ps[1]].set_yticks([])
    axs[qs[3],ps[3]].plot(date,Temp_850_mean, color=colors[sel])
    axs[qs[3],ps[3]].fill_between(date, Temp_850_mean-Temp_850_std ,Temp_850_mean+Temp_850_std, color=colors[sel], edgecolor=None, alpha=0.5)
    axs[qs[3],ps[3]].set_yticks([-30,-20,-10,0])
    axs[qs[3],ps[3]].grid(which='major', axis='y', color='grey', alpha=0.5, linewidth=1)
    axs[qs[4],ps[4]].plot(date,IWV_mean, color=colors[sel])
    axs[qs[4],ps[4]].fill_between(date, IWV_mean-IWV_std ,IWV_mean+IWV_std, color=colors[sel], edgecolor=None, alpha=0.5)
    axs[qs[4],ps[4]].grid(which='major', axis='y', color='grey', alpha=0.5, linewidth=1)
    
    # legend entries
    l1 = lines.Line2D([],[], linestyle='-', color=colors[sel], label=labels[sel])
    p1 = patches.Patch(color=colors[sel], edgecolor=None, alpha=0.5, label=labels[sel])
    ls.append(tuple((l1,p1)))
ls.reverse()
labels.reverse()
axs[qleg,pleg].axis('off')
axs[qleg,pleg].legend(handles=ls, labels=labels, ncol=3, frameon=False, loc=posleg)  
    
# finalize plot
for i in range(panels):
    
    axs[qs[i],ps[i]].set_xlim(pd.to_datetime(np.min(datetime)-np.timedelta64(1,'D')), pd.to_datetime(np.max(datetime)+np.timedelta64(1,'D')))
    if i==1:
        axs[qs[i],ps[i]].annotate(plabel[i], (0.01,0.75), xycoords='axes fraction')
    else:
        axs[qs[i],ps[i]].annotate(plabel[i], (0.01,pos_label), xycoords='axes fraction')
        
    # mark the 6 exemplary days in time series
    for t,val in enumerate(Dates):
        axs[qs[i],ps[i]].axvline(pd.to_datetime(np.datetime64(Dates[t])), color='black', linewidth=0.6, zorder=9)
        
    # axes ticks and labels
    axs[qs[i],ps[i]].xaxis.set_minor_locator(dates.DayLocator(interval=1))
    axs[qs[i],ps[i]].xaxis.set_major_locator(dates.DayLocator(interval=7))
    if qs[i] != rows-1:
        axs[qs[i],ps[i]].set_xticklabels('')
    else:
        axs[qs[i],ps[i]].set_xlabel('Date in 2022')
        axs[qs[i],ps[i]].xaxis.set_major_formatter(dates.DateFormatter('%d %b'))
axs[qs[0],ps[0]].set_ylabel('Mean sea level\npressure (hPa)')
axs[qs[2],ps[2]].set_ylabel('2$\,$m air\ntemperature (°C)')
axs[qs[1],ps[1]].set_ylabel('10$\,$m\n'+'wind\n'+r'(kn)')
axs[qs[3],ps[3]].set_ylabel('850$\,$hPa\ntemperature (°C)')
axs[qs[4],ps[4]].set_ylabel(r'IWV ($\mathrm{kg\,m^{-2}}$)')
 
    
#plt.savefig('plots/Campaign_overview_TimeSeries_wo_Calendar_wind_kn.pdf')
plt.show()