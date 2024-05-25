# -*- coding: utf-8 -*-
"""
plot anomaly maps for ERA5 assessment of HALO-(AC)3 campaign
written 2024 by Hanno Müller, Leipzig University
"""

import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs

# define paths
path_data =  "/path/to/data/"
path_plots = "/path/to/plots/"

# read in data & downsample
ds_psurf_all = xr.open_dataset(path+'era5_psurf.nc')
ds_t2_all = xr.open_dataset(path_data+'era5_t2m.nc')
ds_t850_all = xr.open_dataset(path_data+'era5_t850hpa.nc')
ds_tcwv_all = xr.open_dataset(path_data+'era5_tcwv.nc')

ds_psurf_all_daily = ds_psurf_all.resample(time='D').mean()
ds_t2_all_daily = ds_t2_all.resample(time='D').mean()
ds_t850_all_daily = ds_t850_all.resample(time='D').mean()
ds_tcwv_all_daily = ds_tcwv_all.resample(time='D').mean()

# conversion to data arrays
ds_psurf_all_daily1 = ds_psurf_all_daily.msl
da_t2_all_daily = ds_t2_all_daily.t2m
da_t850_all_daily = ds_t850_all_daily.t
da_tcwv_all_daily = ds_tcwv_all_daily.tcwv

# preparation for averaging
#total: 2022-03-07 til 2022-04-12
total_average_psurf = np.zeros([ds_psurf_all_daily1.shape[1],ds_psurf_all_daily1.shape[2],len(np.arange(1979,2022+1))])
total_average_t2 = np.zeros([da_t2_all_daily.shape[1],da_t2_all_daily.shape[2],len(np.arange(1979,2022+1))])
total_average_t850 = np.zeros([da_t850_all_daily.shape[1],da_t850_all_daily.shape[2],len(np.arange(1979,2022+1))])
total_average_tcwv = np.zeros([da_tcwv_all_daily.shape[1],da_tcwv_all_daily.shape[2],len(np.arange(1979,2022+1))])

#WAI: 2022-03-11 til 2022-03-20
wai_average_psurf = np.zeros([ds_psurf_all_daily1.shape[1],ds_psurf_all_daily1.shape[2],len(np.arange(1979,2022+1))])
wai_average_t2 = np.zeros([da_t2_all_daily.shape[1],da_t2_all_daily.shape[2],len(np.arange(1979,2022+1))])
wai_average_t850 = np.zeros([da_t850_all_daily.shape[1],da_t850_all_daily.shape[2],len(np.arange(1979,2022+1))])
wai_average_tcwv = np.zeros([da_tcwv_all_daily.shape[1],da_tcwv_all_daily.shape[2],len(np.arange(1979,2022+1))])

#CAO: 2022-03-21 til 2022-04-12
cao_average_psurf = np.zeros([ds_psurf_all_daily1.shape[1],ds_psurf_all_daily1.shape[2],len(np.arange(1979,2022+1))])
cao_average_t2 = np.zeros([da_t2_all_daily.shape[1],da_t2_all_daily.shape[2],len(np.arange(1979,2022+1))])
cao_average_t850 = np.zeros([da_t850_all_daily.shape[1],da_t850_all_daily.shape[2],len(np.arange(1979,2022+1))])
cao_average_tcwv = np.zeros([da_tcwv_all_daily.shape[1],da_tcwv_all_daily.shape[2],len(np.arange(1979,2022+1))])

# yearly averaging
for i,year in enumerate(np.arange(1979,2022+1)):
    total_average_psurf[:, :, i] = ds_psurf_all_daily1[
                               np.where(ds_psurf_all_daily1.time == np.datetime64(str(year) + '-03-07'))[0][0]:
                               np.where(ds_psurf_all_daily1.time == np.datetime64(str(year) + '-04-13'))[0][0], :,
                               :].mean(axis=0)
    wai_average_psurf[:, :, i] = ds_psurf_all_daily1[
                               np.where(ds_psurf_all_daily1.time == np.datetime64(str(year) + '-03-11'))[0][0]:
                               np.where(ds_psurf_all_daily1.time == np.datetime64(str(year) + '-03-21'))[0][0], :,
                               :].mean(axis=0)
    cao_average_psurf[:, :, i] = ds_psurf_all_daily1[
                              np.where(ds_psurf_all_daily1.time == np.datetime64(str(year) + '-03-21'))[0][0]:
                              np.where(ds_psurf_all_daily1.time == np.datetime64(str(year) + '-04-13'))[0][0], :,
                              :].mean(axis=0)

    total_average_t2[:, :, i] = da_t2_all_daily[
                               np.where(da_t2_all_daily.time == np.datetime64(str(year) + '-03-07'))[0][0]:
                               np.where(da_t2_all_daily.time == np.datetime64(str(year) + '-04-13'))[0][0], :,
                               :].mean(axis=0)
    wai_average_t2[:, :, i] = da_t2_all_daily[
                               np.where(da_t2_all_daily.time == np.datetime64(str(year) + '-03-11'))[0][0]:
                               np.where(da_t2_all_daily.time == np.datetime64(str(year) + '-03-21'))[0][0], :,
                               :].mean(axis=0)
    cao_average_t2[:, :, i] = da_t2_all_daily[
                              np.where(da_t2_all_daily.time == np.datetime64(str(year) + '-03-21'))[0][0]:
                              np.where(da_t2_all_daily.time == np.datetime64(str(year) + '-04-13'))[0][0], :,
                              :].mean(axis=0)

    total_average_t850[:, :, i] = da_t850_all_daily[
                                 np.where(da_t850_all_daily.time == np.datetime64(str(year) + '-03-07'))[0][0]:
                                 np.where(da_t850_all_daily.time == np.datetime64(str(year) + '-04-13'))[0][0], :,
                                 :].mean(axis=0)
    wai_average_t850[:, :, i] = da_t850_all_daily[
                                 np.where(da_t850_all_daily.time == np.datetime64(str(year) + '-03-11'))[0][0]:
                                 np.where(da_t850_all_daily.time == np.datetime64(str(year) + '-03-21'))[0][0], :,
                                 :].mean(axis=0)
    cao_average_t850[:, :, i] = da_t850_all_daily[
                                np.where(da_t850_all_daily.time == np.datetime64(str(year) + '-03-21'))[0][0]:
                                np.where(da_t850_all_daily.time == np.datetime64(str(year) + '-04-13'))[0][0], :,
                                :].mean(axis=0)

    total_average_tcwv[:, :, i] = da_tcwv_all_daily[
                                 np.where(da_tcwv_all_daily.time == np.datetime64(str(year) + '-03-07'))[0][0]:
                                 np.where(da_tcwv_all_daily.time == np.datetime64(str(year) + '-04-13'))[0][0], :,
                                 :].mean(axis=0)
    wai_average_tcwv[:, :, i] = da_tcwv_all_daily[
                                 np.where(da_tcwv_all_daily.time == np.datetime64(str(year) + '-03-11'))[0][0]:
                                 np.where(da_tcwv_all_daily.time == np.datetime64(str(year) + '-03-21'))[0][0], :,
                                 :].mean(axis=0)
    cao_average_tcwv[:, :, i] = da_tcwv_all_daily[
                                np.where(da_tcwv_all_daily.time == np.datetime64(str(year) + '-03-21'))[0][0]:
                                np.where(da_tcwv_all_daily.time == np.datetime64(str(year) + '-04-13'))[0][0], :,
                                :].mean(axis=0)

# average year of campaign
total_average_psurf_2022=total_average_psurf[:,:,-1]
total_average_t2_2022=total_average_t2[:,:,-1]
total_average_t850_2022=total_average_t850[:,:,-1]
total_average_tcwv_2022=total_average_tcwv[:,:,-1]

wai_average_psurf_2022=wai_average_psurf[:,:,-1]
wai_average_t2_2022=wai_average_t2[:,:,-1]
wai_average_t850_2022=wai_average_t850[:,:,-1]
wai_average_tcwv_2022=wai_average_tcwv[:,:,-1]

cao_average_psurf_2022=cao_average_psurf[:,:,-1]
cao_average_t2_2022=cao_average_t2[:,:,-1]
cao_average_t850_2022=cao_average_t850[:,:,-1]
cao_average_tcwv_2022=cao_average_tcwv[:,:,-1]

# average climatology years
total_average_psurf_1979_2022=np.zeros([ds_psurf_all_daily1.shape[1],ds_psurf_all_daily1.shape[2]])
total_average_t2_1979_2022=np.zeros([da_t850_all_daily.shape[1],da_t850_all_daily.shape[2]])
total_average_t850_1979_2022=np.zeros([da_t850_all_daily.shape[1],da_t850_all_daily.shape[2]])
total_average_tcwv_1979_2022=np.zeros([da_t850_all_daily.shape[1],da_t850_all_daily.shape[2]])

wai_average_psurf_1979_2022=np.zeros([ds_psurf_all_daily1.shape[1],ds_psurf_all_daily1.shape[2]])
wai_average_t2_1979_2022=np.zeros([da_t850_all_daily.shape[1],da_t850_all_daily.shape[2]])
wai_average_t850_1979_2022=np.zeros([da_t850_all_daily.shape[1],da_t850_all_daily.shape[2]])
wai_average_tcwv_1979_2022=np.zeros([da_t850_all_daily.shape[1],da_t850_all_daily.shape[2]])

cao_average_psurf_1979_2022=np.zeros([ds_psurf_all_daily1.shape[1],ds_psurf_all_daily1.shape[2]])
cao_average_t2_1979_2022=np.zeros([da_t850_all_daily.shape[1],da_t850_all_daily.shape[2]])
cao_average_t850_1979_2022=np.zeros([da_t850_all_daily.shape[1],da_t850_all_daily.shape[2]])
cao_average_tcwv_1979_2022=np.zeros([da_t850_all_daily.shape[1],da_t850_all_daily.shape[2]])

for lat in range(ds_psurf_all_daily1.shape[1]):
    for lon in range(ds_psurf_all_daily1.shape[2]):
        total_average_psurf_1979_2022[lat, lon] = np.nanmean(total_average_psurf[lat, lon, :])
        wai_average_psurf_1979_2022[lat, lon] = np.nanmean(wai_average_psurf[lat, lon, :])
        cao_average_psurf_1979_2022[lat, lon] = np.nanmean(cao_average_psurf[lat, lon, :])

for lat in range(da_t2_all_daily.shape[1]):
    for lon in range(da_t2_all_daily.shape[2]):
        total_average_t2_1979_2022[lat, lon] = np.nanmean(total_average_t2[lat, lon, :])
        wai_average_t2_1979_2022[lat, lon] = np.nanmean(wai_average_t2[lat, lon, :])
        cao_average_t2_1979_2022[lat, lon] = np.nanmean(cao_average_t2[lat, lon, :])

for lat in range(da_t850_all_daily.shape[1]):
    for lon in range(da_t850_all_daily.shape[2]):
        total_average_t850_1979_2022[lat, lon] = np.nanmean(total_average_t850[lat, lon, :])
        wai_average_t850_1979_2022[lat, lon] = np.nanmean(wai_average_t850[lat, lon, :])
        cao_average_t850_1979_2022[lat, lon] = np.nanmean(cao_average_t850[lat, lon, :])

for lat in range(da_tcwv_all_daily.shape[1]):
    for lon in range(da_tcwv_all_daily.shape[2]):
        total_average_tcwv_1979_2022[lat, lon] = np.nanmean(total_average_tcwv[lat, lon, :])
        wai_average_tcwv_1979_2022[lat, lon] = np.nanmean(wai_average_tcwv[lat, lon, :])
        cao_average_tcwv_1979_2022[lat, lon] = np.nanmean(cao_average_tcwv[lat, lon, :])

# preparation for plot
label_time_period = {"total": "07 Mar 2022 to 12 Apr 2022",
                     "wai": "11 Mar 2022 to 20 Mar 2022",
                     "cao": "21 Mar 2022 to 12 Apr 2022"}

contour_colormaps = {"psurf": "PRGn",
                     "t2": "RdBu_r",
                     "t": "RdBu_r",
                     "tcwv": "BrBG"}

levels = {"psurf":[995,1000,1005,1010,1015,1020],
          "t2": [250,260,270],
          "t": [250, 255, 260, 265, 270],
          "tcwv": [2, 4, 6, 8]}

fmt_levels = {"psurf": "%1.0f",
              "t2": "%1.0f",
              "t": "%1.0f",
              "tcwv": "%1.1f"}

deviation_levels = {"psurf": np.linspace(-10,10,100),
                    "t2": np.linspace(-8,8,100),
                    "t": np.linspace(-8, 8, 100),
                    "tcwv": np.linspace(-3, 3, 100)}

xtick_range = {"psurf": np.linspace(-10,10,11),
               "t2": np.linspace(-8, 8, 9),
               "t": np.linspace(-8, 8, 9),
               "tcwv": np.linspace(-3, 3, 7)}

xlabel_configs = {"psurf": ["$\mathrm{MSLP}$", " (hPa)"],
                  "t2": ["$\mathrm{T_{2\,m}}$", " (K)"],
                  "t": ["${\mathrm{T_{850\,hPa}}}$", " (K)"],
                  "tcwv": ["$\mathrm{IWV}$",
                           " (kg$\,\mathrm{m}^{-2}$)"]}

labelpad_configs = {'psurf':-5,
                    't2': None,
                    't': None,
                    'tcwv': None}

# define plot function for single panel
def plot_climatology_panel(data_all,data_campaign,da_all_daily_for_latlon,ax1,figureletter,var_of_interest="tcwv",time_period='campaign',draw_cb=False,draw_date=False,draw_boxes=True):

    ax1.set_extent([-15, 30, 65, 90], crs=ccrs.Geodetic())

    x_lon, y_lat = np.meshgrid(da_all_daily_for_latlon.longitude.values, da_all_daily_for_latlon.latitude.values)

    # iso lines of overall average
    CS = ax1.contour(x_lon, y_lat, data_all, levels=levels[var_of_interest],
                     transform=ccrs.PlateCarree(), linewidths=1.0, linestyles="--",
                     colors="black",zorder=11)

    ax1.clabel(CS, fontsize=14, fmt=fmt_levels[var_of_interest], inline=1,inline_spacing=15,zorder=11)

    # deviation from long-term climatology
    C1 = ax1.contourf(x_lon, y_lat, data_campaign - data_all, cmap=contour_colormaps[var_of_interest],
                      levels=deviation_levels[var_of_interest],
                      transform=ccrs.PlateCarree(),extend="both")

    # plot layout
    ax1.coastlines(resolution="50m", color="black")
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                       x_inline=False, y_inline=False)
    gl.xlocator = mticker.FixedLocator([-80,-60,-40,-20, 0, 20,40,60,80])
    gl.top_labels = False
    gl.right_labels = False

    # colorbar
    if draw_cb == True:
        axins = inset_axes(ax1,width="5%",height="100%",loc='center right',borderpad=-4.3)
        cb = plt.colorbar(C1,cax=axins)
        cb.set_label("$\Delta$ " + xlabel_configs[var_of_interest][0] + xlabel_configs[var_of_interest][1],fontsize=18,labelpad=labelpad_configs[var_of_interest])
        cb.set_ticks(xtick_range[var_of_interest])
        cb.ax.tick_params(labelsize=16)

    # draw boxes with area of interest
    if draw_boxes ==True:
        color1='black'
        ax1.plot(np.linspace(-9,16,100),np.linspace(75,75,100), transform=ccrs.Geodetic(),zorder=10,color=color1)
        ax1.plot(np.linspace(-9,16,100),np.linspace(81.5,81.5,100), transform=ccrs.Geodetic(),zorder=10,color=color1)
        ax1.plot(np.linspace(-9,-9,100),np.linspace(75,81.5,100), transform=ccrs.Geodetic(),zorder=10,color=color1)
        ax1.plot(np.linspace(16,16,100),np.linspace(75,81.5,100), transform=ccrs.Geodetic(),zorder=10,color=color1)

        ax1.plot(np.linspace(0,23,100),np.linspace(70.6,70.6,100), transform=ccrs.Geodetic(),zorder=10,color=color1)
        ax1.plot(np.linspace(0,23,100),np.linspace(75,75,100), transform=ccrs.Geodetic(),zorder=10,color=color1)
        ax1.plot(np.linspace(0,0,100),np.linspace(70.6,75,100), transform=ccrs.Geodetic(),zorder=10,color=color1)
        ax1.plot(np.linspace(23,23,100),np.linspace(70.6,75,100), transform=ccrs.Geodetic(),zorder=10,color=color1)

        ax1.plot(np.linspace(-9,30,100),np.linspace(81.5,81.5,100), transform=ccrs.Geodetic(),zorder=10,color=color1)
        ax1.plot(np.linspace(-54,-9,100),np.linspace(84.5,84.5,100), transform=ccrs.Geodetic(),zorder=10,color=color1)
        ax1.plot(np.linspace(-54,30,100),np.linspace(89.3,89.3,100), transform=ccrs.Geodetic(),zorder=10,color=color1)
        ax1.plot(np.linspace(-54,-54,100),np.linspace(84.5,89.3,100), transform=ccrs.Geodetic(),zorder=10,color=color1)
        ax1.plot(np.linspace(-9,-9,100),np.linspace(81.5,84.5,100), transform=ccrs.Geodetic(),zorder=10,color=color1)
        ax1.plot(np.linspace(30,30,100),np.linspace(81.5,89.3,100), transform=ccrs.Geodetic(),zorder=10,color=color1)

    if draw_date == True:
        ax1.set_title(label_time_period[time_period],fontsize=18)
    ax1.text(-0.12,0.95,'('+figureletter+')',transform=ax1.transAxes,fontsize=18)

#plot all panels in one figure
fig, axs = plt.subplots(4, 3, figsize=(16, 25),subplot_kw={"projection": ccrs.NorthPolarStereo(central_longitude=5)})
set_font = 14
matplotlib.rcParams.update({'font.size': set_font})
plt.subplots_adjust(hspace=0.16,wspace=0.14)

ax1=axs[0,0]#psurf total
figureletter='a'
plot_climatology_panel(total_average_psurf_1979_2022/100,total_average_psurf_2022/100,ds_psurf_all_daily1,ax1,figureletter,var_of_interest='psurf',time_period='total',draw_date=True)
ax1=axs[0,1]#psurf WAI
figureletter='b'
plot_climatology_panel(wai_average_psurf_1979_2022/100.,wai_average_psurf_2022/100.,ds_psurf_all_daily1,ax1,figureletter,var_of_interest='psurf',time_period='wai',draw_date=True)
ax1=axs[0,2]#psurf CAO
figureletter='c'
plot_climatology_panel(cao_average_psurf_1979_2022/100.,cao_average_psurf_2022/100.,ds_psurf_all_daily1,ax1,figureletter,var_of_interest='psurf',time_period='cao',draw_cb=True,draw_date=True)

ax1=axs[1,0]#T2 total
figureletter='d'
plot_climatology_panel(total_average_t2_1979_2022,total_average_t2_2022,da_t2_all_daily,ax1,figureletter,var_of_interest='t2',time_period='total')
ax1=axs[1,1]#T2 WAI
figureletter='e'
plot_climatology_panel(wai_average_t2_1979_2022,wai_average_t2_2022,da_t2_all_daily,ax1,figureletter,var_of_interest='t2',time_period='wai')
ax1=axs[1,2]#T2 CAO
figureletter='f'
plot_climatology_panel(cao_average_t2_1979_2022,cao_average_t2_2022,da_t2_all_daily,ax1,figureletter,var_of_interest='t2',time_period='cao',draw_cb=True)

ax1=axs[2,0]#T850 total
figureletter='g'
plot_climatology_panel(total_average_t850_1979_2022,total_average_t850_2022,da_t850_all_daily,ax1,figureletter,var_of_interest='t',time_period='total',draw_date=True)
ax1=axs[2,1]#T850 WAI
figureletter='h'
plot_climatology_panel(wai_average_t850_1979_2022,wai_average_t850_2022,da_t850_all_daily,ax1,figureletter,var_of_interest='t',time_period='wai',draw_date=True)
ax1=axs[2,2]#T850 CAO
figureletter='i'
plot_climatology_panel(cao_average_t850_1979_2022,cao_average_t850_2022,da_t850_all_daily,ax1,figureletter,var_of_interest='t',time_period='cao',draw_cb=True,draw_date=True)


ax1=axs[3,0]#IWV total
figureletter='j'
plot_climatology_panel(total_average_tcwv_1979_2022,total_average_tcwv_2022,da_tcwv_all_daily,ax1,figureletter,var_of_interest='tcwv',time_period='total')
ax1=axs[3,1]#IWV WAI
figureletter='k'
plot_climatology_panel(wai_average_tcwv_1979_2022,wai_average_tcwv_2022,da_tcwv_all_daily,ax1,figureletter,var_of_interest='tcwv',time_period='wai')
ax1=axs[3,2]#IWV CAO
figureletter='l'
plot_climatology_panel(cao_average_tcwv_1979_2022,cao_average_tcwv_2022,da_tcwv_all_daily,ax1,figureletter,var_of_interest='tcwv',time_period='cao',draw_cb=True)

# save plot with all panels
plt.savefig(f"{path_plots}era5_anomalies.png", bbox_inches="tight",dpi=300)
