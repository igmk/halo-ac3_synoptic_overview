def run_HAMP_bright_band_sea_ice(path_sic, path_hamp, path_cartopy_background, path_plots):

    """
    Visualize the bright band measured by HALO's radar, which is part of the microwave package
    HAMP, over sea ice.

    Parameters:
    -----------
    path_sic : str
        Path of the sea ice concentration data.
    path_hamp : str
        Path of the HALO HAMP radar data.
    path_cartopy_background : str
        Path of the cartopy background data.
    path_plots : str
        Path where plots are saved to.
    """

    import os
    import sys
    import pdb
    import glob

    import numpy as np
    import xarray as xr
    import matplotlib as mpl
    from cmcrameri import cm            # new colourmaps
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.patheffects as mpl_pe
    from pyproj import Proj,CRS

    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt

    os.environ['CARTOPY_USER_BACKGROUNDS'] = path_cartopy_background


    # settings:
    set_dict = {'save_figures': True,       # if True, saved to .png, if False, just show plot
                }


    # import SIC data and average over the campaign duration:
    DS = xr.open_dataset(path_sic + "sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_20220313.nc")
    SIC_mean = np.flipud(DS.sic_merged).astype(np.float64)

    # regions where SIC_mean > 100 = land; therefore, remove:
    SIC_mean[SIC_mean > 100.0] = np.nan

    # get lat, lon coordinates from projection:
    x = DS.x
    y = DS.y
    projection = DS.polar_stereographic     # To convert from x and y to lat/lon we need the information about the grid contained in this variable
    m=Proj(projection.attrs["spatial_ref"]) # define a function for conversion that gets the information from the string contained in projection
    xx,yy = np.meshgrid(x.values,y.values)  # create a meshgrid from x and y to use in function m()
    lon, lat = m(xx,yy, inverse=True)

    del x, y, DS, m, projection, xx, yy


    # import HAMP data:
    HAMP_DS = xr.open_dataset(path_hamp + "HALO_HALO_AC3_radar_unified_RF03_20220313_v2.6.nc")
    HAMP_DS['radar_flag'] = HAMP_DS.radar_flag.where(~np.isnan(HAMP_DS.radar_flag), 0.0)

    # plot:
    fs = 16
    fs_small = 14
    fs_dwarf = 12


    # colourmaps:
    # adapt the colormap: change the default length to certain desired lengths:
    def change_colormap_len(cmap, n_new):
        len_cmap = cmap.shape[0]
        n_rgba = cmap.shape[1]

        cmap_new = np.zeros((n_new, n_rgba))
        for m in range(n_rgba):
            cmap_new[:,m] = np.interp(np.linspace(0, 1, n_new), np.linspace(0, 1, len_cmap), cmap[:,m])

        cmap_new = mpl.colors.ListedColormap(cmap_new)
        return cmap_new

    cmap = cm.batlow(range(len(cm.batlow.colors)))
    bounds_dBZ = np.arange(-25,-9.9,0.5)                # in dBZ
    n_levels = len(bounds_dBZ)
    cmap_dBZ = change_colormap_len(cmap, n_levels)

    sic_levels = np.arange(-5., 101., 2.5)
    n_lev_sic = len(sic_levels)
    reversed_map = mpl.cm.get_cmap('Blues_r', n_lev_sic)


    min_lat = 76        # lower left corner latitude
    min_lon = -15       # lower left corner longitude
    max_lat = 86        # upper right corner latitude
    max_lon = 15        # upper right corner longitude
    extent = [min_lon, max_lon, min_lat, max_lat]
    big_extent=[min_lon-10,max_lon+15,min_lat-11,90]

    # station labels and coords:
    coordinates= {"Kiruna": (20.336, 67.821),
                    "Longyearbyen": (15.46, 78.25),}


    x1, y1 = coordinates["Kiruna"]   
    x2, y2 = coordinates["Longyearbyen"]


    # Create a GeoAxes in the tile's projection:
    x = np.linspace(-90,90,41)
    y = np.linspace(55,90,93)
    x_grid, y_grid = np.meshgrid(x,y)
    white_overlay = np.zeros(x_grid.shape)
    plt.rcdefaults()

    f1 = plt.figure(figsize=(12,5))
    a1 = plt.subplot2grid((1,3), (0,0), colspan=2)  # HAMP plot
    a2 = plt.subplot2grid((1,3), (0,2), projection=ccrs.NorthPolarStereo(central_longitude=5))      # map plot

    a2.set_extent(extent, crs=ccrs.Geodetic())
    a2.coastlines(resolution="50m")
    a2.add_feature(cartopy.feature.BORDERS)
    a2.add_feature(cartopy.feature.OCEAN, color=(1,1,1), zorder=0)


    # axis limits: 
    x_lim = np.array([np.datetime64("2022-03-13T14:15:00"), np.datetime64("2022-03-13T15:05:00")])
    y_lim = np.array([0.0, 8000.0])


    # Plot HAMP data:
    ldr_contour = a1.contourf(HAMP_DS.time, HAMP_DS.height, HAMP_DS.LDRg.T, cmap=cmap_dBZ, levels=bounds_dBZ, extend='both')
    a2.background_img(name='NaturalEarthRelief', resolution='high')

    # add sea ice concentration mask:
    a1_add = a1.inset_axes(bounds=[0.0, -0.05, 1.00, 0.05], transform=a1.transAxes)
    sic_contour = a1_add.contourf(HAMP_DS.time.values, HAMP_DS.height.values, HAMP_DS.radar_flag.T.values*100., 
                                    cmap=reversed_map, levels=sic_levels)


    # Plot sea-ice concentration
    contourf_plot = a2.pcolormesh(lon, lat, SIC_mean[1:,1:], cmap=reversed_map, shading="flat",
                                    transform=ccrs.PlateCarree(), vmin=0, vmax=100, zorder=5)


    # plot flight track:
    a2.plot(HAMP_DS.lon, HAMP_DS.lat, color=(1,1,1), linewidth=1.5,
                path_effects=[mpl_pe.Stroke(linewidth=3.0, foreground=(0,0,0)), mpl_pe.Normal()],
                label='HALO flight track', transform=ccrs.PlateCarree(), zorder=10)
    a2.plot(HAMP_DS.lon.sel(time=slice(x_lim[0],x_lim[1])), HAMP_DS.lat.sel(time=slice(x_lim[0],x_lim[1])), 
                color=(0,0.85,1), linewidth=2,
                path_effects=[mpl_pe.Stroke(linewidth=4.0, foreground=(0,0,0)), mpl_pe.Normal()],
                label='Selected segment', transform=ccrs.PlateCarree(), zorder=10)


    # add figure panel identifiers:
    a1.text(0.01, 1.01, "(a)", fontsize=fs, ha='left', va='bottom', transform=a1.transAxes)
    a2.text(0.01, 1.01, "(b)", fontsize=fs, ha='left', va='bottom', transform=a2.transAxes)


    # legend, colorbar:
    lh, ll = a2.get_legend_handles_labels()
    lol = a2.legend(lh, ll, loc='upper left', fontsize=fs_dwarf)
    lol.set(zorder=15)

    c_ax = a1.inset_axes(bounds=[1.005, 0., 0.015, 1.0], transform=a1.transAxes)
    cb1 = f1.colorbar(mappable=ldr_contour, cax=c_ax, extend='both', orientation='vertical')
    cb1.set_label(label="Linear depolarization ratio (dB)", fontsize=fs_small)
    cb1.ax.tick_params(labelsize=fs_dwarf)

    c_ax2 = a1.inset_axes(bounds=[0.75, 0.975, 0.25, 0.025], transform=a1.transAxes)
    cb2 = f1.colorbar(mappable=sic_contour, cax=c_ax2, orientation='horizontal')
    cb2.set_ticks([0.0, 100.])
    cb2.set_label(label="SIC (%)", fontsize=fs_small, labelpad=-9)
    cb2.ax.xaxis.set_label_position('top')
    cb2.ax.tick_params(labelsize=fs_dwarf)
    c_ax2.xaxis.set_ticks_position("top")


    # set axis limits:
    a1.set_xlim(x_lim)
    a1.set_ylim(y_lim)
    a1_add.set_ylim(np.array([0.0, 2.0]))
    a1_add.set_xlim(x_lim)

    # set ticks and tick labels and parameters:
    a1_add.tick_params(axis='both', labelsize=fs_dwarf)
    a1.tick_params(axis='both', labelsize=fs_dwarf)
    a1_add.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")) # (e.g. "12:00"))
    a1.set_xticklabels([])
    a1_add.set_yticklabels([])


    # set labels:
    a1.set_ylabel("Height (m)", fontsize=fs)
    a1_add.set_xlabel("Time on 13 March 2022 (UTC)", fontsize=fs)


    # adjust axis positions:
    plt.tight_layout()
    a1_pos = a1.get_position().bounds
    a1.set_position([a1_pos[0], a1_pos[1]+0.05, a1_pos[2], a1_pos[3]*0.9])

    if set_dict['save_figures']:
        plotname = "HALO-AC3_HALO_HAMP_radar_bright_band_RF03_20220313"
        plotfile = path_plots + plotname
        f1.savefig(plotfile + ".png", dpi=300, bbox_inches='tight')
        print(f"Saved {plotfile}....")
    else:
        plt.show()
        pdb.set_trace()