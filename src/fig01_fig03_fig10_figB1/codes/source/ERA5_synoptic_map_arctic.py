def run_ERA5_synoptic_map_arctic(path_data, path_plots):

    """
    Quick script to visualize mean sea level pressure, 500 hPa geopotential height and 850 hPa
    equiv. pot. temperature (and ERA5 sea ice concentration) for three cases during HALO-AC3 or
    for the warm period and cold period as average.

    Parameters:
    -----------
    path_data : dict
        Dictionary containing the paths of the ERA5 single level (key 'single_level') and 
        pressure level (key 'multi_level') data, as well as the cartopy background image
		(key 'cartopy').
    path_plots : str
        Path where plots are saved to.
    """

    import sys
    import os
    import pdb

    import numpy as np
    import xarray as xr
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as mpl_pe
    from matplotlib import patheffects
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt

    from met_tools import equiv_pot_temperature, Z_from_GP

    os.environ['CARTOPY_USER_BACKGROUNDS'] = path_data['cartopy']


    # settings:
    set_dict = {'save_figures': True,       # if True, saved to .png, if False, just show plot
                'toi': np.array([np.datetime64("2022-03-15T12:00:00"), 
                                np.datetime64("2022-03-21T12:00:00"), 
                                np.datetime64("2022-04-01T12:00:00")]),
                'periods': {'warm': slice(np.datetime64("2022-03-11T00:00:00"), np.datetime64("2022-03-20T23:59:00")),
                            'cold': slice(np.datetime64("2022-03-21T00:00:00"), np.datetime64("2022-04-12T23:59:00"))},
                'incl_theta_e': True,       # if False, 850 hPa equiv. pot. temperature will not be plotted
                'mean_periods': True,       # if True, the mean geopot and mean sea level pres over the warm
                                            # and cold period are displayed
                }


    # import data:
    E_DS = xr.open_dataset(path_data['single_level'] + "ERA5_single_level_MSLP_whole_arctic_2022.nc")
    E_DS = E_DS.load()
    if set_dict['mean_periods']:
        EP_DS = xr.open_dataset(path_data['multi_level'] + "ERA5_pressure_levels_gp_whole_arctic_2022.nc")
        EP_DS = EP_DS.load()
        set_dict['incl_theta_e'] = False        # no theta_e
    else:
        EP_DS = xr.open_dataset(path_data['multi_level'] + "ERA5_pressure_levels_T_q_gp_hydmet_whole_arctic_2022.nc")
        EP_DS = EP_DS.load()


        # compute 850 hPa equiv pot T:
        EP_DS['pres'] = xr.DataArray(np.full(EP_DS.t.shape, 0.0), dims=EP_DS.t.dims)        # pressure in Pa
        EP_DS['pres'][:,0,...] = 50000.0
        EP_DS['pres'][:,1,...] = 85000.0
        EP_DS['theta_e'] = equiv_pot_temperature(EP_DS.t, EP_DS.pres, relhum=EP_DS.r*0.01)

    EP_DS['Z'] = Z_from_GP(EP_DS.z)     # geopotential height in m


    # visualize:
    fs = 16
    fs_small = 14
    fs_dwarf = 12


    min_lat = 70        # lower left corner latitude
    min_lon = -60       # lower left corner longitude
    max_lat = 90        # upper right corner latitude
    max_lon = 60        # upper right corner longitude
    extent = [min_lon, max_lon, min_lat, max_lat]
    extent = 4000000

    # station labels and coords:
    coordinates = { "Kiruna": (20.336, 67.821),
                    "Longyearbyen": (15.46, 78.25),
                    "North_Pole": (0.0, 90.0),
                    }
    x1, y1 = coordinates["Kiruna"]   
    x2, y2 = coordinates["Longyearbyen"]
    x3, y3 = coordinates["North_Pole"]
    time_label = ["(a) 15 March 2022", "(b) 21 March 2022", "(c) 01 April 2022"]
    period_label = ["(a) Warm period", "(b) Cold period"]


    # colormaps and levels:
    levels_mslp = np.arange(950.0, 1050.1, 5.0)
    levels_gp500 = np.arange(468., 602.1, 6.0)
    levels_theta_3 = np.arange(-32.0, 32.1, 2.0)
    if set_dict['incl_theta_e']: 
        n_levels = len(levels_theta_3)
    else:
        n_levels = len(levels_gp500)
    cmap = mpl.cm.get_cmap('turbo', n_levels)


    if set_dict['mean_periods']:
        f1, a1 = plt.subplots(1,2, figsize=(12,7), subplot_kw={"projection": ccrs.NorthPolarStereo(central_longitude=0)})

        periods = ["warm", "cold"]
        for k, ax in enumerate(a1):
            # select ERA5 data for the respective period:
            p_l = periods[k]
            E_DS_sel = E_DS.sel(time=set_dict['periods'][p_l]).mean('time')
            EP_DS_sel = EP_DS.sel(time=set_dict['periods'][p_l]).mean('time')

            ax.set_extent((-extent,extent,-extent,extent), crs=ccrs.NorthPolarStereo())
            ax.coastlines(resolution="50m", linewidth=0.65, zorder=20)
            ax.add_feature(cartopy.feature.BORDERS, zorder=20)

            # dummy plots for legend:
            ax.plot([np.nan, np.nan], [np.nan, np.nan], color='blue', linewidth=1.5, label="15$\,\%$ sea ice concentration")
            ax.plot([np.nan, np.nan], [np.nan, np.nan], color=(1,1,1), linewidth=2, 
                        path_effects=[mpl_pe.Stroke(linewidth=2.5, foreground=(0,0,0)), mpl_pe.Normal()], label="Mean sea level pressure (hPa)")


            # plot geopotent height:
            contour_theta_e = ax.contourf(EP_DS_sel.longitude, EP_DS_sel.latitude, EP_DS_sel.Z*0.1, 
                                    cmap=cmap, levels=levels_gp500, extend='both',
                                    transform=ccrs.PlateCarree(), zorder=10)


            # Plot sea-ice concentration
            sea_ice_contour = ax.contour(E_DS_sel.longitude, E_DS_sel.latitude, E_DS_sel.siconc, 
                                        levels=np.array([0.15]), colors='blue', linewidths=1.0, linestyles='solid',
                                        transform=ccrs.PlateCarree(), zorder=11)


            # plot MSLP and geopotential:
            mslp_contour = ax.contour(E_DS_sel.longitude, E_DS_sel.latitude, E_DS_sel.msl*0.01,
                                        levels=levels_mslp, colors='white', linewidths=1.5, linestyles='solid',
                                        transform=ccrs.PlateCarree(), zorder=17)
            mslp_contour.set(path_effects=[mpl_pe.Stroke(linewidth=2.25, foreground=(0,0,0)), mpl_pe.Normal()])
            mslp_clabels = ax.clabel(mslp_contour, levels=levels_mslp[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf, use_clabeltext=True)
            plt.setp(mslp_clabels, path_effects=[mpl_pe.Stroke(linewidth=2, foreground=(0,0,0)), mpl_pe.Normal()])


            # also highlight the main 500 hPa contour line (552 gpdm):
            gp_lw_val = 1.00
            if set_dict['incl_theta_e']: gp_lw_val = 1.75
            gp_lw = np.full(levels_gp500.shape, gp_lw_val)
            gp_lw[np.where(levels_gp500 == 552.)[0]] = 2.25
            gp500_contour = ax.contour(EP_DS_sel.longitude, EP_DS_sel.latitude, EP_DS_sel.Z*0.1,        # in deca metres
                                        levels=levels_gp500, colors='black', linewidths=gp_lw, linestyles='solid',
                                        transform=ccrs.PlateCarree(), zorder=18)
            ax.clabel(gp500_contour, levels=levels_gp500[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)


            # Draw the HALO-AC3 synoptic overview boxes:
            boxes_lon = {'southern': [np.linspace(0.0, 23.0, 1000), np.array([23.0, 23.0]), np.linspace(23.0, 0.0, 1000), np.array([0.0, 0.0])],
                         'central': [np.linspace(-9.0, 16.0, 1000), np.array([16.0, 16.0]), np.linspace(16.0, -9.0, 1000), np.array([-9.0, -9.0])],
                         'northern': [np.linspace(-9.0, 30.0, 1000), np.array([30.0, 30.0]), np.linspace(30.0, -54.0, 1000), 
                                      np.array([-54.0, -54.0]), np.linspace(-54.0, -9.0, 1000), np.array([-9.0, -9.0])]}
            boxes_lat = {'southern': [np.linspace(70.6, 70.6, 1000), np.array([70.6, 75.0]), np.linspace(75.0, 75.0, 1000), np.array([75.0, 70.6])],
                         'central': [np.linspace(75.0, 75.0, 1000), np.array([75.0, 81.5]), np.linspace(81.5, 81.5, 1000), np.array([81.5, 75.0])],
                         'northern': [np.linspace(81.5, 81.5, 1000), np.array([81.5, 89.3]), np.linspace(89.3, 89.3, 1000), 
                                      np.array([89.3, 84.5]), np.linspace(84.5, 84.5, 1000), np.array([84.5, 81.5])]}
            for b_key in boxes_lon.keys():
                # c_box = (0.871, 0.722, 0.530) # colour of the box
                c_box = (1,0,1) # colour of the box
                # if b_key == 'central': c_box = (1.0, 0.549, 0.0)      # highlight central box
                for edge_lon, edge_lat in zip(boxes_lon[b_key], boxes_lat[b_key]):
                    ax.plot(edge_lon, edge_lat, color=c_box, linewidth=1.5, transform=ccrs.PlateCarree(), 
                            path_effects=[mpl_pe.Stroke(linewidth=2.0, foreground=(1,1,1)), mpl_pe.Normal()], zorder=25)


            # mark Kiruna and Longyearbyen
            ax.plot(x1, y1, color=(1,1,1), linestyle='none', marker='.', markersize=15, markeredgecolor="k", 
                    transform=ccrs.PlateCarree(), zorder=25)
            ax.plot(x2, y2, color=(1,1,1), linestyle='none', marker='.', markersize=15, markeredgecolor="k", 
                    transform=ccrs.PlateCarree(), zorder=25)
            ax.plot(x3, y3, color=(1,1,1), linestyle='none', marker='.', markersize=15, markeredgecolor="k", 
                    transform=ccrs.PlateCarree(), zorder=25)
            if k == 0:
                ax.text(x2 + 2, y2 - 1.35, "Longyearbyen", fontsize=fs_dwarf,
                        transform=ccrs.PlateCarree(), color=(1,1,1),
                        path_effects=[mpl_pe.Stroke(linewidth=1.7, foreground=(0,0,0)), mpl_pe.Normal()], zorder=25)
                ax.text(x1 + 1.2, y1 - 0.2, "Kiruna", fontsize=fs_dwarf,
                        transform=ccrs.PlateCarree(), color=(1,1,1),
                        path_effects=[mpl_pe.Stroke(linewidth=1.7, foreground=(0,0,0)), mpl_pe.Normal()], zorder=25)
                ax.text(x3 + 3.2, y3, "  North Pole", fontsize=fs_dwarf,
                        transform=ccrs.PlateCarree(), color=(1,1,1), ha='left', va='center',
                        path_effects=[mpl_pe.Stroke(linewidth=1.7, foreground=(0,0,0)), mpl_pe.Normal()], zorder=25)


            # Place H and L markers for Lows and Highs. Their positions were read out manually:
            if k == 0:
                syn_labels = ['L', 'L', 'H', 'L', 'H']
                syn_poss = np.array([[84.0, -107.5], [63.2, -38.0], [66.3, 164.3], [54.7, -151], [55.3, 27.7]])
            if k == 1:
                syn_labels = ['H', 'L', 'L', 'H', 'L', 'L']
                syn_poss = np.array([[85.7, -166.7], [71.2, 65.6], [65.3, 27.5], [73.2, -29.2], [50.4, -47.5], [56.2, -153.]])

            for syn_label, syn_pos in zip(syn_labels, syn_poss):
                ax.text(syn_pos[1], syn_pos[0], syn_label, fontsize=fs+8, fontweight='bold', color=(1,1,1),
                        transform=ccrs.PlateCarree(), va='center', ha='center',
                        path_effects=[mpl_pe.Stroke(linewidth=2.0, foreground=(0,0,0)), mpl_pe.Normal()], zorder=26)


            # axis label identifier
            ax.text(0.01, 1.01, period_label[k], fontsize=fs, ha='left', va='bottom', transform=ax.transAxes)


    else:
        f1, a1 = plt.subplots(1,3, figsize=(18,7), subplot_kw={"projection": ccrs.NorthPolarStereo(central_longitude=0)})

        for k, ax in enumerate(a1):
            ax.set_extent((-extent,extent,-extent,extent), crs=ccrs.NorthPolarStereo())
            ax.coastlines(resolution="50m", linewidth=0.65, zorder=20)
            ax.add_feature(cartopy.feature.BORDERS, zorder=20)

            # dummy plots for legend:
            ax.plot([np.nan, np.nan], [np.nan, np.nan], color='blue', linewidth=1.5, label="15$\,\%$ sea ice concentration")
            ax.plot([np.nan, np.nan], [np.nan, np.nan], color=(1,1,1), linewidth=2, 
                        path_effects=[mpl_pe.Stroke(linewidth=2.5, foreground=(0,0,0)), mpl_pe.Normal()], label="Mean sea level pressure (hPa)")


            # plot 850 hPa equiv pot temperature:
            if set_dict['incl_theta_e']:
                contour_theta_e = ax.contourf(EP_DS.longitude, EP_DS.latitude, EP_DS.theta_e.sel(time=set_dict['toi'][k], level=850)-273.15, 
                                        cmap=cmap, levels=levels_theta_3, extend='both',
                                        transform=ccrs.PlateCarree(), zorder=10)
                ax.plot([np.nan, np.nan], [np.nan, np.nan], color=(0,0,0), linewidth=2, label="500 hPa geopotential height (gpdm)")
            else:
                contour_theta_e = ax.contourf(EP_DS.longitude, EP_DS.latitude, EP_DS.Z.sel(time=set_dict['toi'][k], level=500)*0.1, 
                                        cmap=cmap, levels=levels_gp500, extend='both',
                                        transform=ccrs.PlateCarree(), zorder=10)


            # Plot sea-ice concentration
            sea_ice_contour = ax.contour(E_DS.longitude, E_DS.latitude, E_DS.siconc.sel(time=set_dict['toi'][k]), 
                                        levels=np.array([0.15]), colors='blue', linewidths=1.0, linestyles='solid',
                                        transform=ccrs.PlateCarree(), zorder=11)


            # plot MSLP and geopotential:
            mslp_contour = ax.contour(E_DS.longitude, E_DS.latitude, E_DS.msl.sel(time=set_dict['toi'][k])*0.01,
                                        levels=levels_mslp, colors='white', linewidths=1.5, linestyles='solid',
                                        transform=ccrs.PlateCarree(), zorder=17)
            mslp_contour.set(path_effects=[mpl_pe.Stroke(linewidth=2.25, foreground=(0,0,0)), mpl_pe.Normal()])
            mslp_clabels = ax.clabel(mslp_contour, levels=levels_mslp[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf, use_clabeltext=True)
            plt.setp(mslp_clabels, path_effects=[mpl_pe.Stroke(linewidth=2, foreground=(0,0,0)), mpl_pe.Normal()])


            # also highlight the main 500 hPa contour line (552 gpdm):
            gp_lw_val = 1.00
            if set_dict['incl_theta_e']: gp_lw_val = 1.75
            gp_lw = np.full(levels_gp500.shape, gp_lw_val)
            gp_lw[np.where(levels_gp500 == 552.)[0]] = 2.25
            gp500_contour = ax.contour(EP_DS.longitude, EP_DS.latitude, EP_DS.Z.sel(time=set_dict['toi'][k], level=500)*0.1,        # in deca metres
                                        levels=levels_gp500, colors='black', linewidths=gp_lw, linestyles='solid',
                                        transform=ccrs.PlateCarree(), zorder=18)
            ax.clabel(gp500_contour, levels=levels_gp500[::2], inline=True, fmt="%i", inline_spacing=8, fontsize=fs_dwarf)


            # Draw the HALO-AC3 synoptic overview boxes:
            boxes_lon = {'southern': [np.linspace(0.0, 23.0, 1000), np.array([23.0, 23.0]), np.linspace(23.0, 0.0, 1000), np.array([0.0, 0.0])],
                         'central': [np.linspace(-9.0, 16.0, 1000), np.array([16.0, 16.0]), np.linspace(16.0, -9.0, 1000), np.array([-9.0, -9.0])],
                         'northern': [np.linspace(-9.0, 30.0, 1000), np.array([30.0, 30.0]), np.linspace(30.0, -54.0, 1000), 
                                      np.array([-54.0, -54.0]), np.linspace(-54.0, -9.0, 1000), np.array([-9.0, -9.0])]}
            boxes_lat = {'southern': [np.linspace(70.6, 70.6, 1000), np.array([70.6, 75.0]), np.linspace(75.0, 75.0, 1000), np.array([75.0, 70.6])],
                         'central': [np.linspace(75.0, 75.0, 1000), np.array([75.0, 81.5]), np.linspace(81.5, 81.5, 1000), np.array([81.5, 75.0])],
                         'northern': [np.linspace(81.5, 81.5, 1000), np.array([81.5, 89.3]), np.linspace(89.3, 89.3, 1000), 
                                      np.array([89.3, 84.5]), np.linspace(84.5, 84.5, 1000), np.array([84.5, 81.5])]}
            for b_key in boxes_lon.keys():
                # c_box = (0.871, 0.722, 0.530) # colour of the box
                c_box = (1,0,1) # colour of the box
                # if b_key == 'central': c_box = (1.0, 0.549, 0.0)      # highlight central box
                for edge_lon, edge_lat in zip(boxes_lon[b_key], boxes_lat[b_key]):
                    ax.plot(edge_lon, edge_lat, color=c_box, linewidth=1.5, transform=ccrs.PlateCarree(), 
                            path_effects=[mpl_pe.Stroke(linewidth=2.0, foreground=(1,1,1)), mpl_pe.Normal()], zorder=25)


            # mark Kiruna and Longyearbyen
            ax.plot(x1, y1, color=(1,1,1), linestyle='none', marker='.', markersize=15, markeredgecolor="k", 
                    transform=ccrs.PlateCarree(), zorder=25)
            ax.plot(x2, y2, color=(1,1,1), linestyle='none', marker='.', markersize=15, markeredgecolor="k", 
                    transform=ccrs.PlateCarree(), zorder=25)
            ax.plot(x3, y3, color=(1,1,1), linestyle='none', marker='.', markersize=15, markeredgecolor="k", 
                    transform=ccrs.PlateCarree(), zorder=25)
            if k == 0:
                ax.text(x2 + 2, y2 - 1.35, "Longyearbyen", fontsize=fs_dwarf,
                        transform=ccrs.PlateCarree(), color=(1,1,1),
                        path_effects=[mpl_pe.Stroke(linewidth=1.7, foreground=(0,0,0)), mpl_pe.Normal()], zorder=25)
                ax.text(x1 + 1.2, y1 - 0.2, "Kiruna", fontsize=fs_dwarf,
                        transform=ccrs.PlateCarree(), color=(1,1,1),
                        path_effects=[mpl_pe.Stroke(linewidth=1.7, foreground=(0,0,0)), mpl_pe.Normal()], zorder=25)
                ax.text(x3 + 3.2, y3, "North Pole  ", fontsize=fs_dwarf,
                        transform=ccrs.PlateCarree(), color=(1,1,1), ha='right', va='center',
                        path_effects=[mpl_pe.Stroke(linewidth=1.7, foreground=(0,0,0)), mpl_pe.Normal()], zorder=25)


            # Place H and L markers for Lows and Highs. Their positions were read out manually:
            if k == 0:
                syn_labels = ['L', 'L', 'H']
                syn_poss = np.array([[77.4, -18.5], [66., -39.4], [62.65, 50.0]])
            if k == 1:
                syn_labels = ['L', 'H', 'L']
                syn_poss = np.array([[77.8, 25.5], [73.6, -40.8], [82.5, -116.0]])
            if k == 2:
                syn_labels = ['H', 'H', 'L', 'L', 'L']
                syn_poss = np.array([[81, -155.8], [70.6, -41.0], [74.3, 15.3], [72.5, 69.2], [69.0, 105.]])

            for syn_label, syn_pos in zip(syn_labels, syn_poss):
                ax.text(syn_pos[1], syn_pos[0], syn_label, fontsize=fs+8,fontweight='bold', color=(1,1,1),
                        transform=ccrs.PlateCarree(), va='center', ha='center',
                        path_effects=[mpl_pe.Stroke(linewidth=2.0, foreground=(0,0,0)), mpl_pe.Normal()], zorder=26)


            # axis label identifier
            ax.text(0.01, 1.01, time_label[k], fontsize=fs, ha='left', va='bottom', transform=ax.transAxes)


    # legend:
    lh, ll = a1[-1].get_legend_handles_labels()
    lol = a1[-1].legend(lh, ll, loc='upper right', fontsize=fs_small, framealpha=0.9)
    lol.set(zorder=10000.0)

    cbar_ax = f1.add_axes([0.1, 0.08, 0.8, 0.04])
    cb_var = f1.colorbar(mappable=contour_theta_e, cax=cbar_ax, extend='max', orientation='horizontal', 
                            fraction=0.06, pad=0.04, shrink=1.00)
    if set_dict['incl_theta_e']:
        cb_var.set_label(label="850 hPa equivalent-potential temperature ($^{\circ}$C)", fontsize=fs_small)
    else:
        cb_var.set_label(label="500 hPa geopotential height (gpdm)", fontsize=fs_small)
    cb_var.ax.tick_params(labelsize=fs_dwarf)


    # Adjust axis width and position remove horizontal and vertical spacing between subplots:
    plt.tight_layout()
    for ax in a1:
        ax_pos = ax.get_position().bounds
        ax.set_position([ax_pos[0], 0.15, ax_pos[2], 0.79])


    if set_dict['save_figures']:
        plot_add = ""
        if set_dict['incl_theta_e']: plot_add = "theta_e_"
        if set_dict['mean_periods']:
            plotname = "HALO-AC3_ERA5_synoptic_map_avg_periods_Arctic"
        else:
            plotname = f"HALO-AC3_ERA5_synoptic_map_{plot_add}Arctic"
        plotfile = path_plots + plotname
        f1.savefig(plotfile + ".pdf", dpi=300, bbox_inches='tight')
        print(f"Saved {plotfile}....")
    else:
        plt.show()
        pdb.set_trace()