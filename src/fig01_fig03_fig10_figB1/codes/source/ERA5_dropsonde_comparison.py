def run_ERA5_dropsonde_comparison(path_data, path_sic, path_dropsondes_halo, path_plots):


    """
    Script to compare ERA5 model level data with HALO dropsondes.

    Parameters:
    -----------
    path_data : str
        Path of the processed and collocated ERA5 model level data.
    path_sic : str
        Path of the ERA5 single levle sea ice concentration data.
    path_dropsondes_halo : str
        Path of the HALO dropsonde data.
    path_plots : str
        Path where to save plots to.
    """

    import os
    import sys
    import glob
    import pdb
    import datetime as dt

    import numpy as np
    import xarray as xr
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as mpl_pe

    from halo_classes import P5_dropsondes, dropsondes
    from data_tools import compute_RMSE_profile
    from met_tools import convert_spechum_to_relhum, convert_rh_to_spechum, u_v_to_wspeed_wdir


    fs = 16
    fs_small = 14
    fs_dwarf = 12

    # colours:
    colours = {'ice': (0.4,0.7,0.79), 'ocean': (0.67,0.51,0.31), 'all': (0,0,0)}
    colours_N = {'ice': (0,0,1,0.5), 'ocean': (0,0,1), 'all': (0,0,1)}
    line_styles = {'ice': 'dashed', 'ocean': 'solid', 'all': 'solid'}


    def dropsonde_collocation(
        SIC_DS, 
        DS):

        """
        Collocate ERA5 data with dropsonde launch positions. First, select closest grid points. This
        yields a (time, lat, lon) = (n_sondes, n_sondes, n_sondes) data set. To reduce this further
        to (time,) = (n_sondes,), another step for collocation is required.

        Parameters:
        -----------
        SIC_DS : xarray dataset
            ERA5 single level sea ice concentration data in an xarray dataset.
        DS : xarray dataset
            Dropsonde data set that must contain launch_time, ref_lat and ref_lon for the overlap.
        """

        # preselect ERA5 data: closest lat, lon, time: But, now, each of these dimensions has 
        SIC_DS_presel = SIC_DS.sel(latitude=DS.ref_lat.values, longitude=DS.ref_lon.values, 
                                time=DS.launch_time.values, method='nearest')
        SIC_DS_presel = SIC_DS_presel.load()


        # reduce reanalysis data dimensions further: select the correct latitude and longitude dimension for 
        # the respective times (i.e., choose lat index 0 for the first launch, lat index 1 for the second launch...):
        SIC_DS_sel = xr.Dataset(coords={'time': SIC_DS_presel.time})
        for dv in ['siconc', 'latitude', 'longitude']:
            SIC_DS_sel[dv] = xr.DataArray(np.full((len(SIC_DS_presel.time),), np.nan), dims=['time'])


        # loop through ERA5_M_DS time and check which radiosonde launch time is closest
        for k, e_time in enumerate(SIC_DS_presel.time.values):
            
            # loop through data vars of the ERA5 dataset and put correct selection into the _red dataset:
            for dv in ['siconc']:
                SIC_DS_sel[dv][k] = SIC_DS_presel[dv].isel(time=k, latitude=k, longitude=k)

            # get lat and lon into SIC_DS_sel:
            SIC_DS_sel['latitude'][k] = SIC_DS_presel['latitude'][k]
            SIC_DS_sel['longitude'][k] = SIC_DS_presel['longitude'][k]

        SIC_DS_presel = SIC_DS_presel.close()
        del SIC_DS_presel
        return SIC_DS_sel


    def get_errors(
        E_DS_col, 
        DS,
        set_dict):

        """
        Get errors (RMSE, bias, bias-corrected RMSE) of the ERA5 data using dropsonde data as reference.

        Parameters:
        -----------
        E_DS_col : xarray dataset
            ERA5 data collocated with dropsonde observations and brought to the same height grid.
        DS : xarray dataset
            Dropsonde data set used as reference.
        set_dict : dict
            Dictionary containing auxiliary information.
        """

        # initialize arrays:
        ERR_DS = xr.Dataset(coords={'height': E_DS_col.height.values})
        conv_dict = {'temp': np.array([0, 1.0]), # additive and multiplicative
                        'q': np.array([0, 1000.]),
                        'rh': np.array([0, 100.]),
                        'u': np.array([0, 1.0]),
                        'v': np.array([0, 1.0]),
                        'wspeed': np.array([0, 1.0]),
                        'pres': np.array([0, 0.01])}

        # compute errors:
        for dv in ['temp', 'q', 'rh', 'u', 'v', 'wspeed', 'pres']:
            # only consider dropsondes that have enough non-nan values:
            x_stuff = (DS[dv].values + conv_dict[dv][0])*conv_dict[dv][1]
            y_stuff = (E_DS_col[dv].values + conv_dict[dv][0])*conv_dict[dv][1]
            no_nan_idx = np.where(np.count_nonzero(~np.isnan(x_stuff+y_stuff), axis=1) > 0.25*new_shape[1])[0]
            x_stuff = x_stuff[no_nan_idx,:]
            y_stuff = y_stuff[no_nan_idx,:]

            ERR_DS[dv+"_N"] = xr.DataArray(np.count_nonzero(~np.isnan(x_stuff+y_stuff), axis=0), dims='height')
            ERR_DS[dv+"_rmse"] = xr.DataArray(compute_RMSE_profile(y_stuff, x_stuff, which_axis=0), dims='height')
            ERR_DS[dv+"_bias"] = xr.DataArray(np.nanmean(y_stuff - x_stuff, axis=0), dims='height')
            ERR_DS[dv+"_std"] = xr.DataArray(compute_RMSE_profile(y_stuff - ERR_DS[dv+"_bias"].values, x_stuff, which_axis=0), dims='height')

        return ERR_DS


    # settings:
    set_dict = {'save_figures': True,       # if True, saved to .png, if False, just show plot
                'all_in_one': True,         # if True, a huge multi-panel plot is created that shows all errors
                'separate_ice': True,       # if True, errors over ocean and over ice will be handled separately
                'err_distrib': True,        # if True, error freq occurrence distribution for a certain height will
                                            # be plotted; SHOULD BE TRUE TO FILTER BAD DROPSONDES
                }


    # import dropsonde data for lat-lon positions:
    print("Importing dropsonde data....")
    DS_DS = dropsondes(path_dropsondes_halo, 'raw', return_DS=True)
    DS = DS_DS.DS

    # new height grid:
    DS = DS.interp(coords={'height': np.arange(0., 8000.1, 100.)})


    # import SIC data:
    if set_dict['separate_ice']:
        SIC_DS = xr.open_dataset(path_sic + "ERA5_single_level_SIC_march_april_2022.nc")
        SIC_DS = dropsonde_collocation(SIC_DS, DS)
        set_dict['all_in_one'] = True       # not implemented for single-data-plots



    # list ERA5 files:
    print("Importing ERA5 data....")
    files = sorted(glob.glob(path_data + "HALO_AC3_ERA5-HALO_collocated_*.nc"))
    E_DS = xr.open_mfdataset(files, concat_dim='time', combine='nested')
    E_DS = E_DS.load()


    # interpolate ERA5 to DS height grid:
    E_DS_col = xr.Dataset(coords={'time': E_DS.time.values, 'height': DS.height.values})
    new_shape = (len(E_DS_col.time), len(E_DS_col.height))
    for dv in ['temp', 'q', 'u', 'v', 'pres']:
        E_DS_col[dv] = xr.DataArray(np.full(new_shape, np.nan), dims=['time', 'height'])
        for k in range(new_shape[0]):
            E_DS_col[dv][k,...] = np.interp(E_DS_col.height.values, E_DS.Z.values[k,:], E_DS[dv].values[k,:])


    # compute relative humidity and spec humidity:
    E_DS_col['rh'] = convert_spechum_to_relhum(E_DS_col.temp, E_DS_col.pres, E_DS_col.q)
    DS['q'] = convert_rh_to_spechum(DS.temp, DS.pres, DS.rh)


    # compute wind speed and dir:
    wspeed, wdir = u_v_to_wspeed_wdir(E_DS_col.u.values, E_DS_col.v.values, convention='from')
    E_DS_col['wspeed'] = xr.DataArray(wspeed, dims=['time', 'height'])
    E_DS_col['wdir'] = xr.DataArray(wdir, dims=['time', 'height'])


    # freq. occurrence of errors at certain height:
    if set_dict['err_distrib']:
        hgt = 5000
        E_DS_hgt = E_DS_col.sel(height=hgt)
        DS_hgt = DS.sel(height=hgt)
        errors = dict()
        for dv in ['temp', 'q', 'rh', 'wspeed', 'pres']:
            errors[dv] = np.abs(E_DS_hgt[dv].values - DS_hgt[dv].values)

        # filter one dropsonde:
        where_ok = errors['pres'] <= 2000
        DS = DS.sel(launch_time=where_ok)
        E_DS_col = E_DS_col.sel(time=where_ok)

        label_dict = {'temp': "T", 'q': "q", 'rh': "rh", 'pres': "p",
                        'u': "u", 'v': "v", 'wspeed': "wind speed"}
        panel_id = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


    # create error profiles:
    ERR_DS = {}
    if set_dict['separate_ice']:
        # sea ice mask:
        if set_dict['err_distrib']: SIC_DS = SIC_DS.sel(time=where_ok)
        ice_mask = SIC_DS.siconc.values > 0

        ERR_DS['ice'] = get_errors(E_DS_col.sel(time=ice_mask), DS.sel(launch_time=ice_mask), set_dict)
        ERR_DS['ocean'] = get_errors(E_DS_col.sel(time=(~ice_mask)), DS.sel(launch_time=(~ice_mask)), set_dict)
    else:
        ERR_DS['all'] = get_errors(E_DS_col, DS, set_dict)



    # visualize:

    # save limits:
    x_lim_N = np.array([0, len(DS.launch_time)])
    bias_lim = {'temp': np.array([-2.,2.]), 'q': np.array([-0.2, 0.2]), 
                'rh': np.array([-15., 15.]), 'pres': np.array([-2., 2.]),
                'wspeed': np.array([-2, 2])}
    rmse_lim = {'temp': np.array([0,3.]), 'q': np.array([0, 0.6]), 
                'rh': np.array([0, 30.]), 'pres': np.array([0, 3.]),
                'wspeed': np.array([0, 4.])}
    units_dict = {'temp': "K", 'q': "g$\,$kg$^{-1}$", 'rh': "%", 'pres': "hPa",
                    'wspeed': "m$\,$s$^{-1}$"}
    label_dict = {'temp': "T", 'q': "q", 'rh': "rh", 'pres': "p",
                    'wspeed': "wind speed\n"}


    if not set_dict['all_in_one']:
        y_lim = np.array([E_DS_col.height.values[0], E_DS_col.height.values[-1]])
        for dv in ['temp', 'q', 'rh', 'wspeed', 'pres']:
            f1 = plt.figure(figsize=(12,6))
            a1 = plt.subplot2grid((1,2), (0,0))
            a2 = plt.subplot2grid((1,2), (0,1))

            a1.vlines(0, y_lim[0], y_lim[1], color=(0,0,0), linewidths=1.0)
            a1.plot(ERR_DS[dv+"_bias"], ERR_DS.height, linewidth=1.5, color=(0,0,0))
            a1_add = a1.twiny()
            a1_add.plot(ERR_DS[dv+"_N"], ERR_DS.height, linewidth=1.0, color=(0,0,1))


            a2.plot(ERR_DS[dv+"_rmse"], ERR_DS.height, linewidth=1.5, color=(0,0,0), label='RMSE')
            a2.plot(ERR_DS[dv+"_std"], ERR_DS.height, linewidth=1.5, linestyle='dashed', 
                    color=(0,0,0), label='RMSE$_{\mathrm{corr}}$')

            a1.text(0.0, 1.05, "(a)", fontsize=fs, ha='right', va='bottom', transform=a1.transAxes)
            a2.text(0.0, 1.05, "(b)", fontsize=fs, ha='left', va='bottom', transform=a2.transAxes)

            lh, ll = a2.get_legend_handles_labels()
            a2.legend(lh, ll, loc='upper right', fontsize=fs_small)

            a1.tick_params(axis='both', labelsize=fs_dwarf)
            a1_add.tick_params(axis='both', labelsize=fs_dwarf, color=(0,0,1), labelcolor=(0,0,1))
            a2.tick_params(axis='both', labelsize=fs_dwarf)

            a1.set_xlim(bias_lim[dv])
            a1.set_ylim(y_lim)
            a2.set_ylim(y_lim)
            a2.set_xlim(rmse_lim[dv])

            a1_add.set_xlabel("Number of nonnan dropsondes", fontsize=fs, color=(0,0,1))
            a1.set_xlabel("Bias$_{\mathrm{" + label_dict[dv] + "}}$" + f"({units_dict[dv]})", fontsize=fs)
            a1.set_ylabel("Height (m)", fontsize=fs)
            a2.set_xlabel("RMSE$_{\mathrm{" + label_dict[dv] + "}}$" + f"({units_dict[dv]})", fontsize=fs)


            if set_dict['save_figures']:
                plotname = f"HALO-AC3_ERA5_dropsonde_comp_{dv}"
                plotfile = path_plots + plotname
                f1.savefig(plotfile + ".png", dpi=300, bbox_inches='tight')
                print(f"Saved {plotfile}....")
            else:
                plt.show()
                pdb.set_trace()


    else:
            f1 = plt.figure(figsize=(10,11))
            a1 = plt.subplot2grid((4,2), (0,0))
            a2 = plt.subplot2grid((4,2), (0,1))
            a3 = plt.subplot2grid((4,2), (1,0))
            a4 = plt.subplot2grid((4,2), (1,1))
            a5 = plt.subplot2grid((4,2), (2,0))
            a6 = plt.subplot2grid((4,2), (2,1))
            a7 = plt.subplot2grid((4,2), (3,0))
            a8 = plt.subplot2grid((4,2), (3,1))
            # a9 = plt.subplot2grid((5,2), (4,0))
            # a10 = plt.subplot2grid((5,2), (4,1))
            # a11 = plt.subplot2grid((6,2), (5,0))
            # a12 = plt.subplot2grid((6,2), (5,1))

            # figure panel identifiers:
            panel_id_left = ["(a)", "(c)", "(e)", "(g)", "(i)", "(k)"]
            panel_id_right = ["(b)", "(d)", "(f)", "(h)", "(j)", "(l)"]
            k = 0


            # change units of height to km:
            y_lim = [DS.height.values[0]*0.001, DS.height.values[-1]*0.001]     # y limits in km
            for key, err_ds in ERR_DS.items():
                err_ds['height'] = err_ds.height*0.001

            for ax, dv in zip([a1,a3,a5,a7], ['temp', 'rh', 'wspeed', 'pres']):

                ax.vlines(0, y_lim[0], y_lim[1], color=(0,0,0), linewidths=1.0)     # dummy

                
                # plot the data: loop over ERR_DS keys:
                ax_add = ax.twiny()
                for key, err_ds in ERR_DS.items():
                    ax.plot(err_ds[dv+"_bias"], err_ds.height, linewidth=1.5, color=colours[key], label=f'Bias {key}')
                    ax_add.plot(err_ds[dv+"_N"], err_ds.height, linewidth=1.0, linestyle=line_styles[key], color=colours_N[key],
                                label=f'N {key}')   # number of dropsonde obs


                ax_add.text(0.03, 0.96, panel_id_left[k], fontsize=fs, bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(1,1,1,0.5)),
                            ha='left', va='top', transform=ax.transAxes)
                ax_add.text(0.98, 0.98, label_dict[dv] + f" ({units_dict[dv]})", fontsize=fs, fontweight='bold', color=(0.9,0.9,0.9),
                        path_effects=[mpl_pe.Stroke(linewidth=2, foreground=(0,0,0)), mpl_pe.Normal()],
                        ha='right', va='top', transform=ax.transAxes, zorder=99)

                if k == 0: 
                    lh, ll = ax_add.get_legend_handles_labels()
                    ax_add.legend(lh, ll, loc='center right', handlelength=1., fontsize=fs_dwarf)

                if k == 1:
                    lh, ll = ax.get_legend_handles_labels()
                    ax.legend(lh, ll, loc='center right', handlelength=1., fontsize=fs_dwarf)


                ax.tick_params(axis='both', labelsize=fs_dwarf)
                ax_add.tick_params(axis='both', labelsize=fs_dwarf, color=(0,0,1), labelcolor=(0,0,1))
                if k > 0: ax_add.set_xticklabels([])

                ax.set_xlim(bias_lim[dv])
                ax_add.set_xlim(x_lim_N)
                ax.set_ylim(y_lim)

                ax.set_ylabel("Height (km)", fontsize=fs)
                if k == 0: ax_add.set_xlabel("Number of dropsondes", fontsize=fs, color=(0,0,1))
                if k == 3: ax.set_xlabel(f"Bias", fontsize=fs)

                k += 1


            k = 0
            for ax, dv in zip([a2,a4,a6,a8], ['temp', 'rh', 'wspeed', 'pres']):

                # plot data:
                for key, err_ds in ERR_DS.items():
                    ax.plot(err_ds[dv+"_rmse"], err_ds.height, linewidth=1.5, color=colours[key], label=f'RMSD {key}')


                ax.text(0.03, 0.96, panel_id_right[k], fontsize=fs, bbox=dict(facecolor=(1,1,1,0.5), edgecolor=(1,1,1,0.5)),
                            ha='left', va='top', transform=ax.transAxes)
                ax.text(0.98, 0.98, label_dict[dv] + f" ({units_dict[dv]})", fontsize=fs, fontweight='bold', color=(0.9,0.9,0.9),
                        path_effects=[mpl_pe.Stroke(linewidth=2, foreground=(0,0,0)), mpl_pe.Normal()],
                        ha='right', va='top', transform=ax.transAxes, zorder=99)

                lh, ll = ax.get_legend_handles_labels()
                if k == 0: ax.legend(lh, ll, loc='center right', fontsize=fs_dwarf)

                ax.tick_params(axis='both', labelsize=fs_dwarf)
                ax.set_yticklabels([])

                ax.set_ylim(y_lim)
                ax.set_xlim(rmse_lim[dv])

                if k == 3: ax.set_xlabel(f"Root mean square deviation (RMSD)", fontsize=fs)
                k += 1


            plt.tight_layout()

            if set_dict['save_figures']:
                plot_add = "_all"
                if set_dict['separate_ice']: plot_add = "_ice_vs_ocean"
                plotname = f"HALO-AC3_ERA5_dropsonde_comp{plot_add}"
                plotfile = path_plots + plotname
                f1.savefig(plotfile + ".pdf", dpi=300, bbox_inches='tight')
                print(f"Saved {plotfile}....")
            else:
                plt.show()
                pdb.set_trace()