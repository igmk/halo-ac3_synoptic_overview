import os
import sys
import glob
import pdb
import datetime as dt

wdir = os.getcwd() + "/"
path_data_base = os.path.dirname(wdir[:-10]) + "/data/"
path_tools = os.path.dirname(wdir[:-1]) + "/tools/"

import numpy as np
import xarray as xr

sys.path.insert(0, path_tools)
from halo_classes import P5_dropsondes, dropsondes
from data_tools import find_files_daterange


"""
    Quick and dirty script to collocate ERA5 model level data with HALO dropsondes.
"""


# paths: 
path_data = path_data_base + "ERA5_data/model_level/"
path_dropsondes_halo = path_data_base + "HALO/dropsondes/"
path_output =  path_data_base + "ERA5_data/model_level/"

# create output path if not existing:
os.makedirs(os.path.dirname(path_output), exist_ok=True)


# import dropsonde data for lat-lon positions:
DS_DS = dropsondes(path_dropsondes_halo, 'raw', return_DS=True)


# list ERA5 files:
files = sorted(glob.glob(path_data + "ERA5_model_level_HALO-AC3_*.nc"))

date_list = np.array([np.datetime64("2022-03-12"),
                        np.datetime64("2022-03-13"),
                        np.datetime64("2022-03-14"),
                        np.datetime64("2022-03-15"),
                        np.datetime64("2022-03-16"),
                        np.datetime64("2022-03-20"),
                        np.datetime64("2022-03-21"),
                        np.datetime64("2022-03-28"),
                        np.datetime64("2022-03-29"),
                        np.datetime64("2022-03-30"),
                        np.datetime64("2022-04-01"),
                        np.datetime64("2022-04-04"),
                        np.datetime64("2022-04-07"),
                        np.datetime64("2022-04-08"),
                        np.datetime64("2022-04-10"),
                        np.datetime64("2022-04-11"),
                        np.datetime64("2022-04-12")])

for date in date_list:
    # select dropsondes of this data:
    date_str = str(date)
    
    DS = DS_DS.DS.sel(launch_time=str(date))

    # find correct ERA5 file:
    file = find_files_daterange(files, dt.datetime.strptime(date_str, "%Y-%m-%d"),  
                                dt.datetime.strptime(date_str, "%Y-%m-%d"), [-11,-3],
                                file_dt_fmt="%Y%m%d")
    if len(file) == 1:
        E_DS = xr.open_dataset(file[0])
    else:
        pdb.set_trace()


    # preselect ERA5 data: closest lat, lon, time: But, now, each of these dimensions has 
    E_DS_presel = E_DS.sel(latitude=DS.ref_lat.values, longitude=DS.ref_lon.values, 
                            time=DS.launch_time.values, method='nearest')
    E_DS_presel = E_DS_presel.load()


    # reduce reanalysis data dimensions further: select the correct latitude and longitude dimension for 
    # the respective times (i.e., choose lat index 0 for the first launch, lat index 1 for the second launch...):
    E_DS_sel = xr.Dataset(coords={'time': E_DS_presel.time, 'level': E_DS_presel.level})
    for dv in ['temp', 'q', 'u', 'v', 'pres', 'Z']:
        E_DS_sel[dv] = xr.DataArray(np.full((len(E_DS_presel.time), len(E_DS_presel.level)), np.nan), dims=['time', 'level'])
    for dv in ['z_sfc', 'pres_sfc', 'Z_sfc', 'latitude', 'longitude']:
        E_DS_sel[dv] = xr.DataArray(np.full((len(E_DS_presel.time),), np.nan), dims=['time'])


    # loop through ERA5_M_DS time and check which radiosonde launch time is closest
    for k, e_time in enumerate(E_DS_presel.time.values):
        
        # loop through data vars of the ERA5 dataset and put correct selection into the _red dataset:
        for dv in ['temp', 'q', 'u', 'v', 'pres', 'Z']:
            E_DS_sel[dv][k,...] = E_DS_presel[dv].isel(time=k, latitude=k, longitude=k)

        for dv in ['z_sfc', 'pres_sfc', 'Z_sfc']:
            E_DS_sel[dv][k] = E_DS_presel[dv].isel(time=k, latitude=k, longitude=k)

        # also get latitude and longitude information:
        E_DS_sel['latitude'][k] = E_DS_presel['latitude'][k]
        E_DS_sel['longitude'][k] = E_DS_presel['longitude'][k]


    # export and attributes:
    E_DS_sel.attrs['history'] = f"{dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')}: processed with ERA5_dropsonde_collocation.py"

    outfile = f"HALO_AC3_ERA5-HALO_collocated_{date_str.replace('-', '')}.nc"
    E_DS_sel.to_netcdf(path_output + outfile, mode='w', format="NETCDF4")
    E_DS_sel = E_DS_sel.close()
    print(f"Saved {path_output + outfile}")
    del E_DS_sel