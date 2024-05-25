import numpy as np
import datetime as dt
import xarray as xr
import pdb
import sys
import glob
import os

from data_tools import numpydatetime64_to_reftime


# Dictionary translating Research Flight numbers and dates:
RF_dict = {
            '20220225': "RF00",
            "20220311": "RF01",
            "20220312": "RF02",
            "20220313": "RF03",
            "20220314": "RF04",
            "20220315": "RF05",
            "20220316": "RF06",
            "20220320": "RF07",
            "20220321": "RF08",
            "20220328": "RF09",
            "20220329": "RF10",
            "20220330": "RF11",
            "20220401": "RF12",
            "20220404": "RF13",
            "20220407": "RF14",
            "20220408": "RF15",
            "20220410": "RF16",
            "20220411": "RF17",
            "20220412": "RF18",
            }


class dropsondes:

    """
        HALO dropsondes launched during the field campaign(s) HALO-(AC)3. Several versions are 
        supported (see dataset_type and version). All dropsondes will be merged into a
        (launch_time, height) grid. Variable names will be unified in the class attributes
        (also in self.DS).
        

        For initialisation, we need:
        path_data : str
            String indicating the path of the dropsonde data. Subfolders may exist, depending on the
            dropsonde data version.
        dataset_type : str
            Indicates the type of dropsonde data. Options: "raw"
        version : str
            Indicates the version of the dropsonde data type.

        **kwargs:
        return_DS : bool
            If True, the imported xarray dataset will also be set as a class attribute.
        height_grid : 1D array of floats
            1D array of floats indicating the new height grid (especially raw) dropsonde data
            is interpolated to. Units: m
    """

    def __init__(self, path_data, dataset_type, version="", **kwargs):

        # init attributes:
        self.temp = np.array([])            # air temperature in K
        self.pres = np.array([])            # air pressure in Pa
        self.rh = np.array([])              # relative humidity in [0, 1]
        self.height = np.array([])          # height in m
        self.launch_time = np.array([])     # launch time in sec since 1970-01-01 00:00:00 (for HALO-AC3)
        self.time = np.array([])            # time since launch_time in seconds
        self.u = np.array([])               # zonal wind component in m s-1
        self.v = np.array([])               # meridional wind component in m s-1
        self.wspeed = np.array([])          # wind speed in m s-1
        self.wdir = np.array([])            # wind direction in deg
        self.lat = np.array([])             # latitude in deg N
        self.lon = np.array([])             # longitude in deg E
        self.DS = None                      # xarray dataset
        self.height_grid = np.array([])     # height grid in m
        self.n_hgt = 0                      # number of height levels

        if ('height_grid' in kwargs.keys()): 
            self.height_grid = kwargs['height_grid']
            self.n_hgt = len(self.height_grid)
        else:
            # create default height grid:
            self.height_grid = np.arange(0.0, 16000.001, 10.0)
            self.n_hgt = len(self.height_grid)


        # Importing dropsonde files, depending on the type and version:
        if dataset_type == 'raw':

            # dictionary to change variable names: (also filters for relevant variables)
            translator_dict = { 'launch_time': 'launch_time',
                                'time': 'time',
                                'pres': 'pres',
                                'tdry': 'temp',
                                'rh': 'rh',
                                'u_wind': 'u',
                                'v_wind': 'v',
                                'wspd': 'wspeed',
                                'wdir': 'wdir',
                                'lat': 'lat',
                                'lon': 'lon',
                                'alt': 'height',
                                }

            # dictionary for final units:
            unit_dict = {   'launch_time': "seconds since 1970-01-01 00:00:00",
                            'time': "seconds since 1970-01-01 00:00:00",
                            'pres': "Pa",
                            'temp': "K",
                            'rh': "[0,1]",
                            'u': "m s-1",
                            'v': "m s-1",
                            'wspeed': "m s-1",
                            'wdir': "deg",
                            'lat': "deg N",
                            'lon': "deg E",
                            'ref_lat': "deg N",
                            'ref_lon': "deg E",}

            # search for daily subfolders and in them for *QC.nc:
            path_contents = os.listdir(path_data)
            subfolders = []
            for subfolder in path_contents:

                joined_contents = os.path.join(path_data, subfolder)
                if os.path.isdir(joined_contents):
                    subfolders.append(joined_contents + "/")

            subfolders = sorted(subfolders)

            # find ALL dropsonde data files:
            # check if subfolders contain "Level_1", which should exist for *QC.nc:
            files_nc = []                   # will contain all dropsonde files
            for subfolder in subfolders:    # this loop basically loops over the daily dropsondes:

                subfolder_contents = os.listdir(subfolder)
                if "Level_1" in subfolder_contents:
                    files_nc = files_nc + sorted(glob.glob(subfolder + "Level_1/D*QC.nc"))

                else:
                    print(f"Could not find Level_1 dropsonde data in {subfolder} :(")

            # check if nc files were detected:
            if len(files_nc) == 0: raise RuntimeError("Where's the dropsonde data?? I can't find it.\n")


            # import data: importing with mfdataset costs a lot of memory and is therefore discarded here:
            DS_dict = dict()        # keys will indicate the dropsonde number of that day
            for k, file in enumerate(files_nc): DS_dict[str(k)] = xr.open_dataset(file)

            # interpolate dropsonde data to new height grid for all sondes; initialise array
            self.n_sondes = len(DS_dict.keys())
            vars_ip = dict()
            for var in translator_dict.keys(): vars_ip[var] = np.full((self.n_sondes, self.n_hgt), np.nan)

            # set reference latitudes and longitudes: highest non-nan latitude/longitude:
            self.ref_lat = np.asarray([DS_dict[key].lat.values[np.where(~np.isnan(DS_dict[key].lat.values))[0][-1]] for key in DS_dict.keys()])
            self.ref_lon = np.asarray([DS_dict[key].lon.values[np.where(~np.isnan(DS_dict[key].lon.values))[0][-1]] for key in DS_dict.keys()])

            for k, key in enumerate(DS_dict.keys()):

                # need to neglect nans:
                idx_nonnan = np.where(~np.isnan(DS_dict[key].alt.values))[0]
                alt_label = 'alt'       # use this altitude variable in general
                if len(idx_nonnan) < 30: # then use gps altitude because meteo measurements seem to be broken
                    idx_nonnan = np.where(~np.isnan(DS_dict[key].gpsalt.values))[0]
                    alt_label = 'gpsalt'        # use this altitude variable instead when 'alt' is broken

                # interpolate to new grid:
                for var in translator_dict.keys():
                    if var not in ['launch_time', 'time']:
                        try:
                            vars_ip[var][k,:] = np.interp(self.height_grid, DS_dict[key][alt_label].values[idx_nonnan],
                                                            DS_dict[key][var].values[idx_nonnan], left=np.nan, right=np.nan)
                        except ValueError:
                            continue    # then, array for interpolation seems empty --> just leave nans is it

                    elif var == 'time':
                        # catch errors (empty array):
                        try:
                            vars_ip[var][k,:] = np.interp(self.height_grid, DS_dict[key][alt_label].values[idx_nonnan], 
                                                            DS_dict[key][var].values[idx_nonnan].astype("float64")*(1e-09),
                                                            left=np.nan, right=np.nan)
                        except ValueError:
                            continue    # then, array for interpolation seems empty --> just leave nans is it

                """
                # Uncomment if you would like to plot raw and interpolated dropsonde data (i.e., to check for correct procedures):
                if k%15 == 0:       # test some samples

                    f1, a1 = plt.subplots(1,3)
                    a1 = a1.flatten()
                    a1[0].plot(vars_ip['tdry'][k,:], self.height_grid, color=(0,0,0), label='new')
                    a1[0].plot(DS_dict[key].tdry.values[idx_nonnan], DS_dict[key].alt.values[idx_nonnan], color=(1,0,0), linestyle='dashed', label='old')
                    a1[1].plot(vars_ip['pres'][k,:], self.height_grid, color=(0,0,0), label='new')
                    a1[1].plot(DS_dict[key].pres.values[idx_nonnan], DS_dict[key].alt.values[idx_nonnan], color=(1,0,0), linestyle='dashed', label='old')
                    a1[2].plot(vars_ip['rh'][k,:], self.height_grid, color=(0,0,0), label='new')
                    a1[2].plot(DS_dict[key].rh.values[idx_nonnan], DS_dict[key].alt.values[idx_nonnan], color=(1,0,0), linestyle='dashed', label='old')

                    for ax in a1:
                        ax.legend()
                        ax.set_ylabel("Height (m)")
                    a1[0].set_xlabel("tdry (degC)")
                    a1[1].set_xlabel("pres (hPa)")
                    a1[2].set_xlabel("rh (\%)")
                    a1[1].set_title(f"{DS_dict[key].launch_time.values.astype('datetime64[D]')}")

                    f1.savefig(f"/net/blanc/awalbroe/Plots/HALO_AC3/CSSC/dropsonde_ip_vs_original_{str(DS_dict[key].launch_time.values.astype('datetime64[D]')).replace('-','')}_{k}.png", 
                                dpi=300, bbox_inches='tight')
                    plt.close()
                    gc.collect()
                """


            # convert units of vars_ip to SI units:
            vars_ip['tdry'] = vars_ip['tdry'] + 273.15
            vars_ip['pres'] = vars_ip['pres']*100.0
            vars_ip['rh'] = vars_ip['rh']*0.01

            # create launch_time array:
            launch_time = np.zeros((self.n_sondes,))
            launch_time_npdt = np.full((self.n_sondes,), np.datetime64("1970-01-01T00:00:00.000000000"))
            for kk, key in enumerate(DS_dict.keys()):
                launch_time[kk] = DS_dict[key].launch_time.values.astype(np.float64)*(1e-09)
                launch_time_npdt[kk] = DS_dict[key].launch_time.values

            # compute time difference between launch times and true dropsonde measured times
            vars_ip['time_delta'] = np.full((self.n_sondes, self.n_hgt), np.nan)
            for k in range(self.n_sondes):
                vars_ip['time_delta'][k,:] = vars_ip['time'][k,:] - launch_time[k]


            # set class attributes:
            self.temp = vars_ip['tdry']
            self.pres = vars_ip['pres']
            self.rh = vars_ip['rh']
            self.height = self.height_grid
            self.launch_time = launch_time
            self.launch_time_npdt = launch_time_npdt
            self.time = vars_ip['time_delta']
            self.u = vars_ip['u_wind']
            self.v =  vars_ip['v_wind']
            self.wspeed = vars_ip['wspd']
            self.wdir = vars_ip['wdir']
            self.lat = vars_ip['lat']
            self.lon =  vars_ip['lon']


            # build new dataset with (launch_time, height) grid:
            if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:

                DS = xr.Dataset(coords={'launch_time': (['launch_time'], self.launch_time_npdt),
                                        'height': (['height'], self.height_grid, {'units': "m"})})


                for key in unit_dict.keys():
                    if key not in ['launch_time', 'time', 'ref_lat', 'ref_lon']:
                        DS[key] = xr.DataArray(self.__dict__[key], dims=['launch_time', 'height'], 
                                                                attrs={'units': unit_dict[key]})
                    elif key == 'time':

                        DS[key] = xr.DataArray(self.time, dims=['launch_time', 'height'],
                                                                attrs={'units': "seconds since launch_time"})

                    elif key in ['ref_lat', 'ref_lon']:
                        DS[key] = xr.DataArray(self.__dict__[key], dims=['launch_time'], 
                                                                attrs={'units': unit_dict[key]})

                DS.attrs['title'] = "HALO-(AC)3 HALO dropsondes Level_1 interpolated to (launch_time, height) grid"

                self.DS = DS


    def update_meteo_attrs(self):

        """
        Update meteorological profiles of dropsondes
        """

        pdb.set_trace()


class P5_dropsondes:

    """
        P5 dropsondes launched during the field campaign(s) HALO-(AC)3. Currently, only the 
        raw level 1 dropsondes are supported. All dropsondes will be merged into a 
        (launch_time, height) grid. Variable names will be unified in the class attributes
        (also in self.DS).
        

        For initialisation, we need:
        path_data : str
            String indicating the path of the dropsonde data. Subfolders may exist, depending on the
            dropsonde data version.
        dataset_type : str
            Indicates the type of dropsonde data. Options: "raw"

        **kwargs:
        return_DS : bool
            If True, the imported xarray dataset will also be set as a class attribute.
        height_grid : 1D array of floats
            1D array of floats indicating the new height grid (especially raw) dropsonde data
            is interpolated to. Units: m
    """

    def __init__(self, path_data, dataset_type, **kwargs):

        # init attributes:
        self.temp = np.array([])            # air temperature in K
        self.pres = np.array([])            # air pressure in Pa
        self.rh = np.array([])              # relative humidity in [0, 1]
        self.height = np.array([])          # height in m
        self.launch_time = np.array([])     # launch time in sec since 1970-01-01 00:00:00 (for HALO-AC3)
        self.time = np.array([])            # time since launch_time in seconds
        self.u = np.array([])               # zonal wind component in m s-1
        self.v = np.array([])               # meridional wind component in m s-1
        self.wspeed = np.array([])          # wind speed in m s-1
        self.wdir = np.array([])            # wind direction in deg
        self.lat = np.array([])             # latitude in deg N
        self.lon = np.array([])             # longitude in deg E
        self.DS = None                      # xarray dataset
        self.height_grid = np.array([])     # height grid in m
        self.n_hgt = 0                      # number of height levels

        if ('height_grid' in kwargs.keys()): 
            self.height_grid = kwargs['height_grid']
            self.n_hgt = len(self.height_grid)
        else:
            # create default height grid:
            self.height_grid = np.arange(0.0, 16000.001, 10.0)
            self.n_hgt = len(self.height_grid)


        # Importing dropsonde files, depending on the type and version:
        if dataset_type == 'raw':

            # dictionary to change variable names: (also filters for relevant variables)
            translator_dict = { 'launch_time': 'launch_time',
                                'time': 'time',
                                'pres': 'pres',
                                'tdry': 'temp',
                                'rh': 'rh',
                                'u_wind': 'u',
                                'v_wind': 'v',
                                'wspd': 'wspeed',
                                'wdir': 'wdir',
                                'lat': 'lat',
                                'lon': 'lon',
                                'alt': 'height',
                                }

            # dictionary for final units:
            unit_dict = {   'launch_time': "seconds since 1970-01-01 00:00:00",
                            'time': "seconds since 1970-01-01 00:00:00",
                            'pres': "Pa",
                            'temp': "K",
                            'rh': "[0,1]",
                            'u': "m s-1",
                            'v': "m s-1",
                            'wspeed': "m s-1",
                            'wdir': "deg",
                            'lat': "deg N",
                            'lon': "deg E",
                            'ref_lat': "deg N",
                            'ref_lon': "deg E",}

            # search for daily subfolders and in them for *QC.nc:
            path_contents = os.listdir(path_data)
            subfolders = []
            for subfolder in path_contents:

                joined_contents = os.path.join(path_data, subfolder)
                if os.path.isdir(joined_contents):
                    subfolders.append(joined_contents + "/")

            subfolders = sorted(subfolders)

            # find ALL dropsonde data files:
            # check if subfolders contain "Level_1", which should exist for *QC.nc:
            files_nc = []                   # will contain all dropsonde files
            for subfolder in subfolders:    # this loop basically loops over the daily dropsondes:

                subfolder_contents = os.listdir(subfolder)
                if "Level_1" in subfolder_contents:
                    files_nc = files_nc + sorted(glob.glob(subfolder + "Level_1/D*QC.nc"))

                else:
                    raise ValueError(f"Could not find Level_1 dropsonde data in {subfolder} :(")

            # check if nc files were detected:
            if len(files_nc) == 0: raise RuntimeError("Where's the dropsonde data?? I can't find it.\n")


            # import data: importing with mfdataset costs a lot of memory and is therefore discarded here:
            DS_dict = dict()        # keys will indicate the dropsonde number of that day
            for k, file in enumerate(files_nc): DS_dict[str(k)] = xr.open_dataset(file)

            # interpolate dropsonde data to new height grid for all sondes; initialise array
            self.n_sondes = len(DS_dict.keys())
            vars_ip = dict()            # interpolated data variable
            for var in translator_dict.keys(): vars_ip[var] = np.full((self.n_sondes, self.n_hgt), np.nan)

            # set reference latitudes and longitudes: highest non-nan latitude/longitude:
            self.ref_lat = np.asarray([DS_dict[key].lat.values[np.where(~np.isnan(DS_dict[key].lat.values))[0][-1]] for key in DS_dict.keys()])
            self.ref_lon = np.asarray([DS_dict[key].lon.values[np.where(~np.isnan(DS_dict[key].lon.values))[0][-1]] for key in DS_dict.keys()])

            for k, key in enumerate(DS_dict.keys()):

                # need to neglect nans:
                idx_nonnan = np.where(~np.isnan(DS_dict[key].alt.values))[0]
                alt_label = 'alt'       # use this altitude variable in general
                if len(idx_nonnan) < 30: # then use gps altitude because meteo measurements seem to be broken
                    idx_nonnan = np.where(~np.isnan(DS_dict[key].gpsalt.values))[0]
                    alt_label = 'gpsalt'        # use this altitude variable instead when 'alt' is broken

                # interpolate to new grid:
                for var in translator_dict.keys():
                    if var not in ['launch_time', 'time']:
                        try:
                            vars_ip[var][k,:] = np.interp(self.height_grid, DS_dict[key][alt_label].values[idx_nonnan],
                                                            DS_dict[key][var].values[idx_nonnan], left=np.nan, right=np.nan)
                        except ValueError:
                            continue    # then, array for interpolation seems empty --> just leave nans in it

                    elif var == 'time':
                        # catch errors (empty array):
                        try:
                            vars_ip[var][k,:] = np.interp(self.height_grid, DS_dict[key][alt_label].values[idx_nonnan], 
                                                            DS_dict[key][var].values[idx_nonnan].astype("float64")*(1e-09),
                                                            left=np.nan, right=np.nan)
                        except ValueError:
                            continue    # then, array for interpolation seems empty --> just leave nans in it


            # convert units of vars_ip to SI units:
            vars_ip['tdry'] = vars_ip['tdry'] + 273.15
            vars_ip['pres'] = vars_ip['pres']*100.0
            vars_ip['rh'] = vars_ip['rh']*0.01

            # create launch_time array:
            launch_time = np.zeros((self.n_sondes,))
            launch_time_npdt = np.full((self.n_sondes,), np.datetime64("1970-01-01T00:00:00.000000000"))
            for kk, key in enumerate(DS_dict.keys()):
                launch_time[kk] = DS_dict[key].time.min().values.astype('datetime64[s]').astype(np.float64)
                launch_time_npdt[kk] = DS_dict[key].time.min().values

            # compute time difference between launch times and true dropsonde measured times
            vars_ip['time_delta'] = np.full((self.n_sondes, self.n_hgt), np.nan)
            for k in range(self.n_sondes):
                vars_ip['time_delta'][k,:] = vars_ip['time'][k,:] - launch_time[k]


            # set class attributes:
            self.temp = vars_ip['tdry']
            self.pres = vars_ip['pres']
            self.rh = vars_ip['rh']
            self.height = self.height_grid
            self.launch_time = launch_time
            self.launch_time_npdt = launch_time_npdt
            self.time = vars_ip['time_delta']
            self.u = vars_ip['u_wind']
            self.v =  vars_ip['v_wind']
            self.wspeed = vars_ip['wspd']
            self.wdir = vars_ip['wdir']
            self.lat = vars_ip['lat']
            self.lon =  vars_ip['lon']


            # build new dataset with (launch_time, height) grid:
            if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:

                DS = xr.Dataset(coords={'launch_time': (['launch_time'], self.launch_time_npdt),
                                        'height': (['height'], self.height_grid, {'units': "m"})})


                for key in unit_dict.keys():
                    if key not in ['launch_time', 'time', 'ref_lat', 'ref_lon']:
                        DS[key] = xr.DataArray(self.__dict__[key], dims=['launch_time', 'height'], 
                                                                attrs={'units': unit_dict[key]})
                    elif key == 'time':

                        DS[key] = xr.DataArray(self.time, dims=['launch_time', 'height'],
                                                                attrs={'units': "seconds since launch_time"})

                    elif key in ['ref_lat', 'ref_lon']:
                        DS[key] = xr.DataArray(self.__dict__[key], dims=['launch_time'], 
                                                                attrs={'units': unit_dict[key]})

                DS.attrs['title'] = "HALO-(AC)3 P5 dropsondes Level_1 interpolated to (launch_time, height) grid"

                self.DS = DS


class MWR:
    """
        RAW microwave radiometer onboard HALO (part of HAMP). Time will be given in seconds since
        2017-01-01 00:00:00 UTC.

        For initialisation we need:
        path : str
            Path of raw HALO HAMP-MWR data. The path must simply link to HALO and then contain 
            the datefurther folders that then link to the HAMP MWR receivers (KV, 11990, 183): Example:
            path = "/data/obs/campaigns/eurec4a/HALO/" -> contains "./20020205/radiometer/" +
            ["KV/", "11990/", "183/"].
        which_date : str
            Marks the flight day that shall be imported. To be specified in yyyymmdd (e.g. 20200213)!
        version : str
            Version of the HAMP MWR data. Options available: 'raw', 'halo_ac3_raw', 'synthetic_dropsonde'

        **kwargs:
        return_DS : bool
            If True, the imported xarray dataset will also be set as a class attribute.
    """

    def __init__(self, path, which_date, version='halo_ac3_raw', **kwargs):

        assert len(which_date) == 8
        reftime = "2017-01-01 00:00:00"     # in UTC

        # init attributes:
        self.freq = dict()
        self.time = dict()
        self.time_npdt = dict()
        self.TB = dict()
        self.flag = dict()

        if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
            self.DS = dict()

        if version == 'raw':
            path_date = path + f"{which_date}/radiometer/"

            # import data: Identify receivers of the current day:
            self.avail = {'KV': False, '11990': False, '183': False}
            files = dict()
            
            for key in self.avail.keys():
                files[key] = sorted(glob.glob(path_date + key + f"/{which_date[2:]}[0-9][0-9].BRT.NC"))

                if len(files[key]) > 0:
                    self.avail[key] = True

                    # import data: cycle through receivers and import when possible:
                    DS = xr.open_mfdataset(files[key][1:], concat_dim='time', combine='nested')

                    # reduce unwanted dimensions:
                    DS['frequencies'] = DS.frequencies[0,:]
                    
                    # Unify variable names by defining class attributes:
                    self.freq[key] = DS.frequencies.values          # in GHz
                    self.time_npdt[key] = DS.time.values            # in numpy datetime64
                    self.time[key] = numpydatetime64_to_reftime(DS.time.values, reftime) # in seconds since 2017-01-01 00:00:00 UTC
                    self.TB[key] = DS.TBs.values                    # in K, time x freq
                    self.flag[key] = DS.rain_flag.values            # rain flag, no units

                    if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
                        self.DS[key] = DS

                else:
                    print(f"No {key} data on {which_date} from HAMP MWR.\n")

        elif version == 'halo_ac3_raw':

            # import data: Identify receivers of the current day:
            self.avail = {'KV': False, '11990': False, '183': False}
            files = dict()
            
            for key in self.avail.keys():
                files[key] = sorted(glob.glob(path + f"hamp_{key.lower()}/HALO-AC3_HALO_hamp_*_{which_date}*.nc"))

                if len(files[key]) > 0:
                    self.avail[key] = True

                    # import data: cycle through receivers and import when possible:
                    DS = xr.open_dataset(files[key][0])
                    
                    # Unify variable names by defining class attributes:
                    self.freq[key] = DS.Freq.values                 # in GHz
                    self.time_npdt[key] = DS.time.values            # in numpy datetime64
                    self.time[key] = numpydatetime64_to_reftime(DS.time.values, reftime) # in seconds since 2017-01-01 00:00:00 UTC
                    self.TB[key] = DS.TBs.values                    # in K, time x freq
                    self.flag[key] = DS.RF.values           # rain flag, no units

                    if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
                        self.DS[key] = DS

                else:
                    print(f"No {key} data on {which_date} from HAMP MWR.\n")

        elif version == 'synthetic_dropsonde':

            files = dict()

            # import data:
            key = 'dropsonde'
            files[key] = sorted(glob.glob(path + f"HALO-AC3_HALO_Dropsondes_{which_date}_{RF_dict[which_date]}/" +
                                "*.nc"))

            if len(files[key]) > 0:

                # import data: cycle through receivers and import when possible:
                DS = xr.open_mfdataset(files[key], concat_dim='grid_x', combine='nested')

                # Post process PAMTRA simulations: reduce undesired dimensions and          
                # unify variable names by defining class attributes:
                self.freq[key] = DS.frequency.values            # in GHz
                self.time_npdt[key] = DS.datatime.values.flatten()  # in numpy datetime64
                self.time[key] = numpydatetime64_to_reftime(DS.datatime.values.flatten(), reftime) # in seconds since 2017-01-01 00:00:00 UTC
                self.TB[key] = DS.tb.values[:,0,0,0,:,:].mean(axis=-1)      # in K, time x freq

                # apply double side band average:
                self.TB[key], self.freq[key] = Fband_double_side_band_average(self.TB[key], self.freq[key])
                self.TB[key], self.freq[key] = Gband_double_side_band_average(self.TB[key], self.freq[key])

                if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
                    self.DS[key] = DS

            else:
                print(f"No {key} data on {which_date} from HAMP MWR.\n")


class radar:
    """
        Cloud radar onboard HALO (part of HAMP).

        For initialisation we need:
        path : str
            Path of unified HALO HAMP radar data.
        which_date : str
            Marks the flight day that shall be imported. To be specified in 
            yyyymmdd (e.g. 20200213)!
        version : str
            Specifies the data version. Valid option depends on the instrument.

        **kwargs:
        return_DS : bool
            If True, the imported xarray dataset will also be set as a class attribute.
    """

    def __init__(self, path, which_date, version='raw', **kwargs):

        # import data: Identify correct data (version, dates):
        filename = sorted(glob.glob(path + "hamp_mira/" + f"HALO-AC3_HALO_hamp_mira_{which_date}*.nc"))
        if len(filename) == 0:
            raise RuntimeError(f"Could not find and import {filename}.")
        else:
            filename = filename[0]

        if version == 'raw':
            data_DS = xr.open_dataset(filename)
        else:
            raise RuntimeError("Other versions than 'raw' radar data have not yet been implemented.")
        
        # Unify variable names by defining class attributes:
        self.time = numpydatetime64_to_reftime(data_DS.time.values, "2017-01-01 00:00:00")  # in sec since 2017-01-01 00:00:00 UTC
        self.time_npdt = data_DS.time.values        # in numpy datetime64
        self.range = data_DS.range.values           # in m from aircraft on
        self.Z = data_DS.Zg.values                  # radar reflectivity im mm^6 m^-3
        self.dBZ = 10*np.log10(self.Z)              # equivalent reflectivity factor in dBZ

        if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
            self.DS = data_DS


class BAHAMAS:
    """
        BAHAMAS data from HALO for time axis (and eventually other stuff later). Time will be converted
        to 2017-01-01 00:00:00 UTC. 

        For initialisation we need:
        path : str
            Path where HALO BAHAMAS data is located.
        which_date : str
            Marks the flight day that shall be imported. To be specified in yyyymmdd (e.g. 20200213)!
        version : str
            Version of the BAHAMAS data. Options available: 'nc_raw', 'halo_ac3_raw'

        **kwargs:
        return_DS : bool
            If True, the imported xarray dataset will also be set as a class attribute.
    """

    def __init__(self, path, which_date, version='halo_ac3_raw', **kwargs):
        
        if version == 'nc_raw': 
            # Identify correct time:
            files = [file for file in sorted(glob.glob(path + "*.nc")) if which_date in file]

            if len(files) == 1: # then the file is unambiguous
                files = files[0]

            elif len(files) == 0:
                raise RuntimeError(f"No BAHAMAS files found for {which_date} in {path}.")

            else:
                print(f"Multiple potential BAHAMAS files found for {which_date} in {path}. Choose wisely... " +
                        "I'll choose the first file")
                files = files[0]

            # import data:
            DS = xr.open_dataset(files)

            # set attributes:
            reftime = "2017-01-01 00:00:00"         # in UTC
            self.time_npdt = DS.TIME.values         # np.datetime64 array
            self.time = numpydatetime64_to_reftime(DS.TIME.values, reftime) # in seconds since 2017-01-01 00:00:00 UTC

        elif version == 'halo_ac3_raw':
            # Identify correct time: /data/obs/campaigns/ac3airborne/ac3cloud_server/halo-ac3/halo/BAHAMAS/HALO-AC3_HALO_BAHAMAS_20220313_RF03
            path += f"/HALO-AC3_HALO_BAHAMAS_{which_date}_{RF_dict[which_date]}/"
            files = [file for file in sorted(glob.glob(path + "*BAHAMAS*.nc")) if which_date in file]

            if len(files) == 1: # then the file is unambiguous
                files = files[0]

            elif len(files) == 0:
                raise RuntimeError(f"No BAHAMAS files found for {which_date} in {path}.")

            else:
                print(f"Multiple potential BAHAMAS files found for {which_date} in {path}. Choose wisely... " +
                        "I'll choose the first file")
                files = files[0]

            # import data:
            DS = xr.open_dataset(files)

            # eventually convert time units from millisec since midnight to unixtime
            if "midnight" in DS.TIME.long_name:
                t_axis = np.full((len(DS.TIME),), np.datetime64(f"{dt.datetime.strptime(which_date, '%Y%m%d').strftime('%Y-%m-%d')}T00:00:00"))
                t_axis = t_axis + DS.TIME.values.astype("timedelta64[ms]")
                DS['TIME'] = xr.DataArray(t_axis, dims=['tid'], attrs={'long_name': "time in seconds since 1970-1-1 (UTC)",
                                                                        'standard_name': " ",
                                                                        'units': "seconds since 1970-1-1 00:00:00 +00:00"})

            # set attributes:
            reftime = "2017-01-01 00:00:00"         # in UTC
            self.time_npdt = DS.TIME.values         # np.datetime64 array
            self.time = numpydatetime64_to_reftime(DS.TIME.values, reftime) # in seconds since 2017-01-01 00:00:00 UTC

            # provide time as dataset coordinates:
            DS = DS.assign_coords({'tid': DS.TIME})

        if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
            self.DS = DS