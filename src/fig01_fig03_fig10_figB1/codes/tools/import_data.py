import numpy as np
import netCDF4 as nc
import datetime as dt
# import pandas as pd
# import xarray as xr
import copy
import pdb
import os
import glob
import sys
import warnings
import csv
from met_tools import convert_rh_to_spechum, compute_IWV_q


def import_single_NYA_RS_radiosonde(
    filename,
    keys='all',
    height_grid=np.array([]),
    verbose=0):

    """
    Imports single NYA-RS radiosonde data for Ny Alesund. Converts to SI units
    and interpolates to a height grid with 5 m resolution from 0 to 15000 m. 

    Parameters:
    -----------
    filename : str
        Name (including path) of radiosonde data file.
    keys : list of str or str, optional
        This describes which variable(s) will be loaded. Specifying 'all' will import all variables.
        Specifying 'basic' will load the variables the author consideres most useful for his current
        analysis.
        Default: 'all'
    height_grid : array of floats or None
        If not None, height_grid contains a 1D array of floats indicating the new height grid to 
        which the radiosonde data is interpolated to.
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    """
        Loaded values are imported in the following units:
        T: in K
        P: in hPa, will be converted to Pa
        RH: in [0-1]
        Altitude: in m
        time: will be converted to sec since 1970-01-01 00:00:00 UTC
    """

    file_nc = nc.Dataset(filename)

    if (not isinstance(keys, str)) and (not isinstance(keys, list)):
        raise TypeError("Argument 'key' must be a list of strings or 'all'.")

    if keys == 'all':
        keys = file_nc.variables.keys()
    elif keys == 'basic':
        keys = ['time', 'temp', 'press', 'rh', 'alt']

    sonde_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in radiosonde file." % key)

        sonde_dict[key] = np.asarray(file_nc.variables[key])
        if key != "IWV" and len(sonde_dict[key]) == 0: # 'and': second condition only evaluated if first condition True
            return None

        if key in ['lat', 'lon']:   # only interested in the first lat, lon position
            sonde_dict[key] = sonde_dict[key][0]

    # convert units:
    if 'P' in keys:     # from hPa to Pa
        sonde_dict['P'] = sonde_dict['P']*100
    if 'time' in keys:  # from int64 to float64
        time_unit = file_nc.variables['time'].units
        time_offset = (dt.datetime.strptime(time_unit[-19:], "%Y-%m-%dT%H:%M:%S") - dt.datetime(1970,1,1)).total_seconds()
        sonde_dict['time'] = np.float64(sonde_dict['time']) + time_offset
        sonde_dict['launch_time'] = sonde_dict['time'][0]

    # interpolate to new height grid:
    if len(height_grid) == 0:
        height_grid = np.arange(0,15001,5)      # in m
    keys = [*keys]      # converts dict_keys to a list
    for key in keys:
        if sonde_dict[key].shape == sonde_dict['time'].shape:
            if key not in ['time', 'lat', 'lon', 'alt']:
                sonde_dict[key + "_ip"] = np.interp(height_grid, sonde_dict['alt'], sonde_dict[key])
            elif key == 'alt':
                sonde_dict[key + "_ip"] = height_grid


    # Renaming variables to a standard convention
    renaming = {'press': 'pres', 'alt': 'height', 'press_ip': 'pres_ip', 'alt_ip': 'height_ip'}
    for ren_key in renaming.keys():
        if ren_key in sonde_dict.keys():
            sonde_dict[renaming[ren_key]] = sonde_dict[ren_key]

    return sonde_dict


def import_radiosonde_daterange(
    path_data,
    date_start,
    date_end,
    s_version='level_2',
    with_wind=False,
    remove_failed=False,
    extend_height_grid=False,
    height_grid=np.array([]),
    ip_type='lin',
    verbose=0):

    """
    Imports radiosonde data (several versions supported, see below) and concatenates the files 
    into time series x height. E.g. temperature profile will have the dimension: n_sondes x n_height

    Parameters:
    -----------
    path_data : str
        Path of radiosonde data.
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    s_version : str, optional
        Specifies the radiosonde version that is to be imported. Possible options: 'mossonde',
        'psYYMMDDwHH', 'level_2', 'nya-rs', 'GRUAN', 'level_3'. Default: 'level_2' (published by 
        Marion Maturilli)
    with_wind : bool, optional
        This describes if wind measurements are included (True) or not (False). Does not work with
        s_version='psYYMMDDwHH'. Default: False
    remove_failed : bool, optional
        If True, failed sondes with unrealistic IWV values will be removed (currently only implmented
        for s_version in ['level_2', 'GRUAN', 'level_3']). It also includes "height_check" to avoid 
        sondes that burst before reaching > 10000 m.
    extend_height_grid : bool
        If True, the new height grid, to which the radiosonde data is interpolated to is 0, 10, 20, ...
        25000 m. If False, it's 0, 5, 10, 15, ..., 15000 m.
    height_grid : array of floats or None
        If not None, height_grid contains a 1D array of floats indicating the new height grid to 
        which the radiosonde data is interpolated to.
    ip_type : str
        String indicating the interpolation type. Option: 'lin' for linear interpolation using 
        np.interp; 'avg' for using interp_w_avg, which averages data over layers centered on the
        height_grid. 'avg' is only available for s_version in ['level_2', 'level_3', 'GRUAN'].
    verbose : int, optional
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    def time_prematurely_bursted_sondes():

        """
        This little function merely returns time stamps of MOSAiC radiosondes, whose
        burst altitude was <= 10000 m. (Or other errors occurred.)
        """

        failed_sondes_dt = np.array([dt.datetime(2019, 10, 7, 11, 0),
                            dt.datetime(2019, 10, 15, 23, 0),
                            dt.datetime(2019, 11, 4, 11, 0),
                            dt.datetime(2019, 11, 17, 17, 0),
                            dt.datetime(2019, 12, 17, 5, 0),
                            dt.datetime(2019, 12, 24, 11, 0),
                            dt.datetime(2020, 1, 13, 11, 0),
                            dt.datetime(2020, 2, 1, 11, 0),
                            dt.datetime(2020, 2, 6, 23, 0),
                            dt.datetime(2020, 3, 9, 23, 0),
                            dt.datetime(2020, 3, 9, 11, 0), # unrealistic temperature and humidity values at the surface
                            dt.datetime(2020, 3, 11, 17, 0),
                            dt.datetime(2020, 3, 29, 5, 0),
                            dt.datetime(2020, 5, 14, 17, 0),
                            dt.datetime(2020, 6, 14, 17, 0),
                            dt.datetime(2020, 6, 19, 11, 0),
                            dt.datetime(2020, 9, 27, 9, 0)])

        reftime = dt.datetime(1970,1,1)
        failed_sondes_t = np.asarray([datetime_to_epochtime(fst) for fst in failed_sondes_dt])
        failed_sondes_t = np.asarray([(fst - reftime).total_seconds() for fst in failed_sondes_dt])
        
        return failed_sondes_t, failed_sondes_dt

    if not isinstance(s_version, str): raise TypeError("s_version in import_radiosonde_daterange must be a string.")

    # extract day, month and year from start date:
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    if s_version == 'nya-rs':
        all_radiosondes_nc = sorted(glob.glob(path_data + "NYA-RS_*.nc"))

        # inquire the number of radiosonde files (date and time of launch is in filename):
        # And fill a list which will include the relevant radiosonde files.
        radiosondes_nc = []
        for rs_nc in all_radiosondes_nc:
            rs_date = rs_nc[-15:-3]     # date of radiosonde from filename
            yyyy = int(rs_date[:4])
            mm = int(rs_date[4:6])
            dd = int(rs_date[6:8])
            rs_date_dt = dt.datetime(yyyy,mm,dd)
            if rs_date_dt >= date_start and rs_date_dt <= date_end:
                radiosondes_nc.append(rs_nc)


    # number of sondes:
    n_sondes = len(radiosondes_nc)

    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days

    # basic variables that should always be imported:
	if s_version == 'nya-rs':
        geoinfo_keys = ['lat', 'lon', 'launch_time']
        time_height_keys = ['pres', 'temp', 'rh', 'height']
        if with_wind: time_height_keys = time_height_keys + ['wspeed', 'wdir']
    else:
        raise ValueError("s_version in import_radiosonde_daterange must be 'nya-rs'.")
    all_keys = geoinfo_keys + time_height_keys


    # sonde_master_dict (output) will contain all desired variables on specific axes:
    # Time axis (one sonde = 1 timestamp) = axis 0; height axis = axis 1
    if len(height_grid) == 0:
        if extend_height_grid:
            new_height_grid = np.arange(0,25001,10)
        else:
            new_height_grid = np.arange(0,15001,5)
    else:
        new_height_grid = height_grid
    n_height = len(new_height_grid) # length of the interpolated height grid
    sonde_master_dict = dict()
    for gk in geoinfo_keys: sonde_master_dict[gk] = np.full((n_sondes,), np.nan)
    for thk in time_height_keys: sonde_master_dict[thk] = np.full((n_sondes, n_height), np.nan)

    if s_version == 'nya-rs':
        all_keys_import = ['lat', 'lon', 'press', 'temp', 'rh', 'alt', 'time']
        if with_wind: all_keys_import = all_keys_import + ['wdir', 'wspeed']


        # cycle through all relevant sonde files:
        for rs_idx, rs_nc in enumerate(radiosondes_nc):
            
            if verbose >= 1:
                # rs_date = rs_nc[-19:-3]
                print("\rWorking on Radiosonde, " + rs_nc, end="")

            sonde_dict = import_single_NYA_RS_radiosonde(rs_nc, keys=all_keys_import)
            
            # save to sonde_master_dict:
            for key in all_keys:
                if key in geoinfo_keys:
                    sonde_master_dict[key][rs_idx] = sonde_dict[key]

                elif key in time_height_keys:
                    sonde_master_dict[key][rs_idx, :] = sonde_dict[key + "_ip"]     # must use the interpolated versions!

                else:
                    raise KeyError("Key '" + key + "' not found in radiosonde dictionary after importing it with " +
                                    "import_single_NYA_RS_radiosonde")


    if verbose >= 1: print("")

    return sonde_master_dict


def import_ny_alesund_radiosondes_pangaea_tab(
    files):

    """
    Imports radiosonde data from Ny-Alesund published to PANGAEA, i.e., 
    https://doi.org/10.1594/PANGAEA.845373 , https://doi.org/10.1594/PANGAEA.875196 , 
    https://doi.org/10.1594/PANGAEA.914973 . The Integrated Water Vapour will be computed using 
    the saturation water vapour pressure according to Hyland and Wexler 1983. Measurements will be
    given in SI units.
    The radiosonde data will be stored in a dict with keys being the sonde index and
    the values are 1D arrays with shape (n_data_per_sonde,). Since we have more than one sonde per
    .tab file, single sondes must be identified via time difference (i.e., 900 seconds) or the
    provided sonde ID (latter is only available for sondes before 2017).

    Parameters:
    -----------
    files : str
        List of filename + path of the Ny-Alesund radiosonde data (.tab) published on PANGAEA.
    """


    # Ny-Alesund radiosonde .tab files are often composits of multiple radiosondes (i.e., one month
    # or a year). Therefore, first, just load the data and sort out single radiosondes later:
    n_data_per_file = 1000000       # assumed max. length of a file for data array initialisation

    # loop through files and load the data into a temporary dictionary:
    data_dict = dict()
    for kk, file in enumerate(files):

        f_handler = open(file, 'r')

        # # automatised inquiry of file length (is slower than just assuming a max number of lines):
        # n_data_per_file = len(f_handler.readlines())
        # f_handler.seek(0) # set position of pointer back to beginning of file

        translator_dict = {'Date/Time': "time",
                            'Altitude [m]': "height",
                            'PPPP [hPa]': "pres",
                            'TTT [°C]': "temp",
                            'RH [%]': "relhum",
                            'ff [m/s]': "wspeed",
                            'dd [deg]': "wdir"}     # translates naming from .tab files to the convention used here

        print(f"\rImporting {file}", end='')
        str_kk = str(kk)    # string index of file
        data_dict[str_kk] = {'time': np.full((n_data_per_file,), np.nan),       # in sec since 1970-01-01 00:00:00 UTC or numpy datetime64
                            'height': np.full((n_data_per_file,), np.nan),      # in m
                            'pres': np.full((n_data_per_file,), np.nan),        # in Pa
                            'temp': np.full((n_data_per_file,), np.nan),        # in K
                            'relhum': np.full((n_data_per_file,), np.nan),      # in [0,1]
                            'wspeed': np.full((n_data_per_file,), np.nan),      # in m s^-1
                            'wdir': np.full((n_data_per_file,), np.nan)}        # in deg
        if "NYA_UAS_" in file:
            translator_dict['ID'] = "ID"
            data_dict[str_kk]['ID'] = np.full((n_data_per_file,), 20*" ")


        mm = 0      # runs though all data points of one radiosonde and is reset to 0 for each new radiosonde
        data_line_indicator = -1        # if this is no longer -1, then the line where data begins has been identified
        for k, line in enumerate(f_handler):

            if data_line_indicator == -1:
                data_line_indicator = line.find("*/")       # string indicating the beginning of data

            else:   # data begins:
                current_line = line.strip().split("\t")     # split by tabs

                if 'Date/Time' in current_line: 
                    data_descr = current_line   # list with data description

                    # identify which column of a line represents which data type:
                    data_col_id = dict()
                    for data_key in translator_dict.keys():
                        data_col_id[translator_dict[data_key]] = data_descr.index(data_key)

                else:

                    # extract data:
                    for data_key in translator_dict.values():

                        try:
                            if data_key == 'time': 
                                data_dict[str_kk][data_key][mm] = np.datetime64(current_line[data_col_id[data_key]])
                            elif data_key == 'pres': 
                                data_dict[str_kk][data_key][mm] = float(current_line[data_col_id[data_key]])*100.0
                            elif data_key == 'temp':
                                data_dict[str_kk][data_key][mm] = float(current_line[data_col_id[data_key]]) + 273.15
                            elif data_key == 'relhum':
                                data_dict[str_kk][data_key][mm] = float(current_line[data_col_id[data_key]])*0.01
                            elif data_key == 'ID':
                                data_dict[str_kk][data_key][mm] = current_line[data_col_id[data_key]]

                            else:
                                data_dict[str_kk][data_key][mm] = float(current_line[data_col_id[data_key]])

                        except IndexError:      # wind direction or wind speed data missing:
                            data_dict[str_kk]['wspeed'][mm] = float('nan')
                            data_dict[str_kk]['wdir'][mm] = float('nan')


                        except ValueError:      # then at least one measurement is missing:
                            current_line[current_line.index('')] = 'nan'        # 'repair' the data for import

                            if data_key == 'time': 
                                data_dict[str_kk][data_key][mm] = np.datetime64(current_line[data_col_id[data_key]])
                            elif data_key == 'pres': 
                                data_dict[str_kk][data_key][mm] = float(current_line[data_col_id[data_key]])*100.0
                            elif data_key == 'temp':
                                data_dict[str_kk][data_key][mm] = float(current_line[data_col_id[data_key]]) + 273.15
                            elif data_key == 'relhum':
                                data_dict[str_kk][data_key][mm] = float(current_line[data_col_id[data_key]])*0.01
                            elif data_key == 'ID':
                                data_dict[str_kk][data_key][mm] = current_line[data_col_id[data_key]]

                            else:
                                data_dict[str_kk][data_key][mm] = float(current_line[data_col_id[data_key]])

                    mm += 1

        # truncate data_dict of current file:
        for key in data_dict[str_kk].keys():
            data_dict[str_kk][key] = data_dict[str_kk][key][:mm]

    print("")

    # concatenate all data_dict:
    data_dict_all = dict()
    for key in translator_dict.values():
        for k, str_kk in enumerate(data_dict.keys()):
            if k == 0:
                data_dict_all[key] = data_dict[str_kk][key]

                # also add ID if available:
                if 'ID' in data_dict[str_kk].keys():
                    data_dict_all['ID'] = data_dict[str_kk]['ID']
            else:
                data_dict_all[key] = np.concatenate((data_dict_all[key], data_dict[str_kk][key]), axis=0)
                if 'ID' in data_dict[str_kk].keys():
                    data_dict_all['ID'] = np.concatenate((data_dict_all['ID'], data_dict[str_kk]['ID']), axis=0)


    # clear memory:
    del data_dict


    # set faulty data to nan:
    idx_broken = np.where(data_dict_all['relhum'] > 2.00)[0]
    data_dict_all['relhum'][idx_broken] = np.nan


    # identify single radiosondes based on time difference or sonde ID:
    new_sonde_idx = []
    for k in range(len(data_dict_all['ID'])-1):
        if data_dict_all['ID'][k+1] != data_dict_all['ID'][k]:
            new_sonde_idx.append(k)
    new_sonde_idx = np.asarray(new_sonde_idx)
    len(new_sonde_idx)

    # identify the remaining radiosondes via time stamp differences and concatenate both identifier arrays:
    new_sonde_idx_time = np.where(np.abs(np.diff(data_dict_all['time'][new_sonde_idx[-1]+1:])) > 900.0)[0] + new_sonde_idx[-1]+1
                                                                            # indicates the last index belonging to the current sonde
                                                                            # i.e.: np.array([399,799,1194]) => 1st sonde: [0:400]
                                                                            # 2nd sonde: [400:800], 3rd sonde: [800:1195], 4th: [1195:]
                                                                            # ALSO negative time diffs must be respected because it
                                                                            # happens that one sonde may have been launched before the
                                                                            # previous one burst
    new_sonde_idx = np.concatenate((new_sonde_idx, new_sonde_idx_time), axis=0)
    # new_sonde_idx = np.where(np.abs(np.diff(data_dict_all['time'])) > 900.0)[0]   # this line would be for pure time-based sonde detect.

    n_sondes = len(new_sonde_idx) + 1
    rs_dict = dict()

    # loop over new_sonde_idx to identify single radiosondes and save their data:
    for k, nsi in enumerate(new_sonde_idx):
        k_str = str(k)

        # initialise rs_dict for each radiosonde:
        rs_dict[k_str] = dict()

        if (k > 0) & (k < n_sondes-1):
            for key in translator_dict.values():
                rs_dict[k_str][key] = data_dict_all[key][new_sonde_idx[k-1]+1:nsi+1]
        elif k == 0:
            for key in translator_dict.values():
                rs_dict[k_str][key] = data_dict_all[key][:nsi+1]

    # last sonde must be treated separately:
    rs_dict[str(n_sondes-1)] = dict()
    for key in translator_dict.values():
        rs_dict[str(n_sondes-1)][key] = data_dict_all[key][new_sonde_idx[k]+1:]


    # clear memory:
    del data_dict_all


    # finally, compute specific humidity and IWV, looping over all sondes:
    time_nya_uas_limit = 1491044046.0
    for s_idx in rs_dict.keys():

        # limit profiles of the NYA_UAS radiosondes to 10 km height:
        if rs_dict[s_idx]['time'][-1] < time_nya_uas_limit:
            idx_hgt = np.where(rs_dict[s_idx]['height'] <= 10000.0)[0]
            for key in rs_dict[s_idx].keys():
                rs_dict[s_idx][key] = rs_dict[s_idx][key][idx_hgt]

        # compute specific humidity and IWV:
        rs_dict[s_idx]['q'] = convert_rh_to_spechum(rs_dict[s_idx]['temp'], rs_dict[s_idx]['pres'], 
                                                    rs_dict[s_idx]['relhum'])
        rs_dict[s_idx]['IWV'] = compute_IWV_q(rs_dict[s_idx]['q'], rs_dict[s_idx]['pres'], nan_threshold=0.5, scheme='balanced')

    return rs_dict

