Data requirements:
------------------

- (Dropsonde data will soon be published)

- Get ERA5 data on model levels: 
    Execute .../codes/source/era5_model_levels_request.py and move the output file 
    ERA5_model_level_HALO-AC3.nc to .../data/ERA5_data/model_level_raw/. Afterwards, execute 
    .../codes/source/ERA5_process_model_levels.py and then ERA5_dropsonde_collocation.py.

- Get ERA5 sea ice concentration data:
    Execute .../codes/source/get_era5_halo_ac3_sic.py and move the output file to 
    .../data/ERA5_data/single_level/.

- Get ERA5 single level mean sea level pressure data:
    Execute .../codes/source/get_era5_halo_ac3_mslp.py and move the output file to
    .../data/ERA5_data/single_level/.

- Get ERA5 500 hPa geopotential data:
    Execute .../codes/source/get_era5_halo_ac3_gp.py and move the output file to
    .../data/ERA5_data/multi_level/.

- HALO HAMP radar observations:
    Download HALO_HALO_AC3_radar_unified_RF03_20220313_v2.6.nc from https://doi.org/10.1594/PANGAEA.963250
    and move it to .../data/HALO/HAMP/.

- Sea ice concentration data from MODIS-AMSR2:
    Download data from https://data.seaice.uni-bremen.de/modis_amsr2/netcdf/Arctic/2022/ 
    (7 March - 12 April 2022) and move to .../data/sea_ice_modis_amsr2/.

- cartopy background data: Following https://docs.dkrz.de/doc/visualization/sw/python/source_code/python-matplotlib-example-high-resolution-background-image-plot.html
    Download high resolution version ("Download large size") from 
    https://www.naturalearthdata.com/downloads/10m-natural-earth-1/10m-natural-earth-1-with-shaded-relief-water-and-drainages/ ,
    unpack NE1_HR_LC_SR_W_DR.tif and convert to png. Save

{"__comment__": "JSON file specifying background images. env CARTOPY_USER_BACKGROUNDS, ax.background_img()",
  "NaturalEarthRelief": {
    "__comment__": "Natural Earth I with shaded Relief, water, and drainage",
    "__source__": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/raster/NE1_HR_LC_SR_W_DR.zip",
    "__projection__": "PlateCarree",
    "high": "NE1_HR_LC_SR_W_DR.png"
  }
}
    to images.json. Move NE1_HR_LC_SR_W_DR.png and images.json to .../data/cartopy_background/. 


---------------------------------------------------------------------------------------------------


Python packages:
----------------
For all figures created with 'create_plots.py':
- python version: 3.9.16 | packaged by conda-forge | (main, Feb  1 2023, 21:39:03) [GCC 11.3.0]
- numpy: 1.26.4
- netCDF4: 1.6.5
- matplotlib: 3.8.2
- xarray: 2024.1.1
- pandas: 2.2.0
- cartopy: 0.22.0
- cmcrameri: 1.8
- pyproj: 3.6.1
- cdsapi: version unknown
