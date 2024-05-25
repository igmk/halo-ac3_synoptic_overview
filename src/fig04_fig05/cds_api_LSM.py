import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': 'land_sea_mask',
        'year': '2022',
        'month': '04',
        'day': '01',
        'time': '12:00',
        'area': [
            90, -80, 60,
            80,
        ],
        'format': 'netcdf',
    },
    'ERA5_Land-Sea-Mask.nc')