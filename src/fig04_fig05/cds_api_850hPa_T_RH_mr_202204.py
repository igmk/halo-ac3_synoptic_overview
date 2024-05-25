import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'geopotential', 'relative_humidity', 'specific_cloud_ice_water_content',
            'specific_cloud_liquid_water_content', 'specific_humidity', 'specific_rain_water_content',
            'specific_snow_water_content', 'temperature',
        ],
        'pressure_level': '850',
        'year': '2022',
        'month': '04',
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '12:00',
        'area': [
            90, -80, 60,
            80,
        ],
        'format': 'netcdf',
    },
    'ERA5_850_T_h_RH_mixing_202204_60_90_-80_80.nc')