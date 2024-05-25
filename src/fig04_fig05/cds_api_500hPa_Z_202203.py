import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': 'geopotential',
        'pressure_level': '500',
        'year': '2022',
        'month': '03',
        'day': [
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': '12:00',
        'area': [
            90, -80, 60,
            80,
        ],
        'format': 'netcdf',
    },
    'ERA5_500_h_202203_60_90_-80_80.nc')