import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
            'mean_sea_level_pressure', 'total_column_water_vapour',
        ],
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
    'ERA5_surf_T_p_uv_IWV_202204_60_90_-80_80.nc')