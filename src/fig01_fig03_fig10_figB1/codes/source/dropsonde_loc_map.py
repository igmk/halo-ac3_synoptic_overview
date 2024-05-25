def run_dropsonde_loc_map(
    path_sic, 
    path_dropsondes_halo, 
    path_dropsondes_p5, 
    path_cartopy_background, 
    path_plots):

    """
    Visualize the dropsonde positions launched from HALO and P5 during HALO-AC3. Mean sea ice
    concentration will be shown in the background.

    Parameters:
    -----------
    path_sic : str
        Path of the sea ice concentration data.
    path_dropsondes_halo : str
        Path of the HALO dropsonde data.
    path_dropsondes_p5 : str
        Path of the P5 dropsonde data.
    path_cartopy_background : str
        Path of the cartopy background data.
    path_plots : str
        Path where plots are saved to.
    """

    import glob
    import gc
    import os
    import datetime as dt
    import sys
    import pdb
    import re

    import xarray as xr
    import numpy as np
    import matplotlib as mpl
    mpl.rcParams.update({"font.size":16})
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.patheffects as mpl_pe
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt

    from pyproj import Proj,CRS     # for projection/coordinate reference systems
    from halo_classes import P5_dropsondes, dropsondes

    # background image for cartopy:
    os.environ['CARTOPY_USER_BACKGROUNDS'] = path_cartopy_background


    def create_time_axis(DS):

        # Create time axis along new dimension:
        DS = DS.expand_dims(dim='time')

        # set the time based on the filename:
        filename = DS.encoding['source']
        file_date = re.search(r'\d{4}\d{2}\d{2}', filename).group()
        time = np.array([np.datetime64(f"{file_date[:4]}-{file_date[4:6]}-{file_date[6:8]}T12:00:00")])
        DS = DS.assign_coords({'time': time.astype('datetime64[ns]')})

        return DS


    # settings:
    set_dict = {'save_figures': True,
                'date_0': "2022-03-07",         # start date in yyyy-mm-dd
                'date_1': "2022-04-12",         # end date in yyyy-mm-dd
                'mean_sic': True,               # if True, mean SIC over the HALO-AC3 campaign is computed and plotted
                }


    # import dropsonde data for lat-lon positions:
    print("Importing dropsonde data....")
    DS_DS = dropsondes(path_dropsondes_halo, 'raw', return_DS=True)

    # import P5 dropsondes and get reference latitude and longitude (highest available lat/lon):
    P5_DS = P5_dropsondes(path_dropsondes_p5, 'raw', return_DS=True)


    # import SIC data and average over the campaign duration:
    print("Importing sea ice concentration data....")
    files = sorted(glob.glob(path_sic + "*.nc"))

    if set_dict['mean_sic']:
        DS = xr.open_mfdataset(files, concat_dim='time', combine='nested', preprocess=create_time_axis)
        DS = DS.load()
        DS['polar_stereographic'] = DS.polar_stereographic.isel(time=0)
        SIC_mean = np.flipud(DS.sic_merged.mean('time')).astype(np.float64)     # flipping is required; see Janna's script

    else:   # just one date is selected and plotted
        DS = xr.open_dataset(path_sic + "sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_20220307.nc")
        SIC_mean = np.flipud(DS.sic_merged).astype(np.float64)

    # regions where SIC_mean > 100 = land; therefore, remove:
    SIC_mean[SIC_mean > 100.0] = np.nan

    # get lat, lon coordinates from projection:
    x = DS.x
    y = DS.y
    projection = DS.polar_stereographic     # To convert from x and y to lat/lon we need the information about the grid contained in this variable
    m=Proj(projection.attrs["spatial_ref"]) # define a function for conversion that gets the information from the string contained in projection
    xx,yy = np.meshgrid(x.values,y.values)  # create a meshgrid from x and y to use in function m()
    lon, lat = m(xx,yy, inverse=True)

    del x, y, DS, m, projection, xx, yy


    # plot:
    reversed_map = mpl.cm.get_cmap('Blues_r')
    fs = 16
    fs_small = 14
    fs_dwarf = 12


    llcrnlat = 76       # lower left corner latitude
    llcrnlon = -5       # lower left corner longitude
    urcrnlat = 82       # upper right corner latitude
    urcrnlon = 15       # upper right corner longitude
    extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
    big_extent=[llcrnlon-10,urcrnlon+15,llcrnlat-11,90]

    # station labels and coords:
    coordinates= {'EDMO': (11.28, 48.08), 
                    "Kiruna": (20.336, 67.821),
                    "Longyearbyen": (15.46, 78.25),
                    "Meiningen": (10.38, 50.56),
                    "Lerwick": (-1.18, 60.13),
                    "Ittoqqortoormiit": (-21.95, 70.48),
                    "Tasiilaq": (-37.63, 65.60)}

    # radiosonde stations:
    rs_stations = {'Bj\u00F8rn\u00F8ya': (18.9986, 74.5040),
                    'Jan Mayen': (-8.7323, 70.9143),
                    'Danmarkshavn': (-18.6667, 76.7667),
                    'Ny-\u00C5lesund': (11.9222, 78.9250),
                    'And\u00F8ya': (16.01, 69.28)}


    # ocean basin labels and coords:
    ocean_basins = {'Central Arctic': (17.5, 86.0),
                    'Greenland Sea': (-8.0, 73.5),
                    'Barents Sea': (23.0, 72.5),
                    'Norwegian Sea': (5.0, 69.0)}


    x1, y1 = coordinates["Kiruna"]   
    x2, y2 = coordinates["Longyearbyen"]


    # Create a GeoAxes in the tile's projection:
    x=np.linspace(-90,90,41)
    y=np.linspace(55,90,93)
    x_grid,y_grid=np.meshgrid(x,y)
    white_overlay= np.zeros(x_grid.shape)
    plt.rcdefaults()


    f1, a1 = plt.subplots(1,1, figsize=(12,9), subplot_kw={"projection": ccrs.NorthPolarStereo(central_longitude=5)})

    a1.set_extent(big_extent, crs=ccrs.Geodetic())
    a1.background_img(name='NaturalEarthRelief', resolution='high')
    a1.coastlines(resolution="50m")
    a1.add_feature(cartopy.feature.BORDERS)
    a1.add_feature(cartopy.feature.OCEAN, color=(1,1,1), zorder=-1.0)

    gl = a1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                              x_inline=False, y_inline=False)


    # plot radiosonde stations:
    for rs_s in rs_stations.keys():
        a1.plot(rs_stations[rs_s][0], rs_stations[rs_s][1], linestyle='none', marker='^',
                markersize=10, color=(239/255,179/255,0), markeredgecolor=(0,0,0), mew=0.75,
                transform=ccrs.PlateCarree())

        # transformer to offset the text to the marker position:
        PC_mpl_tranformer = ccrs.PlateCarree()._as_mpl_transform(a1)
        if rs_s in ['Danmarkshavn', 'Jan Mayen']: 
            text_ha = 'center'
        else:
            text_ha = 'left'
        text_transform = mpl.transforms.offset_copy(PC_mpl_tranformer, units='dots',
                                                    x=0, y=30)

        a1.text(rs_stations[rs_s][0], rs_stations[rs_s][1], rs_s, fontsize=fs_dwarf,
                transform=text_transform, color=(239/255,179/255,0), ha=text_ha,
                path_effects=[mpl_pe.Stroke(linewidth=2.0, foreground=(0,0,0)), mpl_pe.Normal()])


    # Plot sea-ice concentration
    contourf_plot = a1.pcolormesh(lon, lat, SIC_mean[1:,1:], cmap=reversed_map, shading="flat",
                                    transform=ccrs.PlateCarree(), vmin=0, vmax=100, zorder=0)


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
        c_box = (0.871, 0.722, 0.530)   # colour of the box
        if b_key == 'central': c_box = (1.0, 0.549, 0.0)        # highlight central box
        for edge_lon, edge_lat in zip(boxes_lon[b_key], boxes_lat[b_key]):
            a1.plot(edge_lon, edge_lat, color=c_box, linewidth=4.0, transform=ccrs.PlateCarree())


    # station markers and text:
    a1.text(x2 + 2, y2 - 1.35, "Longyearbyen\n(LYR)", fontsize=fs_dwarf,
             transform=ccrs.PlateCarree(), color="red",
             bbox=dict(facecolor='lightgrey', edgecolor="black"))
    a1.text(x1 + 1.2, y1 - 0.2, "Kiruna (KRN)", fontsize=fs_dwarf,
             transform=ccrs.PlateCarree(), color="red",
             bbox=dict(facecolor='lightgrey', edgecolor="black"))
    a1.plot(x1, y1, '.r', markersize=15, markeredgecolor="k", transform=ccrs.PlateCarree())
    a1.plot(x2, y2, '.r', markersize=15, markeredgecolor="k", transform=ccrs.PlateCarree())


    # Plot dropsonde positions:
    a1.plot(DS_DS.DS.ref_lon, DS_DS.DS.ref_lat, linestyle='none', marker='o', color=(0.729, 0.333, 0.827), ms=5,
            label='HALO', transform=ccrs.PlateCarree(), markeredgecolor=(0,0,0), mew=0.75)
    a1.plot(P5_DS.DS.ref_lon, P5_DS.DS.ref_lat, linestyle='none', marker='o', color=(0.0, 0.467, 0.039, 0.6), ms=5,
            label='P5', transform=ccrs.PlateCarree(), markeredgecolor=(0,0,0), mew=0.75)


    # ocean basin labels:
    for o_l in ocean_basins.keys():
        a1.text(ocean_basins[o_l][0], ocean_basins[o_l][1], o_l, fontsize=fs_small, ha='center', va='center',
                color=(0.031,0.188,0.420), transform=ccrs.PlateCarree(), 
                path_effects=[mpl_pe.Stroke(linewidth=3.0, foreground=(1,1,1)), mpl_pe.Normal()])


    # legend:
    lh, ll = a1.get_legend_handles_labels()
    a1.legend(lh, ll, loc='upper right', fontsize=fs_small, framealpha=0.75)


    # gridline settings and labels:
    gl.bottom_labels = False
    gl.right_labels = False
    gl.rotate_labels = False
    gl.xlabel_style = {'rotation': 0, 'size': fs_small}
    gl.ylabel_style = {'rotation': 0, 'size': fs_small}
    gl.bottom_labels = False
    gl.right_labels = False

    if set_dict['save_figures']:
        plotname = "HALO-AC3_amsr2_modis_sic_dropsonde_loc_map"
        plotfile = path_plots + plotname + ".png"
        f1.savefig(plotfile, dpi=300, bbox_inches='tight')
        print(f"Saved {plotfile}....")
    else:
        plt.show()