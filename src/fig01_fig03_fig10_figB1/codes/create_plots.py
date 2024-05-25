import os
import sys
import pdb

wdir = os.getcwd() + "/"


"""
    Script to manage and create plots for the synoptic overview paper. Execute this script
    in the following way:
    To create Figure 1: python3 create_plots.py "1"
    - Figure 3: python3 create_plots.py "3"
    - Figure 10: python3 create_plots.py "10"
    - Figure B1: python3 create_plots.py "B1"
"""

# Paths:
path_scripts = os.path.dirname(wdir) + "/source/"
path_tools = os.path.dirname(wdir) + "/tools/"
path_data_base = os.path.dirname(wdir[:-1]) + "/data/"
path_plots_base = os.path.dirname(wdir[:-1]) + "/plots/"

sys.path.insert(0, path_scripts)
sys.path.insert(0, path_tools)


# settings:
which_figure = sys.argv[1]  # set the number of the figure of the manuscript as string,
                            # e.g., "1" for Figure 1, "B1" for Figure B1 in Appendix B



# Fig. 1: Dropsonde :
if which_figure == "1":
    from dropsonde_loc_map import run_dropsonde_loc_map
    path_sic = path_data_base + "sea_ice_modis_amsr2/"
    path_dropsondes_halo = path_data_base + "HALO/dropsondes/"
    path_dropsondes_p5 = path_data_base + "P5/dropsondes/"
    path_cartopy_background = path_data_base + "cartopy_background/"
    path_plots = path_plots_base
    os.makedirs(os.path.dirname(path_plots), exist_ok=True) # create path if not existing

    run_dropsonde_loc_map(path_sic, path_dropsondes_halo, path_dropsondes_p5, path_cartopy_background, path_plots)


# Fig. 3: ERA5 dropsonde comparison:
if which_figure == "3":
    from ERA5_dropsonde_comparison import run_ERA5_dropsonde_comparison
    path_data = path_data_base + "ERA5_data/model_level/"
    path_sic = path_data_base + "ERA5_data/single_level/"
    path_dropsondes_halo = path_data_base + "HALO/dropsondes/"
    path_plots = path_plots_base
    os.makedirs(os.path.dirname(path_plots), exist_ok=True)

    run_ERA5_dropsonde_comparison(path_data, path_sic, path_dropsondes_halo, path_plots)


# Fig. 10: HAMP radar bright band:
if which_figure == "10":
    from HAMP_bright_band_sea_ice import run_HAMP_bright_band_sea_ice
    path_sic = path_data_base + "sea_ice_modis_amsr2/"
    path_hamp = path_data_base + "HALO/HAMP/"
    path_cartopy_background = path_data_base + "cartopy_background/"
    path_plots = path_plots_base
    os.makedirs(os.path.dirname(path_plots), exist_ok=True)

    run_HAMP_bright_band_sea_ice(path_sic, path_hamp, path_cartopy_background, path_plots)


# Fig. B1: Synoptic map:
if which_figure == "B1":
    from ERA5_synoptic_map_arctic import run_ERA5_synoptic_map_arctic
    path_data = {'single_level': path_data_base + "ERA5_data/single_level/",
                'multi_level': path_data_base + "ERA5_data/multi_level/",
				'cartopy': path_data_base + "cartopy_background/"}
    path_plots = path_plots_base
    os.makedirs(os.path.dirname(path_plots), exist_ok=True)

    run_ERA5_synoptic_map_arctic(path_data, path_plots)

