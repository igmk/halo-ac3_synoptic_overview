These sections summarize the required steps to reproduce the figures presented in the publication. The figures are numbered according to the publication. The code to reproduce the figures can be found in the "src" folder. 


# Figure 01
Credit: Andreas Walbröl

Figure 01 shows the study area and mean sea ice concentration as well as the locations of the dropsondes launched by HALO and P5 during the (AC)3 campaign.

The code to reproduce Figure 01 (as well as Figure 03, 10 and B1) can be found [here](../src/fig01_fig03_fig10_figB1/).

## Requirements
A list of required python packages to reproduce Figure 01 (as well as Figure 03, 10 and B1) can be found in the file [readme.txt](../src/fig01_fig03_fig10_figB1/readme.txt).

## Data
The steps to download the data needed to reproduce Figure 01 (as well as Figure 03, 10 and B1) can be found in the file [readme.txt](../src/fig01_fig03_fig10_figB1/readme.txt). The code to download parts of the data is provided in this [folder](../src/fig01_fig03_fig10_figB1/codes/source).

## Steps to reproduce figure 
To reproduce the plot, run the script [create_plots.py](../src/fig01_fig03_fig10_figB1/codes/create_plots.py) as specified in the header of the script.



# Figure 02
Credit: Nils Slättberg

Figure 02 shows a time series of Ny-Ålesund radiosondes of temperature profiles (with tropopause height and wind), specific humidity profiles (with integrated water vapor) and precipitation.

The code to reproduce Figure 02 can be found [here](../src/fig02/HALOAC3_radiosondes.m).

## Requirements
The script runs with MATLAB R2022a.

## Data
The data needed to reproduce Figure 02 is specified in the header of the MATLAB script [HALOAC3_radiosondes.m](../src/fig02/HALOAC3_radiosondes.m).

## Steps to reproduce figure
After downloading the required data, it should be organized as specified in the header of the MATLAB script [HALOAC3_radiosondes.m](../src/fig02/HALOAC3_radiosondes.m). Then, run the script to reproduce the plot of Figure 02.


# Figure 03
Credit: Andreas Walbröl

Figure 03 shows ERA5 errors of temperature, relative humidity, wind speed and pressure using dropsonde observations as reference. 

The code to reproduce Figure 03 (as well as Figure 01, 10 and B1) can be found [here](../src/fig01_fig03_fig10_figB1/).

The description of requirements, data and steps to reproduce the figure can be found in the section for Figure 01.


# Figure 04 and 05
Credit: Sebastian Becker

Figure 04 shows an overview over the time series of synoptic variables from ERA5 during the HALO-(AC)3 flight campaign.
Figure 05 shows ERA5 maps for six exemplary days of the HALO-(AC)3 flight campaign.

The code to reproduce Figure 04 and Figure 05 can be found [here](../src/fig04_fig05/).

## Requirements
- numpy
- pandas
- xarray
- pyhdf
- matplotlib
- mpl_toolkits

## Data
The sea ice concentration data to reproduce Figure 05 can be downloaded from [here](https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/n3125/2022/mar/Arctic3125/) for March and [here](https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/n3125/2022/apr/Arctic3125/) for April 2022. The respective latitude and longitude grid can be downloaded [here](https://data.seaice.uni-bremen.de/grid_coordinates/n3125/LongitudeLatitudeGrid-n3125-Arctic3125.hdf). In the [script directory](../src/fig04_fig05/), create a folder named "Satellite_Icefraction" (without quotation marks) and move the sea ice and grid data to that created folder. The ERA5 data needed for Figure 04 and Figure 05 can be downloaded by running [download_ERA5_data.sh](../src/fig04_fig05/download_ERA5_data.sh).

## Steps to reproduce figure
To reproduce the figures, run [plot_for_paper_new.py](../src/fig04_fig05/plot_for_paper_new.py). Note that all paths to the data files need to be adjusted in the script.



# Figure 06
Credit: Benjamin Kirbus

Figure 06 shows daily means of (a) the northwards component of integrated water vapour transport, (b) Marine Cold Air Outbreak (MCAO) index, comparing the year 2022 to the climatology 1979-2022. The plot is based on ERA5 reanalysis data.

The code to reproduce Figure 06 can be found [here](../src/fig06/).

## Requirements
- bash
- cdo
- python

Python packages:
- cdsapi (see https://cds.climate.copernicus.eu/api-how-to )
- os
- numpy
- datetime
- matplotlib
- locale
- xarray

## Data
The data needed is automatically downloaded by the bash scripts specified below.

## Steps to reproduce figure
To reproduce the plot of Figure 06, run the scripts in the following order:

1. step1_download_all.sh 
> initiates the automatic download of all required ERA5 datasets from the Copernicus climate data storage (CDS) api. 
> It requires the 6 attached python scripts cds_api_*.py to be in the same folder. 
> Output: 6 downloaded grib files.

2. step2_convert_grib.sh 
> This converts the downloaded grib files to netCDF.
> Output: 6 converted netCDF files.

3. step3_calculations.sh 
> Selects only the Fram Strait box, calculates daily means of the data, and saves that as new netCDF files. 
> Output: 4 netCDF files, which only contain data for the Fram Strait box, and daily means of a) 2022 vs. climatology, for  b) IVT_north and the MCAO index (2x2 combinations).

4. Py_plot_MCAO_IVTN.py 
> plots the 4 final netCDF files
> It requires the above netCDF files to be in the same folder
> Output: HALO_AC3_WAI_CAO.png and HALO_AC3_WAI_CAO.pdf 




# Figure 07
Credit: Hanno Müller

Figure 07 shows anomaly maps of mean sea level pressure, 2m temperature, 850hPa temperature and integrated water vapour based on ERA5 for the entire HALO–(AC)3 campaign, the warm and the cold period.

The code to reproduce Figure 07 can be found [here](src/fig07/).

## Requirements
The following list shows the needed python packages to run the code:
- xarray (0.21.1)
- numpy (1.22.1)
- matplotlib (3.5.2)
- cartopy (0.20.0)

## Data
- Download the variables “Mean sea level pressure”, “2m temperature” and “Total column water vapour” from 1979 to 2022 for all days from 01 March to 30 April from the dataset “ERA5 hourly data on single levels from 1940 to present” in the Copernicus Climate Data Store in NetCDF and rename the downloads to “era5_psurf.nc”, “era5_t2m.nc” and “era5_tcwv.nc”.
- Download the variable “Temperature“ from 1979 to 2022 for all days from 01 March to 30 April for the pressure level 850hPa from the dataset “ERA5 hourly data on pressure levels from 1940 to present” in the Copernicus Climate Data Store in NetCDF and rename the download file to “era5_t850hPa.nc”.


## Steps to reproduce figure
Run the python script [era5_assessment_halo_ac3.py](../src/fig07/era5_assessment_halo_ac3.py) after downloading the needed data to reproduce Figure 07.



# Figure 08
Credit: Henning Dorff

Figure 08 shows the climatological (1979–2022) distribution of central latitudes of Atmospheric Rivers (ARs) as a function of mean AR integrated water vapour transport as well as cases categorized as ARs during the HALO-(AC)3 campaign.

The code to reproduce Figure 09 can be found [here](src/fig08/).

## Requirements
The following list shows the needed python packages to run the code:
- numpy
- pandas
- xarray

## Data
AR climatology relies on AR catalogue by M. Lauer (IGMK, Cologne) that adapted the AR detection by Guan & Waliser (2015) for ERA5 input.

Data of the instruments aboard the airplanes are not publically available yet. Please contact the authors for further information.

## Steps to reproduce figure
To reproduce the plot of Figure 08, run the script [Synoptic_AR_IVT_climatological_comparison.ipynb](../src/fig08/Synoptic_AR_IVT_climatological_comparison.ipynb). Note that paths need to be adjusted in the script.



# Figure 09
Credit: Melanie Lauer

Figure 09 shows hourly averaged precipitation, snowfall and rain rate derived from ERA5 for 11–20 March 2022, and the corresponding climatology.

The code to reproduce Figure 09 can be found [here](src/fig09). 

## Requirements
The following list shows the needed python packages to run the code:
- cdsapi
- numpy
- netCDF4
- matplotlib
- mpl_toolkits


## Data
Data is downloaded from ERA5. Use  The needed variables are snowfall (sf), total precipitation (tp), and sea ice concentration (siconc).
The data can be downloaded by the two python scripts specified below. Save the data to the same directory as the scripts.
Make sure to have cdsapi installed: https://cds.climate.copernicus.eu/api-how-to.


## Steps to reproduce figure
To reproduce the plot of Figure 09, run the scripts in the following order:

1. Download data: 
> data_HALO.py: script to download variables from 12 March - 20 March for 1979 - 2022
> data_HALO_11.py: script to download variables for 11 March for 1979 - 2022

2. data_tr_sf_global_11.py
> calculates tr (tp - sf)
> computes hourly averaged precipitation rate (sum up hourly data and divide by the number of hours) for:
		the campaign (_fraction_campaign)
		the climatology (_fraction_full)
		the anomaly (_anomaly)
		the deviation (_deviation)
> writes data in netCDF file

3. tr_sf_global_11.py 
> produces the plot of Figure 09 using nbtools.py

Note that paths need to be adjusted in the scripts.


# Figure 10
Credit: Andreas Walbröl

Figure 10 shows the linear depolarization ratio measured by HALO’s radar during research flight 3 as well as the corresponding HALO flight track and the sea ice concentration.

The code to reproduce Figure 10 (as well as Figure 01, 03 and B1) can be found [here](../src/fig01_fig03_fig10_figB1/).

The description of requirements, data and steps to reproduce the figure can be found in the section for Figure 01.


# Figure 11
Credit: Henning Dorff

Figure 11 shows ERA5-based duration and strength of MCAOs in the central domain for the period 1979–2022 as well as the MCAO cases categorized during HALO-(AC)3.

The code to reproduce Figure 11 can be found [here](src/fig11/).

## Requirements
The following list shows the needed python packages to run the code:
- numpy
- pandas
- xarray
- matplotlib
- seaborn

## Data
CAO climatology relies on catalogue by Bernd Heinold (Tropos, Leipzig) using ERA5 input.

## Steps to reproduce figure
To reproduce the plot of Figure 11, run the script [Synoptic_Paper_CAO_climate_framing.ipynb](../src/fig11/Synoptic_Paper_CAO_climate_framing.ipynb).


# Figure 12
Credit: Janna Rückert

Figure 12 shows average sea ice concentration for two three–day time periods during the HALO–(AC)3 campaign. Data is from the MODIS-AMSR2 product at 1km grid resolution.
It further shows sea ice concentration time series and climatological data from the OSI–SAF sea ice concentration climate data record.

The code to reproduce Figure 12 can be found [here](src/fig12/).

## Requirements
An environment file containing all needed python packages is provided: [environment.yml](../src/fig12/environment.yml)

## Data
OSI-SAF Sea Ice Concentration Climate Data Record can be downloaded from EUMETSAT (https://eoportal.eumetsat.int) and the AMSR2-MODIS Sea ice concentration data is available at https://seaice.uni-bremen.de/sea-ice-concentration/modis-amsr2/. In [HALO_AC3_SeaIce.py](../src/fig12/src/HALO_AC3_SeaIce.py) the path to the datasets is currently given to an internal raid at Uni Bremen, this needs to be adapted in this file if one wants to use this code to reproduce the figures.

## Steps to reproduce figure
Figure 12 is generated using the jupyter notebook [HALO_seaice_results-github.ipynb](../src/fig12/HALO_seaice_results-github.ipynb).
All functions needed for the jupyter notebook can be found in [HALO_AC3_SeaIce.py](../src/fig12/src/HALO_AC3_SeaIce.py).


# Figure A1
Credit: Henning Dorff

Figure A1 shows Integrated Water Vapour Transport (IVT) from ERA5 for Moist Air Intrusions/ Atmospheric Rivers during the HALO-(AC)3 campaign.

The code to reproduce Figure A1 can be found [here](src/figA1/).


# Figure B1
Credit: Andreas Walbröl

Figure B1 shows maps of mean sea level pressure and 500hPa geopotential height from ERA5 data averaged over two different periods of time (the warm period and the cold period).

The code to reproduce Figure B1 (as well as Figure 01, 03 and 10) can be found [here](../src/fig01_fig03_fig10_figB1/).

The description of requirements, data and steps to reproduce the figure can be found in the section for Figure 01.

