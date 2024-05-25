#!/bin/bash

# Script to process the converted netCDF files
# processing involves the selection of the "Fram Strait" box
# and calculation of daily means and the MCAO index, all using cdo 

#-------------------------------
#Set parameters for CDO
#-------------------------------

# filenames
IVTN_clim="clim_IVTN_2D"
IVTN_2022="2022_IVTN_2D"
T850_clim="clim_T850"
T850_2022="2022_T850"
sfc_clim="clim_SKT_PS_2D"
sfc_2022="2022_SKT_PS_2D"

# Precision, format  and compression
prec="-b F32"
form="-f nc4c -z zip"


#-------------------------------
# process IVT_N 
#-------------------------------

#processing, 2022
cdo ${prec} ${form} -sellonlatbox,-9,16,75,81.5 ${IVTN_2022}.nc ${IVTN_2022}_FS.nc
cdo ${prec} ${form}  -daymean ${IVTN_2022}_FS.nc ${IVTN_2022}_FS_daymean.nc

#processing, clim
cdo ${prec} ${form} -sellonlatbox,-9,16,75,81.5 ${IVTN_clim}.nc ${IVTN_clim}_FS.nc
cdo ${prec} ${form}  -daymean ${IVTN_clim}_FS.nc ${IVTN_clim}_FS_daymean.nc


#-------------------------------
#process MCAO index 
#-------------------------------

#processing, 2022
cdo ${prec} ${form} -sellonlatbox,-9,16,75,81.5 ${sfc_clim}.nc ${sfc_clim}_FS.nc
cdo ${prec} ${form} -sellonlatbox,-9,16,75,81.5 ${T850_clim}.nc ${T850_clim}_FS.nc
cdo ${prec} ${form} -daymean ${sfc_clim}_FS.nc ${sfc_clim}_FS_daymean.nc
cdo ${prec} ${form} -daymean ${T850_clim}_FS.nc ${T850_clim}_FS_daymean.nc
cdo ${prec} ${form} merge ${sfc_clim}_FS_daymean.nc ${T850_clim}_FS_daymean.nc clim_tmp.nc
cdo ${prec} ${form} -expr,'MCAO_index_SKT=SKT*((100000/SP)^0.286)-T*((1000/850)^0.286)' clim_tmp.nc clim_MCAO_FS_daymean.nc 

rm clim_tmp.nc

#processing, clim
cdo ${prec} ${form} -sellonlatbox,-9,16,75,81.5 ${sfc_2022}.nc ${sfc_2022}_FS.nc
cdo ${prec} ${form} -sellonlatbox,-9,16,75,81.5 ${T850_2022}.nc ${T850_2022}_FS.nc
cdo ${prec} ${form} -daymean ${sfc_2022}_FS.nc ${sfc_2022}_FS_daymean.nc
cdo ${prec} ${form} -daymean ${T850_2022}_FS.nc ${T850_2022}_FS_daymean.nc
cdo ${prec} ${form} merge ${sfc_2022}_FS_daymean.nc ${T850_2022}_FS_daymean.nc 2022_tmp.nc
cdo ${prec} ${form} -expr,'MCAO_index_SKT=SKT*((100000/SP)^0.286)-T*((1000/850)^0.286)' 2022_tmp.nc 2022_MCAO_FS_daymean.nc 

rm 2022_tmp.nc


exit
