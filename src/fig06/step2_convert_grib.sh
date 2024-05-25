#!/bin/bash

# Precision, format  and compression
prec="-b F32"
form="-f nc4c -z zip"
# Program directory
progdir=${PWD}

mkdir output

for var2d in SKT_PS IVTN
do
    (
#    file_2D="clim_${var2d}_2D" #<< climatology 
    file_2D="2022_${var2d}_2D"  #<< year 2022

    ##Conversion of grib_2D
    #-----------------------
    #fix wrongly classified vars, or vars not classified at all
    cdo ${prec} ${form} --reduce_dim -t ecmwf copy -invertlat -chname,var72,VIWVN ${file_2D}.grib ${file_2D}.nc
    #fix var72/T162
    ncatted -O -a long_name,VIWVN,o,c,"Vertical integral of northward water vapour flux" ${file_2D}.nc
    ncatted -O -a units,VIWVN,o,c,"kg m*-1 s*-1" ${file_2D}.nc
    ) &
done

##Conversion of grib_3D
#-----------------------
#file_3D="clim_T850" #<< climatology 
file_3D="2022_T850"  #<< year 2022
cdo ${prec} ${form} -t ecmwf copy -invertlat -chname,plev,lev -chname,t,T ${file_3D}.grib ${file_3D}.nc &

wait

exit
