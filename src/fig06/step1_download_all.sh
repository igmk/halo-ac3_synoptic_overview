#!/bin/bash

# Skript to download all ERA5 datasets

echo "Initiate downloads..."

python cds_api_2D_IVTN_2022.py &
python cds_api_2D_IVTN_clim.py &
python cds_api_2D_SKT_PS_2022.py &
python cds_api_2D_SKT_PS_clim.py &
python cds_api_3D_T_2022.py &
python cds_api_3D_T_clim.py &

wait

echo "Finished..!"

exit
