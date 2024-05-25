#!/bin/bash

# Skript to download all ERA5 datasets

echo "Initiate downloads..."

python3 cds_api_500hPa_Z_202203.py &
python3 cds_api_500hPa_Z_202204.py &
python3 cds_api_850hPa_T_RH_mr_202203.py &
python3 cds_api_850hPa_T_RH_mr_202204.py &
python3 cds_api_LSM.py &
python3 cds_api_sfc_T_p_uv_IWV_202203.py &
python3 cds_api_sfc_T_p_uv_IWV_202204.py &

wait

mkdir ERA5
mv ERA5*.nc ./ERA5/

echo "Finished..!"

exit
