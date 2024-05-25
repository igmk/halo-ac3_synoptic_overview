#!/usr/bin/env python
import cdsapi

c = cdsapi.Client()
date_str = ("2022-03-12/2022-03-13/2022-03-14/2022-03-15/2022-03-16/2022-03-20/2022-03-21/" +
            "2022-03-28/2022-03-29/2022-03-30/2022-04-01/2022-04-04/2022-04-07/2022-04-08/2022-04-10/" +
            "2022-04-11/2022-04-12")    # date given in yyyy-mm-dd; for date lists, separate dates by "/", e.g., '2020-01-01/2020-01-25'
date_str_file = date_str.replace("-","").replace("/","")

c.retrieve("reanalysis-era5-complete", {
    "date": date_str,    # adapt date
    "levelist": "1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50/51/52/53/54/55/56/57/58/59/60/61/62/63/64/65/66/67/68/69/70/71/72/73/74/75/76/77/78/79/80/81/82/83/84/85/86/87/88/89/90/91/92/93/94/95/96/97/98/99/100/101/102/103/104/105/106/107/108/109/110/111/112/113/114/115/116/117/118/119/120/121/122/123/124/125/126/127/128/129/130/131/132/133/134/135/136/137",    # 1: top of atmosphere, 137: lowest model level
    "levtype": "ml",
    "param": "129/130/131/132/133/152", # Full information at https://apps.ecmwf.int/codes/grib/param-db/ ; 129, 130, 133, 152 must always be included
    "stream": "oper",
    "time": '04/to/19/by/1', # adapt time; for 00, 03, 06, ..., 21 UTC, use: '00/to/23/by/3'
    "type": "an",
    "area": '89.75/-56/70/30', # north/west/south/east
    "grid": '0.25/0.25',      # latitude/longitude grid
    "format": 'netcdf'
}, f"ERA5_model_level_HALO-AC3.nc") # output name