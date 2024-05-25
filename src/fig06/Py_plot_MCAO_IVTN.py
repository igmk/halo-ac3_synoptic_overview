# -*- coding: utf-8 -*-
"""
Calc Fram Strait box statistics and plot MCAO index and IVT_N
written 2022 by Benjamin Kirbus, University of Leipzig
"""

def load_nc(directory, fname, varname):
    """Input: dir, fname for netcdf.nc file
    Output: lev,time, data arrays"""
    import xarray as xr
    import numpy as np
    #access corresponding dataset
    xr_data = xr.open_dataset(directory+fname)
    #access data
    data_vals = xr_data[varname]
    #close, return values
    xr_data.close()

    return data_vals


def main():
    print("Booting Python..")
    import os
    import numpy as np
    import datetime
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import locale
    import matplotlib.patches as patches
    cwd = os.getcwd()
    directory = cwd + '/'

    ############# WAI #############
    ###   clim   ###
    nc_clim_time=load_nc(directory, "clim_IVTN_2D_FS_daymean.nc", "time")
    nc_clim_IVTN=load_nc(directory, "clim_IVTN_2D_FS_daymean.nc", "VIWVN")
    date_len=int(len(nc_clim_time)/44) #number of days per year
    clim_IVTN_daily = {} #empty dict for daily values, fill 
    #create empty dict for data
    clim_daily = {}
    #create empty lists for median, mean, pctls..
    clim_daily['mean'] = []
    clim_daily['median'] = []
    clim_daily['pctl10'] = []
    clim_daily['pctl25'] = []
    clim_daily['pctl75'] = []
    clim_daily['pctl90'] = []
    #start cycling over days
    for day in range(date_len):
        #now cycle over years
        clim_IVTN_daily[day]=[]
        for year in range(44):
            index=day+year*date_len
            clim_IVTN_daily[day].append(np.nanmean(nc_clim_IVTN[index,:,:]))
        #now calculate mean, pctls including median
        clim_daily['mean'].append(np.nanmean(clim_IVTN_daily[day]))
        clim_daily['median'].append(np.nanpercentile(clim_IVTN_daily[day],50))
        clim_daily['pctl10'].append(np.nanpercentile(clim_IVTN_daily[day],10))
        clim_daily['pctl25'].append(np.nanpercentile(clim_IVTN_daily[day],25))
        clim_daily['pctl75'].append(np.nanpercentile(clim_IVTN_daily[day],75))
        clim_daily['pctl90'].append(np.nanpercentile(clim_IVTN_daily[day],90))

    ####   2022   ###
    nc_2022_time=load_nc(directory, "2022_IVTN_2D_FS_daymean.nc", "time")
    nc_2022_IVTN=load_nc(directory, "2022_IVTN_2D_FS_daymean.nc", "VIWVN")
    daily_2022=[]
    for i in range(len(nc_2022_time.values)-1): #-1 as there is a wrong empty array at end?
        nc_2022_time_sel=str(nc_2022_time.values[i])[0:10] 
        nc_2022_IVTN_sel=nc_2022_IVTN[i,:,:]
        daily_2022.append(np.nanmean(nc_2022_IVTN_sel))

    ############# CAO #############
    ###   clim   ###
    nc_clim_time_CAO=load_nc(directory, "clim_MCAO_FS_daymean.nc", "time")
    nc_clim_IVTN_CAO=load_nc(directory, "clim_MCAO_FS_daymean.nc", "MCAO_index_SKT")
#    print(nc_clim_time) #1979-2022 #44 years
    date_len_CAO=int(len(nc_clim_time_CAO)/44) #number of days per year
    clim_IVTN_daily_CAO = {} #empty dict for daily values, fill 
    #create empty dict for data
    clim_daily_CAO = {}
    #create empty lists for median, mean, pctls..
    clim_daily_CAO['mean'] = []
    clim_daily_CAO['median'] = []
    clim_daily_CAO['pctl10'] = []
    clim_daily_CAO['pctl25'] = []
    clim_daily_CAO['pctl75'] = []
    clim_daily_CAO['pctl90'] = []
    #start cycling over days
    for day in range(date_len_CAO):
        #now cycle over years
        clim_IVTN_daily_CAO[day]=[]
        for year in range(44):
            index=day+year*date_len_CAO
            clim_IVTN_daily_CAO[day].append(np.nanmean(nc_clim_IVTN_CAO[index,:,:]))
        #now calculate mean, pctls including median
        clim_daily_CAO['mean'].append(np.nanmean(clim_IVTN_daily_CAO[day]))
        clim_daily_CAO['median'].append(np.nanpercentile(clim_IVTN_daily_CAO[day],50))
        clim_daily_CAO['pctl10'].append(np.nanpercentile(clim_IVTN_daily_CAO[day],10))
        clim_daily_CAO['pctl25'].append(np.nanpercentile(clim_IVTN_daily_CAO[day],25))
        clim_daily_CAO['pctl75'].append(np.nanpercentile(clim_IVTN_daily_CAO[day],75))
        clim_daily_CAO['pctl90'].append(np.nanpercentile(clim_IVTN_daily_CAO[day],90))

    ####   2022   ###
    nc_2022_time_CAO=load_nc(directory, "2022_MCAO_FS_daymean.nc", "time")
    nc_2022_IVTN_CAO=load_nc(directory, "2022_MCAO_FS_daymean.nc", "MCAO_index_SKT")
    daily_2022_CAO=[]
    for i in range(len(nc_2022_time.values)-1): #-1 as there is a wrong empty array at end?!
        nc_2022_time_sel_CAO=str(nc_2022_time_CAO.values[i])[0:10] #or keep np.dt64 for plotting
        nc_2022_IVTN_sel_CAO=nc_2022_IVTN_CAO[i,:,:]
        daily_2022_CAO.append(np.nanmean(nc_2022_IVTN_sel_CAO))

    ############# plot #############

    print("\n\t Preparing your figure...")
    fig, (ax, ax2) = plt.subplots(2,1,sharex=True,figsize=(9,7))

    ax.set_ylabel('IVT$_\mathrm{north}$ (kg m$^{-1}$ s$^{-1}$)',size=12)
    ax.grid(zorder=0,color='grey',linewidth=0.4,alpha=0.5,axis='y')
    #plot median values
    ax.plot(nc_2022_time,clim_daily['mean'],color='lightgray',linewidth=2,
        alpha=1,zorder=30,label='1979-2022')

    #fill between lines
    ax.fill_between(nc_2022_time, clim_daily['pctl25'], clim_daily['pctl75'],
        color='dimgray',alpha=0.5,zorder=2) #,label='interquartile')
    ax.fill_between(nc_2022_time, clim_daily['pctl10'], clim_daily['pctl90'],
        color='silver',alpha=0.5,zorder=1) #,label='10th/90th percentile')

    ax.plot(nc_2022_time[:-1],daily_2022,color='black',alpha=1,zorder=100,label='2022')

    #legend
    ax.legend(loc='best',framealpha=1)

    x_start = np.datetime64('2022-03-06T21:00','m')
    x_end = np.datetime64('2022-04-12T23:00','m')
    ax.set_xlim([x_start, x_end])
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7)) 
    monthyearFmt = mdates.DateFormatter(' ')
    ax.xaxis.set_major_formatter(monthyearFmt)

    #days at 12 UTC: 13 March, 15 March, 21 March, 28 March, 01 April, 08 April 2022 
    t_lines = [np.datetime64('2022-03-13T12:00','m'),np.datetime64('2022-03-15T12:00','m'),
        np.datetime64('2022-03-21T12:00','m'),np.datetime64('2022-03-28T12:00','m'),
        np.datetime64('2022-04-01T12:00','m'),np.datetime64('2022-04-08T12:00','m')]

    for t_line in t_lines:
        ax.axvline(t_line,ls='--',color='black',lw=0.8)

    #minor format: weekdays
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    dayFmt = mdates.DateFormatter('%d')
    plt.setp( ax.get_xmajorticklabels(), rotation=0, ha='center' )
    ax.axhline(y=0, color='steelblue',linewidth=1.0, linestyle='-',zorder=90)

    #   add MCAO index #
    ax2.set_ylabel('MCAO index (K)',size=12)
    ax2.set_ylim([-12,12]) 
    ax2.yaxis.set_ticks(np.arange(-12,12+1, 4))
    ax2.set_xlabel("Date",size=12)
    ax2.grid(zorder=0,color='grey',linewidth=0.4,alpha=0.5,axis='y')
    #plot median values
    ax2.plot(nc_2022_time_CAO,clim_daily_CAO['mean'],color='lightgray',linewidth=2,
        alpha=1,zorder=30,label='1979-2022')
    lw_sel=0.8

    #fill between lines
    ax2.fill_between(nc_2022_time_CAO, clim_daily_CAO['pctl25'], clim_daily_CAO['pctl75'],
        color='dimgray',alpha=0.5,zorder=2) #,label='interquartile')
    ax2.fill_between(nc_2022_time_CAO, clim_daily_CAO['pctl10'], clim_daily_CAO['pctl90'],
        color='silver',alpha=0.5,zorder=1) #,label='10th/90th percentile')

    ax2.plot(nc_2022_time_CAO[:-1],daily_2022_CAO,color='black',alpha=1,zorder=100,label='2022')
    locale.setlocale(locale.LC_ALL,'en_US.utf8')
    ax2.axhline(y=0, color='steelblue',linewidth=1.0, linestyle='-',zorder=90)

    x_start = np.datetime64('2022-03-06T21:00','m')
    x_end = np.datetime64('2022-04-12T23:00','m')
    ax2.set_xlim([x_start, x_end])
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7)) 
    monthyearFmt = mdates.DateFormatter('%d %b')
    ax2.xaxis.set_major_formatter(monthyearFmt)

    for t_line in t_lines:
        ax2.axvline(t_line,ls='--',color='black',lw=0.8)

    #minor format: weekdays
    ax2.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    dayFmt = mdates.DateFormatter('%d')

    #add a/b
    txt_x, txt_y, txt_size = -0.078, 1, 19
    ax.text(txt_x, txt_y, '(a)', horizontalalignment='center',size=txt_size,
         verticalalignment='center', transform=ax.transAxes)
    ax2.text(txt_x, txt_y, '(b)', horizontalalignment='center',size=txt_size,
         verticalalignment='center', transform=ax2.transAxes)

    #boxes
    for ax_sel in [ax,ax2]:
        ax_sel.add_patch(
             patches.Rectangle(
                (0.110, 0),
                0.269,
                1,
                fill=False,      # remove background
                zorder=9e3,
                lw=2.5,
                color='red',
                transform=ax_sel.transAxes
             ) ) 
        ax_sel.add_patch(
             patches.Rectangle(
                (0.384, 0),
                0.616,
                1,
                fill=False,      # remove background
                zorder=9e3,
                lw=2.5,
                color='blue',
                transform=ax_sel.transAxes
             ) ) 

    figname = directory + 'HALO_AC3_WAI_CAO.png' # to save as png
    plt.savefig(figname, format='png', dpi=720,bbox_inches='tight')

    figname = directory + 'HALO_AC3_WAI_CAO.pdf' # to save as pdf
    plt.savefig(figname, format='pdf', dpi=720,bbox_inches='tight')


    print("\t\t\t ...saved as: " + str(figname))
    plt.close()

if __name__=="__main__":
    main()


exit()
