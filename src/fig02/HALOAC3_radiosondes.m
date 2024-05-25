%% Matlab script to analyse radiosonde data from the HALO-(AC)3 campaign
% Nils Slaettberg (nils.slaettberg@awi.de)
%
%% OUTPUT: 
% A plot of temperature, specific humidity, precipitation, tropopause
% height, integrated water vapor and cloud occurrence.
% (Tested for the HALO-(AC)3 campaign period (7 March-13 April 2022)
% using MATLAB 9.12.0.1975300 (R2022a))
%
%% DATA:
% RADIOSONDE 
% Maturilli, M.: High resolution radiosonde measurements from station 
% Ny-Ålesund (2017-04 et seq), PANGAEA - Data Publisher for Earth & 
% Environmental Science [data set], https://doi.org/10.1594/PANGAEA.914973
%
% CEILOMETER 
% Maturilli, M.: Ceilometer cloud base height from station Ny-Ålesund 
% (2017-08 et seq), PANGAEA - Data Publisher for Earth & Environmental 
% Science [data set], https://doi.org/10.1594/PANGAEA.942331, 2022. 
%
% PRECIPITATION
% Downloaded from https://seklima.met.no 
%
% TROPOPAUSE HEIGHT AND INTEGRATED WATER VAPOR (IWV)
% Sommer, M.; von Rohden, C.; Simeonov, T.; Dirksen, R.; Fiedler-Krüger, M.; 
% Friedrich, H.; Körner, S.; Naebert, T.; Oelsner, P.; Tietz, R. (2022): 
% RS41 GRUAN Data Product Version 1 (RS41-GDP.1). GRUAN Lead Centre (DWD), 
% https://doi.org/10.5676/GRUAN/RS41-GDP.1
%
%% For the code below, the data should be stored as the following variables:
%
% radio: struct with the radiosonde data, organised with one field 
% for each file, as n rows times 10 columns. The 10 columns are:
%   1.  Pressure [hPa]
%   2.  Geopot height [m]
%   3.  GPS height [m] % Note: GPS signals not good in the lower 500m! (reflecting snow, mountains)
%   4.  Temp [K]
%   5.  RH [%]
%   6.  Specific humidity [g/kg] (can be calculated using the function hywex 
%       under "Local Functions" below, i.e. based on Hyland & Wexler:
%       Hyland, R. W. and A. Wexler, Formulations for the Thermodynamic
%       Properties of the saturated Phases of H2O from 173.15K to 473.15K, 
%       ASHRAE Trans, 89(2A), 500-519, 1983.)
%   7.  Wind direction [deg E]
%   8.  Wind speed [m/s]
%   9.  Longitude
%   10. Latitude
%
% gruanData: Struct for tropopause height and IWV, with the fields
%   sonde_IWV
%   sonde_launchtime
%   sonde_tropopause_geopotheight
%
% cloud: with the ceilomater data (daily cloud ocurrence, %)
%
% prcp:  the precipitation data (total daily precipitation sums)
%
% prcpDt: datetime array with the data for the cloud and precipitation data
%
%
%
% The colormaps for temperature and IWV are included as functions at the
% end of this script, but were originally from the sources below: 
%  The colormap for temperature was generated with the BrewerMap function. 
%    Code and licence are available at
%    https://github.com/DrosteEffect/BrewerMap  (2014-2020, Stephen Cobeldick) 
%    BrewerMap includes color specifications & designs developed by Cynthia Brewer.
%    See the ColorBrewer website for further information about each colorscheme,
%    colour-blind suitability, licensing, and citations: http://colorbrewer.org/
%  The IWV colormap was generated with PyColormap4Matlab,
%    which import colormaps from matplotlib into Matlab.
%    Code and licence are available at
%    https://github.com/f-k-s/PyColormap4Matlab/blob/master/pyplotCMap2txt.py
%    (Version 1.1; K. Schumacher 08.2018)
%
% The wind barbs are drawn with the windbarb function (in which markersize was changed from 2 to 3 at line 134):
%    Available at: https://github.com/dmhuehol/ASOS-Tools/blob/main/windbarb.m
%    Written by: Laura Tomkins
%    Last updated May 2017
%    Modified by: Daniel Hueholt
%    Last updated May 2020
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. PREPARE THE DATA
%% Interpolate radiosonde data to common geopot height levels up to 15 km
names = fieldnames(radio);
% preallocate vars with NaN
tempInt = nan(1500, length(names)); wdInt = tempInt; 
presInt = tempInt; wsInt = tempInt;
altInt = nan(1500, 1); % only one column needed, same fixed levs for all obs 
shumInt = tempInt; eqptInt = tempInt;
lonInt = tempInt; latInt = tempInt;
c = 0; % counter
for s = 1:length(names) % Loop over all soundings
    altRS = radio.(names{s})(:,2); % gph for sounding s
    tempRS= radio.(names{s})(:,4); % temp for sounding s ...
    shumRS= radio.(names{s})(:,6);
    wsRS  = radio.(names{s})(:,8);
    wdRS  = radio.(names{s})(:,7);
    presRS= radio.(names{s})(:,1);
    lonRS = radio.(names{s})(:,9);
    latRS = radio.(names{s})(:,10);
    
    for ii = 1:1500 % 1500 * 10 m steps = 15 km altitude
        alt = ii*10;
        altInt(ii, 1) = alt; % The fixed geopotential height levels
        for jj = 2:length(altRS)
            
            if altRS(jj) >= alt && altRS(jj-1) < alt 
                % The variables on this fixed levels (a distance weighted mean of the two points closest to level l)
                tempInt(ii,s) = tempRS(jj)-(((tempRS(jj)-tempRS(jj-1))/(altRS(jj)-altRS(jj-1)))*(altRS(jj)-alt));
                shumInt(ii,s) = shumRS(jj)-(((shumRS(jj)-shumRS(jj-1))/(altRS(jj)-altRS(jj-1)))*(altRS(jj)-alt));
                wsInt(ii,s)   = wsRS(jj)  -(((wsRS(jj)-  wsRS(jj-1))/  (altRS(jj)-altRS(jj-1)))*(altRS(jj)-alt));
                wdInt(ii,s)   = wdRS(jj)  -(((wdRS(jj)-  wdRS(jj-1))/  (altRS(jj)-altRS(jj-1)))*(altRS(jj)-alt));
                presInt(ii,s) = presRS(jj)-(((presRS(jj)-presRS(jj-1))/(altRS(jj)-altRS(jj-1)))*(altRS(jj)-alt));
                lonInt(ii,s)  = lonRS(jj) - (((lonRS(jj)- lonRS(jj-1))/(altRS(jj)-altRS(jj-1)))*(altRS(jj)-alt));
                latInt(ii,s)  = latRS(jj) - (((latRS(jj) -latRS(jj-1))/(altRS(jj)-altRS(jj-1)))*(altRS(jj)-alt));
            end
        end
    end
    
    c = c+1;
    percentageRun = round(c/length(names) * 100) % shows progress of the loop
end

% % Sanity check the interpolation
% figure
% subplot(2,1,1)
% for s = 1:length(names)
%     hold on
%     plot(tempInt(:,s), altInt(:), 'k-', 'linewi', 2)
%     plot(radio.(names{s})(:,4), radio.(names{s})(:,2), 'r:', 'linewi', 2)
%     ylim([1 30000])
% end
% title('Interpolated vs raw temp')
% 
% subplot(2,1,2)
% for s = 1:length(names)
%     hold on
%     plot(shumInt(:,s), altInt(:), 'k-', 'linewi', 2)
%     plot(radio.(names{s})(:,6), radio.(names{s})(:,2), 'r:', 'linewi', 2)
%     ylim([1 30000])
% end
% title('Interpolated vs raw shum')
clear alt *RS c ii jj percentageRun s
%% Prepare tropopause height and IWV 
% Shift GRUAN dates to same standard launch times, so everything can be plotted together
gruanData.dt = datetime(datestr(gruanData.sonde_launchtime));
gDt = dateshift(gruanData.dt, 'end', 'Hour');
gDt.Hour(gDt.Hour == 5) = 6;
gDt.Hour(gDt.Hour == 11) = 12;
gDt.Hour(gDt.Hour == 17) = 18;
gDt.Hour(gDt.Hour == 23) = 0;
gDt(gDt.Hour == 0) = gDt(gDt.Hour == 0) +1;
% unique(gDt.Hour) % check
% clf; scatter(gDt, gruanData.dt) % check

% Fill missing tropo and IWV data with nan 
TT = timetable(dt4h, nan(size(dt4h)),  nan(size(dt4h)));
TT.Properties.VariableNames = {'IWV', 'tropo'};
for n = 1:length(dt4h)
    
    thisDate = dt4h(n);
    
    if sum(ismember(gDt, thisDate)) > 0
        ix = find(ismember(gDt, thisDate));
        ix = ix(1);
        TT.IWV(n) = gruanData.sonde_IWV(ix);
        TT.tropo(n) = gruanData.sonde_tropopause_geopotheight(ix);
    end
end

clear ix n gDt
%% Copy previous sounding if soundings are missing (i.e. get all on evenly spaced 4h time axis)
% First fill with nans where no data
temp = nan(1500, length(dt4h)); % temperature
shum = temp; % specific humidity
pres = temp; % pressure
wisp = temp; wdir = temp; % wind speed and dir
lon = temp; lat = temp;
ix = find(ismember(dt4h, dt));
for n = 1:length(dt)
    temp(:,ix(n)) = tempInt(:,n);
    shum(:,ix(n)) = shumInt(:,n);
    wisp(:,ix(n)) = wsInt(:,n);
    wdir(:,ix(n)) = wdInt(:,n);
    pres(:,ix(n)) = presInt(:,n);
    lon(:,ix(n)) = lonInt(:,n);
    lat(:,ix(n)) = latInt(:,n);
end

% Then loop, if the sounding doesn't exist, take previous sounding
for n = 1:size(temp, 2) 
    if n > 3
        if sum(~isnan(temp(:,n))) == 0 % If this column only has nans, it should be because there is no sounding for this timestep
            
            if sum(~isnan(temp(:,n-1))) > 0 % if the previous timestep (n-1) has data...
                temp(:,n) = temp(:,n-1); % ...use this data
                shum(:,n) = shum(:,n-1);
                pres(:,n) = pres(:,n-1);
                wisp(:,n) = wisp(:,n-1);
                wdir(:,n) = wdir(:,n-1);
                lon(:,n)  =  lon(:,n-1);
                lat(:,n)  =  lat(:,n-1);
                TT.IWV(n) = TT.IWV(n-1);
                TT.tropo(n) = TT.tropo(n-1);
            elseif sum(~isnan(temp(:,n-2))) > 0  % Otherwise, if the timestep before the previous one (n-2) has data...
                temp(:,n) = temp(:,n-2); % ...use this data
                shum(:,n) = shum(:,n-2);
                pres(:,n) = pres(:,n-2);
                wisp(:,n) = wisp(:,n-2);
                wdir(:,n) = wdir(:,n-2);
                lon(:,n)  =  lon(:,n-2);
                lat(:,n)  =  lat(:,n-2);
                TT.IWV(n) = TT.IWV(n-2);
                TT.tropo(n) = TT.tropo(n-2);
            elseif sum(~isnan(temp(:,n-3))) > 0 % Otherwise, if the timestep before that (n-3) has data...
                temp(:,n) = temp(:,n-3); % ...use this data
                shum(:,n) = shum(:,n-3);
                pres(:,n) = pres(:,n-3);
                wisp(:,n) = wisp(:,n-3);
                wdir(:,n) = wdir(:,n-3);
                lon(:,n)  =  lon(:,n-3);
                lat(:,n)  =  lat(:,n-3);
                TT.IWV(n) = TT.IWV(n-3);
                TT.tropo(n) = TT.tropo(n-3);
            end
        end
    end
end

% % Plot to check
% clf
% hold on; % for whatever reason, hold on removes the little tick marks around the plot
% %imagesc(pres);
% imagesc(datenum(dt4h), altInt, temp)
% %imagesc(wdir)
% datetick('x','dd mmm')
% colorbar

radioInt = cat(3, temp, shum, wisp, wdir, pres, lon, lat);
radioIntNames = ["temp", "shum", "wisp", "wdir", "pres", "lon", "lat"];

clear wisp wdir shum temp pres s n ix tempInt presInt shumInt w*Int lon lat eqpt*
%% Get prcp and cloud data on same time axis as the other data
prcpDt.Hour = 0; 
prcpDt.Minute = 0; prcpDt.Second = 0;
prcp2 = nan(length(dt4h),1);
ix = ismember(dt4h,prcpDt);
prcp2(ix)=prcp;
cloud2 = nan(length(dt4h),1);
cloud2(ix)=cloud;
% tmp = timetable(dt4h, prcp2); % for checking
%% Convert temperature from K to degC
radioInt(:,:,1) = radioInt(:,:,1) - 273.15;

%% 2. THE PLOT
tLim = [193 277]; tLim = tLim - 273.15; % colormap limits for temperature
[x,y] = meshgrid(datenum(dt4h), altInt/1000);
xt = datenum(dt4h(25:8:end));
cm = getSpectralCm(); % cm = flipud(brewermap(42, 'Spectral')); 
kFactor = 1.943844;
sFac = 0.01; % scaling factor for the wind barbs 

fs = 12; % fontsize 
lw = 1.5; % linewidth
w = 0.74; % subplot width
h = 0.27; % subplot height
v = 0.025; % for adjusting subplots vertically
ho = 0.0; % for adjusting suplots horisontally

f = figure; 
f.Position = [78 36 1161 1216]; % size and position of fig, may depend on screen
s1 = subplot(3,1,1);
s1.Position = [ 0.1300-ho , 0.7093-v, w, h];  %[left bottom width height]

% TEMPERATURE
imagesc(datenum(dt4h), altInt/1000, radioInt(:,:,1), 'AlphaData',~isnan(radioInt(:,:,1))); % AlphaData part makes nan transparent!
set(gca, 'XTick', xt, 'XTickLabel', '{}') 
colormap(gca, cm);
caxis(tLim);
cb = colorbar('westoutside');
pos = cb.Position; 
pos(1) = pos(1) - 0.12; % left
pos(2) = pos(2) + 0.02; % height
pos(4) = pos(4) - 0.05; % length
set(cb, 'Position',pos);
title(cb, 'T ({\circ}C)', 'fontsize', fs);
set(gca, 'YDir', 'normal');
% add vertical dashed black lines at 00Z of 2022-03-13, 2022-03-15, 2022-03-21, 2022-03-28, 2022-04-01, 2022-04-08 
xline(datenum(dt4h(49)), 'k--', 'LineWidth',lw, 'Alpha',1);
xline(datenum(dt4h(57)), 'k--', 'LineWidth',lw, 'Alpha',1);
xline(datenum(dt4h(81)), 'k--', 'LineWidth',lw, 'Alpha',1);
xline(datenum(dt4h(109)),'k--', 'LineWidth',lw, 'Alpha',1);
xline(datenum(dt4h(125)),'k--', 'LineWidth',lw, 'Alpha',1);
xline(datenum(dt4h(153)),'k--', 'LineWidth',lw, 'Alpha',1);

% WIND BARBS
for br = 100:100:1400
    for bc = 31:4:167 % winds at 12.00 UTC
        windbarb(x(br,bc), y(br,bc), radioInt(br,bc,3)*kFactor, radioInt(br,bc,4), sFac, 1, [0 0 0], 1);
    end
end

% TROPOPAUSE HEIGHT
hold on
plot(datenum(TT.dt4h), TT.tropo/1000, 'linewi', 2.5, 'linesty', '-','marker','none','color', [0 0 0]);
xlim([datenum(dt4h(25)), datenum(dt4h(end-11))]);

ylim([0.02 15]);
yl = ylabel('Geopotential altitude (km)');
pos = yl.Position; pos(2) = pos(2) -7.5; pos(1) = pos(1) -0.6;
set(yl,'Position',pos);

ax = gca;
ax.FontSize = fs;
text(datenum(dt4h(27))-0.2, 14, '(a)','Fontsize', 22);

% Make legend
ang = 75;
col = [0 0 0];
xDiff = ax.XLim(2)/1000000;
yDiff = ax.YLim(2)/20;
xPos = ax.XLim(2) + 1.2*xDiff;
yPos = ax.YLim(2) -0.2; % + 0.75;
fac = 1.8; % factor for adjusting distance between arrow and text
bScale = 0.017; %0.035; % scaling factor for barbs
kWs = 0:5:150; % legend wind speeds in knots
mWs = kWs/kFactor; % legend wind speeds in m/s

% DRAW THE LEGEND
%windbarb(x,y,spd,dir,scale,width,color,barb)
% The barbs:
windbarb(xPos, yPos,          kWs(1), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-yDiff,    kWs(2), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-2*yDiff,  kWs(3), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-3*yDiff,  kWs(4), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-4*yDiff,  kWs(5), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-5*yDiff,  kWs(6), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-6*yDiff,  kWs(7), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-7*yDiff,  kWs(8), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-8*yDiff,  kWs(9), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-9*yDiff,  kWs(10), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-10*yDiff, kWs(11), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-11*yDiff, kWs(12), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-12*yDiff, kWs(13), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-13*yDiff, kWs(14), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-14*yDiff, kWs(15), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-15*yDiff, kWs(16), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-16*yDiff, kWs(17), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-17*yDiff, kWs(18), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-18*yDiff, kWs(19), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-19*yDiff, kWs(20), ang, bScale, 1, col, 1); 
windbarb(xPos, yPos-20*yDiff, kWs(21), ang, bScale, 1, col, 1); 
%windbarb(xPos, yPos-21*yDiff, kWs(22), ang, bScale, 1, col, 1); 
%windbarb(xPos, yPos-22*yDiff, kWs(23), ang, bScale, 1, col, 1); 
% windbarb(xPos, yPos-23*yDiff, kWs(24), ang, bScale, 1, col, 1); 
% The text:
text(xPos-1.3,       yPos+1.75*yDiff, 'Wind speed (m s^{-1})', 'fontsize', fs); % header for legend
fs = 9;
text(xPos+xDiff*fac, yPos, "<2.6", 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-yDiff, string(round(mWs(2), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-2*yDiff, string(round(mWs(3), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-3*yDiff, string(round(mWs(4), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-4*yDiff, string(round(mWs(5), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-5*yDiff, string(round(mWs(6), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-6*yDiff, string(round(mWs(7), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-7*yDiff, string(round(mWs(8), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-8*yDiff, string(round(mWs(9), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-9*yDiff, string(round(mWs(10), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-10*yDiff, string(round(mWs(11), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-11*yDiff, string(round(mWs(12), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-12*yDiff, string(round(mWs(13), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-13*yDiff, string(round(mWs(14), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-14*yDiff, string(round(mWs(15), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-15*yDiff, string(round(mWs(16), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-16*yDiff, string(round(mWs(17), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-17*yDiff, string(round(mWs(18), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-18*yDiff, string(round(mWs(19), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-19*yDiff, string(round(mWs(20), 1)), 'fontsize', fs+3)
text(xPos+xDiff*fac, yPos-20*yDiff, string(round(mWs(21), 1)), 'fontsize', fs+3)
%text(xPos+xDiff*fac, yPos-21*yDiff, string(round(mWs(22), 1)), 'fontsize', fs+3)
%text(xPos+xDiff*fac, yPos-22*yDiff, string(round(mWs(23), 1)), 'fontsize', fs+3)
% text(xPos+xDiff*fac, yPos-23*yDiff, string(round(mWs(24), 1)), 'fontsize', fs+3)
fs = 12;

% Plot Specific humidity and IWV
% Colormap for IWV, from % https://github.com/f-k-s/PyColormap4Matlab/blob/master/pyplotCMap2txt.py
cm = getTerrainCm; % calls function with colormap (inserted at the end of this script)
cm = flipud(cm(:,1:3)); 
cm = cm(1:4:end,:);

qLim = radioInt(:,:,2);
qLim = [min(qLim(:)), max(qLim(:))];
qLim = round(qLim);

s2 = subplot(3,1,2); % SPECIFIC HUM AND IWV 
s2.Position =  [0.1300-ho  0.4096, w, h]; 
yyaxis left
imagesc(datenum(dt4h), altInt/1000, radioInt(:,:,2)); %, 'AlphaData',~isnan(radioInt(:,:,2))) % AlphaData part makes nan transparent!
set(gca, 'XTick', xt, 'XTickLabel', '{}') 
colormap(gca, cm);
caxis(qLim);
cb = colorbar('westoutside');
cb.Ticks = qLim(1):0.5:qLim(2); 
pos = cb.Position; 
pos(1) = pos(1) - 0.12; % left
pos(2) = pos(2) + 0.02; % height
pos(4) = pos(4) - 0.05; % length
set(cb, 'Position',pos);
title(cb, 'q (g kg^{-1})', 'fontsize', fs);
set(gca, 'YDir', 'normal');

% add vertical dashed black lines at 00Z of 2022-03-13, 2022-03-15, 2022-03-21, 2022-03-28, 2022-04-01, 2022-04-08 
xline(datenum(dt4h(49)), 'k--', 'LineWidth',lw, 'Alpha',1);
xline(datenum(dt4h(57)), 'k--', 'LineWidth',lw, 'Alpha',1);
xline(datenum(dt4h(81)), 'k--', 'LineWidth',lw, 'Alpha',1);
xline(datenum(dt4h(109)),'k--', 'LineWidth',lw, 'Alpha',1);
xline(datenum(dt4h(125)),'k--', 'LineWidth',lw, 'Alpha',1);
xline(datenum(dt4h(153)),'k--', 'LineWidth',lw, 'Alpha',1);

yyaxis right
plot(datenum(TT.dt4h), TT.IWV, 'linewi', 2, 'color', 'k')
ylabel('Integrated water vapor (kg m^{-2})');

ax = gca;
ax.YAxis(1).Color = 'k';
ax.YAxis(2).Color = 'k';
ax.FontSize = fs;
ax.YTick = (0:2:14);
text(datenum(dt4h(27))-0.2, 14, '(b)','Fontsize', 22);

xlim([datenum(dt4h(25)), datenum(dt4h(end-11))]);
ylim([0 15]);

s3 = subplot(3,1,3); % Precipitation + cloud occurence
s3.Position = [0.1300-ho  0.1100+v w h];
yyaxis left;
b = bar(datenum(dt4h), prcp2);
set(gca, 'XTick', xt, 'XTickLabel',xt) 
datetick('x','dd mmm', 'keepticks')
xlim([datenum(dt4h(25)), datenum(dt4h(end-11))]);
ylim([0 45]);
text(datenum(dt4h(27))-0.2, 42.3, '(c)','Fontsize', 22);
xl=xlabel('Date in 2022');
ylabel('Daily precipitation sum (mm)');
ax = gca;
ax.FontSize = fs;
b.BarWidth = 3.5;
pCol = [.21 .1 .77];
b.EdgeColor = pCol; b.FaceColor = pCol;

% add vertical dashed black lines at 00Z of 2022-03-13, 2022-03-15, 2022-03-21, 2022-03-28, 2022-04-01, 2022-04-08 
xline(datenum(dt4h(49)), 'k--', 'LineWidth',lw, 'Alpha',1);
xline(datenum(dt4h(57)), 'k--', 'LineWidth',lw, 'Alpha',1);
xline(datenum(dt4h(81)), 'k--', 'LineWidth',lw, 'Alpha',1);
xline(datenum(dt4h(109)),'k--', 'LineWidth',lw, 'Alpha',1);
xline(datenum(dt4h(125)),'k--', 'LineWidth',lw, 'Alpha',1);
xline(datenum(dt4h(153)),'k--', 'LineWidth',lw, 'Alpha',1);

yyaxis right
plot(datenum(dt4h),cloud2,'ko-','LineWidth',3)
ylim([0 110]);
ylabel('Daily mean cloud occurrence (%)');
ax = gca;
ax.YAxis(1).Color = pCol;
ax.YAxis(2).Color = 'k';


clear ax pCol b s1 s2 s3

%% 3. LOCAL FUNCTIONS
function shum = hywex(temp,press,rh)

% shum = hywex(temp,press,rh)
% calculate specific humidity [kg/kg] from temperature, pressure and relative
% humidity, using vapor pressure formulas of Hyland and Wexler, 1983:
% Hyland, R. W. and A. Wexler, Formulations for the Thermodynamic
% Properties of the saturated Phases of H2O from 173.15K to 473.15K, 
% ASHRAE Trans, 89(2A), 500-519, 1983.

% input:
%   temp...temperature in K
%   press...pressure in hPa
%   rh...relative humidity, 0 <= rh <=1

% This version assumes saturation always occurs over liquid surfaces!
% Uncomment last line to allow ice too


% spec.humidity over water
    satdd =  exp(-5800.2206 ./ temp...
                + 1.3914993 ...
                - 0.048640239    .* temp ...
                + 0.000041764768 .* temp.^2  ...
                - (0.14452093e-7).* temp.^3 ...
                + 6.5459673      .* log(temp)); 

    vaporpressure = rh.*satdd./100; % /100:  Pascal -->  hPa 
   

    mischv  = (0.622.*(vaporpressure)./ (press - (vaporpressure)));
    
    
    % Over ice

  satdd_Eis = exp( -0.56745359e4 ./temp  ...
              + 0.63925247e1           ...
              - 0.96778430e-2 .*temp    ... 
              + 0.62215701e-6 .*temp.^2  ...
              + 0.20747825e-8 .*temp.^3  ...
              - 0.94840240e-12.*temp.^4  ...
              + 0.41635019e1  .*log(temp));
          
          
    vaporpressure = rh.*satdd_Eis./100; % /100:  Pascal -->  hPa 
    mischv2  = (0.622.*(vaporpressure)./ (press - (vaporpressure)));
    
    
    shum1= mischv./(1+mischv); 
    shum2= mischv2./(1+mischv2); 
    shum=NaN.*zeros(size(temp));
    shum=shum1;
    %shum(temp<=273.16)=shum2(temp<=273.16); % <-- THIS LINE WOULD GIVE
    % SPECIFIC HUMIDITY FROM SAT. VAPOR PRESSURE OVER ICE.
    % BUT NOT RECCOMENDED FOR THESE VAISALA RADIOSONDES SINCE THEY ALWAYS GIVE
    % RH WITH RESPECT TO LIQUID SURFACES ACCORDING TO MANUFACTURER
   
end    


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function terrain = getTerrainCm()
terrain = [0.2000    0.2000    0.6000    1.0000
    0.1948    0.2105    0.6105    1.0000
    0.1895    0.2209    0.6209    1.0000
    0.1843    0.2314    0.6314    1.0000
    0.1791    0.2418    0.6418    1.0000
    0.1739    0.2523    0.6523    1.0000
    0.1686    0.2627    0.6627    1.0000
    0.1634    0.2732    0.6732    1.0000
    0.1582    0.2837    0.6837    1.0000
    0.1529    0.2941    0.6941    1.0000
    0.1477    0.3046    0.7046    1.0000
    0.1425    0.3150    0.7150    1.0000
    0.1373    0.3255    0.7255    1.0000
    0.1320    0.3359    0.7359    1.0000
    0.1268    0.3464    0.7464    1.0000
    0.1216    0.3569    0.7569    1.0000
    0.1163    0.3673    0.7673    1.0000
    0.1111    0.3778    0.7778    1.0000
    0.1059    0.3882    0.7882    1.0000
    0.1007    0.3987    0.7987    1.0000
    0.0954    0.4092    0.8092    1.0000
    0.0902    0.4196    0.8196    1.0000
    0.0850    0.4301    0.8301    1.0000
    0.0797    0.4405    0.8405    1.0000
    0.0745    0.4510    0.8510    1.0000
    0.0693    0.4614    0.8614    1.0000
    0.0641    0.4719    0.8719    1.0000
    0.0588    0.4824    0.8824    1.0000
    0.0536    0.4928    0.8928    1.0000
    0.0484    0.5033    0.9033    1.0000
    0.0431    0.5137    0.9137    1.0000
    0.0379    0.5242    0.9242    1.0000
    0.0327    0.5346    0.9346    1.0000
    0.0275    0.5451    0.9451    1.0000
    0.0222    0.5556    0.9556    1.0000
    0.0170    0.5660    0.9660    1.0000
    0.0118    0.5765    0.9765    1.0000
    0.0065    0.5869    0.9869    1.0000
    0.0013    0.5974    0.9974    1.0000
         0    0.6059    0.9824    1.0000
         0    0.6137    0.9588    1.0000
         0    0.6216    0.9353    1.0000
         0    0.6294    0.9118    1.0000
         0    0.6373    0.8882    1.0000
         0    0.6451    0.8647    1.0000
         0    0.6529    0.8412    1.0000
         0    0.6608    0.8176    1.0000
         0    0.6686    0.7941    1.0000
         0    0.6765    0.7706    1.0000
         0    0.6843    0.7471    1.0000
         0    0.6922    0.7235    1.0000
         0    0.7000    0.7000    1.0000
         0    0.7078    0.6765    1.0000
         0    0.7157    0.6529    1.0000
         0    0.7235    0.6294    1.0000
         0    0.7314    0.6059    1.0000
         0    0.7392    0.5824    1.0000
         0    0.7471    0.5588    1.0000
         0    0.7549    0.5353    1.0000
         0    0.7627    0.5118    1.0000
         0    0.7706    0.4882    1.0000
         0    0.7784    0.4647    1.0000
         0    0.7863    0.4412    1.0000
         0    0.7941    0.4176    1.0000
    0.0039    0.8008    0.4008    1.0000
    0.0196    0.8039    0.4039    1.0000
    0.0353    0.8071    0.4071    1.0000
    0.0510    0.8102    0.4102    1.0000
    0.0667    0.8133    0.4133    1.0000
    0.0824    0.8165    0.4165    1.0000
    0.0980    0.8196    0.4196    1.0000
    0.1137    0.8227    0.4227    1.0000
    0.1294    0.8259    0.4259    1.0000
    0.1451    0.8290    0.4290    1.0000
    0.1608    0.8322    0.4322    1.0000
    0.1765    0.8353    0.4353    1.0000
    0.1922    0.8384    0.4384    1.0000
    0.2078    0.8416    0.4416    1.0000
    0.2235    0.8447    0.4447    1.0000
    0.2392    0.8478    0.4478    1.0000
    0.2549    0.8510    0.4510    1.0000
    0.2706    0.8541    0.4541    1.0000
    0.2863    0.8573    0.4573    1.0000
    0.3020    0.8604    0.4604    1.0000
    0.3176    0.8635    0.4635    1.0000
    0.3333    0.8667    0.4667    1.0000
    0.3490    0.8698    0.4698    1.0000
    0.3647    0.8729    0.4729    1.0000
    0.3804    0.8761    0.4761    1.0000
    0.3961    0.8792    0.4792    1.0000
    0.4118    0.8824    0.4824    1.0000
    0.4275    0.8855    0.4855    1.0000
    0.4431    0.8886    0.4886    1.0000
    0.4588    0.8918    0.4918    1.0000
    0.4745    0.8949    0.4949    1.0000
    0.4902    0.8980    0.4980    1.0000
    0.5059    0.9012    0.5012    1.0000
    0.5216    0.9043    0.5043    1.0000
    0.5373    0.9075    0.5075    1.0000
    0.5529    0.9106    0.5106    1.0000
    0.5686    0.9137    0.5137    1.0000
    0.5843    0.9169    0.5169    1.0000
    0.6000    0.9200    0.5200    1.0000
    0.6157    0.9231    0.5231    1.0000
    0.6314    0.9263    0.5263    1.0000
    0.6471    0.9294    0.5294    1.0000
    0.6627    0.9325    0.5325    1.0000
    0.6784    0.9357    0.5357    1.0000
    0.6941    0.9388    0.5388    1.0000
    0.7098    0.9420    0.5420    1.0000
    0.7255    0.9451    0.5451    1.0000
    0.7412    0.9482    0.5482    1.0000
    0.7569    0.9514    0.5514    1.0000
    0.7725    0.9545    0.5545    1.0000
    0.7882    0.9576    0.5576    1.0000
    0.8039    0.9608    0.5608    1.0000
    0.8196    0.9639    0.5639    1.0000
    0.8353    0.9671    0.5671    1.0000
    0.8510    0.9702    0.5702    1.0000
    0.8667    0.9733    0.5733    1.0000
    0.8824    0.9765    0.5765    1.0000
    0.8980    0.9796    0.5796    1.0000
    0.9137    0.9827    0.5827    1.0000
    0.9294    0.9859    0.5859    1.0000
    0.9451    0.9890    0.5890    1.0000
    0.9608    0.9922    0.5922    1.0000
    0.9765    0.9953    0.5953    1.0000
    0.9922    0.9984    0.5984    1.0000
    0.9961    0.9950    0.5979    1.0000
    0.9882    0.9849    0.5936    1.0000
    0.9804    0.9749    0.5894    1.0000
    0.9725    0.9649    0.5852    1.0000
    0.9647    0.9548    0.5809    1.0000
    0.9569    0.9448    0.5767    1.0000
    0.9490    0.9347    0.5725    1.0000
    0.9412    0.9247    0.5682    1.0000
    0.9333    0.9147    0.5640    1.0000
    0.9255    0.9046    0.5598    1.0000
    0.9176    0.8946    0.5555    1.0000
    0.9098    0.8845    0.5513    1.0000
    0.9020    0.8745    0.5471    1.0000
    0.8941    0.8645    0.5428    1.0000
    0.8863    0.8544    0.5386    1.0000
    0.8784    0.8444    0.5344    1.0000
    0.8706    0.8344    0.5301    1.0000
    0.8627    0.8243    0.5259    1.0000
    0.8549    0.8143    0.5216    1.0000
    0.8471    0.8042    0.5174    1.0000
    0.8392    0.7942    0.5132    1.0000
    0.8314    0.7842    0.5089    1.0000
    0.8235    0.7741    0.5047    1.0000
    0.8157    0.7641    0.5005    1.0000
    0.8078    0.7540    0.4962    1.0000
    0.8000    0.7440    0.4920    1.0000
    0.7922    0.7340    0.4878    1.0000
    0.7843    0.7239    0.4835    1.0000
    0.7765    0.7139    0.4793    1.0000
    0.7686    0.7038    0.4751    1.0000
    0.7608    0.6938    0.4708    1.0000
    0.7529    0.6838    0.4666    1.0000
    0.7451    0.6737    0.4624    1.0000
    0.7373    0.6637    0.4581    1.0000
    0.7294    0.6536    0.4539    1.0000
    0.7216    0.6436    0.4496    1.0000
    0.7137    0.6336    0.4454    1.0000
    0.7059    0.6235    0.4412    1.0000
    0.6980    0.6135    0.4369    1.0000
    0.6902    0.6035    0.4327    1.0000
    0.6824    0.5934    0.4285    1.0000
    0.6745    0.5834    0.4242    1.0000
    0.6667    0.5733    0.4200    1.0000
    0.6588    0.5633    0.4158    1.0000
    0.6510    0.5533    0.4115    1.0000
    0.6431    0.5432    0.4073    1.0000
    0.6353    0.5332    0.4031    1.0000
    0.6275    0.5231    0.3988    1.0000
    0.6196    0.5131    0.3946    1.0000
    0.6118    0.5031    0.3904    1.0000
    0.6039    0.4930    0.3861    1.0000
    0.5961    0.4830    0.3819    1.0000
    0.5882    0.4729    0.3776    1.0000
    0.5804    0.4629    0.3734    1.0000
    0.5725    0.4529    0.3692    1.0000
    0.5647    0.4428    0.3649    1.0000
    0.5569    0.4328    0.3607    1.0000
    0.5490    0.4227    0.3565    1.0000
    0.5412    0.4127    0.3522    1.0000
    0.5333    0.4027    0.3480    1.0000
    0.5255    0.3926    0.3438    1.0000
    0.5176    0.3826    0.3395    1.0000
    0.5098    0.3725    0.3353    1.0000
    0.5020    0.3625    0.3311    1.0000
    0.5059    0.3675    0.3379    1.0000
    0.5137    0.3776    0.3484    1.0000
    0.5216    0.3876    0.3589    1.0000
    0.5294    0.3976    0.3694    1.0000
    0.5373    0.4077    0.3799    1.0000
    0.5451    0.4177    0.3904    1.0000
    0.5529    0.4278    0.4009    1.0000
    0.5608    0.4378    0.4115    1.0000
    0.5686    0.4478    0.4220    1.0000
    0.5765    0.4579    0.4325    1.0000
    0.5843    0.4679    0.4430    1.0000
    0.5922    0.4780    0.4535    1.0000
    0.6000    0.4880    0.4640    1.0000
    0.6078    0.4980    0.4745    1.0000
    0.6157    0.5081    0.4850    1.0000
    0.6235    0.5181    0.4955    1.0000
    0.6314    0.5282    0.5060    1.0000
    0.6392    0.5382    0.5165    1.0000
    0.6471    0.5482    0.5271    1.0000
    0.6549    0.5583    0.5376    1.0000
    0.6627    0.5683    0.5481    1.0000
    0.6706    0.5784    0.5586    1.0000
    0.6784    0.5884    0.5691    1.0000
    0.6863    0.5984    0.5796    1.0000
    0.6941    0.6085    0.5901    1.0000
    0.7020    0.6185    0.6006    1.0000
    0.7098    0.6285    0.6111    1.0000
    0.7176    0.6386    0.6216    1.0000
    0.7255    0.6486    0.6322    1.0000
    0.7333    0.6587    0.6427    1.0000
    0.7412    0.6687    0.6532    1.0000
    0.7490    0.6787    0.6637    1.0000
    0.7569    0.6888    0.6742    1.0000
    0.7647    0.6988    0.6847    1.0000
    0.7725    0.7089    0.6952    1.0000
    0.7804    0.7189    0.7057    1.0000
    0.7882    0.7289    0.7162    1.0000
    0.7961    0.7390    0.7267    1.0000
    0.8039    0.7490    0.7373    1.0000
    0.8118    0.7591    0.7478    1.0000
    0.8196    0.7691    0.7583    1.0000
    0.8275    0.7791    0.7688    1.0000
    0.8353    0.7892    0.7793    1.0000
    0.8431    0.7992    0.7898    1.0000
    0.8510    0.8093    0.8003    1.0000
    0.8588    0.8193    0.8108    1.0000
    0.8667    0.8293    0.8213    1.0000
    0.8745    0.8394    0.8318    1.0000
    0.8824    0.8494    0.8424    1.0000
    0.8902    0.8595    0.8529    1.0000
    0.8980    0.8695    0.8634    1.0000
    0.9059    0.8795    0.8739    1.0000
    0.9137    0.8896    0.8844    1.0000
    0.9216    0.8996    0.8949    1.0000
    0.9294    0.9096    0.9054    1.0000
    0.9373    0.9197    0.9159    1.0000
    0.9451    0.9297    0.9264    1.0000
    0.9529    0.9398    0.9369    1.0000
    0.9608    0.9498    0.9475    1.0000
    0.9686    0.9598    0.9580    1.0000
    0.9765    0.9699    0.9685    1.0000
    0.9843    0.9799    0.9790    1.0000
    0.9922    0.9900    0.9895    1.0000
    1.0000    1.0000    1.0000    1.0000];

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function cm = getSpectralCm()
cm = [ 0.3686    0.3098    0.6353
    0.3260    0.3668    0.6669
    0.2836    0.4212    0.6939
    0.2417    0.4747    0.7175
    0.2002    0.5280    0.7391
    0.1942    0.5849    0.7424
    0.2477    0.6464    0.7146
    0.3154    0.7052    0.6762
    0.3855    0.7532    0.6490
    0.4570    0.7868    0.6442
    0.5252    0.8136    0.6444
    0.5904    0.8368    0.6457
    0.6526    0.8595    0.6445
    0.7141    0.8843    0.6352
    0.7784    0.9101    0.6155
    0.8379    0.9344    0.5966
    0.8867    0.9545    0.5914
    0.9221    0.9690    0.6081
    0.9526    0.9814    0.6349
    0.9775    0.9913    0.6717
    0.9949    0.9980    0.7201
    0.9994    0.9888    0.7220
    0.9996    0.9629    0.6686
    0.9997    0.9328    0.6174
    0.9983    0.8995    0.5705
    0.9947    0.8634    0.5287
    0.9942    0.8205    0.4862
    0.9952    0.7718    0.4438
    0.9947    0.7196    0.4045
    0.9908    0.6659    0.3708
    0.9879    0.6047    0.3390
    0.9832    0.5383    0.3087
    0.9724    0.4730    0.2815
    0.9521    0.4174    0.2593
    0.9290    0.3690    0.2614
    0.9037    0.3233    0.2818
    0.8721    0.2799    0.3027
    0.8307    0.2392    0.3095
    0.7826    0.1967    0.3043
    0.7314    0.1482    0.2940
    0.6771    0.0898    0.2788
    0.6196    0.0039    0.2588];

end


