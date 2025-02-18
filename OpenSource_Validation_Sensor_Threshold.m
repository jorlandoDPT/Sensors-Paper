%% Automated Processing of H5 files: Open Source

        %Code initially written by Mayumi Mohan, significantly modified and
        %updated by Julie Orlando with section contributions from Jocelyn Hafer (signal
        %processing expert) and in consultation with Laura Prosser, Beth Smith,
        %Athylia Paremski, Matthew Amodeo and Michele Lobo

%This code requires the following functions:

%importTimingfile.m
%epoch2datestr.m
%MultiplyQuaternions.m
%RotateVector.m

% This code requires an input file named: Timestamps.xls with 6 columns:
% filename, 
% startTime (selected by user in motion studio), 
% stopTime (selected by user in motion studio), 
% sensor ID number,
% button press 1- start of session,
% button press 2- end of session

%the filename starts with "EMO" followed by the participant ID. This is
%used to identify to participant ID in the following code. 

% Note: The quaternions produced by APDM's sensors express the rotation from the sensor frame to
% the Earth frame in which the XYZ axes are aligned respectively with magnetic North, West, and Up.
%
% Note: There are two nearly equivalent time columns, column 2 is "Sync Count" and column 3 is "Time".
% Column 2 can be converted to seconds by dividing by 2560
% Column 3 can be converted to seconds by dividing by 1e6
% Both represent time elaspsed since Jan 1, 1970 0:00
% 
% This script estimates user acceleration by using
%       A_gravity_p = [0,0,9.806]
%       A_user_p = A_total_p - A_gravity_p;      
%       A_user_mag = | A_user_p |
% The _p notation indicates North, West, Up coordinate frame 

addpath(genpath('Datasets'));
close all
clear all

% output file where all the data is to be saved
saveFileName = 'outputData_Individual_Files.xlsx';
% Read the file with the time stamps in it
timingFile = 'Timestamps.xlsx';
timingData = importTimingfile('Timestamps.xlsx','Sheet1','A1:F10'); %change file here too

%% folder with datsasets in it
% put all the subfolders into a folder named Datasets and add it to this
% folder 
dataPath = '/Users/ENTER/DATA/PATH'; %(Edit this based on the computer and the user) 
% get subfolders each of which contain data files
dataDirectories = dir(dataPath);
% get only names of folders
dataDirectoryNamesAll = {dataDirectories(:).name};
% the first two values are current and previous folder; so remove them
dataDirectoryNames = dataDirectoryNamesAll(3:end);

%% Declarations and initializations for the code

time_zone_offset_hours = 4;     % adjust time zone of time data, only effects screen readout, not critical

%%
% Master Data Table where all the data will be store
tableLen = length(timingData); % get the total number of datasets that we want to process
masterTable = table; % initialize a table to store all the values
masterTable.FileName = cell(tableLen, 1); % this is the column for all filenames
masterTable.BabyID = cell(tableLen, 1); % this is the column for all filenames
masterTable.StartTime = zeros(tableLen,1);% column of start times
masterTable.StopTime = zeros(tableLen,1);% column of stop times
masterTable.Duration = zeros(tableLen,1);% column of duration
masterTable.DurationMins = zeros(tableLen,1);% column of duration
masterTable.UserStart = cell(tableLen,1);% column of start times user readable format
masterTable.UserStop = cell(tableLen,1);% column of stop timesuser readable format
masterTable.totalAcceleration = zeros(tableLen,1);% column of total Accerleration
masterTable.timeNormalizedAcc = zeros(tableLen,1);% column of time normalized acceleration
masterTable.timeNormalizedAccMins = zeros(tableLen,1);% column of time normalized acceleration in minutes
masterTable.thresholdTotalAcceleration = zeros(tableLen,1);% column of total Accerleration
masterTable.thresholdTimeNormalizedAcc = zeros(tableLen,1);% column of time normalized acceleration
masterTable.thresholdTimeNormalizedAccMins = zeros(tableLen,1);% column of time normalized acceleration in minutes
masterTable.Sum_Sensor_Active_Time_Duration = zeros(tableLen,1);% column of Total Duration of Active time 
masterTable.Min_Sensor_Active_Time_Duration = zeros(tableLen,1);% column of Minimal Duration of Active time
masterTable.Max_Sensor_Active_Time_Duration = zeros(tableLen,1);% column of Maximum Duration of Active time
masterTable.count_Sensor_active_time_bouts = zeros(tableLen,1);% column of numober/ count of Actime time bouts
masterTable.Sum_Sensor_NONActive_Time_Duration = zeros(tableLen,1);% column of Total Duration of Active time 
masterTable.TruePositive= zeros(tableLen,1);% 
masterTable.TrueNegative= zeros(tableLen,1);% 
masterTable.FalsePositive= zeros(tableLen,1);% 
masterTable.FalseNegative= zeros(tableLen,1);% 
masterTable.TruePositive_Perc= zeros(tableLen,1);% 
masterTable.TrueNegative_Perc= zeros(tableLen,1);% 
masterTable.FalsePositive_Perc= zeros(tableLen,1);% 
masterTable.FalseNegative_Perc= zeros(tableLen,1);% 
masterTable.TruePositive_Perc_tol= zeros(tableLen,1);%  
masterTable.TrueNegative_Perc_tol= zeros(tableLen,1);% ;% 
masterTable.FalsePositive_Perc_tol= zeros(tableLen,1);% ;% 
masterTable.FalseNegative_Perc_tol= zeros(tableLen,1);% ;% 
masterTable.OptimalThreshold= zeros(tableLen,1);% 
masterTable.AUC= zeros(tableLen,1);% 
% Corresponds to the row of the timestamps.xlsx file (no header in file) 
%Can update this based on the row. I suggest batches of 15-20 rows at a
%time. Note the output file will re-write when you hit "run" so you will
%need to copy and paste the output data to a new document when running
%smaller batches. 

start = 10; %update as needed
endfile =  10; %update as needed
%endfile = length(dataDirectoryNames); %This will run full batch

VideoCode_filename = ('Analysis.xlsm');
    [VideoCode_filename_num, VideoCode_filename_text, VideoCode_filename_raw] = xlsread(VideoCode_filename, 'LL028_ten', 'A:H');

% Iterate through each folder
for ii = start:endfile  %user updates
    %% Load data from h5 file 
    % Set up file name, find the participant ID from the file name and use
    % the particpant ID. Read in the H5 file. 
    
    % Current File name
    currentFile = timingData{ii,1};
    % Get the baby id and thus the required folder
    startInd = strfind(currentFile, 'LLO');
    babyID = currentFile(startInd:startInd+4);
    babyFile = [dataPath, babyID, '/', currentFile];
    SensorID2 = timingData{ii,4};

    %% Use this if the file was before 2021 otherwise comment out this section and see below
    % caseID = hdf5read(currentFile, '/', 'CaseIdList');
    % groupName = caseID.data;
    % time = double(hdf5read(currentFile, [groupName '/Time']));% time has been converted to double since hdf5 extracts it as uint64
    % A_total = hdf5read(currentFile, [groupName '/Calibrated/Accelerometers'])';
    % q = hdf5read(currentFile, [groupName '/Calibrated/Orientation'])';


%% if the file was from after 2021, the structure of the h5 file changed.

    %to identify the sensorID, uncomment this section. Then look in the
    %command window, the sensor ID will be located after 'Group
    %'/Processed/' within the text. Copy this sensor ID and input the
    %sensor ID into the timestamps file for files created after 2021. 
   
    %h5disp(currentFile); %use to view file info inluding sensor ID number & location
    %in command window
    % info = hdf5info(currentFile);
    SensorID2 = num2str(SensorID2);
    time_filename = ['Sensors/' SensorID2]; 
    time = double(hdf5read(currentFile, ['Sensors/' SensorID2 '/Time']));% time has been converted to double since hdf5 extracts it as uint64
    A_total = hdf5read(currentFile, ['Sensors/' SensorID2 '/Accelerometer'])';
    q = hdf5read(currentFile, ['Processed/' SensorID2 '/Orientation'])';

    %% Load time data and print human readable date and time to screen
    time = time./1e6; % [sec] Load time data and convert from milliseconds to seconds
    time_start_string = epoch2datestr(time(1),  time_zone_offset_hours); % Convert start time to human readable date and time
    time_end_string   = epoch2datestr(time(end),time_zone_offset_hours); % Convert end time to human readable date and time
  
   % Indicate when the button to sync sensors or indicate start of session
   % was pressed
    button_on_annotation = timingData{ii,5}; %import data
    button_off_annotation = timingData{ii,6}; %import data
    button_1 = str2double(button_on_annotation);
    button_2 = str2double(button_off_annotation);
    button_1_sec = button_1./1e6; %convert to seconds
    button_event_on = epoch2datestr(button_1_sec,  time_zone_offset_hours); %convert to readable time and date
    button_2_sec = button_2./1e6; %convert to seconds
    button_event_off = epoch2datestr(button_2_sec,  time_zone_offset_hours); %convert to readable time and date
    sensor_recorded_time = (button_2_sec-button_1_sec); %Time of session between button pushes (i.e., time of intersest)
    start_button_push_diff = button_1_sec - time(1); %Time elpased before button push (before start of session)
    stop_button_push_diff = time(end) - button_2_sec; %Time elapsed after button push (after end of session)
    time_total_diff = time(end)- time(1); %Total recoreded time

    %synch sensor and video data EXAMPLE PROVIDED FOR ONE PARTICIPANT. 

    % %%used with video data for LL028
    video_time_removal = 1840.842; %in sec
    sensor_time_removal = 1838.854; %in sec
    video_time_start = 0; %in sec
    video_time_in_place = 10.088; %in sec, time in the video when "button is pressed"
    sensor_time_in_place = 8.100; %in sec, time of button press 1 on sensor
    video_sensor_difference = sensor_time_in_place - video_time_in_place;

    % %Check for duration
    duration_video = video_time_removal - video_time_in_place;
    duration_sensor_check =  sensor_time_removal - sensor_time_in_place;
    
    %Convert time to seconds after start
    time = time - time(1);
    % t_start_input_file = timingData{ii,2};
    % t_stop_input_file = timingData{ii,3};
    % 
    %Trim data based on video code NOTE: If there is no video file and you
    %want to trim based on pre-identified start/stop times from visual data
    %inspection, adust this to use the t_start_input_file and
    %t_stop_input_file above. 

    ix_time_new = time >= sensor_time_in_place & time <= sensor_time_removal; %logical
    time = time(ix_time_new,:); %indexing from logical
        
    %convert sensor to video_code time
    time_reconciled = time - video_sensor_difference;
 
    %index Acceleration total from time cut-offs
    A_total = A_total(ix_time_new,:);
    q = q(ix_time_new,:);

    %make time start at zero using reconciled time, use time_zero for all
    %time values in remaining code
     if time_reconciled(1) > 0
        time_zero = time_reconciled - time_reconciled(1); 
     else
        time_zero = time_reconciled;
     end 
        
    %figure 1: raw data   
    hf1SaveName = 'raw_acceleration_components'
    hf1=figure; hold on; grid on;
    h1x = plot(time_zero,A_total(:,1),'-r');
    h1y = plot(time_zero,A_total(:,2),'-g');
    h1z = plot(time_zero,A_total(:,3),'-b');
    title('Total Acceleration Components');
    legend([h1x,h1y,h1z],'A_x','A_y','A_z');
    xlabel('Time (sec)');
    ylabel('Acceleration (m/s/s)');
    ylim([-30,30]);
    %xlim([0,1900])
    set(gca, 'fontsize', 16);
    % saveas(hf1,[pwd, '/Datasets/', babyID, '/Graphs/' hf1SaveName]);  


    %% Calculate the total acceleration in North, West, and Up coordinate frame
    A_total_p = A_total; % Initialize array
    for jj = 1:length(A_total(:,1))
        A_total_p(jj,:) = RotateVector(A_total(jj,:),q(jj,:)); % Rotate gravity vector into sensor coordinate frame
    end
    
        %figure 2: raw data in NWU for comparision
    figure;grid; hold on;
    h2x = plot(time_zero,A_total_p(:,1),'-r');
    h2y =plot(time_zero,A_total_p(:,2),'-g');
    h2z =plot(time_zero,A_total_p(:,3),'-b');
    legend([h2x,h2y,h2z],'A_x','A_y','A_z');
    xlabel('Time (sec)')
    ylabel('Acceleration (m/s/s)')
    title('Raw Acceleration Components in North, West, Up');
    ylim([-30,30]);
    %xlim([0,1900]);
    set(gca, 'fontsize', 16);

    % Calculate user acceration by removing gravity from the median position detected
    %A_user_p = A_total_p - A_total_p(medI,:); %filtered
    A_gravity_p = [0,0,9.806];
    A_user_p = A_total_p - A_gravity_p; %filtered
    
    %figure 2: raw data in NWU for comparision
    figure;grid; hold on;
    h2x = plot(time_zero,A_user_p(:,1),'-r');
    h2y =plot(time_zero,A_user_p(:,2),'-g');
    h2z =plot(time_zero,A_user_p(:,3),'-b');
    legend([h2x,h2y,h2z],'A_x','A_y','A_z');
    xlabel('Time (sec)')
    ylabel('Acceleration (m/s/s)')
    title('Raw Acceleration Components in North, West, Up with Gravity Removed');
    ylim([-30,30]);
    %xlim([0,1900]);
    set(gca, 'fontsize', 16);

    %Calculate user acceleration magnitude from filtered data
    
    % Calculate acceleration magnitudes
    A_user_mag  = sqrt(A_user_p(:,1).^2 + A_user_p(:,2).^2 + A_user_p(:,3).^2); %filtered accelration resultant, gravity removed
    
   %% Filter user_acceleration components raw data in N, W, Up (filter first, then get resultant)
         
    % Fourier transform (FFT): This will transform data from time domain
    % to frequency doamin
    Fs = 128; %sampling frequency/sampling rate (of the sensor) - A_total_p
    dt = 1/128; %change in time 
    tol = time_zero; 
    n = length(time_zero); % should this be magnitude data or the raw data - use raw data NWU
    fhat = fft(A_user_mag,n);
      
    %Power Spectral Density Analysis: Once data is in the frequency domain, we can do a power spectral density analysis to look power of the signal and decide
    %where to filter the data    
    PSD = fhat.*conj(fhat)/n; %power spectrum (power per freq)
    freq = 1/(dt*n)*(0:n);   %x-axis of freq in Hz
    L = 1:floor(n/2); %plot the first half of freqs
    half_freq = freq(L);
         

    %Filter the data:
    %Determine how much of the data to inlcude and update in Norm_Q >= .XXX
    Q = cumtrapz(PSD(L));
    Q_index_length = length(Q);
    Norm_Q = Q/Q(end); 
    idx_Q = find(Norm_Q >= .95);%find row index and then index out of freq(L) 
    idx_95 = idx_Q(1); %index of first frame of PSD(L) that exceeds 95% of the PSD AUC
    filter_Q_95 =  half_freq(idx_95); 
    filter_Q = 19.889; %Filter parameters from the merged participants data

    %figure 3:
    figure; grid on; hold on;
    plot(freq(L),PSD(L),'k');
    xline(filter_Q,'c', 'LineWidth', 1);
    xlabel('f (Hz)')
    ylabel('Power')
    title('Power Spectral Density Plot');
    ylim([0,50]);
    set(gca, 'fontsize', 16); 

    %lowpass Butterworth filter at the frequency defined above
    [b,a] = butter(4,filter_Q/(Fs/2)); %4th order lowpass butterworth filter with a cutoff frequency of filter_Q Hz
    A_user_mag_filt = abs(filtfilt(b,a,A_user_mag)); %dual-pass filter

%down sample for video coding- comment out if this is not being used for
%validation. 
    A_user_mag_filt = downsample(A_user_mag_filt,4);
    time_zero = downsample(time_zero,4);


    %Calculate trapezoidal numerical integration of user acceleration
    %magnitude %filtered without threshold applied
    integrate_A_user_mag_modified = trapz(time_zero,A_user_mag_filt);
    
    %figure 5: Plot Resultant User Acceleration
    figure; 
    grid on; hold on;
    h4u = plot(time_zero,A_user_mag_filt, 'DisplayName', babyID);
    title('Filtered Resultant Acceleration Magnitude');
    legend(h4u,'A_{user filtered}'); %{_makes it subscript} don't need brackets for 1 (duncan) 
    xlabel('Time (sec)');
    ylabel('Acceleration (m/s/s)'); 
    %xlim([0,1900]);
    set(gca, 'fontsize', 16);
    
%ndex active and sedentary time ID'd from video from sensor
%data and then plot as histograms to find threshold. 
    
    active_time_video_cell = VideoCode_filename_raw(strcmp(VideoCode_filename_raw(:,8),'a'),:); % in cell array
    sedentary_time_video_cell = VideoCode_filename_raw(strcmp(VideoCode_filename_raw(:,8),'s'),:);
    picked_up_removed_time_video_cell = VideoCode_filename_raw(strcmp(VideoCode_filename_raw(:,8),'r'),:);

    active_time_video = cell2mat(active_time_video_cell(1:end, 1:end-1)); %converted to double array
    sedentary_time_video = cell2mat(sedentary_time_video_cell(1:end, 1:end-1));
    picked_up_removed_time_video = cell2mat(picked_up_removed_time_video_cell(1:end, 1:end-1));

    %create active time windows
    %preallocate
    L = length(active_time_video); 
    start_active_time = nan([1, L]); 
    stop_active_time = nan([1, L]); 
    video_active_time_windows = nan([1, L]); 
    % vectorize
    start_active_time = active_time_video(1:length(active_time_video), 5) - video_time_in_place;
    stop_active_time = active_time_video(1:length(active_time_video),6) - video_time_in_place;
    video_active_time_windows = stop_active_time - start_active_time;
    sum_video_active_time = sum(video_active_time_windows)
 
%create sedentary time windows
    L = length(sedentary_time_video ); 
    start_sedentary_time = nan([1, L]); 
    stop_sedentary_time = nan([1, L]); 
    video_sedentary_time_windows = nan([1, L]); 
    % vectorize
    start_sedentary_time = sedentary_time_video(1:length(sedentary_time_video), 5) - video_time_in_place;
    stop_sedentary_time = sedentary_time_video(1:length(sedentary_time_video),6) - video_time_in_place;
    video_sedentary_time_windows = stop_sedentary_time - start_sedentary_time;
    sum_video_sedentary_time = sum(video_sedentary_time_windows)

    percent_sedentary = sum_video_sedentary_time/(sum_video_sedentary_time+sum_video_active_time)*100
    percent_active = sum_video_active_time/(sum_video_sedentary_time+sum_video_active_time)*100

    %create picked up/removed time windows
    %preallocate
    % size_array = size(picked_up_removed_time_video);
    % start_picked_up_removed_time = nan([1, size_array(1)]); 
    % stop_picked_up_removed_time = nan([1, size_array(1)]); 
    % picked_up_removed_time_windows = nan([1, size_array(1)]);
    % % vectorize
    % start_picked_up_removed_time = picked_up_removed_time_video(1:size_array(i), 5) - video_time_in_place;
    % stop_picked_up_removed_time = picked_up_removed_time_video(1:size_array(i),6) - video_time_in_place;
    % picked_up_removed_time_windows = stop_picked_up_removed_time - start_picked_up_removed_time;

    %% Index from sensor data the video ID'd active time
    %preallocate
    L = length(start_active_time); 
    active_data_start_idx = nan([1, L]); %add active here 
    active_data_end_idx = nan([1, L]);   
    for m=1:(length(start_active_time))  
    active_temp = find(time_zero>=start_active_time(m)&time_zero<=stop_active_time(m));  %This will give you the indices of the time array that fall between the video_code timepoints
    active_data_start_idx(m) = active_temp(1);
    active_data_end_idx(m) = active_temp(end);
    end 
    
    %preallocate
    time_video_active = NaN(length(time_zero),1);%initializing variables
    active_video_A_user_mag_filt = NaN(length(A_user_mag_filt),1);

   time_video_active_sedentary_annotate=zeros(length(active_video_A_user_mag_filt),1);
   ones_annotate=ones(length(active_video_A_user_mag_filt),1);

    for n= 1:length(active_data_start_idx)
        time_video_active_sedentary_annotate(active_data_start_idx(n):active_data_end_idx(n))= ones_annotate(active_data_start_idx(n):active_data_end_idx(n)); 
        time_video_active(active_data_start_idx(n):active_data_end_idx(n))= time_zero(active_data_start_idx(n):active_data_end_idx(n)); 
        active_video_A_user_mag_filt(active_data_start_idx(n):active_data_end_idx(n)) = A_user_mag_filt(active_data_start_idx(n):active_data_end_idx(n));
    end

    sum_nan_active = sum(isnan(active_video_A_user_mag_filt));

    %% Index from sensor data the video ID'd sedentary time
    %preallocate
    L = length(start_sedentary_time); 
    sedentary_data_start_idx = nan([1, L]); 
    sedentary_data_end_idx = nan([1, L]);   
    for m=1:(length(start_sedentary_time))  
    sedentary_temp = find(time_zero>=start_sedentary_time(m)&time_zero<=stop_sedentary_time(m));  %This will give you the indices of the time array that fall between the video_code timepoints
    sedentary_data_start_idx(m) = sedentary_temp(1);
    sedentary_data_end_idx(m) = sedentary_temp(end);
    end 
    
    %preallocate
    time_video_sedentary = NaN(length(time_zero),1);%initializing variables
    sedentary_video_A_user_mag_filt = NaN(length(A_user_mag_filt),1);
    for n= 1:length(sedentary_data_start_idx)
        time_video_sedentary(sedentary_data_start_idx(n):sedentary_data_end_idx(n))= time_zero(sedentary_data_start_idx(n):sedentary_data_end_idx(n)); 
        sedentary_video_A_user_mag_filt(sedentary_data_start_idx(n):sedentary_data_end_idx(n)) = A_user_mag_filt(sedentary_data_start_idx(n):sedentary_data_end_idx(n));
    end

sum_nan_sedentary = sum(isnan(sedentary_video_A_user_mag_filt));
 check = [active_video_A_user_mag_filt sedentary_video_A_user_mag_filt];

   mycolors = [0 0.4470 0.7410];
    %plot video active data
    %figure 7
    activetimefigure = figure;
    plot(time_zero, active_video_A_user_mag_filt); 
    hold on; grid on;
    ax = gca;
    % Set x and y font sizes.
    ax.XAxis.FontSize = 14;
    ax.YAxis.FontSize = 14;
    ylabel('Acceleration (m/s/s)', 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold')
    xlabel('Time (sec)', 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold')
    title({'Filtered Acceleration Magnitude of Active Time', 'Identified with Video Coding'},'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold');
    xlim([0,1900]);
    ax.ColorOrder = mycolors;
    saveas(activetimefigure, [babyID '_' 'FilteredActiveTime.jpg']);

    mycolors = [0.8500 0.3250 0.0980];

    %figure 7
    sedentarytimefigure = figure;
    hold on; grid on;
    plot(time_zero, sedentary_video_A_user_mag_filt); 
    ax = gca;
    ax.YAxis.Exponent = 0;
    ax.YAxis.TickLabelFormat = '%.0f';
    % Set x and y font sizes.
    ax.XAxis.FontSize = 14;
    ax.YAxis.FontSize = 14;
    ylabel('Acceleration (m/s/s)', 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold')
    xlabel('Time (sec)', 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold')
    title({'Filtered Acceleration Magnitude of Sedentary Time', 'Identified with Video Coding'},'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold');
    xlim([0,1900]);
    ax.ColorOrder = mycolors;
    saveas(sedentarytimefigure, [babyID '_' 'FilteredSedentaryTime.jpg']);



%Histogram of active and sedentary times

hist1_Figure = figure;
hold on; grid on;
hist1 = histogram(sedentary_video_A_user_mag_filt);
hist2 = histogram(active_video_A_user_mag_filt);
title({'Histogram of Active and Sedentary Time', 'Identified with Behavioral Coding'}, 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold');
legend([hist1,hist2],'Filtered Acceleration Magnitude of Sedentary Time','Filtered Acceleration Magnitude of Active Time', 'FontSize', 12, 'FontName','Ariel'); 
xlim([0,10]);
%ylim([0, 80000]);
hist1.BinWidth = 0.07;
hist2.BinWidth = 0.07;
% Get handle to current axes.
ax = gca;
ax.YAxis.Exponent = 0;
ax.YAxis.TickLabelFormat = '%.0f';
% Set x and y font sizes.
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
ylabel('Counts', 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold')
xlabel('Acceleration (m/s/s)', 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold')
saveas(hist1_Figure, [babyID '_' 'Histogram1.jpg']);


[X,Y,T,AUC,OPTROCPT] = perfcurve(time_video_active_sedentary_annotate, A_user_mag_filt,1);

ROC = figure; hold on; 
plot(X,Y,'linewidth',1);
plot(xlim, ylim, '--k','linewidth',1);
ax = gca;
% Set x and y font sizes.
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
plot(OPTROCPT(1),OPTROCPT(2),'.r', 'MarkerSize',30);
xlabel('False positive rate', 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold') 
ylabel('True positive rate', 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold')
title('Receiver Operating Characteristic Curve', 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold')
saveas(ROC, [babyID '_' 'ROC.jpg']);

Optimal_threshold = T((X==OPTROCPT(1))&(Y==OPTROCPT(2)))

%Threshold = Optimal_threshold;

Threshold =  .417; %
  
%% filtered user accleration with threshold cutoff for Active Time
      A_user_mag_filt_threshold = NaN(length(A_user_mag_filt),1); %initializing variable
      active_time_threshold = zeros(length(time_zero),1); %probably remove
      active_accel_threshold = zeros(length(time_zero),1);
      sedentary_time_threshold = zeros(length(time_zero),1); %probably remove
      sedentary_accel_threshold = NaN(length(time_zero),1);
      active_time_threshold_annotate = NaN(length(A_user_mag_filt),1);
      sedentary_time_threshold_annotate = NaN(length(A_user_mag_filt),1);

    %Apply threshold this method removes spikes below the threshold (fill
    %in zero for time and acceleration magnitude if below the threshold to
    %ID active time using sensor threshold (active=1, sedentary=0) %need to
    % apply filter ONLY if 2 cells remain
    %add a cell at the start 

    for k = 2:(length(A_user_mag_filt))-1
        if A_user_mag_filt(k)> Threshold & A_user_mag_filt(k+1)> Threshold;
            active_time_threshold_annotate(k)=1;
           active_time_threshold(k) = time_zero(k);
           active_accel_threshold(k) = A_user_mag_filt(k);
        elseif A_user_mag_filt(k)> Threshold & A_user_mag_filt(k-1)> Threshold;
            active_time_threshold_annotate(k)=1;
           active_time_threshold(k) = time_zero(k);
           active_accel_threshold(k) = A_user_mag_filt(k);
        else 
            active_time_threshold_annotate(k)=0;
           active_time_threshold(k) = NaN;  
           active_accel_threshold(k) = NaN; 
        end
    end  

    %fills in the first cell which is skipped in the previous for-loop
if A_user_mag_filt(1)>Threshold && A_user_mag_filt(2)>Threshold
           active_time_threshold_annotate(1)=1;
           active_time_threshold(1) = time_zero(1);
          active_accel_threshold(1) = A_user_mag_filt(1);
else
           active_time_threshold_annotate(1)=0;
           active_time_threshold(1) = 0;
           active_accel_threshold(1) = NaN; 
end

if A_user_mag_filt(end)>Threshold && A_user_mag_filt(end-1)>Threshold
           active_time_threshold_annotate(end)=1;
           active_time_threshold(end) = time_zero(end);
          active_accel_threshold(end) = A_user_mag_filt(end);
else
           active_time_threshold_annotate(end)=0;
           active_time_threshold(end) = 0;
           active_accel_threshold(end) = NaN; 
end

%Remove single spikes of active time from the sedentary accelerations to
%finish removing any single spikes of active time either within active or
%sedentary

for i = 1:length(active_accel_threshold)
if isnan(active_accel_threshold(i)) 
    sedentary_accel_threshold(i) = A_user_mag_filt(i);
    sedentary_time_threshold_annotate(i) = 1;
else
    sedentary_time_threshold_annotate(i) = 0;
end
end

for i = 1:length(active_accel_threshold)
if sedentary_accel_threshold(i)> Threshold
    sedentary_accel_threshold(i) = NaN;
    sedentary_time_threshold_annotate(i) = 0;
end
end

if A_user_mag_filt(1)>Threshold && A_user_mag_filt(2)>Threshold
           sedentary_time_threshold_annotate(1)=1;
           sedentary_time_threshold(1) = time_zero(1);
          sedentary_accel_threshold(1) = A_user_mag_filt(1);
else
           sedentary_time_threshold_annotate(1)=0;
           sedentary_time_threshold(1) = 0;
           sedentary_accel_threshold(1) = NaN;
end

if A_user_mag_filt(end)>Threshold && A_user_mag_filt(end-1)>Threshold
           sedentary_time_threshold_annotate(end)=1;
           sedentary_time_threshold(end) = time_zero(end);
          sedentary_accel_threshold(end) = A_user_mag_filt(end);
else
           sedentary_time_threshold_annotate(end)=0;
           sedentary_time_threshold(end) = 0;
           sedentary_accel_threshold(end) = NaN; 
end

 %Now we have sedentary and active annotations and acceleration and time
 %with the threshold applied and active bouts of 1 spike only removed from
 %both active and sedentary time. 

HistogramFigure2 = figure;
hold on; grid on;
hist1 = histogram(sedentary_accel_threshold);
hist2 = histogram(active_accel_threshold);
title({'Histogram of Active and Sedentary Time', 'Identified with Threshold'}, 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold');
ax = gca;
xlim([0,10]);
hist1.BinWidth = 0.07;
hist2.BinWidth = 0.07;
ax.YAxis.Exponent = 0;
ax.YAxis.TickLabelFormat = '%.0f';
% Set x and y font sizes.
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
legend([hist1,hist2],{'Filtered Acceleration Magnitude of Sedentary Time','Filtered Acceleration Magnitude of Active Time'}, 'FontSize', 12, 'FontName','Ariel'); 
ylabel('Counts', 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold')
xlabel('Acceleration (m/s/s)', 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold')
saveas(HistogramFigure2, [babyID '_' 'Histogram2.jpg']);


Sensor_Sed_Video_Sed = zeros(length(active_time_threshold_annotate),1);
Sensor_Active_Video_Active = zeros(length(active_time_threshold_annotate),1);
Sensor_Sed_Video_Active = zeros(length(active_time_threshold_annotate),1);
Sensor_Active_Video_Sed = zeros(length(active_time_threshold_annotate),1);
Sensor_Active_Video_Sed_accel = NaN(length(A_user_mag_filt),1);
Sensor_Sed_Video_Active_accel = NaN(length(A_user_mag_filt),1);
Sensor_Active_Video_Active_accel = NaN(length(A_user_mag_filt),1);
Sensor_Sed_Video_Sed_accel = NaN(length(A_user_mag_filt),1);

 for i = 1:length(active_time_threshold_annotate)
  if active_time_threshold_annotate(i) == 1 & time_video_active_sedentary_annotate(i) == 1
     Sensor_Active_Video_Active(i)=1;
     Sensor_Active_Video_Active_accel(i)=A_user_mag_filt(i);
  else
      Sensor_Active_Video_Active(i)=0;
  end
  if sedentary_time_threshold_annotate(i) == 1 & time_video_active_sedentary_annotate(i) == 0
     Sensor_Sed_Video_Sed(i)=1;
     Sensor_Sed_Video_Sed_accel(i)=A_user_mag_filt(i);
  else
      Sensor_Sed_Video_Sed(i)=0;
  end
  if active_time_threshold_annotate(i) == 1 & time_video_active_sedentary_annotate(i) == 0
     Sensor_Active_Video_Sed(i)=1;
     Sensor_Active_Video_Sed_accel(i)=A_user_mag_filt(i);
  else
      Sensor_Active_Video_Sed(i)=0;
  end
  if sedentary_time_threshold_annotate(i) == 1 & time_video_active_sedentary_annotate(i) == 1
     Sensor_Sed_Video_Active(i)=1;
     Sensor_Sed_Video_Active_accel(i)=A_user_mag_filt(i);
  else
      Sensor_Sed_Video_Active(i)=0;
  end
 end

Sum_Sensor_Sed_Video_Sed = sum(Sensor_Sed_Video_Sed);
Sum_Sensor_Active_Video_Active = sum(Sensor_Active_Video_Active);
Sum_Sensor_Sed_Video_Active = sum(Sensor_Sed_Video_Active);
Sum_Sensor_Active_Video_Sed = sum(Sensor_Active_Video_Sed);
total_sum= Sum_Sensor_Sed_Video_Sed+Sum_Sensor_Active_Video_Active+Sum_Sensor_Sed_Video_Active+Sum_Sensor_Active_Video_Sed;
Perc_Sensor_Sed_Video_Sed = Sum_Sensor_Sed_Video_Sed/(total_sum)*100;
Perc_Sensor_Active_Video_Active = Sum_Sensor_Active_Video_Active/(total_sum)*100;
Perc_Sensor_Sed_Video_Active = Sum_Sensor_Sed_Video_Active/(total_sum)*100;
Perc_Sensor_Active_Video_Sed = Sum_Sensor_Active_Video_Sed/(total_sum)*100;

sum_true_pos_true_neg = Perc_Sensor_Sed_Video_Sed + Perc_Sensor_Active_Video_Active
sum_perc = Perc_Sensor_Sed_Video_Sed + Perc_Sensor_Active_Video_Active + Perc_Sensor_Sed_Video_Active + Perc_Sensor_Active_Video_Sed;
Perc_Sensor_Active = Perc_Sensor_Active_Video_Active + Perc_Sensor_Active_Video_Sed;
Perc_Sensor_Sedentary = Perc_Sensor_Sed_Video_Sed + Perc_Sensor_Sed_Video_Active;

    fig_Comparison = figure;
    g_1 = gca; 
    hold on; grid on; 
    testfig2 = plot(time_zero, A_user_mag_filt, 'DisplayName', babyID);
    testfig1 = plot(time_zero, active_accel_threshold, 'DisplayName', babyID);
    testfig3 = plot([time_zero(1),time_zero(length(time_zero))],[Threshold,Threshold],'-k', 'Linewidth', 0.5); %add horizontal line at upper threshold
    title({'Comparision of Video vs. Sensor Threshold', 'Identified Active Times'});
    %legend('-DynamicLegend')
    legend([testfig1,testfig2,testfig3],['A_{user filtered}' ' Sensor Threshold'],['A_{user filtered}' ' Without Threshold'], 'Threshold'); %updating legends note {_makes it subscript}
    xlabel('Time (sec)');
    ylabel('Acceleration (m/s/s)');
    %xlim([0,1900]);
    set(gca, 'fontsize', 16);
    saveas(fig_Comparison, [babyID '_' 'Comparison.jpg']);

 %adds tolerance
    % 5 frame tolerance (.125 seconds) (4 frames in each direction)
    ext_tol = zeros(1,5)'; % modify this based on the number of frames within the tolerance
    ext_time_video_active_sedentary_annotate = [ext_tol; time_video_active_sedentary_annotate; ext_tol]; %add 4 to beginning and end; needed for the for loop
    ext_sedentary_time_threshold_annotate = [ext_tol; sedentary_time_threshold_annotate; ext_tol];
    ext_active_time_threshold_annotate = [ext_tol; active_time_threshold_annotate; ext_tol];
    tolerance = 5; 
    tol = 1: length(tolerance); %should be 1x1
    tol_pos = zeros(1,tolerance -1)'; % 4 x 1
    tol_neg = zeros(1,tolerance-1)'; % 4 x 1
    % 

Sensor_Active_Video_Active_tolerance= NaN(1,length(Sensor_Active_Video_Active));
Sensor_Sedentary_Video_Sedentary_tolerance= NaN(1,length(Sensor_Active_Video_Active));
Sensor_Active_Video_Sedentary_tolerance= zeros(1,length(Sensor_Active_Video_Active));
Sensor_Sedentary_Video_Active_tolerance= zeros(1,length(Sensor_Active_Video_Active));


%Active-Active Tolerance
    for i = (1 + tolerance: length(time_video_active_sedentary_annotate) + tolerance)
        for ii = 1:(tolerance - 1) %subtract 1 because you want 3 additional row comparisons (same row + 3 more)
         if ext_time_video_active_sedentary_annotate(i) == 1 & ext_active_time_threshold_annotate(i) ==1
            tol(ii) = 1;
           elseif ext_time_video_active_sedentary_annotate(i) ==  1 & ext_active_time_threshold_annotate(i + ii) ==1
             tol_pos(ii) = 1; 
           elseif ext_time_video_active_sedentary_annotate(i) == 1 & ext_active_time_threshold_annotate(i - ii) ==1
             tol_neg(ii) = 1; 
         else
         tol(ii) = 0;
         tol_pos(ii) = 0;
         tol_neg(ii) = 0;
         true(i - tolerance) = 0;
         end
      
         if tol>=1
             Sensor_Active_Video_Active_tolerance(i - tolerance) = 1;
         elseif sum(tol_pos)>=1
             Sensor_Active_Video_Active_tolerance(i - tolerance) = 1;
         elseif sum(tol_neg)>=1
             Sensor_Active_Video_Active_tolerance(i - tolerance) = 1;  
         else
             Sensor_Active_Video_Active_tolerance(i - tolerance) = 0;
         end   
        end
    end

%Sedentary-Sedentary Tolerance
    for i = (1 + tolerance: length(time_video_active_sedentary_annotate) + tolerance)
        for ii = 1:(tolerance - 1) %subtract 1 because you want 3 additional row comparisons (same row + 3 more)
         if ext_time_video_active_sedentary_annotate(i) == 0 & ext_sedentary_time_threshold_annotate(i)==1
            tol(ii) = 1;
           elseif ext_time_video_active_sedentary_annotate(i) == 0 & ext_sedentary_time_threshold_annotate(i + ii) == 1
             tol_pos(ii) = 1; 
           elseif ext_time_video_active_sedentary_annotate(i) == 0 & ext_sedentary_time_threshold_annotate(i - ii) ==1
             tol_neg(ii) = 1; 
         else
         tol(ii) = 0;
         tol_pos(ii) = 0;
         tol_neg(ii) = 0;
         true(i - tolerance) = 0;
         end
      
         if tol>=1
             Sensor_Sedentary_Video_Sedentary_tolerance(i - tolerance) = 1;
         elseif sum(tol_pos)>=1
             Sensor_Sedentary_Video_Sedentary_tolerance(i - tolerance) = 1;
         elseif sum(tol_neg)>=1
             Sensor_Sedentary_Video_Sedentary_tolerance(i - tolerance) = 1;  
         else
             Sensor_Sedentary_Video_Sedentary_tolerance(i - tolerance) = 0;
         end 

        end
    end

    for i = 1:length(Sensor_Sedentary_Video_Sedentary_tolerance)
if Sensor_Sedentary_Video_Sedentary_tolerance(i) ==0 & Sensor_Active_Video_Active_tolerance(i)==0
    if Sensor_Active_Video_Sed(i) ==1
       Sensor_Active_Video_Sedentary_tolerance(i) =1;
    else
        Sensor_Active_Video_Sedentary_tolerance(i) =0;
    end
    if Sensor_Sed_Video_Active(i) ==1
        Sensor_Sedentary_Video_Active_tolerance(i)=1;
    else
        Sensor_Sedentary_Video_Active_tolerance(i)=0;
    end
end
    end
    %sum sed-sed and active-active; add in the false pos and false neg and
    %make percents here. 

sum_Sensor_Sedentary_Video_Sedentary_tolerance = sum(Sensor_Sedentary_Video_Sedentary_tolerance)
sum_Sensor_Active_Video_Active_tolerance = sum(Sensor_Active_Video_Active_tolerance)
sum_Sensor_Sedentary_Video_Active_tolerance = sum(Sensor_Sedentary_Video_Active_tolerance)
sum_Sensor_Active_Video_Sedentary_tolerance = sum(Sensor_Active_Video_Sedentary_tolerance)
total_tolerance= sum_Sensor_Sedentary_Video_Sedentary_tolerance + sum_Sensor_Active_Video_Active_tolerance+ sum_Sensor_Sedentary_Video_Active_tolerance+sum_Sensor_Active_Video_Sedentary_tolerance
perc_Sensor_Sedentary_Video_Sedentary_tolerance = (sum_Sensor_Sedentary_Video_Sedentary_tolerance/ total_tolerance) *100
perc_Sensor_Active_Video_Active_tolerance = (sum_Sensor_Active_Video_Active_tolerance/total_tolerance) * 100
perc_Sensor_Active_Video_Sedentary_tolerance = (sum_Sensor_Active_Video_Sedentary_tolerance/ total_tolerance) *100
perc_Sensor_Sedentary_Video_Active_tolerance = (sum_Sensor_Sedentary_Video_Active_tolerance/total_tolerance) * 100
total_perc_tolerance= perc_Sensor_Sedentary_Video_Sedentary_tolerance + perc_Sensor_Active_Video_Active_tolerance+ perc_Sensor_Sedentary_Video_Active_tolerance+perc_Sensor_Active_Video_Sedentary_tolerance
sum_true_tolerance = perc_Sensor_Active_Video_Active_tolerance + perc_Sensor_Sedentary_Video_Sedentary_tolerance

sensor_active_tol = perc_Sensor_Active_Video_Active_tolerance + perc_Sensor_Active_Video_Sedentary_tolerance
sensor_sedentary_tol = perc_Sensor_Sedentary_Video_Active_tolerance + perc_Sensor_Sedentary_Video_Sedentary_tolerance


%active_time_threshold needs to have zero instead of NaN
active_time_threshold(isnan(active_time_threshold))=0;

%% ID index of start and stop active time for master table calculations
Time_interval = time_zero(2)-time_zero(1);
Diff = diff(active_time_threshold_annotate(1:end)); %difference between cell 2 and cell 1, cell 3 and cell 2, etc. This is 1 cell less than full array
idx_stop_active_sensor = find(Diff < 0); %This is the last "cell" of active time for indexing
idx_start_active_sensor = find(Diff>0)+1; %This is the first "cell" of active time for indexing

%check if the first cell or last cell should be active time or sedentary
%time and add that to the idx

    if active_time_threshold_annotate(1) ==1
        idx_start_active_sensor = [1; idx_start_active_sensor];
    end

    if active_time_threshold_annotate(end)==1
        idx_stop_active_sensor = [idx_stop_active_sensor; length(time_zero)];
    end

% preallocate
    L = length(idx_start_active_sensor);
    active_Sensor_timebouts = 1:L;
    Duration_Sensor_Active_full = nan([1, L]); 
    Mean_A_user_mag_filt_Sensor_active = nan([1, L]); 
    STD_A_user_mag_filt_Sensor_active = nan([1, L]);
 
for p = 1: length(idx_start_active_sensor)
    active_Sensor_time_start = time_zero(idx_start_active_sensor(p));
    active_Sensor_time_stop = time_zero(idx_stop_active_sensor(p));
    idx_Sensor_active_time = find(time_zero >= active_Sensor_time_start & time_zero<= active_Sensor_time_stop);%find row index and then index out of time
    active_Sensor_time = time_zero(idx_Sensor_active_time);
    active_Sensor_time_acceleration_mag_filt = A_user_mag_filt(idx_Sensor_active_time);
    active_Sensor_timebouts(p) = p;
    duration_Sensor_Active = length(active_Sensor_time) * Time_interval;
    Duration_Sensor_Active_full(p) = duration_Sensor_Active;
    Mean_A_user_mag_filt_Sensor_active(p) = mean(active_Sensor_time_acceleration_mag_filt);
    STD_A_user_mag_filt_Sensor_active(p) = std(active_Sensor_time_acceleration_mag_filt); 
end
% 
    count_Sensor_active_time_bouts = length(active_Sensor_timebouts);
    Sum_Sensor_Active_Time_Duration = sum(Duration_Sensor_Active_full);
    Min_Sensor_Active_Time_Duration = min(Duration_Sensor_Active_full);
    Max_Sensor_Active_Time_Duration = max(Duration_Sensor_Active_full);

    % preallocate
    L = length(active_data_start_idx); 
    active_Video_timebouts = 1:L;
    Duration_Video_Active_full = nan([1, L]); 
    Mean_A_user_mag_filt_active = nan([1, L]); 
    STD_A_user_mag_filt_active = nan([1, L]); 
    for p = 1: length(active_data_start_idx)
        active_Video_time_start = time_zero(active_data_start_idx(p));
        active_Video_time_stop = time_zero(active_data_start_idx(p));
        idx_active_Video_time = find(time_zero >= active_Video_time_start & time_zero<= active_Video_time_stop);%find row index and then index out of time
        active_Video_time = time_zero(idx_active_Video_time);
        active_Video_time_acceleration_mag_filt = A_user_mag_filt(idx_active_Video_time);
        active_Video_timebouts(p) = p;
        count_Video_active_time_bouts = length(active_Video_timebouts);
        duration_Video_Active = length(active_Video_time) * Time_interval;
        Duration_Video_Active_full(p) = duration_Video_Active;
        Mean_A_user_mag_filt_active(p) = mean(active_Video_time_acceleration_mag_filt);
        STD_A_user_mag_filt_active(p) = std(active_Video_time_acceleration_mag_filt); 

        % Makes a figure for each active movement bout; turn the visibility of the
        % figure off, and save once you are done making it, and then close the
        % figure, all inside the loop. This will slow down the program. Use
        % if needed. 
        
        %hf2 = figure('visible', 'off');
        %plot(active_time, active_time_acceleration_mag_filt, 'b'); 
        %hold on; 
        %plot([active_time(1),active_time(length(active_time))],[Mean_A_user_mag_filt_active(p),Mean_A_user_mag_filt_active(p)],'-k', 'Linewidth', 1);
        %xlabel('Time(s)')
        %ylabel('Acceleration (m/s/s)')
        %title('Filtered User Acceleration Active Time from Video');
        %saveas(hf2,[pwd, '/Datasets/', babyID, '/Graphs/' hf1SaveName]); 
     end

% %Summarize Duration Video for MasterTable
Sum_Video_Active_Time_Duration = sum(Duration_Video_Active_full);
Min_Video_Active_Time_Duration = min(Duration_Video_Active_full);
Max_Video_Active_Time_Duration = max(Duration_Video_Active_full);
    

  %% Apply threshold to integrated acceleration
    %Return time_threshold to NAN from 0 except for first cell which may be
    %zero if active time started at zero
    for i= 2:(length(active_time_threshold))
        if active_time_threshold(i) ==0
        active_time_threshold(i) = nan;
        end
    end

    if active_time_threshold_annotate(1)==1
        active_time_threshold(1) = 0;
    else active_time_threshold(1) = nan;
    end

    hf10=figure; 
    g = gca;
    hold on; grid on;
    h10u = plot(time_zero,active_accel_threshold, 'DisplayName', babyID);
    h7u = plot([time_zero(1),time_zero(length(time_zero))],[Threshold,Threshold],'-k', 'Linewidth', 0.5); %add horizontal line at upper threshold
    %plot([time_zero(1),time_zero(length(time_zero))],[Mean_A_user_mag_filt,Mean_A_user_mag_filt],'-k', 'Linewidth', 1);
    title({'Filtered Acceleration Magnitude of Active Time', 'Identified with Sensor Threshold'});
    legend([h10u, h7u],'A_{user filtered} ', 'Threshold'); %updating legends note {_makes it subscript}
    xlabel('Time (sec)');
    ylabel('Acceleration (m/s/s)');
    xlim([0,1900]);
    %ylim([0,10]);
    set(gca, 'fontsize', 16);

    yfigure = ones(1,length(time_video_active));
    yfigure2 = 1.2*yfigure;

    %Testing Figure Active time
    epochactivetimefigure = figure;
    hold on; grid on;
    e1 = plot(time_video_active, yfigure2,'Linewidth', 40, 'Color', [0.4660 0.6740 0.1880]); 
    e2 = plot(active_time_threshold, yfigure,'Linewidth', 40, 'Color', [0.4940 0.1840 0.5560]);
    ax = gca;
    % Set x and y font sizes.
    ax.XAxis.FontSize = 14;
    ax.YAxis.FontSize = 14;
    set(gca, 'YTickLabel', []);
    %ylabel('Acceleration (m/s/s)', 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold')
    xlabel('Time (sec)', 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold')
    title({'Active Time Epochs'},'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold');
    %xlim([1000,1200]);
    ylim([.75,1.75]);
    legend([e1, e2],'Behavioral Coding Active Time', 'Sensor Active Time'); %updating legends note {_makes it subscript}
    set(gca, 'fontsize', 16);
    %ax.ColorOrder = mycolors;
    %saveas(epochactivetimefigure, [babyID '_' 'ActiveTime.jpg']);
    

  %% Apply threshold to integrated acceleration
%Calcuate user acceleration with threshold applied, Trapz funciton takes integral- (area under
%%curve) %use indices of active time to ID time 1/accel 1 and time2/accel 2 and then
%store in new matrix and sum

cumtrap_store = zeros(length(idx_start_active_sensor),1);

for i = 1: length(idx_start_active_sensor)
cumtrap_store(i) = trapz(active_time_threshold(idx_start_active_sensor(i):(idx_stop_active_sensor(i))), active_accel_threshold(idx_start_active_sensor(i):(idx_stop_active_sensor(i))));
end

integral_sum_threshold = sum(cumtrap_store);

    %% write file details and start and stop time to the table
    masterTable.FileName(ii) = cellstr(currentFile);
    masterTable.BabyID(ii) = cellstr(babyID);
    %masterTable.StartTime(ii) = t_start_input_file;
    %masterTable.StopTime(ii) = t_stop_input_file;
    masterTable.Duration(ii) = time_zero(end)-time_zero(1);
    masterTable.DurationMins(ii) = (time_zero(end)-time_zero(1)) /60 ;% column of duration
    masterTable.UserStart(ii) = cellstr(time_start_string);
    masterTable.UserStop(ii) = cellstr(time_end_string);
    masterTable.totalAcceleration(ii) = integrate_A_user_mag_modified;% column of total Accerleration
    masterTable.timeNormalizedAcc(ii) = integrate_A_user_mag_modified/(time(end)-time(1));% column of time normalized acceleration
    masterTable.timeNormalizedAccMins(ii) = integrate_A_user_mag_modified/ masterTable.DurationMins(ii) ;
    masterTable.thresholdTotalAcceleration(ii) = integral_sum_threshold;% column of total Accerleration
    masterTable.thresholdTimeNormalizedAcc(ii) = integral_sum_threshold/(time(end)-time(1));% column of time normalized acceleration
    masterTable.thresholdTimeNormalizedAccMins(ii) = integral_sum_threshold/ masterTable.DurationMins(ii) ;    
    masterTable.Sum_Sensor_Active_Time_Duration(ii) = Sum_Sensor_Active_Time_Duration;% column of Total Duration of Active time 
    masterTable.Min_Sensor_Active_Time_Duration(ii) = Min_Sensor_Active_Time_Duration;% column of Minimal Duration of Active time
    masterTable.Max_Sensor_Active_Time_Duration(ii) = Max_Sensor_Active_Time_Duration ;% column of Maximum Duration of Active time
    masterTable.count_Sensor_active_time_bouts(ii) = count_Sensor_active_time_bouts;% column of numober/ count of Actime time bouts
    masterTable.Sum_Sensor_NONActive_Time_Duration(ii) = (time_zero(end)-time_zero(1)) - Sum_Sensor_Active_Time_Duration;% column of Total Duration of Active time 
    masterTable.TruePositive(ii)= sum_Sensor_Active_Video_Active_tolerance;% 
    masterTable.TrueNegative(ii)= sum_Sensor_Sedentary_Video_Sedentary_tolerance;% 
    masterTable.FalsePositive(ii)= sum_Sensor_Active_Video_Sedentary_tolerance;% 
    masterTable.FalseNegative(ii)= sum_Sensor_Sedentary_Video_Active_tolerance;% 
    masterTable.TruePositive_Perc(ii)= Perc_Sensor_Active_Video_Active;% 
    masterTable.TrueNegative_Perc(ii)= Perc_Sensor_Sed_Video_Sed;% 
    masterTable.FalsePositive_Perc(ii)= Perc_Sensor_Active_Video_Sed;% 
    masterTable.FalseNegative_Perc(ii)= Perc_Sensor_Sed_Video_Active;% 
    masterTable.TruePositive_Perc_tol(ii)= perc_Sensor_Active_Video_Active_tolerance;% 
    masterTable.TrueNegative_Perc_tol(ii)= perc_Sensor_Sedentary_Video_Sedentary_tolerance;% 
    masterTable.FalsePositive_Perc_tol(ii)= perc_Sensor_Active_Video_Sedentary_tolerance;% 
    masterTable.FalseNegative_Perc_tol(ii)= perc_Sensor_Sedentary_Video_Active_tolerance;% 
    masterTable.OptimalThreshold(ii)= Optimal_threshold;
    masterTable.AUC(ii)= AUC;
end 



%%
% write the table to an output file
writetable(masterTable, saveFileName)
