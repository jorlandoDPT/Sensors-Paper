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

% This code requires an input file named: Timestamps.xls with 3 columns:
% filename, startTime, stopTime
%This code will create an output file named: outputData.xlsx

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
% saveFileName = 'Datasets/outputData.xlsx';
saveFileName = 'outputData.xlsx';
% Read the file with the time stamps in it
timingFile = 'Timestamps.xlsx';
%timingFile = 'Timestamps_1_file_each.xlsx';
timingData = importTimingfile('Timestamps.xlsx','Sheet1','A1:C355');%change file here too
%timingData = importTimingfile('Timestamps_1_file_each.xlsx','Sheet1','A1:C355');%change file here too
%% folder with datsasets in it
% put all the subfolders into a folder named Datasets and add it to this
% folder 
%example: 
dataPath = 'Users/juliemorlando/Desktop/iMove_MatLab/Datasets/'; %(Edit this based on the computer and the user) 
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

% Corresponds to the row of the timestamps.xlsx file. 
%Can update this based on the row. I suggest batches of 15-20 rows at a
%time. Note the output file will re-write when you hit "run" so you will
%need to copy and paste the output data to a new document when running
%smaller batches. 

start = 77; %update as needed
endfile = 77; %update as needed
%endfile = length(dataDirectoryNames); %update as needed

figure('visible', 'on', 'position', [300 300 1200 800]);
mainFig = gcf();
mainAx = gca();
tile = tiledlayout(mainFig, 3,3); 


% Iterate through each folder
for ii = start:endfile  %user updates
    
    % Set up file name, find the participant ID from the file name and use
    % the particpant ID. Read in the H5 file. 
    
    % Current File name
    currentFile = timingData{ii,1};
    % Get the participant id and thus the required folder
    startInd = strfind(currentFile, 'EM0');
    babyID = currentFile(startInd:startInd+4);
    babyFile = [dataPath, babyID, '/', currentFile];  
    caseID = hdf5read(currentFile, '/', 'CaseIdList');
    groupName = caseID.data;
    time = double(hdf5read(currentFile, [groupName '/Time']));% time has been converted to double since hdf5 extracts it as uint64
    A_total = hdf5read(currentFile, [groupName '/Calibrated/Accelerometers'])';
    q = hdf5read(currentFile, [groupName '/Calibrated/Orientation'])';
    
    %% Load time data and print human readable date and time to screen
    time = time./1e6; % [sec] Load time data and convert from milliseconds to seconds
    time_start_string = epoch2datestr(time(1),  time_zone_offset_hours); % Convert start time to human readable date and time
    time_end_string   = epoch2datestr(time(end),time_zone_offset_hours); % Convert end time to human readable date and time
    
    %synch sensor and video data
   
    %used with video data for EM039
    video_time_removal = 1860; %in sec
    sensor_time_removal = 1945; %in sec
    video_time_start = 0; %in sec
    video_time_in_place = 39; %in sec
    sensor_time_in_place = 124; %in sec
    video_sensor_difference = sensor_time_in_place - video_time_in_place; %(85 sec)
    
    %Check for duration
    duration_video = video_time_removal - video_time_in_place;
    duration_sensor_check =  sensor_time_removal - sensor_time_in_place;
    
    %Convert time to seconds after start
    time = time - time(1);
    t_start_input_file = timingData{ii,2};
    t_stop_input_file = timingData{ii,3};
    
    %Trim data based on video code
    ix_time_new = time >= sensor_time_in_place & time <= sensor_time_removal;
    time = time(ix_time_new,:);
        
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
    hf1=figure; hold on; grid on;
    h1x = plot(time_zero,A_total(:,1),'-r');
    h1y = plot(time_zero,A_total(:,2),'-g');
    h1z = plot(time_zero,A_total(:,3),'-b');
    title('Total Acceleration Components');
    legend([h1x,h1y,h1z],'A_x','A_y','A_z');
    xlabel('Time (sec)');
    ylabel('Acceleration (m/s/s)');
    ylim([-30,30]);
    xlim([0,1900])
    set(gca, 'fontsize', 16);
    %saveas(hf1,[pwd, hf1SaveName]); %update as needed
    %saveas(hf1,[pwd, '/Datasets/', babyID, '/Graphs/' hf1SaveName]);  
   
    % overall figure 
    ax = nexttile(tile); 
    hold(ax, 'on'); grid(ax, 'on');
    h1x = plot(ax, time_zero,A_total(:,1),'-r');
    h1y = plot(ax, time_zero,A_total(:,2),'-g');
    h1z = plot(ax, time_zero,A_total(:,3),'-b');
    title(ax, 'Total Acceleration Components');
    legend(ax, [h1x,h1y,h1z],'A_x','A_y','A_z');
    xlabel(ax, 'Time (sec)');
    ylabel(ax, 'Acceleration (m/s/s)');
    ylim(ax, [-30,30]);
    xlim(ax, [0,1900]); 


    %% Calculate the total acceleration in North, West, and Up coordinate frame
    A_total_p = A_total; % Initialize array
    for jj = 1:length(A_total(:,1))
        A_total_p(jj,:) = RotateVector(A_total(jj,:),q(jj,:)); % Rotate gravity vector into sensor coordinate frame
    end
    
     %% Filter user_acceleration components raw data in N, W, Up (filter first, then get resultant)
         
    % Fourier transform (FFT): This will transform data from time domain
    % to frequency doamin
    Fs = 128; %sampling frequency/sampling rate (of the sensor) - A_total_p
    dt = 1/128; %change in time 
    t = time_zero; 
    n = length(time_zero); % should this be magnitude data or the raw data - use raw data NWU
    fhat = fft(A_total_p,n);
      
    %Power Spectral Density Analysis: Once data is in the frequency domain, we can do a power spectral density analysis to look power of the signal and decide
    %where to filter the data    
    PSD = fhat.*conj(fhat)/n; %power spectrum (power per freq)
    freq = 1/(dt*n)*(0:n);   %x-axis of freq in Hz
    L = 1:floor(n/2); %plot the first half of freqs
    half_freq = freq(L);
          
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
    xlim([0,1900]);
    set(gca, 'fontsize', 16);
    
    % overall figure 
    ax = nexttile(tile);
    grid(ax, 'on'); hold(ax, 'on');
    h2x = plot(ax, time_zero,A_total_p(:,1),'-r');
    h2y =plot(ax, time_zero,A_total_p(:,2),'-g');
    h2z =plot(ax, time_zero,A_total_p(:,3),'-b');
    legend(ax, [h2x,h2y,h2z],'A_x','A_y','A_z');
    xlabel(ax, 'Time (sec)')
    ylabel(ax, 'Acceleration (m/s/s)')
    title(ax, 'Raw Acceleration Components in North, West, Up');
    ylim(ax, [-30,30]);
    xlim(ax, [0,1900]);  
    
    %Filter the data:
    %Determine how much of the data to inlcude and update in Norm_Q >= .XXX
    Q = cumtrapz(PSD(L));
    Q_index_length = length(Q);
    Norm_Q = Q/Q(end); %find values that exceed .8 (Can work on this threshold)
    idx_Q = find(Norm_Q >= .75);%find row index and then index out of freq(L) 
    idx_75 = idx_Q(1); %index of first frame of PSD(L) that exceeds 75% of the PSD AUC
    filter_Q = half_freq(idx_75);
   
    %figure 3:
    figure; grid on; hold on;
    plot(freq(L),PSD(L),'k');
    xline(filter_Q,'c', 'LineWidth', 1);
    xlabel('f (Hz)')
    ylabel('Power')
    title('Power Spectral Density Plot');
    ylim([0,50]);
    set(gca, 'fontsize', 16); % here too if you want (duncan)
    
    %overall figure
    ax = nexttile(tile);
    grid(ax, 'on'); hold(ax, 'on');
    plot(ax, freq(L),PSD(L),'k');
    xline(ax, filter_Q,'c', 'LineWidth', 1);
    xlabel(ax, 'f (Hz)')
    ylabel(ax, 'Power')
    title(ax, 'Power Spectral Density Plot');
    ylim(ax, [0,50]);

    %Designing lowpass Butterworth filter at the frequency defined above
    [b,a] = butter(4,filter_Q/(Fs/2)); %4th order lowpass butterworth filter with a cutoff frequency of filter_Q Hz
        
    A_total_p_filt = filtfilt(b,a,A_total_p); %dual-pass filter
        
    %figure 4: filtered data in NWU for comparision
    figure;grid on; hold on;
    h3x = plot(time_zero,A_total_p_filt(:,1),'-r');
    h3y = plot(time_zero,A_total_p_filt(:,2),'-g');
    h3z = plot(time_zero,A_total_p_filt(:,3),'-b');
    xlabel('Time (sec)')
    ylabel('Acceleration (m/s/s)')
    title('Filtered Acceleration Components in North, West, Up');
    legend([h3x,h3y,h3z],'A_x','A_y','A_z'); 
    xlim([0,1900]);
    set(gca, 'fontsize', 16);
    
    %overall figure
    ax = nexttile(tile); 
    grid(ax, 'on'); hold(ax, 'on');
    h3x = plot(ax, time_zero,A_total_p_filt(:,1),'-r');
    h3y = plot(ax, time_zero,A_total_p_filt(:,2),'-g');
    h3z = plot(ax, time_zero,A_total_p_filt(:,3),'-b');
    xlabel(ax, 'Time (sec)')
    ylabel(ax, 'Acceleration (m/s/s)')
    title(ax, 'Filtered Acceleration Components in North, West, Up');
    legend(ax, [h3x,h3y,h3z],'A_x','A_y','A_z'); 
    xlim(ax, [0,1900]);
          
    %Calculate user acceleration from filtered data
    
    %Find the index where the median is present
    medI = find(A_total_p_filt(:,3) == median(A_total_p_filt(:,3)),1, 'first');
   
    %if median is not within the dataset, find median
    if isempty(medI)
        Median = median(A_total_p_filt(:,3));
        dist = abs(A_total_p_filt(:,3) - Median);
        min_dist = min(dist(:));
        idx = find(dist == min_dist);
        [indices{1:ndims(A_total_p_filt(:,3))}] = ind2sub(size(A_total_p_filt(:,3)), idx);
        medI = idx(1);
    end

    % Calculate user acceration by removing gravity from the median position detected
    A_user_p_filt = A_total_p_filt - A_total_p_filt(medI,:); %filtered
    A_user_p = A_total_p - A_total_p_filt(medI,:); %UNfiltered- not used. 
    
    % Calculate acceleration magnitudes
    A_total_mag_unfilt = sqrt(A_total(:,1).^2 + A_total(:,2).^2 + A_total(:,3).^2); %UNfiltered accleration resultant, gravity still present in signal 
    A_user_mag = sqrt(A_user_p(:,1).^2 + A_user_p(:,2).^2 + A_user_p(:,3).^2);%UNfilt accleration resultant, gravity removed
    A_user_mag_filt  = sqrt(A_user_p_filt(:,1).^2 + A_user_p_filt(:,2).^2 + A_user_p_filt(:,3).^2); %filtered accelration resultant, gravity removed
    
    %Calculate trapezoidal numerical integration of user acceleration
    %magnitude %filtered without threshold applied
    integrate_A_user_mag_modified = trapz(time_zero,A_user_mag_filt);
    
    %figure 5: Plot Resultant User Acceleration to ID quiet baseline time
    figure; 
    grid on; hold on;
    h4u = plot(time_zero,A_user_mag_filt, 'DisplayName', babyID);
    title('Filtered Resultant Acceleration Magnitude');
    legend(h4u,'A_{user filtered}'); %{_makes it subscript} don't need brackets for 1 (duncan) 
    xlabel('Time (sec)');
    ylabel('Acceleration (m/s/s)'); 
    xlim([0,1900]);
    set(gca, 'fontsize', 16);
    
    %% Automate "finding" quiet baseline: Testing different interval lengths
    %for our target population, 15 seconds was best. Recommend testing in
    %other populations. Other recommended values are 8, 10, or 30 (duncan)
     
    Time_interval = time_zero(2)-time_zero(1);
    interval_length = 15;
    Xsecinterval = floor(interval_length/Time_interval); %data points within both timezero and A_user_mag_filt
    % preallocate  
    L = (length(time_zero) - Xsecinterval); 
    start_interval = 1:L;
    stop_interval = start_interval + Xsecinterval; 
    output_mean = nan([1, L]); 
    output_std = nan([1, L]); 
    for u = 1: (length(time_zero) - Xsecinterval)
%         start_interval(u) = u;
%         stop_interval(u) = u + Xsecinterval;
        output_mean(u) = mean(A_user_mag_filt(start_interval(u):stop_interval(u),:));
        output_std(u) = std(A_user_mag_filt(start_interval(u):stop_interval(u),:));
    end
    
%% 
    %index the mean interval with the lowest (min) acceleration 
    %and use index to get acceleration during quiet interval

    Interval_Min = min(output_mean);
    idx_Interval_Min = find(output_mean ==Interval_Min);
    idx_start_Interval = start_interval(idx_Interval_Min);
    idx_stop_Interval = stop_interval(idx_Interval_Min);
    idx_quiet_interval_start = time_zero(idx_start_Interval);
    idx_quiet_interval_stop = time_zero(idx_stop_Interval);
    idx_quiet_time = find(time_zero >= idx_quiet_interval_start & time_zero<= idx_quiet_interval_stop);%find row index and then index out of time
    quiet_time = time_zero(idx_quiet_time);
    quiet_time_acceleration_mag_filt = A_user_mag_filt(idx_quiet_time);

    %put time of quiet baseline in the command window (use if needed to verify
    %active/sedentary)
    sensor_quiet_time_start = idx_quiet_interval_start + video_time_in_place; 
    sensor_quiet_time_stop = idx_quiet_interval_stop + video_time_in_place;

    %Calculate the mean, standard deviation of quiet time
    Mean_A_user_mag_filt = output_mean(idx_Interval_Min);
    STD_A_user_mag_filt = output_std(idx_Interval_Min);
    Upper_threshold = Mean_A_user_mag_filt + (3*STD_A_user_mag_filt); %modify  here if changing the SD (tested 3 and 5)

%%
 %%
    %Recalculate the threshold based on 1/4 of the data (divided up into 4 even
    %sections to determine if there is consistency between the threshold across
    %the data

    %1-25%
    Interval_25 = floor((length(time_zero))/4);
    % pre-allocate and such  (duncan) 
    L = Interval_25 - Xsecinterval; 
    start_interval_1_25 = 1:L;
    stop_interval_1_25 = start_interval_1_25 + Xsecinterval; 
    output_mean_1_25 = nan([1, L]); 
    output_std_1_25 = nan([1, L]); 
   for u = 1:Interval_25 - Xsecinterval
%        start_interval_1_25(u) = u;
%        stop_interval_1_25(u) = u + Xsecinterval;
       output_mean_1_25(u) = mean(A_user_mag_filt(start_interval_1_25(u):stop_interval_1_25(u),:));
       output_std_1_25(u) = std(A_user_mag_filt(start_interval_1_25(u):stop_interval_1_25(u),:));
   end
   
    Interval_Min_1_25 = min(output_mean_1_25);
    idx_Interval_Min_1_25 = find(output_mean_1_25 ==Interval_Min_1_25);
    idx_start_Interval_1_25 = start_interval_1_25(idx_Interval_Min_1_25);
    idx_stop_Interval_1_25 = stop_interval_1_25(idx_Interval_Min_1_25);
    idx_quiet_interval_start_1_25 = time_zero(idx_start_Interval_1_25);
    idx_quiet_interval_stop_1_25 = time_zero(idx_stop_Interval_1_25);
    idx_quiet_time_1_25 = find(time_zero >= idx_quiet_interval_start_1_25 & time_zero<= idx_quiet_interval_stop_1_25);%find row index and then index out of time
    quiet_time_1_25 = time_zero(idx_quiet_time_1_25);
    quiet_time_acceleration_mag_filt_1_25 = A_user_mag_filt(idx_quiet_time_1_25);
    Mean_A_user_mag_filt_1_25 = output_mean_1_25(idx_Interval_Min_1_25);
    STD_A_user_mag_filt_1_25 = output_std_1_25(idx_Interval_Min_1_25);
    Upper_threshold_1_25 = Mean_A_user_mag_filt_1_25 + (3*STD_A_user_mag_filt_1_25); %modify  here if changing threshold

    %26-50%
    % pre-allocate (duncan) 
    L = (Interval_25 * 2 - Xsecinterval) - (Interval_25 + 1) + 1; 
    start_interval_26_50 = (Interval_25 + 1) : (Interval_25 * 2 - Xsecinterval); 
    stop_interval_26_50 = start_interval_26_50 + Xsecinterval; 
    output_mean_26_50 = nan([1, L]); 
    output_std_26_50 = nan([1, L]); 
   for u =(Interval_25 + 1) : (Interval_25 * 2 - Xsecinterval)
%        start_interval_26_50(u-Interval_25) = u;
%        stop_interval_26_50(u-Interval_25) = u + Xsecinterval;
       output_mean_26_50(u-Interval_25) = mean(A_user_mag_filt(start_interval_26_50(u-Interval_25):stop_interval_26_50(u-Interval_25),:));
       output_std_26_50(u-Interval_25) = std(A_user_mag_filt(start_interval_26_50(u-Interval_25):stop_interval_26_50(u-Interval_25),:));
   end

    Interval_Min_26_50 = min(output_mean_26_50);
    idx_Interval_Min_26_50 = find(output_mean_26_50 ==Interval_Min_26_50);
    idx_start_Interval_26_50 = start_interval_26_50(idx_Interval_Min_26_50);
    idx_stop_Interval_26_50 = stop_interval_26_50(idx_Interval_Min_26_50);
    idx_quiet_interval_start_26_50 = time_zero(idx_start_Interval_26_50);
    idx_quiet_interval_stop_26_50 = time_zero(idx_stop_Interval_26_50);
    idx_quiet_time_26_50 = find(time_zero >= idx_quiet_interval_start_26_50 & time_zero<= idx_quiet_interval_stop_26_50);%find row index and then index out of time
    quiet_time_26_50 = time_zero(idx_quiet_time_26_50);
    quiet_time_acceleration_mag_filt_26_50 = A_user_mag_filt(idx_quiet_time_26_50);
    Mean_A_user_mag_filt_26_50 = output_mean_26_50(idx_Interval_Min_26_50);
    STD_A_user_mag_filt_26_50 = output_std_26_50(idx_Interval_Min_26_50);
    Upper_threshold_26_50 = Mean_A_user_mag_filt_26_50 + (3*STD_A_user_mag_filt_26_50); %modify  here if changing threshold

    %51-75%
    % pre-allocate (duncan) 
    L =  (Interval_25 *3 - Xsecinterval) - (Interval_25*2 + 1) + 1; 
    start_interval_51_75 =  (Interval_25*2 + 1): (Interval_25 *3 - Xsecinterval); 
    stop_interval_51_75 = start_interval_51_75 + Xsecinterval; 
    output_mean_51_75 = nan([1, L]); 
    output_std_51_75 = nan([1, L]); 
   for u = (Interval_25*2 + 1): (Interval_25 *3 - Xsecinterval)
%        start_interval_51_75(u-(Interval_25*2)) = u;
%        stop_interval_51_75(u-(Interval_25*2)) = u + Xsecinterval;
       output_mean_51_75(u-(Interval_25*2)) = mean(A_user_mag_filt(start_interval_51_75(u-(Interval_25*2)):stop_interval_51_75(u-(Interval_25*2)),:));
       output_std_51_75(u-(Interval_25*2)) = std(A_user_mag_filt(start_interval_51_75(u-(Interval_25*2)):stop_interval_51_75(u-(Interval_25*2)),:));
   end
   
    Interval_Min_51_75 = min(output_mean_51_75);
    idx_Interval_Min_51_75 = find(output_mean_51_75 ==Interval_Min_51_75);
    idx_start_Interval_51_75 = start_interval_51_75(idx_Interval_Min_51_75);
    idx_stop_Interval_51_75 = stop_interval_51_75(idx_Interval_Min_51_75);
    idx_quiet_interval_start_51_75 = time_zero(idx_start_Interval_51_75);
    idx_quiet_interval_stop_51_75 = time_zero(idx_stop_Interval_51_75);
    idx_quiet_time_51_75 = find(time_zero >= idx_quiet_interval_start_51_75 & time_zero<= idx_quiet_interval_stop_51_75);%find row index and then index out of time
    quiet_time_51_75 = time_zero(idx_quiet_time_51_75);
    quiet_time_acceleration_mag_filt_51_75 = A_user_mag_filt(idx_quiet_time_51_75);
    Mean_A_user_mag_filt_51_75 = output_mean_51_75(idx_Interval_Min_51_75);
    STD_A_user_mag_filt_51_75 = output_std_51_75(idx_Interval_Min_51_75);
    Upper_threshold_51_75 = Mean_A_user_mag_filt_51_75 + (3*STD_A_user_mag_filt_51_75); %modify  here if changing threshold (duncan)

    %76-100%
    % pre-allocate (duncan) 
    L = ((Interval_25 *4) - Xsecinterval) - ((Interval_25*3)+1) + 1;
    start_interval_76_100 = ((Interval_25*3)+1): ((Interval_25 *4) - Xsecinterval); 
    stop_interval_76_100 =start_interval_76_100 + Xsecinterval; 
    output_mean_76_100 = nan([1, L]); 
    output_std_76_100 = nan([1, L]);
       for u = ((Interval_25*3)+1): ((Interval_25 *4) - Xsecinterval)
%            start_interval_76_100(u-(Interval_25*3)) = u;
%            stop_interval_76_100(u-(Interval_25*3)) = u + Xsecinterval;
           output_mean_76_100(u-(Interval_25*3)) = mean(A_user_mag_filt(start_interval_76_100(u-(Interval_25*3)):stop_interval_76_100(u-(Interval_25*3)),:));
           output_std_76_100(u-(Interval_25*3)) = std(A_user_mag_filt(start_interval_76_100(u-(Interval_25*3)):stop_interval_76_100(u-(Interval_25*3)),:));
       end
   
    Interval_Min_76_100 = min(output_mean_76_100);
    idx_Interval_Min_76_100 = find(output_mean_76_100 ==Interval_Min_76_100);
    idx_start_Interval_76_100 = start_interval_76_100(idx_Interval_Min_76_100);
    idx_stop_Interval_76_100 = stop_interval_76_100(idx_Interval_Min_76_100);
    idx_quiet_interval_start_76_100 = time_zero(idx_start_Interval_76_100);
    idx_quiet_interval_stop_76_100 = time_zero(idx_stop_Interval_76_100);
    idx_quiet_time_76_100 = find(time_zero >= idx_quiet_interval_start_76_100 & time_zero<= idx_quiet_interval_stop_76_100);%find row index and then index out of time
    quiet_time_76_100 = time_zero(idx_quiet_time_76_100);
    quiet_time_acceleration_mag_filt_76_100 = A_user_mag_filt(idx_quiet_time_76_100);
    Mean_A_user_mag_filt_76_100 = output_mean_76_100(idx_Interval_Min_76_100);
    STD_A_user_mag_filt_76_100 = output_std_76_100(idx_Interval_Min_76_100);
    Upper_threshold_76_100 = Mean_A_user_mag_filt_76_100 + (3*STD_A_user_mag_filt_76_100); %modify  here if changing threshold (duncan)

    Upper_threshold_sections = [Upper_threshold_1_25 Upper_threshold_26_50 Upper_threshold_51_75 Upper_threshold_76_100];
    
%% Plot quiet time (manually or automated)
    %figure 6
    figure;hold on; 
    h5u1 = plot(quiet_time, quiet_time_acceleration_mag_filt); 
    h5u2 = plot([quiet_time(1),quiet_time(length(quiet_time))],[Upper_threshold,Upper_threshold],'-k', 'Linewidth', 0.5); %add horizontal line at upper threshold
    h5u3 = plot([quiet_time(1),quiet_time(length(quiet_time))],[Mean_A_user_mag_filt,Mean_A_user_mag_filt],'-k', 'Linewidth', 1);
    xlabel('Time (sec)')
    ylabel('Acceleration (m/s/s)')
    title('Filtered Acceleration Quiet Window');
    legend([h5u1, h5u2, h5u3], ['A_{user filtered} ' ' Quiet Window'], ['Threshold'], ['Mean Acceleration'])
    set(gca, 'fontsize', 16);
%% filtered user accleration with threshold cutoff
      A_user_mag_filt_threshold = zeros(length(A_user_mag_filt),1); %initializing variable
      time_threshold = zeros(length(time_zero),1);

    %Apply threshold ONLY if applies to two datapoints, "start" and
    %"stop", this method puts in data over the threshold (no data left
    %under the threshold line)
    for k = 2:length(A_user_mag_filt) - 1
        if A_user_mag_filt(k)>= Upper_threshold && A_user_mag_filt(k+1)>=Upper_threshold
           time_threshold(k) = time_zero(k);
           A_user_mag_filt_threshold(k) = A_user_mag_filt(k);
        elseif A_user_mag_filt(k)>= Upper_threshold && A_user_mag_filt(k-1)>=Upper_threshold
           time_threshold(k) = time_zero(k);
           A_user_mag_filt_threshold(k) = A_user_mag_filt(k);
        else
           A_user_mag_filt_threshold(k) = 0;
           time_threshold(k) = 0;
        end
    end     
      
    %ID active time bouts from applied threshold:
    ext_time_threshold = [0; time_threshold; 100]; %add extra value at beginning and end to capture start/stop at beginning and end
    D = diff(ext_time_threshold(1:end-1)); 
    idx_stop_active_sensor = find(D < 0); % this would be idx_stop_active_sensor
    idx_start_active_sensor = find(round(D) > Time_interval) + 1;  % this would be idx_start_active sensor

    %if active time starts at zero, this if statement will apply and add "1" to
    %the start to make the start and stop times equal and index the first value
    %of time_zero in the next step
    if length(idx_start_active_sensor) == (length(idx_stop_active_sensor)-1)
        idx_start_active_sensor = [1; idx_start_active_sensor];
    end

    %used time_zero, pulling the active bouts from the sensor ID's active times
    %from time_zero. Above, we have time_threshold which has the active bouts
    %removed, but we need to index out of the full time array to then apply
    %that to the full filtered acceleration file. This lets us pull out the
    %active time bouts from the full data file and analyze them. (counts,
    %duration, etc.)
    
    % pre-allocate
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

    count_Sensor_active_time_bouts = length(active_Sensor_timebouts);
    Sum_Sensor_Active_Time_Duration = sum(Duration_Sensor_Active_full);
    Min_Sensor_Active_Time_Duration = min(Duration_Sensor_Active_full);
    Max_Sensor_Active_Time_Duration = max(Duration_Sensor_Active_full);

    %% This segment will be used when validating sensory data with Video coding data
    %Import video coded data file
    %Video coded file had 8 columns, we used column 8 which was behaviorally coded for active('a'), sdenetary ('s')
    % or picked up/removed time ('r'). Sedentary and pickedup/removed time were
    % considered together as "non-active" time 

    VideoCode_filename = ('EM039_JO.xlsx');
    [VideoCode_filename_num, VideoCode_filename_text, VideoCode_filename_raw] = xlsread(VideoCode_filename, 'Sheet1', 'A:H');

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
    active_time_windows = nan([1, L]); 
    % vectorize
    start_active_time = active_time_video(1:length(active_time_video), 5) - video_time_in_place;
    stop_active_time = active_time_video(1:length(active_time_video),6) - video_time_in_place;
    active_time_windows = stop_active_time - start_active_time;
 

    %create picked up/removed time windows
    %preallocate
    size_array = size(picked_up_removed_time_video);
    start_picked_up_removed_time = nan([1, size_array(1)]); 
    stop_picked_up_removed_time = nan([1, size_array(1)]); 
    picked_up_removed_time_windows = nan([1, size_array(1)]);
    % vectorize
    start_picked_up_removed_time = picked_up_removed_time_video(1:size_array(1), 5) - video_time_in_place;
    stop_picked_up_removed_time = picked_up_removed_time_video(1:size_array(1),6) - video_time_in_place;
    picked_up_removed_time_windows = stop_picked_up_removed_time - start_picked_up_removed_time;

    %% calculate the percent "correct" ID active time based on video data
    %preallocate
    L = length(start_active_time); 
    data_start_idx = nan([1, L]); 
    data_end_idx = nan([1, L]);   
    for m=1:(length(start_active_time))  
    temp = find(time_zero>=start_active_time(m)&time_zero<=stop_active_time(m));  %This will give you the indices of the time array that fall between the video_code timepoints
    data_start_idx(m) = temp(1);
    data_end_idx(m) = temp(end);
    end 
    
    %preallocate
    time_video_active = NaN(length(time_zero),1);%initializing variables
    active_video_A_user_mag_filt = NaN(length(A_user_mag_filt),1);
    for n= 1:length(data_start_idx)
        time_video_active(data_start_idx(n):data_end_idx(n))= time_zero(data_start_idx(n):data_end_idx(n)); 
        active_video_A_user_mag_filt(data_start_idx(n):data_end_idx(n)) = A_user_mag_filt(data_start_idx(n):data_end_idx(n));
    end

    %plot video active data
    %figure 7
    figure;
    plot(time_zero, active_video_A_user_mag_filt, 'r'); 
    hold on; grid on;
    %plot([active_time(1),active_time(length(active_time))],[Mean_A_user_mag_filt_active,Mean_A_user_mag_filt_active],'-k', 'Linewidth', 1);
    xlabel('Time (sec)')
    ylabel('Acceleration (m/s/s)')
    title({'Filtered Acceleration Magnitude of Active Time', 'Identified with Video Coding'});
    xlim([0,1900]);
    set(gca, 'fontsize', 16); 
%% Remove time where baby is moved by therapist/ parent for proof of principle
    %Testing removing picked up time from both sensor and video data to see
    %how this improves our validity (proof of principle- but should not be
    %removed for final analysis 

%indexing picked up/removed times from time_zero, used for ONLY for proof
%of principle
%     for m=1:(length(start_picked_up_removed_time))  
%     temp_v = find(time_zero>=start_picked_up_removed_time(m)&time_zero<=stop_picked_up_removed_time(m));  %This will give you the indices of the time array that fall between the video_code timepoints
%     remove_start_idx(m) = temp_v(1);
%     remove_end_idx(m) = temp_v(end);
%     time_video_active(remove_start_idx(m):remove_end_idx(m))=[];
%     time_threshold(remove_start_idx(m):remove_end_idx(m))=0;
%     end 

%%
    time_threshold_alt = time_threshold;
    time_video_active_alt = time_video_active;

    %Replace 0 in Time_threshold with -100 for comparison between files
    for i = 2: length(time_threshold_alt)
        if time_threshold_alt(i) == 0
            time_threshold_alt(i) = 2;
            A_user_mag_filt_threshold(i) = NaN;
        else
            time_threshold_alt(i) = 1;
            A_user_mag_filt_threshold(i) = A_user_mag_filt_threshold(i);
        end
    end     

    time_video_active_alt(time_video_active_alt > 0)= 1; %Active time =1

    %Replace NaN's with -100 (chose -100 to signal it was a replaced value)
    time_video_active_alt(isnan(time_video_active_alt))= 2; %non-active time = 2

    %isequal function
    % preallocate
    equal_Compare_threshold_video = nan([1, length(time_video_active_alt)]); 
    for m = 1:length(time_video_active_alt)
    equal_Compare_threshold_video(m) = isequaln(time_video_active_alt(m),time_threshold_alt(m));
    end

    Correct_count_check = sum(equal_Compare_threshold_video);
    error_count_check = length(time_video_active_alt) - Correct_count_check; 

    %Count false negatives and Count false positives
    %Preallocate
    false_neg_direct = zeros(length(time_video_active_alt),1); 
    false_pos_direct = zeros(length(time_video_active_alt),1);
    % vectorize and logically index 
    false_neg_direct(time_video_active_alt == 2 & time_threshold_alt ~= 2) = 1;
    false_pos_direct(time_video_active_alt ~= 2 & time_threshold_alt == 2) = 1; 
    direct_false_pos_sum = sum(false_pos_direct);
    direct_false_neg_sum = sum(false_neg_direct);

    %adds tolerance
    % 32 frame tolerance (.25 seconds) 
    ext = zeros(1,32)'; % modify this based on the number of frames within the tolerance
    ext_time_video_active = [ext; time_video_active_alt; ext]; %add 32 to beginning and end, will remove after loop and compare files
    ext_time_threshold_tol = [ext; time_threshold_alt; ext];
    tolerance = 32; 
    t = 1: length(tolerance); %should be 1x1
    t_pos = zeros(1,tolerance -1)'; % 31 x 1
    t_neg = zeros(1,tolerance-1)'; % 31 x 1

    true = zeros(1, length(time_video_active_alt));
    false_neg = zeros(1, length(time_video_active_alt));
    false_pos = zeros(1, length(time_video_active_alt));
    true_neg = zeros(1, length(time_video_active_alt));
    true_pos =  zeros(1, length(time_video_active_alt));
    Compare_threshold_video = zeros(1, length(time_video_active_alt))';

    for r = (1 + tolerance: length(time_video_active_alt) + tolerance)
        for s = 1:(tolerance - 1) %subtract 1 because you want 31 additional row comparisons (same row + 31 more)
         if ext_time_video_active(r) == ext_time_threshold_tol(r)
            t(s) = 1;
           elseif ext_time_video_active(r) == ext_time_threshold_tol(r + s)
             t_pos(s) = 1; 
           elseif ext_time_video_active(r) == ext_time_threshold_tol(r - s) 
             t_neg(s) = 1; 
         else
         t(s) = 0;
         t_pos(s) = 0;
         t_neg(s) = 0;
         true(r - tolerance) = 0;
         end
         if t>=1
             Compare_threshold_video(r - tolerance) = 1;
         elseif sum(t_pos)>=1
             Compare_threshold_video(r - tolerance) = 1;
         elseif sum(t_neg)>=1
             Compare_threshold_video(r - tolerance) = 1;  
         else
             Compare_threshold_video(r - tolerance) = 0;
         end   
        end
    end

    for u = 1:length(Compare_threshold_video)
        if Compare_threshold_video(u) == 0
            if time_video_active_alt(u) == 1 && time_threshold_alt(u) == 2
                false_neg(u) = 1;
                false_pos(u) = 0; 
                true_pos(u) = 0;
                true_neg(u) = 0;
            elseif time_video_active_alt(u) == 2 && time_threshold_alt(u) == 1
                false_neg(u) = 0;
                false_pos(u) = 1;
                true_pos(u) = 0;
                true_neg(u) = 0;
            end
        elseif Compare_threshold_video(u) == 1
              if time_video_active_alt(u) == 2 && time_threshold_alt(u) == 2 
                  true_neg(u) = 1;
                  true_pos(u) = 0;
                  false_neg(u) = 0;
                  false_pos(u) = 0;
              elseif time_video_active_alt(u) == 1 && time_threshold_alt(u) == 1
                  true_neg(u) = 0;
                  true_pos(u) = 1;
                  false_neg(u) = 0;
                  false_pos(u) = 0;
              elseif time_video_active_alt(u) == 2 && time_threshold_alt(u) == 1
                  true_neg(u) = 1;
                  true_pos(u) = 0;
                  false_neg(u) = 0;
                  false_pos(u) = 0;
              elseif time_video_active_alt(u) == 1 && time_threshold_alt(u) == 2
                  true_neg(u) = 0;
                  true_pos(u) = 1;
                  false_neg(u) = 0;
                  false_pos(u) = 0;
              end        
        end   
    end

    sum_true = sum(Compare_threshold_video);
    num_true_neg = sum(true_neg);
    num_true_pos = sum(true_pos);
    num_false_neg = sum(false_neg);
    num_false_pos = sum(false_pos);

    true_neg = true_neg';
    true_pos = true_pos';
    false_neg = false_neg';
    false_pos = false_pos';

    sum_total = num_true_neg + num_true_pos + num_false_neg + num_false_pos;

    %check_isequal = [time_video_active time_threshold Compare_threshold_video]; %visual inspection of arrays combined if needed
    check_isequal = [time_video_active_alt time_threshold_alt Compare_threshold_video true_neg true_pos false_neg false_pos];
    equal_percent_agreement = sum(equal_Compare_threshold_video)/length(equal_Compare_threshold_video) * 100 %original method without tolerance

    tolerance_percent_agreement = sum(Compare_threshold_video)/length(Compare_threshold_video) * 100 % with tolerance added

%% ID active times from video data and plot the active times to find mean/SD 
    % preallocate
    L = length(data_start_idx); 
    active_Video_timebouts = 1:L;
    Duration_Video_Active_full = nan([1, L]); 
    Mean_A_user_mag_filt_active = nan([1, L]); 
    STD_A_user_mag_filt_active = nan([1, L]); 
    for p = 1: length(data_start_idx)
        active_Video_time_start = time_zero(data_start_idx(p));
        active_Video_time_stop = time_zero(data_end_idx(p));
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

%Summarize Duration Video for MasterTable
Sum_Video_Active_Time_Duration = sum(Duration_Video_Active_full);
Min_Video_Active_Time_Duration = min(Duration_Video_Active_full);
Max_Video_Active_Time_Duration = max(Duration_Video_Active_full);
    
  %% Apply threshold to integrated acceleration
    %Return time_threshold to NAN from -100 to match A_user_mag_filt_threshold
    %and then remove both NAN's from both variables
    %Replace 0 in Time_threshold with -100 for comparison between files
    % vectorize 
    time_threshold(time_threshold(2:end) == 0) = nan;  

%% Plots with Non-active acceleration and time as NaN

%plot filtered acceleration magnitude of one user on single plot
     %with time shifted to start at zero and threshold applied
     %figure 8
    hf6=figure; 
    g = gca;
    hold on; grid on;
    h6u = plot(time_zero,A_user_mag_filt_threshold, 'DisplayName', babyID);
    h7u = plot([time_zero(1),time_zero(length(time_zero))],[Upper_threshold,Upper_threshold],'-k', 'Linewidth', 0.5); %add horizontal line at upper threshold
    %plot([time_zero(1),time_zero(length(time_zero))],[Mean_A_user_mag_filt,Mean_A_user_mag_filt],'-k', 'Linewidth', 1);
    title({'Filtered Acceleration Magnitude of Active Time', 'Identified with Sensor Threshold'});
    legend([h6u, h7u],['A_{user filtered} '], ['Threshold']); %updating legends note {_makes it subscript}
    xlabel('Time (sec)');
    ylabel('Acceleration (m/s/s)');
    xlim([0,1900]);
    %ylim([0,10]);
    set(gca, 'fontsize', 16);
    
    hf8 = figure;
    g_1 = gca; 
    hold on; grid on; 
    h8u = plot(time_zero,A_user_mag_filt_threshold, 'DisplayName', babyID);
    h9u = plot([time_zero(1),time_zero(length(time_zero))],[Upper_threshold,Upper_threshold],'-k', 'Linewidth', 0.5); %add horizontal line at upper threshold
    h10u = plot(time_zero, active_video_A_user_mag_filt, 'r'); 
    title('Comparision of Video vs. Sensor Threshold Identified Active Times');
    %legend('-DynamicLegend')
    legend([h8u,h10u],['A_{user filtered}' ' Sensor Threshold'],['A_{user filtered}' ' Video Coding']); %updating legends note {_makes it subscript}
    xlabel('Time (sec)');
    ylabel('Acceleration (m/s/s)');
    xlim([0,1900]);
    set(gca, 'fontsize', 16);
    
%% Remove NaNs to integrate 

    A_user_mag_filt_threshold(isnan(A_user_mag_filt_threshold)) = [];
    time_threshold(isnan(time_threshold)) = [];
    
%% Calcuate user acceleration with threshold applied, Trapz funciton takes integral- (area under
%%curve) 

    integrate_A_user_mag_threshold = trapz(time_threshold,A_user_mag_filt_threshold);


    %% write file details and start and stop time to the table
    masterTable.FileName(ii) = cellstr(currentFile);
    masterTable.BabyID(ii) = cellstr(babyID);
    masterTable.StartTime(ii) = t_start_input_file;
    masterTable.StopTime(ii) = t_stop_input_file;
    masterTable.Duration(ii) = time(end)-time(1);
    masterTable.DurationMins(ii) = (time(end)-time(1)) /60 ;% column of duration
    masterTable.UserStart(ii) = cellstr(time_start_string);
    masterTable.UserStop(ii) = cellstr(time_end_string);
    masterTable.totalAcceleration(ii) = integrate_A_user_mag_modified;% column of total Accerleration
    masterTable.timeNormalizedAcc(ii) = integrate_A_user_mag_modified/(time(end)-time(1));% column of time normalized acceleration
    masterTable.timeNormalizedAccMins(ii) = integrate_A_user_mag_modified/ masterTable.DurationMins(ii) ;
    masterTable.thresholdTotalAcceleration(ii) = integrate_A_user_mag_threshold;% column of total Accerleration
    masterTable.thresholdTimeNormalizedAcc(ii) = integrate_A_user_mag_threshold/(time(end)-time(1));% column of time normalized acceleration
    masterTable.thresholdTimeNormalizedAccMins(ii) = integrate_A_user_mag_threshold/ masterTable.DurationMins(ii) ;    
    masterTable.Sum_Sensor_Active_Time_Duration(ii) = Sum_Sensor_Active_Time_Duration;% column of Total Duration of Active time 
    masterTable.Min_Sensor_Active_Time_Duration(ii) = Min_Sensor_Active_Time_Duration;% column of Minimal Duration of Active time
    masterTable.Max_Sensor_Active_Time_Duration(ii) = Max_Sensor_Active_Time_Duration ;% column of Maximum Duration of Active time
    masterTable.count_Sensor_active_time_bouts(ii) = count_Sensor_active_time_bouts;% column of numober/ count of Actime time bouts
    masterTable.Sum_Sensor_NONActive_Time_Duration(ii) = (time_zero(end)-time_zero(1)) - Sum_Sensor_Active_Time_Duration;% column of Total Duration of Active time 
   
end 

%%
% write the table to an output file
writetable(masterTable, saveFileName)