%% Automated Processing of H5 files: Open Source

        %Code initially written by Mayumi Mohan, significantly modified and
        %updated by Julie Orlando with section contributions from Jocelyn Hafer (signal
        %processing expert) and in consultation with Laura Prosser, Beth Smith,
        %Athylia Paremski, Matthew Amodeo and Michele Lobo

%This code requires the following functions:

%importTimingfile.m
%MergeSensorFilesFunction.m
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
% saveFileName = 'Datasets/outputData.xlsx';
saveFileName = 'outputData_MergedDataFiles.xlsx';
% Read the file with the time stamps in it
timingFile = 'Timestamps.xlsx';

%% folder with datsasets in it
% put all the subfolders into a folder named Datasets and add it to this
% folder 
%example: 
%dataPath = '/Users/juliemorlando/Desktop/iMove_MatLab/Methods Paper/Program';
dataPath = '/Users/orlandoj1/Desktop/Methods Paper/Program'; %(Edit this based on the computer and the user) 
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
tableLen = length(1); % get the total number of datasets that we want to process
masterTable = table; % initialize a table to store all the values
masterTable.FileName = cell(tableLen, 1); % this is the column for all filenames
masterTable.BabyID = cell(tableLen, 1); % this is the column for all filenames
masterTable.Duration = zeros(tableLen,1);% column of duration
masterTable.DurationMins = zeros(tableLen,1);% column of duration
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
masterTable.TruePositive= zeros(tableLen,1);% 
masterTable.TrueNegative= zeros(tableLen,1);% 
masterTable.FalsePositive= zeros(tableLen,1);% 
masterTable.FalseNegative= zeros(tableLen,1);% 
masterTable.TruePositive_Perc= zeros(tableLen,1);% 
masterTable.TrueNegative_Perc= zeros(tableLen,1);% 
masterTable.FalsePositive_Perc= zeros(tableLen,1);% 
masterTable.FalseNegative_Perc= zeros(tableLen,1);% 
masterTable.OptimalThreshold = zeros(tableLen,1);%
%Uses function to concatenate 8 files into one file for validation

babyID = 'Merged_1_8';

[A_total, q, time, time_zero] = MergeSensorFilesFunction(timingFile);

    % %figure 1: raw data   
   frequency_x_axis = 0:2500:150000;
    
    hf1=figure; hold on; grid on;
    h1x = plot(time_zero,A_total(:,1),'-r');
    h1y = plot(time_zero,A_total(:,2),'-g');
    h1z = plot(time_zero,A_total(:,3),'-b');
    title('Raw Acceleration Components');
    legend([h1x,h1y,h1z],'A_x','A_y','A_z');
    xlabel('Time (sec)');
    ylabel('Acceleration (m/s^2)');
    ylim([-75,100]);
    %xlim([0,1900])
    set(gca, 'fontsize', 16, 'XTick', frequency_x_axis);
    saveas(hf1, [babyID '_' 'Raw_Acceleration_Components.jpg']);
 

    %% Calculate the total acceleration in North, West, and Up coordinate frame
    A_total_p = A_total; % Initialize array
    for jj = 1:length(A_total(:,1))
        A_total_p(jj,:) = RotateVector(A_total(jj,:),q(jj,:)); % Rotate gravity vector into sensor coordinate frame
    end
    

    % Calculate user acceration by removing gravity from the median position detected
    %A_user_p= A_total_p - A_total_p(medI,:); 
    A_gravity_p = [0,0,9.806];
    A_user_p = A_total_p - A_gravity_p; %filtered
         
    % %figure 2: raw data in NWU for comparision
    hf2 = figure;grid; hold on;
    h2x = plot(time_zero,A_user_p(:,1),'-r');
    h2y =plot(time_zero,A_user_p(:,2),'-g');
    h2z =plot(time_zero,A_user_p(:,3),'-b');
    legend([h2x,h2y,h2z],'A_x','A_y','A_z');
    xlabel('Time (sec)')
    ylabel('Acceleration (m/s^2)')
    title({'Acceleration Components in North, West, Up', 'Gravity Free'});
    ylim([-75,100]);
    %xlim([0,1900]);
    set(gca, 'fontsize', 16, 'XTick', frequency_x_axis);
    saveas(hf2, [babyID '_' 'NWU_Gravity_Removed.jpg']);

    %Calculate user acceleration from filtered data

    % Calculate acceleration magnitudes
    A_user_mag = sqrt(A_user_p(:,1).^2 + A_user_p(:,2).^2 + A_user_p(:,3).^2);%UNfilt accleration resultant, gravity removed
   
     %% Filter user_acceleration components raw data in N, W, Up (filter first, then get resultant)
         
    % Fourier transform (FFT): This will transform data from time domain
    % to frequency doamin
    Fs = 128; %sampling frequency/sampling rate (of the sensor) - A_total_p
    dt = 1/128; %change in time 
    t = time_zero; 
    n = length(time_zero); % should this be magnitude data or the raw data - use raw data NWU
    fhat = fft(A_user_mag,n);
      
    %Power Spectral Density Analysis: Once data is in the frequency domain, we can do a power spectral density analysis to look power of the signal and decide
    %where to filter the data    
    PSD = fhat.*conj(fhat)/n; %power spectrum (power per freq)
    freq = 1/(dt*n)*(0:n);   %x-axis of freq in Hz
    L = 1:floor(n/2); %plot the first half of freqs
    half_freq = freq(L);

        %Determine how much of the data to inlcude and update in Norm_Q >= .XXX
    Q = cumtrapz(PSD(L));
    Q_index_length = length(Q);
    Norm_Q = Q/Q(end); %find values that exceed .95 (Can work on this threshold)
    idx_Q = find(Norm_Q >= .95);%find row index and then index out of freq(L) 
    idx = idx_Q(1); %index of first frame of PSD(L) that exceeds 95% of the PSD AUC
    filter_Q = half_freq(idx); %10.966
    %Designing lowpass Butterworth filter at the frequency defined above
    [b,a] = butter(4,filter_Q/(Fs/2)); %4th order lowpass butterworth filter with a cutoff frequency of filter_Q Hz
     
    A_user_mag_filt = abs(filtfilt(b,a,A_user_mag)); %dual-pass filter
   
%down sample for video coding- comment out if this is not being used for
% %validation. 
    A_user_mag_filt = downsample (A_user_mag_filt,4);
    time_zero = downsample (time_zero,4);

    % %figure 4: filtered data in NWU for comparision
    hf3 = figure;grid on; hold on;
    h3x = plot(time_zero,A_user_mag_filt(:,1),'-k');
    xlabel('Time (sec)')
    ylabel('Acceleration (m/s^2)')
    title({'Filtered Resultant Acceleration'});
    %legend([h3x,h3y,h3z],'A_x','A_y','A_z'); 
    %xlim([0,1900]);
    set(gca, 'fontsize', 16, 'XTick', frequency_x_axis);
    saveas(hf3, [babyID '_' 'Resultant_Gravity_Removed.jpg']);

    % %figure 3:
    hf4 = figure; grid on; hold on;
    plot(freq(L),PSD(L),'k');
    xline(filter_Q,'c', 'LineWidth', 1);
    xlabel('f (Hz)')
    ylabel('Power')
    title('Power Spectral Density Analysis');
    ylim([0,100]);
    set(gca, 'fontsize', 16); 
    saveas(hf4, [babyID '_' 'PSD.jpg']);

    %Calculate trapezoidal numerical integration of user acceleration
    %magnitude %filtered without threshold applied
    integrate_A_user_mag_modified = trapz(time_zero,A_user_mag_filt);
   
   
%Index active and sedentary time ID'd from video from sensor
%data and then plot as histograms to find threshold. 
    VideoCode_filename = ('Analysis.xlsm');
    [VideoCode_filename_num, VideoCode_filename_text, VideoCode_filename_raw] = xlsread(VideoCode_filename, 'Merge', 'A:I');

    active_time_video_cell = VideoCode_filename_raw(strcmp(VideoCode_filename_raw(:,9),'a'),:); % in cell array
    sedentary_time_video_cell = VideoCode_filename_raw(strcmp(VideoCode_filename_raw(:,9),'s'),:);
    picked_up_removed_time_video_cell = VideoCode_filename_raw(strcmp(VideoCode_filename_raw(:,9),'r'),:);

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
    start_active_time = active_time_video(1:length(active_time_video), 6);%Merged file has sensor set up removed, so we do  not subract video time in place
    stop_active_time = active_time_video(1:length(active_time_video),7);
    active_time_windows = stop_active_time - start_active_time;
    sum_active_time = sum(active_time_windows)
 
%create sedentary time windows
    L = length(sedentary_time_video); 
    start_sedentary_time = nan([1, L]); 
    stop_sedentary_time = nan([1, L]); 
    sedentary_time_windows = nan([1, L]); 
    % vectorize
    start_sedentary_time = sedentary_time_video(1:length(sedentary_time_video), 6);
    stop_sedentary_time = sedentary_time_video(1:length(sedentary_time_video),7);
    sedentary_time_windows = stop_sedentary_time - start_sedentary_time;
    sum_sedentary_time = sum(sedentary_time_windows)

    video_percent_sedentary = sum_sedentary_time/(sum_sedentary_time+sum_active_time)*100
    video_percent_active = sum_active_time/(sum_sedentary_time+sum_active_time)*100

    %create picked up/removed time windows- not currently in use
    %preallocate
    % size_array = size(picked_up_removed_time_video);
    % start_picked_up_removed_time = nan([1, size_array(1)]); 
    % stop_picked_up_removed_time = nan([1, size_array(1)]); 
    % picked_up_removed_time_windows = nan([1, size_array(1)]);
    % % vectorize
    % start_picked_up_removed_time = picked_up_removed_time_video(1:size_array(1), 5) - video_time_in_place;
    % stop_picked_up_removed_time = picked_up_removed_time_video(1:size_array(1),6) - video_time_in_place;
    % picked_up_removed_time_windows = stop_picked_up_removed_time - start_picked_up_removed_time;

    %% Index from sensor data the video ID'd active time
    %preallocate
    L = length(start_active_time); 
    active_data_start_idx = nan([1, L]); 
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

    for n= 1:(length(active_data_start_idx))
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

%     %plot video active data
    %figure 8 Filtered Active Time Identified with Video Coding
    hf5 = figure;
    plot(time_zero, active_video_A_user_mag_filt, 'r'); 
    hold on; grid on;
    xlabel('Time (sec)')
    ylabel('Acceleration (m/s^2)')
    title({'Filtered Acceleration Magnitude of Active Time', 'Identified with Video Coding'});
    %xlim([0,1900]);
    set(gca, 'fontsize', 16, 'XTick', frequency_x_axis);
    saveas(hf5, [babyID '_' 'Active_Video_Coding.jpg']);


% 
%     %figure 9 Filtered Sedentary Time Identified with Video Coding
%     figure;
%     plot(time_zero, sedentary_video_A_user_mag_filt, 'g'); 
%     hold on; grid on;
%     %plot([active_time(1),active_time(length(active_time))],[Mean_A_user_mag_filt_active,Mean_A_user_mag_filt_active],'-k', 'Linewidth', 1);
%     xlabel('Time (sec)')
%     ylabel('Acceleration (m/s/s)')
%     title({'Filtered Acceleration Magnitude of Sedentary Time', 'Identified with Video Coding'});
%     %xlim([0,1900]);
%     set(gca, 'fontsize', 16); 
% 
% % figure 10 Histogram of active and sedentary times
hf6 = figure;
hold on; grid on;
hist1 = histogram(active_video_A_user_mag_filt);
hist1.BinWidth = 0.1;
title({'Histogram of Active and Sedentary Time', 'Identified with Behavioral Coding'},'FontSize', 16, 'FontName','Ariel', 'FontWeight','bold');;
hist2 = histogram(sedentary_video_A_user_mag_filt);
hist2.BinWidth = 0.1;
xlabel('Acceleration (m/s^2)')
xlim([0,5]);
ylim([0, 80000]);
% Get handle to current axes.
ax = gca;
ax.YAxis.Exponent = 0;
ax.YAxis.TickLabelFormat = '%.0f';
% Set x and y font sizes.
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
legend([hist1,hist2],{'Filtered Acceleration Magnitude of Active Time','Filtered Acceleration Magnitude of Sedentary Time'}, 'FontSize', 12, 'FontName','Ariel'); 
ylabel('Counts', 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold')
xlabel('Acceleration (m/s^2)', 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold')
saveas(hf6, [babyID '_' 'histogram1.jpg']);


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

Threshold = T((X==OPTROCPT(1))&(Y==OPTROCPT(2))); %Optimal Threshold ID'd by Matlab function
%Threshold = 0.00 %Manual entry here
 
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
    %adjust this to apply filter ONLY if 2 cells remain
    %add a cell at the start 

    for k = 2:length(A_user_mag_filt)-1
        if A_user_mag_filt(k)> Threshold & A_user_mag_filt(k+1)> Threshold
           active_time_threshold_annotate(k)=1;
           active_time_threshold(k) = time_zero(k);
           active_accel_threshold(k) = A_user_mag_filt(k);
        elseif A_user_mag_filt(k)> Threshold & A_user_mag_filt(k-1)> Threshold
           active_time_threshold_annotate(k)=1;
           active_time_threshold(k) = time_zero(k);
           active_accel_threshold(k) = A_user_mag_filt(k);
        else %A_user_mag_filt(k)<= Threshold;
           active_time_threshold_annotate(k)=0;
           active_time_threshold(k) = 0;
           active_accel_threshold(k) = NaN;
        end
    end  

    %fills in the first cell & last cell which are skipped in the previous for-loop
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
%sedentary time

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

%fills in the first cell & last cell which are skipped in the previous for-loop
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


sum_nan = sum(isnan(sedentary_accel_threshold));
total_rem = length(sedentary_accel_threshold)- sum_nan;
 %note this removes 1.2 percent of the data only

 %Now we have sedentary and active annotations and acceleration and time
 %with the threshold applied and active bouts of 1 spike only removed from
 %both active and sedentary time. 

% %figure 11 Histogram with Active and Sedendatry Time Separated by Threshold
hf7 = figure;
hold on; grid on;
hist1 = histogram(active_accel_threshold);
hist1.BinWidth = 0.1;
title({' Histogram of Active and Sedentary Time', 'Identified with the Threshold'});
hist2 = histogram(sedentary_accel_threshold);
hist2.BinWidth = 0.1;
xlim([0,5]);
ylim([0, 120000]);
% Get handle to current axes.
ax = gca;
ax.YAxis.Exponent = 0;
ax.YAxis.TickLabelFormat = '%.0f';
% Set x and y font sizes.
ax.XAxis.FontSize = 14;
ax.YAxis.FontSize = 14;
title({'Active and Sedentary Histogram', 'Identified with Threshold'}, 'FontSize', 16, 'FontName','Ariel', 'FontWeight','bold');
legend([hist1,hist2],{'Filtered Acceleration Magnitude of Active Time','Filtered Acceleration Magnitude of Sedentary Time'}, 'FontSize', 12, 'FontName','Ariel'); 
ylabel('Counts', 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold')
xlabel('Acceleration (m/s^2)', 'FontSize', 15, 'FontName','Ariel', 'FontWeight','bold')
saveas(hf7, [babyID '_' 'Histogram2.jpg']);

%% Calculate the accuracy of sensor threshold and video identified time for validation
%preallocate
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
Perc_Sensor_Sed_Video_Sed = Sum_Sensor_Sed_Video_Sed/(sum(active_time_threshold_annotate) + sum(sedentary_time_threshold_annotate))*100;
Perc_Sensor_Active_Video_Active = Sum_Sensor_Active_Video_Active/(sum(active_time_threshold_annotate) + sum(sedentary_time_threshold_annotate))*100;
Perc_Sensor_Sed_Video_Active = Sum_Sensor_Sed_Video_Active/(sum(active_time_threshold_annotate) + sum(sedentary_time_threshold_annotate))*100;
Perc_Sensor_Active_Video_Sed = Sum_Sensor_Active_Video_Sed/(sum(active_time_threshold_annotate) + sum(sedentary_time_threshold_annotate))*100;

sum_perc = Perc_Sensor_Sed_Video_Sed + Perc_Sensor_Active_Video_Active + Perc_Sensor_Sed_Video_Active + Perc_Sensor_Active_Video_Sed
Perc_Sensor_Active = Perc_Sensor_Active_Video_Active + Perc_Sensor_Active_Video_Sed
Perc_Sensor_Sedentary = Perc_Sensor_Sed_Video_Sed + Perc_Sensor_Sed_Video_Active
sum_true_pos_true_neg = Perc_Sensor_Sed_Video_Sed + Perc_Sensor_Active_Video_Active

%Calculate cost function based on Mustafa Ghazi paper- not used currently
%Delta_sed_Perc = ((Perc_Sensor_Sed_Video_Sed + Perc_Sensor_Active_Video_Sed)-(Perc_Sensor_Sed_Video_Sed + Perc_Sensor_Sed_Video_Active))/ (Perc_Sensor_Sed_Video_Sed + Perc_Sensor_Active_Video_Sed)
% Delta_sed = ((Sum_Sensor_Sed_Video_Sed + Sum_Sensor_Active_Video_Sed)-(Sum_Sensor_Sed_Video_Sed + Sum_Sensor_Sed_Video_Active))/ (Sum_Sensor_Sed_Video_Sed + Sum_Sensor_Active_Video_Sed)
% Delta_active = ((Sum_Sensor_Sed_Video_Active + Sum_Sensor_Active_Video_Active)-(Sum_Sensor_Active_Video_Sed + Sum_Sensor_Active_Video_Active))/ (Sum_Sensor_Sed_Video_Active + Sum_Sensor_Active_Video_Active)
% cost_function = (abs(Delta_sed) * 100) + (abs(Delta_active) * 100)

%Accuracy of Active time
Accuracy_Active = (Sum_Sensor_Active_Video_Active)/(Sum_Sensor_Active_Video_Active + Sum_Sensor_Active_Video_Sed) * 100
Accuracy_Sedentary = (Sum_Sensor_Sed_Video_Sed)/(Sum_Sensor_Sed_Video_Sed + Sum_Sensor_Sed_Video_Active) * 100

% %figure 12
% figure;
% g_1 = gca; 
% hold on; grid on; 
% testfig2 = plot(time_zero, A_user_mag_filt, 'DisplayName', babyID);
% testfig1 = plot(time_zero, active_accel_threshold, 'DisplayName', babyID);
% testfig3 = plot([time_zero(1),time_zero(length(time_zero))],[Threshold,Threshold],'-k', 'Linewidth', 0.5); %add horizontal line at upper threshold
% title({'Comparision of Video vs. Sensor Threshold', 'Identified Active Times'});
% %legend('-DynamicLegend')
% legend([testfig1,testfig2,testfig3],['A_{user filtered}' ' Sensor Threshold'],['A_{user filtered}' ' Without Threshold'], 'Threshold'); %updating legends note {_makes it subscript}
% xlabel('Time (sec)');
% ylabel('Acceleration (m/s/s)');
% %xlim([0,1900]);
% set(gca, 'fontsize', 16);

%add a tolerance to validation between sensor threshold and video to
%account for imperfect syncing process

    ext_tol = zeros(1,5)'; % modify this based on the number of frames within the tolerance
    ext_time_video_active_sedentary_annotate = [ext_tol; time_video_active_sedentary_annotate; ext_tol]; %add 4 to beginning and end; needed for the for loop
    ext_sedentary_time_threshold_annotate = [ext_tol; sedentary_time_threshold_annotate; ext_tol];
    ext_active_time_threshold_annotate = [ext_tol; active_time_threshold_annotate; ext_tol];
    tolerance = 5; 
    tol = 1: length(tolerance); %should be 1x1
    tol_pos = zeros(1,tolerance -1)'; %
    tol_neg = zeros(1,tolerance-1)'; %

%preallocate
Sensor_Active_Video_Active_tolerance= NaN(1,length(Sensor_Active_Video_Active));
Sensor_Sedentary_Video_Sedentary_tolerance= NaN(1,length(Sensor_Active_Video_Active));
Sensor_Active_Video_Sedentary_tolerance= zeros(1,length(Sensor_Active_Video_Active));
Sensor_Sedentary_Video_Active_tolerance= zeros(1,length(Sensor_Active_Video_Active));

%Active-Active Tolerance
    for i = (1 + tolerance: length(time_video_active_sedentary_annotate) + tolerance)
        for ii = 1:(tolerance - 1) %subtract 1 because you want 4 additional row comparisons (same row + 4 more)
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
        for ii = 1:(tolerance - 1) %subtract 1 because you want 4 additional row comparisons (same row + 4 more)
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
 
%Calculate validation accuracy and percentages with tolerance
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
percent_true = perc_Sensor_Sedentary_Video_Sedentary_tolerance + perc_Sensor_Active_Video_Active_tolerance

%% ID active times from sensor data to calculate number of active bouts, integral below, & other descriptors

%active_time_threshold needs to have zero instead of NaN
active_time_threshold(isnan(active_time_threshold))=0;

%ID index of start and stop active time for master table calculations
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

 %preallocate
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
   
%% ID active times from video data to calculate number of active bouts & other descriptors
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

     end

% %Summarize Duration Video for MasterTable
Sum_Video_Active_Time_Duration = sum(Duration_Video_Active_full);
Min_Video_Active_Time_Duration = min(Duration_Video_Active_full);
Max_Video_Active_Time_Duration = max(Duration_Video_Active_full);
    
  %% Apply threshold to integrated acceleration

    %Return time_threshold to NAN from 0 except for first cell which may be
    %zero if active time started at zero
    for i= 2:(length(active_time_threshold)-1)
        if active_time_threshold(i) ==0
        active_time_threshold(i) = nan;
        end
    end

    if active_time_threshold_annotate(1)==1
        active_time_threshold(1) = 0;
    else active_time_threshold(1) = nan;
    end

    % hf10=figure; 
    % g = gca;
    % hold on; grid on;
    % h10u = plot(time_zero,active_accel_threshold, 'DisplayName', babyID);
    % h7u = plot([time_zero(1),time_zero(length(time_zero))],[Threshold,Threshold],'-k', 'Linewidth', 0.5); %add horizontal line at upper threshold
    % %plot([time_zero(1),time_zero(length(time_zero))],[Mean_A_user_mag_filt,Mean_A_user_mag_filt],'-k', 'Linewidth', 1);
    % title({'Filtered Acceleration Magnitude of Active Time', 'Identified with Sensor Threshold'});
    % legend([h10u, h7u],'A_{user filtered} ', 'Threshold'); %updating legends note {_makes it subscript}
    % xlabel('Time (sec)');
    % ylabel('Acceleration (m/s/s)');
    % %xlim([0,1900]);
    % %ylim([0,10]);
    % set(gca, 'fontsize', 16);
      
%% Calcuate user acceleration with threshold applied, Trapz funciton takes integral- (area under
%%curve) 
%https://www.mathworks.com/matlabcentral/answers/853840-how-to-find-area-under-graph-between-two-points
%use indices of active time to ID time 1/accel 1 and time2/accel 2 and then
%store in new matrix and sum
cumtrap_store = zeros(length(idx_start_active_sensor),1);

for i = 1: length(idx_start_active_sensor)
cumtrap_store(i) = trapz(active_time_threshold(idx_start_active_sensor(i):(idx_stop_active_sensor(i))), active_accel_threshold(idx_start_active_sensor(i):(idx_stop_active_sensor(i))));
end

integral_sum_threshold = sum(cumtrap_store);

currentFile = 'Merged Validation Files';
    %% write file details and start and stop time to the table
    masterTable.FileName = cellstr(currentFile);
    masterTable.BabyID = cellstr(babyID);
    masterTable.Duration = time_zero(end)-time_zero(1);
    masterTable.DurationMins = (time_zero(end)-time_zero(1)) /60 ;% column of duration
    masterTable.totalAcceleration = integrate_A_user_mag_modified;% column of total Accerleration
    masterTable.timeNormalizedAcc = integrate_A_user_mag_modified/(time_zero(end)-time_zero(1));% column of time normalized acceleration
    masterTable.timeNormalizedAccMins = integrate_A_user_mag_modified/ masterTable.DurationMins(1) ;
    masterTable.thresholdTotalAcceleration = integral_sum_threshold;% column of total Accerleration
    masterTable.thresholdTimeNormalizedAcc = integral_sum_threshold/(time_zero(end)-time_zero(1));% column of time normalized acceleration
    masterTable.thresholdTimeNormalizedAccMins = integral_sum_threshold/ masterTable.DurationMins(1) ;    
    masterTable.Sum_Sensor_Active_Time_Duration = Sum_Sensor_Active_Time_Duration;% column of Total Duration of Active time 
    masterTable.Min_Sensor_Active_Time_Duration = Min_Sensor_Active_Time_Duration;% column of Minimal Duration of Active time
    masterTable.Max_Sensor_Active_Time_Duration = Max_Sensor_Active_Time_Duration ;% column of Maximum Duration of Active time
    masterTable.count_Sensor_active_time_bouts = count_Sensor_active_time_bouts;% column of numober/ count of Actime time bouts
    masterTable.TruePositive= sum_Sensor_Active_Video_Active_tolerance;% 
    masterTable.TrueNegative= sum_Sensor_Sedentary_Video_Sedentary_tolerance;% 
    masterTable.FalsePositive= sum_Sensor_Active_Video_Sedentary_tolerance;% 
    masterTable.FalseNegative= sum_Sensor_Sedentary_Video_Active_tolerance;% 
    masterTable.TruePositive_Perc= perc_Sensor_Active_Video_Active_tolerance;% 
    masterTable.TrueNegative_Perc= perc_Sensor_Sedentary_Video_Sedentary_tolerance;% 
    masterTable.FalsePositive_Perc= perc_Sensor_Active_Video_Sedentary_tolerance;% 
    masterTable.FalseNegative_Perc= perc_Sensor_Sedentary_Video_Active_tolerance;% 
    masterTable.OptimalThreshold= Threshold;%
%%
% write the table to an output file
writetable(masterTable, saveFileName)