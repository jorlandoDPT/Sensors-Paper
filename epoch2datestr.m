function[mydatastr] = epoch2datestr(epoch_time,offset_hours)
% This function converts unix epoch time in seconds to a human readable data and time string.
% The unix epoch time is time after Jan 1, 1970 0:00
% Matlab date time strings are time after Jan 1, 1900 0:00 and are in days instead of seconds
% The optional offset hours argument is used to adjust for time zone and / or daylight savings
% Written by John Prosser, November 2016, jjprosser@gmail.com

% Constants    
seconds_per_hour = 60*60;
seconds_per_day = 24*60*60;

% Set optional input to blank if not provided by user and convert to seconds
if nargin < 2, offset_hours = 0; end
offset_time = offset_hours * seconds_per_hour;

% Convert to date string using matlab datesr function
mydatastr = datestr(datenum(1970, 1, 1, 0, 0, 0) + (epoch_time - offset_time)/seconds_per_day,'yyyy-mmm-dd HH:MM:SS');

return;