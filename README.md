# Sensors-Paper
Open Source MATLAB code developed by Julie M. Orlando in collaboration with section contributions from Jocelyn Hafer (signal
processing expert) and in consultation with Laura Prosser, Beth Smith,
Athylia Paremski, Matthew Amodeo and Michele Lobo


Requires 4 functions: 
importTimingfile.m
epoch2datestr.m
MultiplyQuaternions.m
RotateVector.m

2 input files:
Timestamps.xlsx with 3 columns: filename, startTime, stopTime
  the filename starts with "EMO" followed by the participant ID. This is
  used to identify to participant ID in the following code. 
Videocode.xlsx
  Examples of set-up are provided 
  
1 Output file:
outputData.xlsx is created by MATLAB

Set up: 
H5 datafiles from IMU should be in a folder labeled "Datasets"
Individual particpant files should be in individual folders within "Datasets" saved as "EMOXX" where "XX" indicates ID number


