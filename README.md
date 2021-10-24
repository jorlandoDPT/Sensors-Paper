# Sensors-Paper

Open Source MATLAB code developed by Julie M. Orlando in collaboration with section contributions from Jocelyn Hafer (signal processing expert) and in consultation with Laura Prosser, Beth Smith, Athylia Paremski, Matthew Amodeo and Michele Lobo.

To recreate what is seen in the paper, please run `OpenSource_Validation_Sensor_Threshold.m`, which requires 4 main functions: 

1. `importTimingfile.m`

2. `epoch2datestr.m`

3. `MultiplyQuaternions.m`*

4. `RotateVector.m`*

*These functions are available on the [APDM website](http://community.apdm.com/hc/en-us/articles/214504186-Using-orientation-estimates-to-convert-from-sensor-frame-to-Earth-frame-of-refernce) as I do not own them, I will refer you to the source. 

## Inputs 

1. Timestamps.xlsx  

    - 3 columns: filename, startTime, stopTime

    - the filename starts with "EMO" followed by the participant ID. This is used to identify to participant ID in the following code. 

2. Videocode.xlsx

    - Examples of set-up are provided 

## Output

1. outputData.xlsx 

    - created by MATLAB

## Set up 

- H5 datafiles from IMU should be in a folder labeled "Datasets"

- Individual particpant files should be in individual folders within "Datasets" saved as "EMOXX" where "XX" indicates ID number

# Figures 

The figures generated from the code should look something like the figure below: 

![](EXtotal.png)



