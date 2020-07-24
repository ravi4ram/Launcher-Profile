# Launch Vehicle Flight Profile (PSLV, GSLV Mk-II, GSLV Mk-III)
> This script creates profile plots for altitude, velocity, acceleration and dynamic pressure.   

## Table of contents
* [General info](#general-info)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [How to run ](#how)
* [Updates](#updates)
* [To-do list](#to-do)


## Screenshots
Output maps generated by the script.   
####GSLV-MkIII 
![results](./img/gslv-mk3-d2-gsat-29_acel_vs_time.png)   
![results](./img/gslv-mk3-d2-gsat-29_alt_vs_dyn_pressure.png) 
![results](./img/gslv-mk3-d2-gsat-29_merged_display.png)   
####GSLV-MkII    
![results](./img/gslv-mk2-f08-gsat-6a_acel_vs_time.png)   
![results](./img/gslv-mk2-f08-gsat-6a_alt_vs_dyn_pressure.png) 
![results](./img/gslv-mk2-f08-gsat-6a_merged_display.png)   
####PSLV     
![results](./img/pslv-c26-irnss-1c_acel_vs_time.png)
![results](./img/pslv-c26-irnss-1c_alt_vs_dyn_pressure.png)
![results](./img/pslv-c26-irnss-1c_merged_display.png)

## General info
This project was started as a result of non-availability of ISRO launch vehicle profile plots. Data is included for one mission of PSLV, GSLV Mk-II and GSLV Mk-III each and can be extended as explained below.  This script generates plots with altitude, velocity, acceleration and dynamic pressure profiles.  
&nbsp;   
With the lack of publicly available data, I ended-up extracting data from the screen shot of the televised launch.  Used the software [Engauge Digitizer](https://markummitchell.github.io/engauge-digitizer/) 
to extract data points from images of graphs. These image plots contains both altitude and relative velocity against time on the same graph (ISRO's merged display plots).   
![results](./img/gslv-mk3-d2-gsat29-48.jpg)   
![results](./img/gslv-mk2-f08-gsat-6a-42.jpg)
![results](./img/pslv-c26-irnss-1c-11.jpg)    

Extracted data files are   

1.  gslv-mk3-d2-gsat29-48-alt.dat  
2.  gslv-mk3-d2-gsat29-48-vel.dat  
3.  gslv-f08-gsat6a-42-alt.dat  
4.  gslv-f08-gsat6a-42-vel.dat    
5.  pslv-c26-irnss-1c-11-alt.dat  
6.  pslv-c26-irnss-1c-11-vel.dat   

&nbsp;   
Additional data's were collected from flight events listing on the launch vehicle brochure. They are converted as dat files.    

![results](./img/gslv-mk3-d2-flight-events.png)   
![results](./img/gslv-f08-flight-events.png)   
![results](./img/pslv-c26-flight-events.png)   

Extracted data files are    

1.  gslv-mk3-d2-gsat29-flight-events.dat  
2.  gslv-f08-gsat6a-flight-events.dat  
3.  pslv-c26-irnss-1c-flight-events.dat   

&nbsp;   
Atmospheric density information's are interpolated from the tables under the data file named,   

1.  atm_data.dat   

## Setup
Script is written with python (Version: 3.6) on linux. Additional modules required :   

* numpy  (tested with Version: 1.18.4 )
* matplotlib  (tested with Version: 2.1.1 )
* scipy (tested with Version:  0.19.1 )


## How to run   
* Verify and install required modules 
* run `python flight_profile.py`. 
* It generates six png files at the current directory and opens the plot window one by one.  

## Updates   
* [16July2020]  
*   *  Corrected acceleration profile plot 
* [17July2020] 
*   *  Corrected plot font sizes.  
*   *  Included options for noise filtered plot  
*       * Set the flag in line no. 383 of the code `apply_low_noise_filter_flag = True` to view the plot accordingly.
* [24July2020]   
*   *  Replaced savitzky_golay filtering function with Butterworth filter
*   *  Included data files for
*   *   *  GSLV-Mk-II (GSLV-F08-GSAT-6A)
*   *   *  
## To-do list
* Other launch profile plots.

