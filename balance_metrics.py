#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:41:44 2023

The purpose of this module is to hold all functions created to assess balance
for BME 3740 Unit 1.

@author: sammy
"""

#%% Import Packages 

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.integrate import simps

#%% Define functions 



def get_uniform_time(data):
    """
    The purpose of this function is to deal with time recorded in a variety of units. 
    We want all our data to look similar so that we can analyze it using the same processes. 
    Thus, lets turn all our time information into elapsed time in seconds. 
    Thus, the first element in the time series should be 0s. 
    
    Let's additionally remove any columns we dont need so that each dataframe 
    here only has 4 columns: elasped time(s), accel-x (g), accel- y(g), 
    and accel-z (g)
    

    Parameters
    ----------
    data : dataframe of size (x,y), where x= number of data points collected and y=number of columns in data file
        The raw accelerometer data collected for one task.

    Returns
    -------
    adjusted_data : dataframe of size (x,4), where x= number of data points collected and 4=number of columns in data file
        The adjusted accelerometer data for a task in which time is recorded in elapsed seconds. 

    """
    
    adjusted_data = data
    first_col_name = data.columns[0]
    
    if first_col_name == 'Epoch':
        
        # subtract the first time value from all time values, then divide by 
        # 1000 becuase the data appears to be in ms and we want s. How do you
        # know its in ms, not s?
        adjusted_data['Epoch'] = (data['Epoch'] - data['Epoch'][0])/1000
        
        # update column name
        adjusted_data.rename(columns = {'Epoch':'Elasped Time (s)'}, inplace = True) 
        
    elif first_col_name == 'time':
        
        # do you have to change anything here? 
        
        
        #update column name
        adjusted_data.rename(columns = {'time':'Elasped Time (s)'}, inplace = True) 
        
    elif first_col_name == 'epoc (ms)':
        
        # adjust epoc column and get rid of timestamp and elasped cols
        
        # Subtract first time value from all time values & divide by 1000
        adjusted_data['epoc (ms)'] = (data['epoc (ms)'] - data['epoc (ms)'][0])/1000
        
        # update column name
        adjusted_data.rename(columns = {'epoc (ms)': 'Elasped Time (s)'}, inplace = True) 
       
        # Remove cols of index 1 and 2
        adjusted_data = adjusted_data.drop(adjusted_data.columns[[1,2]], axis = 1) # axis = 1 tells us we are removing cols, not rows
        
      
        
     # If there are any other cases we have to deal with, add them here using
     # elif statements.
        
    return adjusted_data
    

def calibrate_accelerometer(data):
    """
    Calibrate the 3-axis accelerometer data to the world frame.
    
    Parameters
    ----------
    data : numpy.ndarray
        3-axis accelerometer data. Shape: (N, 3) where N is the number of samples.
        
    Returns
    -------
    world_data : numpy.ndarray
        Calibrated 3-axis accelerometer data in the world frame. Shape: (N, 3) where N is the number of samples.
    """
   
    # Calculate mean acceleration vector
    mean_accel = np.mean(data, axis=0)
    mean_accel /= np.linalg.norm(mean_accel) # points down
    
    # Calculate cross product with z-axis (AP) to get the ml-axis (left)
    ml_axis = np.cross(mean_accel, [0, 0, 1])
    ml_axis /= np.linalg.norm(ml_axis) # points left
    
    # Calculate cross product between ml-axis and mean_accel to get the ap-axis (forward)
    ap_axis = np.cross(ml_axis, mean_accel)
    ap_axis /= np.linalg.norm(ap_axis) # points forward/backward
    
    # Construct rotation matrix
    rotation_matrix = np.array([ml_axis, mean_accel, ap_axis]) 
    
    
    # Rotate data to world frame
    world_data = np.dot(data, rotation_matrix)
    
    # Plot world data
    # plt.figure()
    # plt.plot(world_data)
    # plt.legend(['x','y','z'])
    # plt.title('Calibrated Data')
    
    return world_data



def get_balance_metrics(data):
    '''
    Computes the Range ML/AP, RMS ML/AP, MV, and JERK metrics for each task, for each subject. 

    Parameters
    ----------
    data : array of floats of size (x,y), in which x=the number of data points collected and y=the number of axes
        Contains the accelerometer data collected of the x, y, and z axes for a subject, for a task.

    Returns
    -------
    metrics : list of size n, where n=the number of metrics being returned
        Contains 5 floats for the Range ML/AP, RMS ML/AP, and MV metrics and 1 array of floats for the JERK metric.

    '''

    ''' NOTE! This is just a framework. You may have to change column or row 
    selections depending on how you choose to structure your data'''
    
    # Compute balance metrics
    # Range
    Range_ML = np.amax(data[:,0]) - np.amin(data[:,0])
    Range_AP = np.amax(data[:,2]) - np.amin(data[:,2])
    # RMS
    RMS_ML = np.sqrt(np.mean(np.square(data[:,0])))
    RMS_AP = np.sqrt(np.mean(np.square(data[:,2])))
    #TODO: class to add other balance metrics here
    
    # get MV
    dt = 1/50   # 1 / sampling frequency 
    integral_ML = simps(data[:,0], dx=dt)
    integral_AP = simps(data[:,2], dx=dt)
    MV = np.sqrt((integral_AP)**2 + (integral_ML)**2)
    
    # get JERK
    derivative_ML = np.gradient(data[:,0])
    derivative_AP = np.gradient(data[:,2])
    time_array = np.arange(0,len(data[:,0]), dt)
    JERK = np.zeros_like(time_array)
    for i, t in enumerate(time_array):
        JERK[i] = 0.5*simps(derivative_ML**2 + derivative_AP**2)
    
    # Assign measures to output array
    metrics = [Range_ML, Range_AP, RMS_ML, RMS_AP, MV, JERK]
    return metrics







