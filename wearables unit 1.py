#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 12:02:19 2024
BME 3740 Unit 1
ISway Replication Analysis

@author: bella
"""


#%% Import Packages

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import balance_metrics as bm
from scipy import stats
from tabulate import tabulate
# from tabulate import tabulate
# import os
# import io
# from scipy.stats import pearsonr
# import seaborn as sb

# %% Define Parameters

fs = 50  # sampling frequency of 50 samples/sec
task_count = 4 # number of tasks
metrics_count = 4 # number of metrics to be calculated
subject_count = 5 # number of subjects
task_numbers = range(1, task_count+1) # ranges number of tasks

#%% Import Data

# create dict
all_data = {}
# import data for each subject
for subject in range (1,subject_count +1):
    # get subject id's
    subject_id = f'subject{subject}'
    # create lists within each subject dict
    all_data[subject_id] = {'raw_data': {}, 'calibrated_data': [], 'balance_metrics': []}
    # import data for each task, for each subject
    for task in range (1, task_count+1):
        # get file path
        file_path = f'data/subject{subject}_task{task}.csv'
        # read files
        task_data = pd.read_csv(file_path)
        # add task data to dictionary
        all_data[subject_id]['raw_data'][f'task{task}'] = task_data

# remove timestamp and elasped time columns from subject 5 data
for task in range(1, task_count + 1):
    all_data['subject5']['raw_data'][f'task{task}'] = pd.DataFrame(all_data['subject5']['raw_data'][f'task{task}']).drop(columns=['timestamp (-0500)', 'elapsed (s)'])

# %% Get Uniform Time

# define looping variable
subject_keys = list(all_data.keys())

# for current_subject in subject_keys:
#     access_subect = all_data[current_subject]['raw_data']
#     for current_task in task_numbers:
#         all_data[current_subject]['raw_data'][f'task{current_task}'] = bm.get_uniform_time(all_data[current_subject]['raw_data'][f'task{current_task}'])

#%% Calibrate Data

# loop through each subject
for current_subject in subject_keys :
    access_subjects = all_data[f'{current_subject}']['raw_data']
    calibrated_subject_data = []
    # loop through each task for each subject
    for current_task in task_numbers:
        current_task_data = access_subjects[f'task{current_task}']
        # calibrate data
        calibrated_task_data = pd.DataFrame(bm.calibrate_accelerometer(current_task_data.iloc[:,1:]), columns = ['Ax (g)', 'Ay (g)', 'Az (g)'])
        # add calibrated task data to calibrated subject data
        calibrated_subject_data.append(calibrated_task_data) 
    # add calibrated subject data to subject dictionary
    all_data[current_subject]['calibrated_data'] = calibrated_subject_data

#%% Question 1

# sort data
# uncalibrated baseline accelerometer data
subject3_task1_x = all_data['subject3']['raw_data']['task1']['X']
subject3_task1_y = all_data['subject3']['raw_data']['task1']['Y']
subject3_task1_z = all_data['subject3']['raw_data']['task1']['Z']
# calibrated baseline accelerometer data
subject3_task1_cx = all_data['subject3']['calibrated_data'][0]['Ax (g)']
subject3_task1_cy = all_data['subject3']['calibrated_data'][0]['Ay (g)']
subject3_task1_cz = all_data['subject3']['calibrated_data'][0]['Az (g)']

#create time array
get_time_length = all_data['subject3']['calibrated_data'][0]['Ax (g)']
baseline_time = np.arange(0, len(get_time_length)/fs, 1/fs)

# plot axes of acceleration, calibrated
plt.figure(2, clear=True)
plt.subplot(1,2,1)
plt.plot(baseline_time, subject3_task1_cx, label='x-axis')
plt.plot(baseline_time, subject3_task1_cy, label='y-axis')
plt.plot(baseline_time, subject3_task1_cz, label='z-axis')
# annotate plot
plt.xlabel('time (s)')
plt.ylabel('acceleration (g)')
plt.title('Subject 3 Calibrated Baseline Acceleration')
plt.legend()
# plot axes of acceleration, uncalibrated
plt.subplot(1,2,2)
plt.plot(baseline_time, subject3_task1_x, label='x-axis')
plt.plot(baseline_time, subject3_task1_y, label='y-axis')
plt.plot(baseline_time, subject3_task1_z, label='z-axis')
# annotate plot
plt.xlabel('time (s)')
plt.ylabel('acceleration (g)')
plt.title('Subject 3 Uncalibrated Baseline Acceleration')
plt.legend()

plt.tight_layout()

# save figure
plt.savefig('subject3_baseline_acceleration.jpg')

# %% Get Metrics & Plot
'''
- 2 figures (1 range, 1 RMS - could do subplots), 8 box plots (4 each)
- range and RMS calculated using formulas from original study report
- figure description in write up needs to differentiate between 2 box plot
- differences can indicate difference in sway/balance
- difference between conditions - ML dizzy/ML nondizzy + AP dizzy/AP nondizzy
- comparison ML baseline vs ML dizzy
- functions online to carry out t-tests

10 figures
    2 figures each subject: 1 RMS, 1 Range
        8 boxplots: 1 AP, 1 ML per task 

'''
# define task index looping variable
task_range = range(0, 4)
# loop through each subject
for current_subject in subject_keys:
    # create task lists within balance metrics list
    all_data[current_subject]['balance_metrics'] = {'task0': [], 'task1': [], 'task2': [], 'task3': []}
    # access current subject
    current_subject_data = all_data[current_subject]['calibrated_data']
    # loop through each task
    for current_task in task_range:
        # access current task
        current_task_data = current_subject_data[current_task]
        # convert to array
        current_task_data_array = current_task_data.values
        # get metrics
        task_balance_metrics = bm.get_balance_metrics(current_task_data_array)
        # add metrics to each task list
        all_data[current_subject]['balance_metrics'][f'task{current_task}'] = task_balance_metrics

# create individual metric lists
RMS_data_ML = {}
RMS_data_AP = {}
Range_data_ML = {}
Range_data_AP = {}
MV_data = {}
JERK_data = {}

# loop through each task
for task in task_range:
    # define metric lists for each task
    RMS_task_ML = []
    RMS_task_AP = []
    Range_task_ML = []
    Range_task_AP = []
    MV_task = []
    JERK_task = []
    # loop through each subject 
    for current_subject in subject_keys:
        # access metrics for current subject and current task
        current_task_metrics = all_data[current_subject]['balance_metrics'][f'task{task}']
        # add metrics to respective lists
        Range_task_ML.append(current_task_metrics[0])
        Range_task_AP.append(current_task_metrics[1])
        RMS_task_ML.append(current_task_metrics[2])
        RMS_task_AP.append(current_task_metrics[3])
        MV_task.append(current_task_metrics[4])
        JERK_task.append(current_task_metrics[5])
    # add each task metric to metric lists
    Range_data_ML[task] = Range_task_ML  
    Range_data_AP[task] = Range_task_AP  
    RMS_data_ML[task] = RMS_task_ML  
    RMS_data_AP[task] = RMS_task_AP  
    MV_data[task] = MV_task
    JERK_data[task] = JERK_task

# plot range values
plt.figure(3, clear=True)
plt.subplot(2,1,1)
plt.boxplot(list(Range_data_ML.values()), positions = np.arange(1,8,2) - 0.15, widths = 0.3)
plt.boxplot(list(Range_data_AP.values()), positions = np.arange(2,9,2) - 0.15, widths = 0.3)
# annotate plot
plt.title('ML and AP Range Values')
plt.xlabel('Task')
plt.ylabel('Range')
plt.xticks(range(1,9), ['Task 1 ML', 'Task 1 AP', 'Task 2 ML', 'Task 2 AP', 'Task 3 ML', 'Task 3 AP', 'Task 4 ML', 'Task 4 AP'], rotation=10)
plt.grid()
plt.tight_layout(pad=.5)

# plot RMS values
plt.subplot(2,1,2)
plt.boxplot(list(RMS_data_ML.values()), positions = np.arange(1,8,2) - 0.15, widths = 0.3)
plt.boxplot(list(RMS_data_AP.values()), positions = np.arange(2,9,2) - 0.15, widths = 0.3)
# annotate plot
plt.title('ML and AP RMS Values')
plt.xlabel('Task')
plt.ylabel('RMS')
plt.xticks(range(1,9), ['Task 1 ML', 'Task 1 AP', 'Task 2 ML', 'Task 2 AP', 'Task 3 ML', 'Task 3 AP', 'Task 4 ML', 'Task 4 AP'], rotation=10)
plt.grid()
plt.tight_layout(pad=.5)

plt.savefig('range_and_rms_plots.jpg')

# plot mean velocity
plt.figure(4, clear=True)
# plt.subplot(2,1,1)
plt.boxplot(list(MV_data.values()), widths = 0.3)
# annotate plot
plt.title('Mean Velocity Values')
plt.xlabel('Task')
plt.ylabel('Mean Velocity')
# plt.xticks(range(1,9), ['Baseline ML', 'Baseline AP', 'Normal, Eyes Closed ML', 'Normal, Eyes Closed AP', 'Toe to Heel, Eyes Closed ML', 'Toe to Heel, Eyes Closed AP', 'Dizzy Bat ML', 'Dizzy Bat AP'], rotation=30)
plt.grid()
plt.tight_layout(pad=.5)
plt.savefig('mv_plot.png')

  
# %% Means and SDs

# create means dictionary
metric_means = {'task0': [], 'task1': [], 'task2': [], 'task3': []}

# get metric means for each task
for current_task in task_range:
    RMS_ML_task_mean = np.mean(RMS_data_ML[current_task])
    RMS_AP_task_mean = np.mean(RMS_data_AP[current_task])
    Range_ML_task_mean = np.mean(Range_data_ML[current_task])
    Range_AP_task_mean =np.mean(Range_data_AP[current_task])
    MV_task_mean = np.mean(MV_data[current_task])
    metric_means[f'task{current_task}'] = [RMS_ML_task_mean, RMS_AP_task_mean, Range_ML_task_mean, Range_AP_task_mean, MV_task_mean]

JERK_plot_data = {}
# loop through each task 
for task in task_range:
    # get plotting data
    JERK_plot_task = []
    # create list to store JERK subject means
    subject_means = []  
    # loop through each subject
    for subject_JERK_data in JERK_data[task]:
        # get JERK mean for each subject
        subject_JERK_mean = np.mean(subject_JERK_data)
        # add to subject means list
        subject_means.append(subject_JERK_mean)
    # get mean for each task
    mean_JERK_task = np.mean(subject_means)
    # plotting data
    JERK_plot_data[task] = subject_means
    # add JERK means to metric means
    metric_means[f'task{task}'].append(mean_JERK_task)

# plot JERK
plt.figure(5, clear = True)    
# plt.subplot(2,1,2)
plt.boxplot(list(JERK_plot_data.values()), widths = 0.3)
# annotate plot
plt.title('Mean JERK Values Per Task')
plt.xlabel('Task')
plt.ylabel('Mean JERK')
plt.grid()
plt.tight_layout()

plt.savefig('jerk_plot.png')

# create std metrics dict
metric_std = {'task0': [], 'task1': [], 'task2': [], 'task3': []}
# get metric std for each task
for current_task in task_range:
    RMS_ML_task_std = np.std(RMS_data_ML[current_task])
    RMS_AP_task_std = np.std(RMS_data_AP[current_task])
    Range_ML_task_std = np.std(Range_data_ML[current_task])
    Range_AP_task_std =np.std(Range_data_AP[current_task])
    MV_task_std = np.std(MV_data[current_task])
    # add to task list
    metric_std[f'task{current_task}'] = [RMS_ML_task_std, RMS_AP_task_std, Range_ML_task_std, Range_AP_task_std, MV_task_std]

for task in task_range:
    subject_std = []
    for subject_JERK_data in JERK_data[task]:
        # get JERK std for each subject
        subject_JERK_std = np.std(subject_JERK_data)
        # add to subject std list
        subject_std.append(subject_JERK_std)
    # get std for each task
    std_JERK_task = np.mean(subject_std)
    # append to std metrics dict
    metric_std[f'task{task}'].append(std_JERK_task)

# create std and means table

table_data = []
headers = ["Task", "RMS_ML Mean", "RMS_AP Mean","Range_ML Mean", "Range_AP Mean",
           "MV Mean", "JERK Mean", "RMS_ML Std", "RMS_AP Std", "Range_ML Std",
           "Range_AP Std", "MV Std", "JERK Std"]

# loop thru metrics for each task
for task in task_range:
    task_data = [f"Task {task}"]
    task_data.extend(metric_means[f'task{task}'])
    task_data.extend(metric_std[f'task{task}'])
    table_data.append(task_data)

# put table into text
table_text = tabulate(table_data, headers=headers)

# Create image
image = Image.new('RGB', (1100, 500), color='white')
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

# Put table into text
table_text = tabulate(table_data, headers=headers)

# Add text to image
draw.text((10, 10), table_text, fill='black', font=font)

# Save image
image.save('table_image.png')

print(table_text)

# %% Stats

for task in task_range:
    # define significance level
    alpha = 0.05
    
    # Range t-test
    range_t_statistic, range_p_value = stats.ttest_rel(Range_data_AP[task], Range_data_ML[task])
    if range_p_value < alpha:
        print(f"For Task {task}: There is a statistically significant difference between Range AP and ML metrics.")
    else:
        print(f"For Task {task}: There is no statistically significant difference between Range AP and ML metrics.")
    
    # RMS t-test
    rms_t_statistic, rms_p_value = stats.ttest_rel(RMS_data_AP[task], RMS_data_ML[task])
    if rms_p_value < alpha:
        print(f"For Task {task}: There is a statistically significant difference between RMS AP and ML metrics.")
    else:
        print(f"For Task {task}: There is no statistically significant difference between RMS AP and ML metrics.")
       
    # MV t-test
    q3_t_statistic, q3_p_value = stats.ttest_rel(MV_data[task], JERK_plot_data[task])
    if q3_p_value < alpha:
        print(f"For Task {task}: There is a statistically significant difference between MV and JERK metrics.")
    else:
        print(f"For Task {task}: There is no statistically significant difference between MV and JERK metrics.")

# combine AP and ML data
all_AP_metrics = [val for sublist in list(Range_data_AP.values()) + list(RMS_data_AP.values()) for val in sublist]
all_ML_metrics = [val for sublist in list(Range_data_ML.values()) + list(RMS_data_ML.values()) for val in sublist]

# t-test
directional_t_statistic, directional_p_value = stats.ttest_ind(all_AP_metrics, all_ML_metrics)
if directional_p_value < alpha:
    print("There is a statistically significant difference between AP and ML metrics across testing conditions.")
else:
    print("There is no statistically significant difference between AP and ML metrics across testing conditions.")
# %% Task Difficulty Scatterplot

plt.figure(6, clear=True)

# get Range AP/ML, MV, and JERK values
for metric_index in range(2, 6):  
    # scatter plot axis values
    x_values = []
    y_values = []
    
    # Loop through each task
    for task in task_range:
        # get mean
        current_task_mean = metric_means[f'task{task}'][metric_index]
        
        # append to axis lists
        x_values.append(task)
        y_values.append(current_task_mean)
        
    # add to scatter plot
    plt.scatter(x_values, y_values, label=f'Metric Index {metric_index}')

# annotate plot 
plot_labels = ['Range_ML', 'Range_AP', 'MV', 'JERK']
plt.legend(labels=plot_labels)
plt.title('Metrics Across Tasks: Do they correlate with task difficulty?')
plt.xlabel('Task')
plt.ylabel('Mean Value')
plt.xticks(range(len(task_range)), labels=['Task 1', 'Task 2', 'Task 3', 'Task 4'])
plt.savefig('task_difficulty_plot.jpg')



#%% Notes
'''
    
sub3: x axis, column 1 = traditionally z-axis, up & down (hovering around 1 G)
sub4: recorded on android, x-axis is inverted (negative values)

RMS & Range look at different axes individually- ML and AP.

Question 3
- write function to calculate jerk
- repeat question 2 with jerk and velocity --> create a function in question 2 to reuse***
- loop through subjects and tests
- write a hypothesis for objective 1 in protocol

Question 4
- do sway metrics correlate with task difficulty? does sway metric increase as task difficulty
increases? 
- tasks linearly increase in task difficulty 
- correlation coefficient using pearson product moment or spearmen


QUESTION 5: develop range for baseline health - eliminate outliers?

hints:
    - create module for functions 

'''


