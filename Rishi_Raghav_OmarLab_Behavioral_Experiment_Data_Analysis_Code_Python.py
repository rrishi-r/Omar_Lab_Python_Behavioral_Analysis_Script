#!/usr/bin/env python #
"""\

Author: Rishi Raghav (Omar Lab: University of Michigan)

Timeframe: The following code sample was developed and revised from January 22 - February 20 as part of a long term behavioral / conditioning experiment

Code Summary:
    Overview: Behavior is measured through the type of nose poke made by the subject (recorded by FED3 device)

    This python script consists of three steps
        1. Restructuring behavioral data to account for erroneous behavior
        2. Generating behavioral classification for two subjects (+ and - for cognitive condition being studied)
        3. Generating efficiency plots for two subjects (+ and - for cognitive condition being studied)

** Note: Details regarding the specific condition and type of behavior being studied have been redacted (will be released once study is published)

"""

# Import libraries necessary for parsing / sorting CSV data and building data visualizations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

from natsort import natsorted, ns

# Initialize access to directory containing behavioral data (contained in CSV files)
os.chdir("/Users/SEQFR2_Sample")
path = os.getcwd()

# Sort CSV files by experimental procedure date
csv_files = sorted(glob.glob('*.CSV'), key=os.path.getmtime)
csv_files = sorted(glob.glob(f'{os.getcwd()}/*.CSV'), key=len)
csv_files = sorted(csv_files, key=lambda x: float(re.findall("(\d+)", x)[0]))
csv_files = natsorted(csv_files, key=lambda y: y.lower())

# Additional parameters for data visualization styling
plt.style.use("seaborn-v0_8")
plt.rcParams.update({'lines.markeredgewidth': 1})

# Loop through all CSV files in set directory and restructure / normalize data to account for erroneous pokes
for f in csv_files:

    # read in csv file
    df = pd.read_csv(f)

    # extract event column as an array from csv file
    extract_event_column = df.iloc[:, 7]
    event_column = df.iloc[:, 7].values

    index = 0

    # Set trials where the first type of poke error occurs to "Error_1"
    # Set trials where the second type of poke error occurs to "Error_2"
    for i in range(len(event_column)):
        index = index + 1
        if event_column[i] == "Start" or event_column[i] == "Incorrect_Timeout" or event_column[i] == "Correct_Timeout":
            continue
        elif (event_column[i-1] == "Incorrect_Timeout" or event_column[i-1] == "Correct_Timeout") and (event_column[i] == "Right"):
            for j in range(i+1, len(event_column)):
                if event_column[j] != "Incorrect_Timeout":
                    if event_column[j] == "Right":
                        event_column[j] = "Error_1"
                    elif event_column[j] == "Left":
                        print("Condition Satisfied")
                        event_column[j] = "Error_2"
                    continue
                else:
                    break
            continue
        elif (event_column[i-1] == "Incorrect_Timeout" or event_column[i-1] == "Correct_Timeout") and (event_column[i] == "RightWithPellet" or event_column[i] == "LeftWithPellet"):
            for j in range(i+1, len(event_column)):
                if event_column[j] != "Incorrect_Timeout":
                    if event_column[j] == "Right":
                        event_column[j] = "Error_1"
                    elif event_column[j] == "Left":
                        event_column[j] = "Error_2"
                    continue
                else:
                    break
            continue
        elif event_column[i-1] == "Start" and (event_column[i] == "Right"):
            for j in range(i+1, len(event_column)):
                if event_column[j] != "Incorrect_Timeout":
                    if event_column[j] == "Right":
                        event_column[j] = "Error_1"
                    elif event_column[j] == "Left":
                        event_column[j] = "Error_2"
                    continue
                else:
                    break
            continue
        elif event_column[i-1] == "Start" and (event_column[i] == "RightWithPellet" or event_column[i] == "LeftWithPellet"):
            for j in range(i+1, len(event_column)):
                if event_column[j] != "Incorrect_Timeout":
                    if event_column[j] == "Right":
                        event_column[j] = "Error_1"
                    elif event_column[j] == "Left":
                        event_column[j] = "Error_2"
                    continue
                else:
                    break
            continue
        elif event_column[i-1] == "Pellet" and (event_column[i] == "Right"):
            for j in range(i+1, len(event_column)):
                if event_column[j] != "Incorrect_Timeout":
                    if event_column[j] == "Right":
                        event_column[j] = "Error_1"
                    elif event_column[j] == "Left":
                        event_column[j] = "Error_2"
                    continue
                else:
                    break
            continue
        elif event_column[i-1] == "Pellet" and (event_column[i] == "RightWithPellet" or event_column[i] == "LeftWithPellet"):
            for j in range(i+1, len(event_column)):
                if event_column[j] != "Incorrect_Timeout":
                    if event_column[j] == "Right":
                        event_column[j] = "Error_1"
                    elif event_column[j] == "Left":
                        event_column[j] = "Error_2"
                    continue
                else:
                    break
            continue
        elif event_column[i-1] == "Left" and event_column[i] == "Left":
            for j in range(i + 1, len(event_column)):
                if event_column[j] != "Incorrect_Timeout":
                    if event_column[j] == "Right":
                        event_column[j] = "Error_1"
                    elif event_column[j] == "Left":
                        event_column[j] = "Error_2"
                    continue
                else:
                    break
            continue

    # Update CSV file to account for revised event column
    df.to_csv(f, index=False)



# Read in updated CSV files for the two subjects
df_1 = pd.read_csv(csv_files[0])
df_2 = pd.read_csv(csv_files[1])


# Extract event column of data (contains behavior type) from updated CSV files
extract_event_column_1 = df_1.iloc[:, 7]
extract_event_column_2 = df_2.iloc[:, 7]


# Pass behavioral type columns to an array for each of the two subjects
event_column_1 = df_1.iloc[:, 7].values
event_column_2 = df_2.iloc[:, 7].values


# Initialize variables corresponding to updated CSV files for two subjects (later passed as an argument to plotting functions)
source_csv_1 = csv_files[0]
source_csv_2 = csv_files[1]

active_bias_smooth_window = 1
efficiency_group_size = 1
retrieval_group_size = 1


# Evaluates each event (poke made by subject) in the data log from the FED3 device as correct, incorrect, error (for both subjects)
# Correct corresponds to "active_poke", incorrect corresponds to "poke_attempt" and error corresponds to "error"
def evaluate_events(df_1,df_2):
    # Creates a list called "conditions" containing all possible events in the CSV file
    # Correct poke sequence by subject results in a pellet being dispensed
    # Incorrect pokes correspond to "Error_1", "Error_2", "RightWithPellet", and "LeftWithPellet"
    conditions = [
        (df_1['Event'] == "Pellet"),
        (df_1['Event'] == "Incorrect_Timeout"),
        (df_1['Event'] == "Correct_Timeout"),
        ((df_1['Event'] == "Error_1")
        |(df_1['Event'] == "Error_2")),
        ((df_1['Event'] == "RightWithPellet") | (df_1['Event'] == "LeftWithPellet")),
        ((df_1['Event'] == "Left") | (df_1['Event'] == "Right")),
    ]

    # Create a list of the values we want to assign for each condition
    values = ['Active_Poke', 'Invalid', 'Invalid', 'Error', 'Bad_Poke', 'Poke_Attempt']

    # Create a new column and use np.select to assign values to it using our lists as arguments
    df_1["Evaluation"] = np.select(conditions, values)
    df_1["Is_Correct"] = df_1["Evaluation"] == "Active_Poke"
    df_1["Is_Poke"] = (df_1["Evaluation"] == "Bad_Poke") | (df_1["Evaluation"] == "Poke_Attempt")


    conditions = [
        (df_2['Event'] == "Pellet"),
        (df_2['Event'] == "Incorrect_Timeout"),
        (df_2['Event'] == "Correct_Timeout"),
        ((df_2['Event'] == "Error_1")
         | (df_2['Event'] == "Error_2")),
        ((df_2['Event'] == "RightWithPellet") | (df_2['Event'] == "LeftWithPellet")),
        ((df_2['Event'] == "Left") | (df_2['Event'] == "Right")),
    ]

    values = ['Active_Poke', 'Invalid', 'Invalid', 'Error', 'Bad_Poke', 'Poke_Attempt']

    df_2["Evaluation"] = np.select(conditions, values)

    df_2["Is_Correct"] = df_2["Evaluation"] == "Active_Poke"
    df_2["Is_Poke"] = (df_2["Evaluation"] == "Bad_Poke") | (df_2["Evaluation"] == "Poke_Attempt")




################################################################
# Calculates cumulative statistics needed for poke efficiency calculations
# REQUIRES: Cumulative cumulative correct pokes and cumulative pokes
def calc_cumulative_stats(df_1, df_2, smooth_window=1):
    df_1["Cumulative_Correct"] = df_1["Is_Correct"].cumsum()
    df_1["Cumulative_Pokes"] = df_1["Is_Poke"].cumsum()
    df_1["Active_Bias"] = df_1["Cumulative_Correct"] / df_1["Cumulative_Pokes"]

    df_1['Smooth_Active_Bias'] = df_1['Active_Bias'].rolling(smooth_window).mean()

    # Converts time to DateTime object and adds time_elapsed column (hours, float)
    df_1["MM:DD:YYYY hh:mm:ss"] = pd.to_datetime(df_1["MM:DD:YYYY hh:mm:ss"], format="%m/%d/%Y %H:%M:%S")
    start_time = df_1["MM:DD:YYYY hh:mm:ss"][0]

    df_1["time_elapsed"] = (df_1["MM:DD:YYYY hh:mm:ss"] - start_time).dt.total_seconds() / 3600


    df_2["Cumulative_Correct"] = df_2["Is_Correct"].cumsum()
    df_2["Cumulative_Pokes"] = df_2["Is_Poke"].cumsum()
    df_2["Active_Bias"] = df_2["Cumulative_Correct"] / df_2["Cumulative_Pokes"]

    df_2['Smooth_Active_Bias'] = df_2['Active_Bias'].rolling(smooth_window).mean()

    df_2["MM:DD:YYYY hh:mm:ss"] = pd.to_datetime(df_2["MM:DD:YYYY hh:mm:ss"], format="%m/%d/%Y %H:%M:%S")
    start_time = df_2["MM:DD:YYYY hh:mm:ss"][0]

    df_2["time_elapsed"] = (df_2["MM:DD:YYYY hh:mm:ss"] - start_time).dt.total_seconds() / 3600


# Initialize variables for poke frequency (for each type of poke)
correct_right_trials_positive = 0
correct_right_trials_negative = 0

incorrect_right_trials_positive = 0
incorrect_right_trials_negative = 0

correct_left_trials_positive = 0
correct_left_trials_negative = 0

incorrect_left_trials_positive = 0
incorrect_left_trials_negative = 0

error_1_positive = 0
error_2_positive = 0

error_1_negative = 0
error_2_negative = 0


# Loop through event column for both subjects and determine frequency for each type of poke
for l in range(len(event_column_1)):
    if event_column_1[l] == "Right" and event_column_1[l - 1] == "Left":
        correct_right_trials_positive = correct_right_trials_positive + 1
    elif event_column_1[l] == "Right" and (
            event_column_1[l - 1] == "Incorrect_Timeout" or event_column_1[l - 1] == "Start" or event_column_1[
        l - 1] == "Pellet"):
        incorrect_right_trials_positive = incorrect_right_trials_positive + 1
    elif event_column_1[l] == "Left" and (
            event_column_1[l - 1] == "Incorrect_Timeout" or event_column_1[l - 1] == "Start" or event_column_1[
        l - 1] == "Pellet"):
        correct_left_trials_positive = correct_left_trials_positive + 1
    elif event_column_1[l] == "Left" and event_column_1[l - 1] == "Left":
        incorrect_left_trials_positive = incorrect_left_trials_positive + 1
    elif event_column_1[l] == "Error_1":
        error_1_positive = error_1_positive + 1
    elif event_column_1[l] == "Error_2":
        error_2_positive = error_2_positive + 1


for l in range(len(event_column_2)):
    if event_column_2[l] == "Right" and event_column_2[l - 1] == "Left":
        correct_right_trials_negative = correct_right_trials_negative + 1
    elif event_column_2[l] == "Right" and (
            event_column_2[l - 1] == "Incorrect_Timeout" or event_column_2[l - 1] == "Start" or event_column_2[
        l - 1] == "Pellet"):
        incorrect_right_trials_negative = incorrect_right_trials_negative + 1
    elif event_column_2[l] == "Left" and (
            event_column_2[l - 1] == "Incorrect_Timeout" or event_column_2[l - 1] == "Start" or event_column_2[
        l - 1] == "Pellet"):
        correct_left_trials_negative = correct_left_trials_negative + 1
    elif event_column_2[l] == "Left" and event_column_2[l - 1] == "Left":
        incorrect_left_trials_negative = incorrect_left_trials_negative + 1
    elif event_column_2[l] == "Error_1":
        error_1_negative = error_2_negative + 1
    elif event_column_2[l] == "Error_2":
        error_2_negative = error_2_negative + 1

# Initialize x-axis labels (Type of Poke / Subject + or -)
categories = ["Correct Right Poke: +","Correct Right Poke: -", "Incorrect Right Poke: +",
              "Incorrect Right Poke: -", "Correct Left Poke: +", "Correct Left Poke: -","Incorrect Left Poke: +",
              "Incorrect Left Poke: -", "Error 1: +",  "Error 1: -",
              "Error 2: +", "Error 2: -"]

number_of_bars = np.arange(len(categories))

# Initialize y-axis values (Poke Frequency)
poke_frequency = [correct_right_trials_positive, correct_right_trials_negative, incorrect_right_trials_positive,
incorrect_right_trials_negative, correct_left_trials_positive, correct_left_trials_negative, incorrect_left_trials_positive,
incorrect_left_trials_negative, error_1_positive,  error_2_positive,  error_1_negative, error_2_negative]

# Plot bar graph for poke classification
fig, ax = plt.subplots()
ax.bar(number_of_bars, poke_frequency,  align='edge', alpha=0.5, ecolor='black', capsize=10, width = 0.5)
ax.set_ylabel("Poke Frequency")
ax.set_xticks(number_of_bars)
ax.set_xticklabels(categories)
ax.set_title('Poke Classification for Two Subjects from Subject 1 and Subject 2')
ax.yaxis.grid(True)
plt.xticks(rotation = 60)


# Save the figure and show in output
plt.tight_layout()
plt.savefig("SEQFR2_Phase_Poke_Classification_Subject_1_and_Subject_2_Day_1.png")
plt.show()


# Plots cumulative poke efficiency as a continuous graph
# REQUIRES: time_elapsed, Smooth_Active_Bias columns
def plot_cont_poke_efficiency(df_1, df_2):
    plt.figure(figsize=(8, 8))
    Cohort_1 = plt.plot(df_1["time_elapsed"],df_1["Smooth_Active_Bias"], 'b-')


    plt.axhline(y=0.5, color='r', linewidth=2)
    plt.ylim(0, 1)
    plt.xlim(0, 3.5)
    plt.title('Poke Efficiency Over Time For Cohort 1 and Cohort 2')
    plt.xlabel("Time since start (hours)")
    plt.ylabel("Poke Efficiency (smoothed)")
    plt.figlegend((Cohort_1), (['Cohort 1']), loc='upper right')

    Cohort_2 = plt.plot(df_2["time_elapsed"], df_2["Smooth_Active_Bias"], 'g-')

    plt.axhline(y=0.5, color='r', linewidth=2)
    plt.ylim(0, 1)
    plt.xlim(0, 3.5)
    plt.title('Poke Efficiency Over Time For Subject 1 and Subject 2')
    plt.xlabel("Time since start (hours)")
    plt.ylabel("Poke Efficiency (smoothed)")
    plt.figlegend((Cohort_2), (['Cohort 2']), loc='upper left')

    # Save the figure and show in output
    plt.savefig("SEQFR2_Phase_Poke_Efficiency_Subject_1_and_Subject_2_Day_1.png")
    plt.show()

# Initialize arguments to efficiency function (CSV files for two subjects)
log_df_1 = pd.read_csv(source_csv_1)
log_df_2 = pd.read_csv(source_csv_2)

# Call functions for evaluating behavioral events and cumulative statistics, and graph efficiency plot
evaluate_events(log_df_1, log_df_2)
calc_cumulative_stats(log_df_1, log_df_2,  smooth_window=active_bias_smooth_window)
plot_cont_poke_efficiency(log_df_1, log_df_2)