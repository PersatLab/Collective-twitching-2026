"""
Utilities for track analysis and plotting
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
from collections import defaultdict

# Define a function to clean up loaded spots.csv dataframe

def load_csv_files_with_spots(directory):
    """
    Loads all CSV files ending with '_spots' from a specified directory into pandas DataFrames.

    Parameters:
    - directory: str
        The path to the directory containing the CSV files.

    Returns:
    - dict
        A dictionary where the keys are the filenames and the values are the corresponding DataFrames.
    """
    
    dataframes = {}
    for filename in os.listdir(directory):
        if filename.endswith("_spots.csv"):
            filepath = os.path.join(directory, filename)
            try:
                # Try loading with utf-8 encoding
                #dataframes[filename] = pd.read_csv(filepath, encoding='utf-8')
                dataframes[filename] = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
                print(f"Loaded {filename} with utf-8 encoding")
            except UnicodeDecodeError:
                print(f"Failed to load {filename} with utf-8 encoding. Trying with latin1 encoding.")
                try:
                    # If utf-8 fails, try loading with latin1 encoding
                    #dataframes[filename] = pd.read_csv(filepath, encoding='latin1')
                    dataframes[filename] = pd.read_csv(filepath, encoding='latin1', low_memory=False)
                    print(f"Loaded {filename} with latin1 encoding")
                except Exception as e:
                    print(f"Failed to load {filename} due to unexpected error: {e}")
    
    return dataframes

def clean_trc_csv(dataframe):
    """
    Clean the loaded spot.csv file from TrackMate.

    Removes rows and columns that are not necessary. 
    Resets the index and renames the ID column to SPOTS_ID for clarity. 

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe that contains the TrackMate spot information, loaded from a csv file.

    Returns
    -------
    dataframe : pandas dataframe
        The cleaned up datafranme.

    Raises
    ------
    TypeError
        If the variable is not a pandas dataframe.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"must be pandas dataframe")        

    # rename ID column label
    if not "SPOT_ID" in dataframe.columns and "ID" in dataframe.columns:
        dataframe = dataframe.rename(columns={"ID": "SPOT_ID"})

        # Remove rows with NaN TRACK_ID
    dataframe = dataframe.dropna(subset=["TRACK_ID"])
    
    col2keep = ["SPOT_ID", "TRACK_ID", "POSITION_X", "POSITION_Y", "FRAME", "ELLIPSE_MAJOR", "ELLIPSE_MINOR", "ELLIPSE_THETA", "ELLIPSE_ASPECTRATIO", "AREA"] # will keep only these colums  
    dataframe = dataframe[col2keep] # removes all unwanted columns

    # remove rows and colums that are not required
    dataframe = dataframe.drop([0,1]) # remove useless rows

    # convert numbers into int or float
    dtypes_map = {"SPOT_ID":int, "TRACK_ID":int, "POSITION_X":float, "POSITION_Y":float, "FRAME":int, "ELLIPSE_MAJOR":float, "ELLIPSE_MINOR":float, "ELLIPSE_THETA":float, "ELLIPSE_ASPECTRATIO":float, "AREA":float}    
    dataframe = dataframe.astype(dtype=dtypes_map)

    # sort by tracks and frames and reset the index
    dataframe = dataframe.sort_values(["TRACK_ID","FRAME"]).reset_index()
    
    return dataframe


def clean_all_csv(dataframes):
    """
    Clean all loaded spot.csv files from TrackMate stored in a dictionary of DataFrames.

    Removes rows and columns that are not necessary, resets the index, and renames the ID column to SPOT_ID for clarity.

    Parameters
    ----------
    dataframes : dict
        A dictionary where keys are filenames and values are DataFrames containing the TrackMate spot information.

    Returns
    -------
    dict
        A dictionary containing the cleaned DataFrames.

    Raises
    ------
    TypeError
        If any variable is not a pandas DataFrame.
    """
    cleaned_dataframes = {}
    for filename, dataframe in dataframes.items():
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"The file {filename} contains data that is not a pandas DataFrame.")

        # Rename ID column label for clarity
        if "ID" in dataframe.columns and "SPOT_ID" not in dataframe.columns:
            dataframe = dataframe.rename(columns={"ID": "SPOT_ID"})

        # Remove rows with NaN TRACK_ID
        dataframe = dataframe.dropna(subset=["TRACK_ID"])
        
        # Keep only necessary columns
        col2keep = ["SPOT_ID", "TRACK_ID", "POSITION_X", "POSITION_Y", "FRAME", "ELLIPSE_MAJOR", "ELLIPSE_MINOR", "ELLIPSE_THETA", "ELLIPSE_ASPECTRATIO", "AREA"]
        dataframe = dataframe[col2keep]

        # Remove potentially problematic rows (e.g., headers repeated within the data)
        dataframe = dataframe.drop([0, 1], errors='ignore')

        # Convert columns to specific data types
        dtypes_map = {
            "SPOT_ID": int, "TRACK_ID": int, "POSITION_X": float, "POSITION_Y": float,
            "FRAME": int, "ELLIPSE_MAJOR": float, "ELLIPSE_MINOR": float,
            "ELLIPSE_THETA": float, "ELLIPSE_ASPECTRATIO": float, "AREA": float
        }
        dataframe = dataframe.astype(dtypes_map)

        # Sort by TRACK_ID and FRAME and reset the index
        dataframe = dataframe.sort_values(["TRACK_ID", "FRAME"]).reset_index(drop=True)

        # Store the cleaned DataFrame back in the dictionary
        cleaned_dataframes[filename] = dataframe

    return cleaned_dataframes




# functions to a track to clean up splitting tracks


# several sub-functions used by the main function
def dist_prev_spot(prev,curr):
    """
    Takes the sub-dataframe of current and previous frame and the distances of the (first) current spot to the two previous spots.
    """
    cm_prev = prev[["POSITION_X","POSITION_Y"]].to_numpy() # coordinates of spot center of mass
    cm_curr = curr[["POSITION_X","POSITION_Y"]].to_numpy() # coordinates of spot center of mass
    length_distance = np.linalg.norm(cm_prev-cm_curr[0], axis=1) # np array containing the distance length of first current spot to previous spots

    return length_distance

def identify_correct_label(length_distance,prev):
    """
    Takes the sub-dataframe of previous frame and distances of current (first) spot to both previous spots to identify the corresponding label for the (first) current spot.
    """
    idx_min_dist = np.argmin(length_distance) # index of minimal distance
    correct_label = prev.iloc[idx_min_dist].loc["LABEL"]

    return correct_label

def flip_label(correct_label,labels):
    """
    Returns the entry in the labels list that is the other one than the identified correct_label, if the list contains only two items
    """
    if len(labels) != 2:
            raise ValueError(f"Incorrect number of items in labels list. Has to be exactly two entries.")

    other_label = labels[~labels.index(correct_label)] # find index of identified correct label, invert it and take the other one
    
    return other_label

# this function does the main job of separating split tracks, then labeling the spots occuring in the same frame corresponding to the distance to previous spot(s)
def separate_split_tracks(dataframe, labels=["A","B"]):
    """
    Takes a dataframe of a single TRACK_ID and separates the spots after a split event to split_tracks.
    Labels can be changed to whatever strings you want, but there should be only two!
    """
    # identify row indexes containing duplicate frames, save as np array
    duplicates_index = dataframe[dataframe.duplicated(subset="FRAME", keep=False)].index.to_numpy() 
    # identify first frame index in which tracks are duplicated
    rows2del = dataframe.index[dataframe.index>=min(duplicates_index)] 
    # make new dataframe of subtrack after splitting event
    split_tracks = dataframe.loc[rows2del]
    # remove frames after splitting event from original dataframe, i.e. shorten the original track
    dataframe = dataframe.drop(labels=rows2del)

    # label and assign spots in split tracks
    frames = np.unique(split_tracks["FRAME"].to_numpy()) # get frame numbers to loop over
    split_tracks["LABEL"]="" # add column and label first frame (that has to be double)
    labels = labels # should be maximux two strings
    
    # label the first frame with two spots
    split_tracks.loc[split_tracks["FRAME"]==frames[0],"LABEL"]=labels
    
    # loop over the remaining frames and label spots corresponding to min distance to previous spots
    for i in range(1,len(frames)):
        # get previous rows
        prev = split_tracks[split_tracks["FRAME"]==frames[i-1]]
        # get current rows
        curr = split_tracks[split_tracks["FRAME"]==frames[i]]
            
        # case 1: current and previous frames have two spots --> identify correct label for first of the two current spots based on distance to previous spot, the other spot gets the other label
        if curr.shape[0]==2 and prev.shape[0]==2:
            # function to calculate vector distance of first current spot coordinates to previous
            length_distance = dist_prev_spot(prev,curr)
    
            # function to identify label of previous spot that corresponds to current spot based on min distance
            correct_label = identify_correct_label(length_distance,prev)
            other_label = flip_label(correct_label,labels) # function to get the other label
    
            # assign labels to correct current spots in split_tracks dataframe
            split_tracks.loc[curr["LABEL"].index[0],["LABEL"]]=correct_label # find row index of first spot, assign correct value
            split_tracks.loc[curr["LABEL"].index[1],["LABEL"]]=other_label # find row index of second spot, assign other value
        
        # case 2: current frame has one spot and previous frame has two --> identify correct label for current spot based on distance to previous spot
        if curr.shape[0]==1 and prev.shape[0]==2:
            
            # function to calculate vector distance of first current spot coordinates to previous
            length_distance = dist_prev_spot(prev,curr)
    
            # function to identify label of previous spot that corresponds to current spot based on min distance
            correct_label = identify_correct_label(length_distance,prev)
            
            # write correct label into split_tracks dataframe
            split_tracks.loc[split_tracks["FRAME"]==frames[i],"LABEL"]=correct_label
    
        # case 3: current frame has two spots and previous frame has one --> identify to which of the two current spots the previous spot belongs
        if curr.shape[0]==2 and prev.shape[0]==1:
            # calculate vector distance of current spot coordinates to previous
            cm_prev = prev[["POSITION_X","POSITION_Y"]].to_numpy() # coordinates of spot center of mass
            cm_curr = curr[["POSITION_X","POSITION_Y"]].to_numpy() # coordinates of spot center of mass
            length_distance = np.linalg.norm(cm_prev-cm_curr, axis=1) # np array containing the distance length of current spot to previous spot(s)
            
            # find which current spot is closer to spot in previous frame
            idx_min_dist = np.argmin(length_distance) # index of minimal distance
            idx_max_dist = np.argmax(length_distance) # index of maximal distance
            correct_label = prev.iloc[0]["LABEL"] # get the label of the previous spot
            other_label = flip_label(correct_label,labels) # function to get the other label
            
            # assign labels to correct current spots in split_tracks dataframe
            split_tracks.loc[curr["LABEL"].index[idx_min_dist],["LABEL"]]=correct_label # find row index of spot with smaller distance to prev spot, assign correct value
            split_tracks.loc[curr["LABEL"].index[idx_max_dist],["LABEL"]]=other_label # find row index of spot with larger distance to prev spot, assign other value
    
        # case 4: current and previous frames have just one spot
        if curr.shape[0]==1 and prev.shape[0]==1:
            raise ValueError(f"Too many consecutive frames with only one spot per frame in TRACK_ID {curr['TRACK_ID'].values[0]}. Potential causes and workarounds:\n"
                             "(A) merge event in track --> go back to tracking software and remove merge event (bacteria typically don't merge)\n"
                             "(B) too many gaps in co-occuring split tracks --> can't find corresponding previous spot --> go back to tracking software and avoid those gaps\n"
                             f"(C) temporary workaround --> exclude problematic track using the exclude=[{curr['TRACK_ID'].values[0]}] option in the reindex_split_tracks function"
                             )
    
    # add column with previoous TRACK_ID
    pos = split_tracks.columns.get_loc("TRACK_ID")
    split_tracks.insert(pos+1, "PREV_TRACK_ID",pd.unique(split_tracks["TRACK_ID"])[0])
    dataframe.insert(pos+1, "PREV_TRACK_ID","")

    return dataframe, split_tracks, labels

def reunite_track(dataframe, split_tracks, labels, TRACK_ID_max):
    """
    Assign new TRACK_ID to split and labelled sub tracks and combine with main track dataframe
    """
    # assign new TRACK_ID to split sub tracks
    sub_track_1 = split_tracks[split_tracks["LABEL"]==labels[0]]
    sub_track_2 = split_tracks[split_tracks["LABEL"]==labels[1]]
    
    # reindex TRACK_IDs
    sub_track_1.loc[:,"TRACK_ID"]=TRACK_ID_max+1
    sub_track_2.loc[:,"TRACK_ID"]=TRACK_ID_max+2
    
    # remove LABEL column
    sub_track_1=sub_track_1.drop("LABEL",axis=1)
    sub_track_2=sub_track_2.drop("LABEL",axis=1)
    
    # add subtracks to main dataframe
    dataframe = pd.concat([dataframe,sub_track_1,sub_track_2])

    return dataframe

# this is the actual function to apply, uses the functions above


def reindex_split_tracks(tracks, track_ids=[], exclude=[]):
    """
    Cleans up tracks that split by separating a splitting track into three subtracks.
        (1) Main track until split event with previous TRACK_ID
        (2) First subtrack during the split event with new TRACK_ID --> correlates spots based on distance to previous frame
        (3) Seconds subtrack during the split event with new TRACK_ID --> correlates spots based on distance to previous frame
        
    Parameters
    ----------
    tracks : pandas dataframe
        The dataframe that contains the TrackMate spot information, loaded from a csv file and cleaned with clean_trc_csv function.
    track_ids (optional) : list of TRACK_IDs
        Specifiy individual or multiple tracks by TRACK_ID
    exclude: list of TRACK_IDs
        Specify individual or multiple TRACK_IDs that will be removed from the dataframe because they are trouble
        
    Returns
    -------
    cleaned_tracks : pandas dataframe
        The cleaned up datafranme with splitting tracks dealt with!
    """
    # identify the highest track number as starting point to add new tracks
    TRACK_ID_max = np.max(tracks["TRACK_ID"].values)
    
    # exclude tracks by removing them
    excluded_tracks = tracks[tracks["TRACK_ID"].isin(exclude)].index
    tracks = tracks.drop(excluded_tracks)
        
    # loop over all tracks to check and deal with split events
    if not track_ids:
        track_ids = np.unique(tracks["TRACK_ID"].to_numpy()) # get frame numbers to loop over if not specific TRACK_IDs were given
        
    cleaned_tracks = pd.DataFrame()
    for id in track_ids:
        dataframe = tracks[tracks["TRACK_ID"].isin([id])] # single track dataframe
        # check if there is a split event (i.e. there are duplicate frames for the same TRACK_ID)
        dups = np.any(dataframe.duplicated(subset="FRAME").values)
        
        if ~dups: # if no duplicates in track frames, just add emtpy column
            pos = dataframe.columns.get_loc("TRACK_ID")
            dataframe.insert(pos+1, "PREV_TRACK_ID","")
        
        if dups: # if duplicates in track frames, continue with separating split tracks, realigning split tracks, and adding them back with new TRACK_ID
            
            # function to identify and separate the spots after/during the split event
            dataframe, split_tracks, labels = separate_split_tracks(dataframe)
            
            # function to add split tracks with new TRACK_ID to main track dataframe
            dataframe = reunite_track(dataframe, split_tracks, labels, TRACK_ID_max)

            # update highest TRACK_ID, if new tracks were added
            TRACK_ID_max = TRACK_ID_max+2 

        cleaned_tracks = pd.concat([cleaned_tracks,dataframe])
        
    return cleaned_tracks



def reindex_split_tracks_all(dataframes, track_ids=[], exclude=[]):
    """
    Applies reindexing and handling of split tracks to multiple DataFrames stored in a dictionary.
    
    Parameters:
    ----------
    dataframes : dict
        Dictionary where keys are filenames and values are pandas DataFrames containing track information.
    track_ids : list, optional
        List of TRACK_IDs to specifically process. If empty, all tracks are processed.
    exclude : list, optional
        List of TRACK_IDs to exclude from processing.

    Returns:
    -------
    dict
        Dictionary of cleaned DataFrames with reindexed and split tracks handled.
    """
    cleaned_dataframes = {}
    for filename, tracks in dataframes.items():
        print(f"Processing {filename}")
        
        # Apply the original reindexing logic to each DataFrame
        cleaned_tracks = reindex_split_tracks(tracks, track_ids, exclude)
        
        # Store the cleaned DataFrame in the dictionary with the filename as the key
        cleaned_dataframes[filename] = cleaned_tracks

    return cleaned_dataframes


# functions to separate split and non-split tracks

def get_non_split_tracks(tracks):
    """
    Separates continuous tracks without any split phases from tracks with split phases. 
    
    Input: pandas.DataFrame (tracks)
        Tracks after separating and reindexing splitting tracks with the reindex_split_tracks function.
    Returns: pandas.DataFrame (tracks)
         Only those tracks with not splitting phases.
    """
    if "PREV_TRACK_ID" not in tracks.columns:
        raise ValueError(f"PREV_TRACK_ID column doesn't exist in tracks dataframe. Probably need to run reindex_split_tracks function first.")
    
    separated_tracks = tracks[tracks["PREV_TRACK_ID"].values=='']

    return separated_tracks

def get_split_tracks(tracks):
    """
    Separates tracks with split phases from tracks with no split phases. 
    
    Input: pandas.DataFrame (tracks)
        Tracks after separating and reindexing splitting tracks with the reindex_split_tracks function.
    Returns: pandas.DataFrame (tracks)
        Only those tracks with splitting phases.
    """
    if "PREV_TRACK_ID" not in tracks.columns:
        raise ValueError(f"PREV_TRACK_ID column doesn't exist in tracks dataframe. Probably need to run reindex_split_tracks function first.")
    
    separated_tracks = tracks[tracks["PREV_TRACK_ID"].values!='']

    return separated_tracks

def get_split_tracks_all(dataframes):
    """
    Separates tracks with split phases for multiple DataFrames stored in a dictionary.
    
    Parameters:
    ----------
    dataframes : dict
        Dictionary where keys are filenames and values are pandas DataFrames containing track information after handling splits.
    
    Returns:
    -------
    dict
        Dictionary of DataFrames, each containing only those tracks with splitting phases.
    """
    split_tracks_dataframes = {}
    for filename, tracks in dataframes.items():
        print(f"Processing {filename}")
        if "PREV_TRACK_ID" not in tracks.columns:
            raise ValueError(f"PREV_TRACK_ID column doesn't exist in tracks dataframe for {filename}. Probably need to run reindex_split_tracks function first.")
        
        separated_tracks = tracks[tracks["PREV_TRACK_ID"] != '']
        split_tracks_dataframes[filename] = separated_tracks

    return split_tracks_dataframes
    

def tracks_table(tracks, track_ids=[], all=False):
    """
    Displays the tracks dataframe either as shortened (default) or complete table (argument all=True).
    Caution! Displaying all tracks can result in a very long list that takes time to display.
    Enter specific TRACK_IDs in the track_ids=[] argument to display only those tracks.
    If no track_ids argument is given, all TRACK_IDs are displayed.

    Input: pandas.DataFrame (tracks)
    """
    
    specific_tracks=np.asarray(track_ids)
    if track_ids:
        tracks=tracks[tracks["TRACK_ID"].isin(specific_tracks)]

        missing_tracks = specific_tracks[np.isin(specific_tracks,tracks["TRACK_ID"].unique(),invert=True)].tolist()
        if missing_tracks:
            print("Warning: Tracks", str(missing_tracks)[1:-1],"were not found in the dataframe.")
    
    if all:
        with pd.option_context('display.max_rows', None,'display.max_columns', None):
            display(tracks)
    else:
        display(tracks)





def display_all_tracks(dataframes, track_ids=[], all=False):
    """
    Displays each DataFrame loaded from CSV files, either as shortened or complete tables based on the parameters.
    This function is intended to display track data from multiple DataFrames.
    
    Parameters:
    - dataframes: dict
        A dictionary where keys are filenames and values are DataFrames loaded from those files.
    - track_ids: list, optional
        List of specific TRACK_IDs to display. If not provided, displays based on the 'all' parameter.
    - all: bool, optional
        If True, displays all data in each DataFrame. If False, shows a truncated view.

    Raises:
    - ValueError: If a specified TRACK_ID is not found in any DataFrame.
    """
    for filename, df in dataframes.items():
        print(f"Displaying data from {filename}:")
        
        if track_ids:
            # Filter DataFrame by specific track IDs if provided
            df_filtered = df[df["TRACK_ID"].isin(track_ids)]
            missing_tracks = np.setdiff1d(track_ids, df_filtered["TRACK_ID"].unique())
            
            if missing_tracks.size > 0:
                print(f"Warning: Tracks {missing_tracks} were not found in {filename}.")
            
            if all:
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    display(df_filtered)
            else:
                display(df_filtered)  # Display only the first few rows by default

        elif all:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                display(df)
        else:
            display(df)  # Display only the first few rows by default



def count_unique_tracks(tracks):
    """
    Print and returns the number of unique TRACK_IDs in the dataframe.

    Input: pandas.DataFrame (tracks)
     Track data with a 'TRACK_ID' column.
    Return: Integer count of unique TRACK_IDs.
    """
    
    unique_track_count = tracks["TRACK_ID"].nunique()
    print(f"Number of unique tracks: {unique_track_count}")

    return unique_track_count




def count_unique_tracks_all(dataframes):
    """
    Prints and returns the number of unique TRACK_IDs for each DataFrame in a dictionary.

    Parameters:
    ----------
    dataframes : dict
        A dictionary where keys are filenames and values are pandas DataFrames, each containing TrackMate track information.

    Returns:
    -------
    dict
        A dictionary with filenames as keys and the count of unique TRACK_IDs as values.

    Raises:
    ------
    ValueError
        If any DataFrame does not contain a 'TRACK_ID' column.
    """
    track_counts = {}
    for filename, df in dataframes.items():
        if "TRACK_ID" not in df.columns:
            raise ValueError(f"DataFrame from {filename} does not contain a 'TRACK_ID' column.")

        unique_track_count = df["TRACK_ID"].nunique()
        print(f"Number of unique tracks in {filename}: {unique_track_count}")
        track_counts[filename] = unique_track_count

    return track_counts


# functions to calculate displacement and speed and add to dataframes

# count length of tracks based on occurrence of TRACK_IDs
def length_of_tracks(tracks):
    """
    Returns pd series with number of frames in every track. Index labels are TRACK_IDs.
    """
    track_length = tracks["TRACK_ID"].value_counts(sort=False)

    return track_length

def length_of_single_track(track):
    """
    Returns number of frames of a single track as int. Only run on the spots dataframe of a single track!
    """
    if len(track["TRACK_ID"].unique())>1:
        raise ValueError(f"Too many tracks. Only one unique track allowed!")
    
    single_track_length = length_of_tracks(track).values[0]

    return single_track_length

# filter out short tracks
def filter_short_tracks(tracks,min_frames=3):
    """
    Returns pd dataframe tracks filterd by number of frames (default min_frames=3, change as desired).
    """
    
    track_lengths = length_of_tracks(tracks)<min_frames # get boolean pd series where tracks are shorter than the cutoff threshold
    trcs_too_short = track_lengths[track_lengths].index.values # get np array of too short track
    idx_too_short = tracks[tracks["TRACK_ID"].isin(trcs_too_short)].index # get pd index of too short track indexes 
    
    tracks = tracks.drop(idx_too_short)

    return tracks

def calculate_angle_and_angular_velocity(track):
    """
    Calculates the angle between consecutive displacement vectors and the angular velocity for a track.
    
    :param track: DataFrame containing the track data.
    :return: Lists of angles (in radians) and angular velocities (in radians per frame interval).
    """
    V_t_t1 = track[["POSITION_X", "POSITION_Y"]].diff().fillna(0)
    angles = [np.nan]  # Initialize with NaN for the first angle, where calculation isn't applicable.
    angular_velocities = [np.nan]  # Initialize with NaN for the first angular velocity, where calculation isn't applicable.
    frame_intervals = track["FRAME"].diff()  # Calculate frame intervals

    for i in range(1, len(V_t_t1)):
        vector_1 = V_t_t1.iloc[i - 1].values
        vector_2 = V_t_t1.iloc[i].values
        if np.linalg.norm(vector_1) == 0 and np.linalg.norm(vector_2) == 0 and i > 1:  # Check for no movement and not at the beginning.
            angle = 0
            angular_velocity = 0
        elif np.linalg.norm(vector_1) > 0 and np.linalg.norm(vector_2) > 0:
            cos_theta = np.clip(np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2)), -1, 1)
            angle = np.arccos(cos_theta)
            # Calculate angular velocity if this isn't the first interval.
            if i > 1 and not np.isnan(angles[-1]) and not np.isnan(frame_intervals.iloc[i]) and frame_intervals.iloc[i] != 0:
                angular_velocity = (angle - angles[-1]) / frame_intervals.iloc[i]
            else:
                angular_velocity = np.nan
        else:
            angle = np.nan
            angular_velocity = np.nan if i > 1 else np.nan  # Keep it as NaN for the second frame since we can't compute the angular velocity.

        angles.append(angle)
        angular_velocities.append(angular_velocity)

    return angles, angular_velocities

def calculate_msd(track):
    #track = track.sort_values(by='FRAME').reset_index(drop=True)
    msd_values = [0]  # MSD is 0 for lag time = 0
    max_lag = len(track) - 1  # Maximum lag time
    
    for lag in range(1, max_lag + 1):
        # Calculate differences for the given lag
        diff_x = track['POSITION_X'].diff(lag).dropna()
        diff_y = track['POSITION_Y'].diff(lag).dropna()
        
        # Calculate squared displacements
        squared_displacement = diff_x**2 + diff_y**2
        
        # Compute mean squared displacement (MSD) by averaging squared displacements
        msd = squared_displacement.mean()
        msd_values.append(msd)
    
    return pd.Series(msd_values, index=np.arange(0, max_lag + 1))
    

def calculate_max_pairwise_distance(track):
    """
    Calculate the maximum straight-line distance between any two spots in a track.
    
    :param track: DataFrame containing th e track data, with 'POSITION_X' and 'POSITION_Y' columns.
    :return: Maximum distance between any two spots in the track.
    """
    # Extract positions as a 2D array
    positions = track[["POSITION_X", "POSITION_Y"]].values
    
    # Calculate all pairwise Euclidean distances
    pairwise_distances = cdist(positions, positions, 'euclidean')
    
    # Find the maximum distance
    max_distance = np.max(pairwise_distances)
    
    return max_distance


def get_track_parameters(tracks, strain, pixel_scale=1, frame_interval=1, min_frames=3, disable_warning=False):
    """
    Calculates the displacement and speed of tracks. Saves frame to frame displacement and displacement vector, speed in spots dataframe (in pixel) and corresponding averages in new tracks dataframe (in pixel/frame and µm/s if given).
    Possibility to provide a cutoff threshold for minimum length of tracks in frames. 
    Requires pixel_scale and frame_interval to calculate scaled displacement and speed. Otherwise provides unscaled values.
    Now includes calculation of mean squared displacement (MSD) for each track in both pixels and micrometers.
    """
    if not isinstance(strain, str):
        raise TypeError("No or wrong strain argument: strain must be a single string specifying the strain name (and condition if required).")
    
    if not pixel_scale or not frame_interval:
        pixel_scale = 1
        frame_interval = 1
        
    if not disable_warning:
        if pixel_scale == 1 or frame_interval == 1:
            print("WARNING: Pixel scale or frame interval not specified or = 1 --> displacement and speed are not scaled.\n"
                  "Provide pixel scale or frame interval to scale displacement and speed properly.\n"
                  "Ignore if pixel scale or frame interval are actually = 1.\n"
                  "Add argument disable_warning=True to disable this warning.")

    tracks = filter_short_tracks(tracks, min_frames)
    
    # Initialize the DataFrame to store spot-level data.
    df_spots = pd.DataFrame()

    df_rows = []  # List to store each track's data as a dictionary

    for trackID in pd.unique(tracks["TRACK_ID"]):
        track = tracks[tracks["TRACK_ID"] == trackID].copy()
        
        # Calculate displacement vectors and their lengths.
        V_t_t1 = track[["POSITION_X", "POSITION_Y"]].diff().fillna(0)
        V_t_t1_length = np.linalg.norm(V_t_t1, axis=1)

        # Calculate difference of frames between subsequent entries, fill NaNs with 0, store as int
        frame_gap = track["FRAME"].diff().fillna(0).astype('int')
        
        # Calculate the start and end positions.
        start_position = track.iloc[0][["POSITION_X", "POSITION_Y"]].values
        end_position = track.iloc[-1][["POSITION_X", "POSITION_Y"]].values
        end_to_end_vector = end_position - start_position
        end_to_end_distance = np.linalg.norm(end_to_end_vector)
        
        # Calculate maximum pairwise distance for the track
        max_distance = calculate_max_pairwise_distance(track)
        
        # Calculate the MSD of each trajectories over time 
        msd = calculate_msd(track)
        
        # Calculate angles and angular velocities.
        angles, angular_velocities = calculate_angle_and_angular_velocity(track)

        # Calculate Mean Directional Change.
        # Exclude the first NaN value before calculating the mean.
        mean_directional_change = np.nanmean(angles[1:])  # Use np.nanmean to safely ignore any NaN values.
        
        # Assign calculated angles between displacement vectors and angular velocities to the track DataFrame.
        track.loc[:, "ANGLE"] = angles
        track.loc[:, "ANGULAR_VELOCITY"] = angular_velocities
        
        # Add calculated displacement vectors and speeds to the track DataFrame.
        track.loc[:, "TOTAL_DISPLACEMENT_VECTOR"] = V_t_t1.values.tolist()
        track.loc[:, "TOTAL_DISPLACEMENT"] = V_t_t1_length
        track.loc[:, "FRAME_GAP"] = frame_gap
        track.loc[:, "STEP_SPEED"] = (track["TOTAL_DISPLACEMENT"] / track["FRAME_GAP"]).fillna(0)

        # Calculate Mean Squared Displacement (MSD)
        track.loc[:, "MSD_LAG_TIME"] = msd.index.tolist()
        track.loc[:, "MSD"] = msd.values.tolist()
        #track.loc[:, "MSD_norm"] = msd.values.tolist()/track[:, "WIDTH"].mean()

        # Concatenate the processed track data into df_spots.
        df_spots = pd.concat([df_spots, track], ignore_index=True)
        
        # Aggregate track-level data and update df_tracks.
        total_displacement_pix = track["TOTAL_DISPLACEMENT"].sum()
        total_displacement_mic = total_displacement_pix * pixel_scale
        end_to_end_distance_pix = end_to_end_distance
        end_to_end_distance_mic = end_to_end_distance_pix * pixel_scale
        
        # Calculate persistence: the ratio between the net displacement (end-to-end distance) and the total distance traveled (sum of all individual displacements between consecutive spots).
        # The ratio is unitless and ranges from 0 to 1. Values close to 0 indicate movement is confined near the starting point, while values close to 1 suggest movement along a straight line.
        persistence = end_to_end_distance_pix / total_displacement_pix
        sinuosity = 1 / persistence
        
        max_distance_pix = max_distance
        max_distance_mic = max_distance_pix * pixel_scale
        mean_directional_change_rad = mean_directional_change
        mean_directional_change_deg = np.rad2deg(mean_directional_change_rad)
        average_speed_pix_fr = track["STEP_SPEED"].iloc[1:].mean()
        average_speed_mic_sec = average_speed_pix_fr * pixel_scale / frame_interval
        average_angular_velocity_rad_fr = np.nanmean(track["ANGULAR_VELOCITY"])  # Handle NaN values appropriately
        average_angular_velocity_deg_sec = np.rad2deg(average_angular_velocity_rad_fr) / frame_interval
    
        # Calculate the track's frame length
        track_frame_length = len(track)

        df_rows.append({
            "IDENTIFIER": strain,
            "TRACK_ID": trackID,
            "TRACK_FRAME_LENGTH": track_frame_length,
            "TOTAL_DISPLACEMENT_PIX": total_displacement_pix,
            "TOTAL_DISPLACEMENT_MIC": total_displacement_mic,
            "END_TO_END_DISTANCE_PIX": end_to_end_distance_pix,
            "END_TO_END_DISTANCE_MIC": end_to_end_distance_mic,
            "PERSISTENCE": persistence,
            "SINUOSITY": sinuosity,
            "MAX_DISTANCE_PIX": max_distance_pix,
            "MAX_DISTANCE_MIC": max_distance_mic,
            "MEAN_DIRECTIONAL_CHANGE_RAD": mean_directional_change_rad,
            "MEAN_DIRECTIONAL_CHANGE_DEG": mean_directional_change_deg,
            "SPEED_PIX_FR": average_speed_pix_fr,
            "SPEED_MIC_SEC": average_speed_mic_sec,
            "AVERAGE_ANGULAR_VELOCITY_RAD_FR": average_angular_velocity_rad_fr,
            "AVERAGE_ANGULAR_VELOCITY_DEG_SEC": average_angular_velocity_deg_sec
        })

    df_tracks = pd.DataFrame(df_rows)
    
    # Return the DataFrame containing spot-level data and the DataFrame containing aggregated track-level data.
    return df_spots, df_tracks
    

def get_track_parameters_all(dataframes, pixel_scale=1, frame_interval=1, min_frames=3, disable_warning=False):
    """
    Calculates and updates track parameters for multiple DataFrames using the filename as a unique identifier.
    
    Parameters:
    ----------
    dataframes : dict
        Dictionary where keys are filenames and values are pandas DataFrames containing track information.
    pixel_scale : float, optional
        Scale to convert from pixels to microns.
    frame_interval : float, optional
        Time interval between frames.
    min_frames : int, optional
        Minimum number of frames a track must have to be included.
    disable_warning : bool, optional
        If True, disables the warning for non-scaled values.
        
    Returns:
    -------
    dict
        A dictionary where keys are filenames and values are tuples containing two DataFrames:
        1. DataFrame with spot-level data including updated parameters.
        2. DataFrame with aggregated track-level data.
    """
    all_spots = {}
    all_tracks = {}
    for filename, tracks in dataframes.items():
        print(f"Processing {filename}")
        # Use filename as a unique identifier instead of strain
        df_spots, df_tracks = get_track_parameters(tracks, filename, pixel_scale, frame_interval, min_frames, disable_warning)
        all_spots[filename] = df_spots
        all_tracks[filename] = df_tracks

    return all_spots, all_tracks



# function to save spots and tracks dataframes as csv files

def save_spots_tracks(all_spots, all_tracks, tracksdir, save_combined=True):
    """
    Saves the spots and tracks dataframes as CSV files into subdirectories within the provided base directory.

    Parameters:
    - all_spots : dict
        Dictionary of DataFrames with spot-level data, keys are filenames.
    - all_tracks : dict
        Dictionary of DataFrames with track-level aggregated data, keys are filenames.
    - tracksdir : str
        Base directory to save the outputs.
    - save_combined : bool, optional
        If True, saves a combined tracks DataFrame for all files processed.
    """
    combined_df_tracks = pd.DataFrame()

    for identifier in all_spots:
        # Removing '_spots.csv' and replacing it with an empty string for folder name
        folder_name = identifier.replace("_spots.csv", "")
        folder_path = os.path.join(tracksdir, folder_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # Ensure the directory exists
            print(f"Created directory: {folder_path}")
        else:
            print(f"Directory already exists: {folder_path}")

        spots_name = folder_name + "_Spots.csv"
        tracks_name = folder_name + "_Tracks.csv"
        
        spots_file_path = os.path.join(folder_path, spots_name)
        tracks_file_path = os.path.join(folder_path, tracks_name)

        all_spots[identifier].to_csv(spots_file_path, index=False)
        all_tracks[identifier].to_csv(tracks_file_path, index=False)

        print(f"Saved spots to {spots_file_path}")
        print(f"Saved tracks to {tracks_file_path}")

        combined_df_tracks = pd.concat([combined_df_tracks, all_tracks[identifier]])
        
        #print(combined_df_tracks.head())

    if save_combined:
        combined_data_dir = os.path.join(tracksdir, "combined_data")
        if not os.path.exists(combined_data_dir):
            os.makedirs(combined_data_dir)
            print(f"Created directory: {combined_data_dir}")
        
        combined_csv_path = os.path.join(combined_data_dir, "Combined_Tracks.csv")
        combined_df_tracks.to_csv(combined_csv_path, index=False)
        print(f"Combined tracks saved to {combined_csv_path}")

def load_tracks(tracksdir, load_combined=False):
    """
    Loads all tracks CSV files from the specified directory structure, with an option to load a combined tracks file.
    
    Parameters:
    - tracksdir: str
        The base directory where track CSV files are stored.
    - load_combined: bool, optional
        If True, loads the combined tracks file named 'Combined_Tracks.csv'.

    Returns:
    - pandas.DataFrame or dict
        If load_combined is True, returns a pandas DataFrame of the combined tracks.
        Otherwise, returns a dictionary where the keys are the filenames (without '_Tracks.csv') and the values are the loaded DataFrames.
    """
    if load_combined:
        combined_path = os.path.join(tracksdir, "combined_data", "Combined_Tracks.csv")
        if os.path.exists(combined_path):
            print(f"Loading combined tracks data from {combined_path}")
            return pd.read_csv(combined_path)
        else:
            raise FileNotFoundError(f"Combined tracks file not found at {combined_path}")

    dataframes = {}
    for dirpath, dirnames, filenames in os.walk(tracksdir):
        for filename in filenames:
            if filename.endswith("_Tracks.csv"):
                if "combined_data" not in dirpath:  # Skip combined data directory
                    file_path = os.path.join(dirpath, filename)
                    identifier = filename.replace("_Tracks.csv", "")
                    dataframes[identifier] = pd.read_csv(file_path)
                    print(f"Loaded {identifier} from {file_path}")

    return dataframes

def load_spots(tracksdir):
    """
    Loads all spots CSV files from the specified directory structure.
    
    Parameters:
    - tracksdir: str
        The base directory where spots CSV files are stored.

    Returns:
    - dict
        A dictionary where the keys are the filenames (without '_Spots.csv') and the values are the loaded DataFrames.
    """
    dataframes = {}
    for dirpath, dirnames, filenames in os.walk(tracksdir):
        for filename in filenames:
            if filename.endswith("_Spots.csv"):
                file_path = os.path.join(dirpath, filename)
                identifier = filename.replace("_Spots.csv", "")
                dataframes[identifier] = pd.read_csv(file_path)
                print(f"Loaded {identifier} from {file_path}")

    return dataframes

def list_column_names(data_frame):
    """
    Lists the column names from the DataFrame.

    Parameters:
    - data_frame: pandas.DataFrame
        The DataFrame from which to list the column names.

    Returns:
    - A list of column names from the DataFrame.
    """
    if data_frame.empty:
        print("No data frame loaded.")
        return []

    # Get the column names from the data frame
    column_names = data_frame.columns.tolist()
    
    print("Available column names:")
    for column in column_names:
        print(column)
    
    return column_names



def raincloud_plot(df, x_col, y_col, save_dir, fig_width=20, fig_height=10, show_outliers=True, save_plot=False, dpi=300):
    """
    Generates a raincloud plot, which combines box plot, half-violin plot, scatter plot elements, and now a point for the mean.
    This visualization is particularly useful for displaying the distribution and key statistics
    of a numeric variable segmented by a categorical variable in a dataset.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the data to be visualized.
    - x_col (str): Column name in 'df' for the categorical data (e.g., 'STRAIN').
    - y_col (str): Column name in 'df' for the numeric data to visualize (e.g., 'SPEED_MIC_SEC').
    - save_dir (str): Directory where the plot should be saved.
    - fig_width (int, optional): Width of the figure; default is 20.
    - fig_height (int, optional): Height of the figure; default is 10.
    - show_outliers (bool, optional): Whether to show outliers in the plot; default is True.
    - save_plot (bool, optional): Whether to save the plot as a file; default is False.
    - dpi (int, optional): Dots per inch (DPI) for the saved plot, which controls the quality; default is 300.

    Outputs:
    - A matplotlib plot displayed inline, showing the raincloud plot of the specified data.
    - If save_plot is True, saves the plot in the specified directory.
    """
    
    # Sort data for better visualization
    df_sorted = df.sort_values(x_col)
    # Get unique categories to separate them for plotting
    unique_categories = df_sorted[x_col].unique()
    # Set up figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    # Define dynamic positions based on the number of categories
    spacing = 0.4 / len(unique_categories)  # Adjust spacing dynamically
    boxplot_positions = np.arange(len(unique_categories))
    violin_positions = boxplot_positions + spacing / 2  # Slight right shift
    scatter_positions = boxplot_positions - spacing  # Slight left shift

    # Prepare data for plotting, removing outliers if requested
    data_for_plotting = [df_sorted[df_sorted[x_col] == category][y_col] for category in unique_categories]
    if not show_outliers:
        data_for_plotting = [remove_outliers(data) for data in data_for_plotting]

    # Create violin plot (only right half)
    vp = ax.violinplot(data_for_plotting, positions=violin_positions, showmeans=False, showextrema=False, showmedians=False, vert=True, widths=0.3)

    # Customize violin plots to show only the right half
    for body in vp['bodies']:
        m = np.mean(body.get_paths()[0].vertices[:, 0])
        body.get_paths()[0].vertices[:, 0] = np.clip(body.get_paths()[0].vertices[:, 0], m, np.inf)
        body.set_facecolor('skyblue')
        body.set_edgecolor('black')

    # Create boxplots using original data, but control the display of outliers
    bp = ax.boxplot([df_sorted[df_sorted[x_col] == category][y_col].values for category in unique_categories], positions=boxplot_positions, patch_artist=True, vert=True, widths=0.075, showfliers=show_outliers)

    # Customize boxplots
    for box, median in zip(bp['boxes'], bp['medians']):
        box.set_facecolor('skyblue')
        box.set_edgecolor('black')
        median.set_color('red')

    # Create scatter plots with or without outliers
    for i, category in enumerate(unique_categories):
        category_data = data_for_plotting[i]
        x = np.full(len(category_data), scatter_positions[i])
        x += np.random.uniform(-spacing / 2, spacing / 2, size=len(category_data))
        ax.scatter(x, category_data, color='gray', edgecolor='black', alpha=0.6)

        # Calculate and plot the mean using the potentially filtered data
        mean_value = category_data.mean()
        ax.scatter(scatter_positions[i], mean_value, color='red', edgecolor='black', label='Mean' if i == 0 else "")

    # Set x-axis labels and title to prevent overlap
    ax.set_xticks(boxplot_positions)
    ax.set_xticklabels(unique_categories, rotation=45, ha='right')
    ax.set_ylabel(y_col)
    ax.set_title('Raincloud Plot of ' + y_col + ' by ' + x_col)

    # Optionally add a legend
    ax.legend()

    if save_plot:
        # Create the raincloud_plots directory within the combined_data folder if it doesn't exist
        raincloud_plot_dir = os.path.join(save_dir, "tracks_combined_data", "raincloud_plots")
        if not os.path.exists(raincloud_plot_dir):
            os.makedirs(raincloud_plot_dir)
            print(f"Created directory: {raincloud_plot_dir}")

        # Save the plot as a file
        plot_filename = f"raincloud_plot_{x_col}_vs_{y_col}.png"
        plot_path = os.path.join(raincloud_plot_dir, plot_filename)
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved raincloud plot to {plot_path}")

    #plt.show()


def remove_outliers(series):
    """
    Helper function to remove outliers using IQR.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return series[~((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR)))]


def calculate_max_extent(dfs, min_frames):
    max_extent = 0
    for df in dfs.values():
        for track_id, track_data in df.groupby("TRACK_ID"):
            if len(track_data) >= min_frames:
                track_data_sorted = track_data.sort_values("FRAME")
                x = track_data_sorted["POSITION_X"] - track_data_sorted.iloc[0]["POSITION_X"]
                y = track_data_sorted["POSITION_Y"] - track_data_sorted.iloc[0]["POSITION_Y"]
                max_extent = max(max_extent, max(abs(x.min()), x.max(), abs(y.min()), y.max()))
    return max_extent

def plot_trajectories(ax, df, min_frames, axis_lims):
    for track_id, track_data in df.groupby("TRACK_ID"):
        if len(track_data) >= min_frames:
            track_data_sorted = track_data.sort_values("FRAME")
            track_data_sorted["POSITION_X"] -= track_data_sorted.iloc[0]["POSITION_X"]
            track_data_sorted["POSITION_Y"] -= track_data_sorted.iloc[0]["POSITION_Y"]
            color = np.random.rand(3,)
            ax.plot(track_data_sorted["POSITION_X"], track_data_sorted["POSITION_Y"], linestyle="-", color=color)
            

###---------------------
### MSD Analysis
###---------------------

# Figures settings
fig_x = 5
fig_y = 5.2
fontsize = 16

 # Set global font to Arial
plt.rcParams['font.family'] = 'Arial'


def extract_condition_from_filename(filename):
    strain_pattern = r"(177|232|459)"
    sc_pattern = r"(SC10pc|SC30pc|SC70pc)"
    
    strain_match = re.search(strain_pattern, filename)
    strain = strain_match.group(0) if strain_match else "Unknown"
    
    sc_match = re.search(sc_pattern, filename)
    surface_coverage = sc_match.group(0) if sc_match else "Unknown"
    
    condition = f"Strain{strain}_Coverage{surface_coverage}"
    
    return condition



def plot_and_save_MSD(tracks, frame_interval, save_dir):
    """
    This function generates and saves MSD plots for all the tracks, 
    and ensures that the 'file' column is added to the combined DataFrame.
    
    - tracks: A dictionary where the key is the filename and the value is a DataFrame with track data.
    - frame_interval: Time between frames (to convert to seconds).
    - save_dir: Directory where the plots and CSV files will be saved.
    """
    
    # Initialize an empty list to collect DataFrames
    track_list = []
    
    # Process each track and combine them
    for filename, track in tracks.items():
        print(f"---------")
        print(f"Processing {filename}...")

        # Extract condition info from the filename
        condition = extract_condition_from_filename(filename)

        # Add the 'file' and 'condition' columns to the track DataFrame
        track['file'] = filename
        track['condition'] = condition
        
        # Append the track DataFrame to the list
        track_list.append(track)
        
        # Save the individual plot for this track (you can call your plotting function here)
        plot_single_MSD(track, save_dir, filename, frame_interval)
        
        # OPTIONAL: Save the individual MSD DataFrame as a CSV
        #plot_filename = os.path.join(save_dir, f"{filename}_MSD_curves.png")
        #csv_filename = os.path.join(save_dir, f"{filename}_MSD.csv")
        #track.to_csv(csv_filename, index=False)
        #print(f"CSV for {filename} saved: {csv_filename}")
    
    # Now concatenate all the DataFrames in track_list into one DataFrame
    combined_df = pd.concat(track_list, ignore_index=True)
    
    # Now combine the MSD data by condition (this should now include the 'file' column)
    # Get unique conditions
    unique_conditions = combined_df['condition'].unique()
    
    for condition in unique_conditions:
        # Extract data for this condition
        condition_data = combined_df[combined_df['condition'] == condition]
        
        # Plot MSD curves for all individual tracks in this condition
        plot_condition_MSD(condition_data, save_dir, condition, frame_interval)
    
    # Call the function to plot all average MSD for all conditions
    plot_all_conditions_MSD(combined_df, save_dir, frame_interval)

    # Process MSD curves by surface coverage
    surface_coverages = combined_df['condition'].str.extract(r'_(\d+pc)').dropna()[0].unique()
    
    for surface_coverage in surface_coverages:
        surface_coverage_data = combined_df[combined_df['condition'].str.contains(surface_coverage)]
        plot_surface_coverage_MSD(surface_coverage_data, save_dir, surface_coverage, frame_interval)
    
    print("-------------------------")
    print("Finished processing and saving MSD plots and CSVs.")



def plot_single_MSD(track, save_dir, filename, frame_interval):
    """
    This function generates two MSD plots for each data file:
    1. The individual MSD curves in a separate plot (different colors for each track with labels).
    2. The individual MSD curves and the average MSD curve in another plot (average in color, individual in light gray).
    
    track: DataFrame containing all the trajectories for a given file.
    save_dir: Directory where the plots will be saved.
    filename: Name of the file for saving the plots.
    frame_interval: Time between frames (in seconds).
    """
    
    # Create an empty DataFrame to store all individual MSD curves' data
    all_tracks_df = pd.DataFrame()

    # Collect all unique 'MSD_LAG_TIME' values from all tracks
    all_lag_times = set()
    for _, subtrack in track.groupby('TRACK_ID'):  # Group by each individual track
        all_lag_times.update(subtrack['MSD_LAG_TIME'].values)  # Add lag times from each track

    all_lag_times = sorted(list(all_lag_times))  # Sort the unique lag times

    # Generate a color palette with as many colors as there are tracks
    num_tracks = len(track['TRACK_ID'].unique())
    color_palette = sns.color_palette("hls", num_tracks)

    # Initialize a list to store data for all tracks
    all_tracks_data = []

    # Plot each individual MSD curve and add the data to the DataFrame
    fig, ax = plt.subplots(figsize=(fig_x, fig_y))
    
    for i, (track_id, subtrack) in enumerate(track.groupby('TRACK_ID')):  # Group by each track
        subtrack = subtrack.sort_values(by='MSD_LAG_TIME')  # Sort by lag time for each track

        # Reindex 'subtrack' on the 'all_lag_times' to ensure all tracks have the same 'MSD_LAG_TIME'
        subtrack_reindexed = subtrack.set_index('MSD_LAG_TIME').reindex(all_lag_times, fill_value=np.nan)

        # Add the track data to the list
        track_data = pd.DataFrame({
            'MSD_LAG_TIME': np.array(all_lag_times) * frame_interval,  # Convert to seconds
            'MSD': subtrack_reindexed['MSD'].values
        })
        
        # Add the track name to each row
        track_data['Track_ID'] = f"Track_{track_id}"

        # Append this track's data to the global list
        all_tracks_data.append(track_data)

        # Plot the individual MSD curve for this track in a unique color and label the track
        ax.plot(subtrack_reindexed.index * frame_interval, subtrack_reindexed['MSD'], 
                color=color_palette[i], alpha=0.7, label=f"Track {track_id}")

    ax.set_xlabel('Lag time (seconds)', fontsize = fontsize)
    ax.set_ylabel('MSD (µm²)', fontsize = fontsize)
    #ax.set_title(f'Individual MSD Curves - File {filename}')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', length=10, width=1)
    ax.tick_params(axis='y', length=10, width=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 1. Save the first figure with only the individual MSD curves (OPTIONAL)
    
    #individual_plot_filename = os.path.join(save_dir, f"{filename}_individual_MSD_curves.png")
    #plt.savefig(individual_plot_filename, transparent=True)
    plt.close()
    
    # 2. Save the second figure with the individual MSD curves and the average MSD curve
    
    # Concatenate all the track data into one DataFrame
    all_tracks_df = pd.concat(all_tracks_data, ignore_index=True)

    # Compute the average MSD across all tracks (over the unified lag times)
    average_msd = all_tracks_df.groupby('MSD_LAG_TIME')['MSD'].mean()
    time_in_seconds_avg = np.array(average_msd.index) #* frame_interval  # Convert lag time to seconds

    # Custom colors for the strain conditions
    strain_colors = {"177": "magenta", "459": "orange", "232": "deepskyblue"}
    strain_condition = track['condition'].iloc[0][6:9]  # Assume all tracks in the file have the same strain condition
    
    fig, ax = plt.subplots(figsize=(fig_x, fig_y))
    #fig, ax = plt.subplots(figsize=(5, 5))

    # Plot the individual MSD curves in light gray
    for i, (track_id, subtrack) in enumerate(track.groupby('TRACK_ID')):  # Group by each track
        subtrack = subtrack.sort_values(by='MSD_LAG_TIME')  # Sort by lag time
        subtrack_reindexed = subtrack.set_index('MSD_LAG_TIME').reindex(all_lag_times, fill_value=np.nan)
        ax.plot(subtrack_reindexed.index * frame_interval, subtrack_reindexed['MSD'], 
                color="0.5", alpha=0.3)

    # Plot the average MSD in black
    ax.plot(time_in_seconds_avg, average_msd, color=strain_colors[strain_condition], linestyle='-', linewidth=3, label='Average MSD')

    ax.set_xscale('log')
    ax.set_yscale('log') 
    ax.set_xlabel('Lag time (seconds)', fontsize = fontsize)
    ax.set_ylabel('MSD (µm²)', fontsize = fontsize)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    #ax.set_title(f'Individual and Average MSD Curves - File {filename}')
    ax.tick_params(axis='x', length=10, width=1)
    ax.tick_params(axis='y', length=10, width=1)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # OPTIONAL
    #average_plot_filename = os.path.join(save_dir, f"{filename}_individual_and_average_MSD.png")
    #plt.savefig(average_plot_filename, transparent=True)
    plt.close()

    # Save the DataFrame containing all individual MSD curves to a CSV file
    #csv_filename = os.path.join(save_dir, f"{filename}_MSD.csv")
    #all_tracks_df['MSD_LAG_TIME'] = all_tracks_df['MSD_LAG_TIME'] * frame_interval  # Convert to seconds
    #all_tracks_df[['MSD_LAG_TIME', 'MSD', 'Track_ID']].to_csv(csv_filename, index=False)
    #print(f"CSV for {filename} saved: {csv_filename}")
 

def plot_condition_MSD(condition_data, save_dir, condition, frame_interval):
    """
    Plot the MSD for all tracks in a given condition and save the plot. Convert x-axis to seconds using frame_interval.
    
    Parameters:
    - condition_data: DataFrame containing all tracks for a specific condition.
    - save_dir: Directory to save the plot.
    - condition: Name of the condition (strain and surface coverage).
    - frame_interval: Time interval between consecutive frames in seconds.
    """
    
    # Ensure save_dir exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  
    
    # Check if 'file' column exists
    if 'file' not in condition_data.columns:
        print(f"Error: 'file' column not found in condition_data for {condition}")
        return
    
    print(f"-------------------------")
    print(f"Processing condition: {condition}")
    
    # Define a color map for different files
    filenames = condition_data['file'].unique()
    num_files = len(filenames)
    color_map = plt.cm.get_cmap("tab20", num_files) 
    
    characteristic_time = 4  # sec
    characteristic_length = 0.8  # µm
    
    # Compute the average MSD for the condition
    average_msd = condition_data.groupby('MSD_LAG_TIME')['MSD'].mean()
    time_in_seconds_avg = average_msd.index * frame_interval 
    # Normalization avg
    average_norm_MSD = average_msd / (characteristic_length ** 2)
    average_norm_time = time_in_seconds_avg / characteristic_time
    
    # Extract strain from the condition (assuming strain is part of the condition name)
    strain = condition.split("_")[0].replace("Strain", "")
    
    # Define colors for each strain
    strain_colors = {
        "177": "magenta",
        "232": "deepskyblue",
        "459": "darkorange",
        "1047": "gray"  # Default for other strains
    }
    
    # Get the color for the corresponding strain
    color = strain_colors.get(strain, "black")  
    
    # Create a plot for "MSD for condition" with individual curves and the average curve on top
    fig, ax = plt.subplots(figsize=(fig_x, fig_y))
    
    # Plot all individual MSD curves for the condition with unique colors for each file
    for i, filename in enumerate(filenames):
        file_data = condition_data[condition_data['file'] == filename]
        
        # Handle NaN values by removing them
        file_data = file_data.dropna(subset=['MSD']) 
        
        # Loop over each track and plot it individually
        for track_id in file_data['TRACK_ID'].unique():
            track_data = file_data[file_data['TRACK_ID'] == track_id]
            # Normalize MSD
            norm_MSD = track_data['MSD'] / (characteristic_length ** 2)
            time_in_seconds = track_data['MSD_LAG_TIME'] * frame_interval
            # Normalize time
            norm_time = time_in_seconds / characteristic_time
            ax.plot(norm_time, norm_MSD, color=color_map(i), alpha=0.7) 
        
        # Add the label for the current file to the legend (only once per file)
        ax.plot([], [], color=color_map(i), label=filename)  
    
    # Plot the average MSD for the condition with the strain-specific color
    ax.plot(average_norm_time, average_norm_MSD, color=color, linestyle='-', linewidth=3, label='Average MSD')

    # Set labels and title
    ax.set_xlabel('Lag time', fontsize = fontsize)
    ax.set_ylabel('MSD', fontsize = fontsize)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', length=10, width=1)
    ax.tick_params(axis='y', length=10, width=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add a legend with filenames and the average MSD
    ax.legend(loc='upper right', frameon = False)

    # Save the plot (OPTIONAL)
    #file_name = os.path.join(save_dir, f"{condition}_MSD.png")
    #plt.savefig(file_name, transparent=True)
    plt.close()

    # Create a new plot for "Individual and Average MSD Curves" with individual curves in light gray
    fig, ax = plt.subplots(figsize=(4,4)) #fig_x, fig_y))

    # Plot all individual MSD curves for the condition in light gray
    for i, filename in enumerate(filenames):
        file_data = condition_data[condition_data['file'] == filename]
        
        # Handle NaN values by removing them
        file_data = file_data.dropna(subset=['MSD'])  
        
        # Loop over each track and plot it individually in light gray
        for track_id in file_data['TRACK_ID'].unique():
            track_data = file_data[file_data['TRACK_ID'] == track_id]
            # Normalize MSD
            norm_MSD = track_data['MSD'] / (characteristic_length ** 2)
            time_in_seconds = track_data['MSD_LAG_TIME'] * frame_interval
            # Normalize time
            norm_time = time_in_seconds / characteristic_time
            ax.plot(norm_time, norm_MSD, color="0.5", alpha=0.3)
    
    # Plot the average MSD for the condition with the strain-specific color on top
    ax.plot(average_norm_time, average_norm_MSD, color=color, linestyle='-', linewidth=5) #, label='Average MSD')

    # Set labels and title for the new plot
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(bottom = 1, top = 10000)
    ax.set_xlim(left=0.8, right = 120)
    ax.set_xlabel('Lag time', fontsize=18, fontweight='bold', fontname='Arial')
    ax.set_ylabel('MSD', fontsize=18, fontweight='bold', fontname='Arial')
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='both', which='minor', length=7, width=1, color='black') 
    ax.tick_params(axis='both', which='major', length=14, width=1, color='black') 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.25)
    ax.spines['bottom'].set_linewidth(1.25)

    # Save the new plot
    fig.tight_layout()
    file_name = os.path.join(save_dir, f"{condition}_individual_and_average_MSD_condition.png")
    plt.savefig(file_name, dpi=300, transparent=True)
    plt.close()



def plot_all_conditions_MSD(combined_df, save_dir, frame_interval):
    """
    Plot the average MSD for all conditions together, in both linear and log-log scale.
    Convert x-axis to seconds.
    
    Parameters:
    - combined_df: DataFrame containing all track data.
    - save_dir: Directory to save the plots.
    - frame_interval: Time interval between consecutive frames in seconds.
    """

    # Get all unique conditions
    conditions = combined_df['condition'].unique()
    #print(conditions)

    # Define colors based on strain
    strain_colors = {
        "177": "magenta",    # WT
        "232": "deepskyblue",    # pilH
        "459": "darkorange",    # pilG
        "1047": "gray"  # Unknown
    }

    # Mapping for strain labels
    strain_labels = {
        '232': r'$\it{\Delta pilH}$',
        '177': 'WT',
        '459': r'$\it{\Delta pilG}$',
        '1047': 'Unknown'
    }

    fit_results = []  # Store fitting results

    ## --- First Plot: Average MSD all conditions (log-log, no fit) ---
    fig_no_fit, ax_no_fit = plt.subplots(figsize=(fig_x, fig_y))

    handles_no_fit = []
    new_labels_no_fit = []

    # Specify the desired order for the strains in the legend: pilH, WT, pilG
    ordered_strains = ['232', '177', '459'] 
    for strain_key in ordered_strains:
        condition = [cond for cond in conditions if strain_key in cond][0]

        color = strain_colors.get(strain_key, "black")
        label_base = strain_labels.get(strain_key, "Unknown")

        condition_data = combined_df[combined_df['condition'] == condition]
        average_msd = condition_data.groupby('MSD_LAG_TIME')['MSD'].mean()
        time_in_seconds = average_msd.index * frame_interval

        ax_no_fit.plot(time_in_seconds, average_msd, color=color, linewidth=5)

        handles_no_fit.append(plt.Line2D([], [], color=color, linewidth=5))
        new_labels_no_fit.append(label_base)

    ax_no_fit.set_xscale('log')
    ax_no_fit.set_yscale('log')
    ax_no_fit.set_xlabel('Lag time (sec)', fontsize=22, fontweight='bold', fontname='Arial')
    ax_no_fit.set_ylabel('MSD (µm²)', fontsize=22, fontweight='bold', fontname='Arial')
    ax_no_fit.set_xlim(left=1, right=1000)
    ax_no_fit.set_ylim(bottom=0.5, top=10000)
    ax_no_fit.tick_params(axis='both', which='minor', length=7, width=1, color='black')
    ax_no_fit.tick_params(axis='both', which='major', length=14, width=1, color='black')
    ax_no_fit.tick_params(axis='x', labelsize=20)
    ax_no_fit.tick_params(axis='y', labelsize=20)
    ax_no_fit.spines['top'].set_linewidth(1.25)
    ax_no_fit.spines['right'].set_linewidth(1.25)
    ax_no_fit.spines['left'].set_linewidth(1.25)
    ax_no_fit.spines['bottom'].set_linewidth(1.25)

    # Add legend with correct order
    ax_no_fit.legend(handles=handles_no_fit, labels=new_labels_no_fit, fontsize=13, loc='upper left')

    fig_no_fit.tight_layout()
    file_name_no_fit = os.path.join(save_dir, "average_MSD_no_fit.png")
    plt.savefig(file_name_no_fit, dpi=300, transparent=True)
    plt.close(fig_no_fit)

    ## --- Second Plot: Average MSD with fit (log-log) ---
    fig_with_fit, ax_with_fit = plt.subplots(figsize=(fig_x, fig_y))

    handles_with_fit = []
    new_labels_with_fit = []

    # Specify the desired order for the strains in the legend: pilH, WT, pilG
    for strain_key in ordered_strains:
        condition = [cond for cond in conditions if strain_key in cond][0]

        color = strain_colors.get(strain_key, "black")
        label_base = strain_labels.get(strain_key, "Unknown")

        condition_data = combined_df[combined_df['condition'] == condition]
        average_msd = condition_data.groupby('MSD_LAG_TIME')['MSD'].mean()
        time_in_seconds = average_msd.index * frame_interval

        # Prepare full log-log data
        mask = (time_in_seconds >= 11) & (time_in_seconds <= 100)
        x_vals = time_in_seconds[mask]
        y_vals = average_msd[mask]

        if len(x_vals) == 0:
            continue

        log_x = np.log10(x_vals)
        log_y = np.log10(y_vals)

        slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
        r_squared = r_value ** 2

        # Plot: x < 10 s as dashed, x >= 10 s as solid
        solid_mask = time_in_seconds >= 10
        dashed_mask = time_in_seconds <= 10

        ax_with_fit.plot(time_in_seconds[solid_mask], average_msd[solid_mask], 
                      color=color, linewidth=5, linestyle='-')

        # Plot fit
        fit_line = 10**(slope * np.log10(x_vals) + intercept)
        ax_with_fit.plot(x_vals, fit_line, linestyle='--', linewidth=2, color='black', alpha=0.7)

        # Update legend
        handles_with_fit.append(plt.Line2D([], [], color=color, linewidth=5))
        new_labels_with_fit.append(f"{label_base}, a = {slope:.2f}") #R² = {r_squared:.0f}

        # Save fit results
        fit_results.append({
            "condition": condition,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared
        })


    # Add legend with correct order for the graph with fit
    ax_with_fit.legend(handles=handles_with_fit, labels=new_labels_with_fit, fontsize=14, loc='upper left')

    ax_with_fit.set_xscale('log')
    ax_with_fit.set_yscale('log')
    ax_with_fit.set_xlabel('Lag time (sec)', fontsize=22, fontweight='bold', fontname='Arial')
    ax_with_fit.set_ylabel('MSD (µm²)', fontsize=22, fontweight='bold', fontname='Arial')
    ax_with_fit.set_xlim(left=1, right=1000)
    ax_with_fit.set_ylim(bottom=0.5, top=10000)
    ax_with_fit.tick_params(axis='both', which='minor', length=7, width=1, color='black')
    ax_with_fit.tick_params(axis='both', which='major', length=14, width=1, color='black')
    ax_with_fit.tick_params(axis='x', labelsize=20)
    ax_with_fit.tick_params(axis='y', labelsize=20)
    ax_with_fit.spines['top'].set_linewidth(1.25)
    ax_with_fit.spines['right'].set_linewidth(1.25)
    ax_with_fit.spines['left'].set_linewidth(1.25)
    ax_with_fit.spines['bottom'].set_linewidth(1.25)

    fig_with_fit.tight_layout()
    file_name_with_fit = os.path.join(save_dir, "average_MSD_with_fit.png")
    plt.savefig(file_name_with_fit, dpi=300, transparent=True)
    plt.close(fig_with_fit) 
    
    ## --- Save fitting results ---
    fit_df = pd.DataFrame(fit_results)
    fit_df.to_csv(os.path.join(save_dir, "MSD_fit_parameters.csv"), index=False)
    
    ## --- Third Plot: Average normalized MSD with fit (log-log) ---

    fig_norm_with_fit, ax_norm_with_fit = plt.subplots(figsize=(fig_x, fig_y))
    
    handles_norm_with_fit = []
    new_labels_norm_with_fit = []
    
    characteristic_time = 4  # sec
    characteristic_length = 0.8  # µm
    
    fit_norm_results = []

    
    # Specify the desired order for the strains in the legend: pilH, WT, pilG
    for strain_key in ordered_strains:
        condition = [cond for cond in conditions if strain_key in cond][0]
    
        color = strain_colors.get(strain_key, "black")
        label_base = strain_labels.get(strain_key, "Unknown")
    
        condition_data = combined_df[combined_df['condition'] == condition]
        average_msd = condition_data.groupby('MSD_LAG_TIME')['MSD'].mean()
        time_in_seconds = average_msd.index * frame_interval
    
        # Normalize time and MSD
        norm_time = time_in_seconds / characteristic_time
        norm_msd = average_msd / (characteristic_length ** 2)
    
        # Prepare log-log fit in desired range (normalized time between ~2.75 and 25)
        mask = (norm_time >= 2.75) & (norm_time <= 25)
        x_vals_norm = norm_time[mask]
        y_vals_norm = norm_msd[mask]
    
        if len(x_vals_norm) == 0:
            continue
    
        normlog_x = np.log10(x_vals_norm)
        normlog_y = np.log10(y_vals_norm)
    
        norm_slope, norm_intercept, norm_r_value, norm_p_value, norm_std_err = linregress(normlog_x, normlog_y)
        norm_r_squared = norm_r_value ** 2
    
        # Plot: normalized time < 2.5 as dashed, >= 2.5 as solid
        solid_mask = norm_time >= 2.5
        dashed_mask = norm_time <= 2.5
    
        ax_norm_with_fit.plot(norm_time[solid_mask], norm_msd[solid_mask],
                              color=color, linewidth=5, linestyle='-')
    
        # Plot fit line
        fit_norm_line = 10**(norm_slope * np.log10(x_vals_norm) + norm_intercept)
        ax_norm_with_fit.plot(x_vals_norm, fit_norm_line, linestyle='--', linewidth=2, color='black', alpha=0.7)
    
        # Update legend
        handles_norm_with_fit.append(plt.Line2D([], [], color=color, linewidth=5))
        new_labels_norm_with_fit.append(f"{label_base}, a = {norm_slope:.2f}")
    
        # Save fit results
        fit_norm_results.append({
            "condition": condition,
            "slope": norm_slope,
            "intercept": norm_intercept,
            "r_squared": norm_r_squared
        })
    
    # Add legend with correct order for the graph with fit
    ax_norm_with_fit.legend(handles=handles_norm_with_fit, labels=new_labels_norm_with_fit, fontsize=14, loc='upper left', frameon = False)
    
    ax_norm_with_fit.set_xscale('log')
    ax_norm_with_fit.set_yscale('log')
    ax_norm_with_fit.set_xlabel('Lag time', fontsize=22, fontweight='bold', fontname='Arial')
    ax_norm_with_fit.set_ylabel('MSD', fontsize=22, fontweight='bold', fontname='Arial')
    ax_norm_with_fit.set_xlim(left=0.8, right=120)
    ax_norm_with_fit.set_ylim(bottom=1, top=10000)
    ax_norm_with_fit.tick_params(axis='both', which='minor', length=7, width=1, color='black')
    ax_norm_with_fit.tick_params(axis='both', which='major', length=14, width=1, color='black')
    ax_norm_with_fit.tick_params(axis='x', labelsize=20)
    ax_norm_with_fit.tick_params(axis='y', labelsize=20)
    ax_norm_with_fit.spines['top'].set_visible(False)
    ax_norm_with_fit.spines['right'].set_visible(False)
    ax_norm_with_fit.spines['left'].set_linewidth(1.25)
    ax_norm_with_fit.spines['bottom'].set_linewidth(1.25)
    
    fig_norm_with_fit.tight_layout()
    
    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)
    
    file_name_norm_with_fit = os.path.join(save_dir, "average_MSD_norm_with_fit.png")
    plt.savefig(file_name_norm_with_fit, dpi=300, transparent=True)
    plt.close(fig_norm_with_fit)


    ## --- Save fitting results ---
    fit_norm_df = pd.DataFrame(fit_norm_results)
    fit_norm_df.to_csv(os.path.join(save_dir, "MSD_norm_fit_parameters.csv"), index=False)

        

def plot_and_save_MSD_exp_and_simus(tracks, frame_interval, save_dir, simulations_dir):
    """
    This function generates and saves MSD plots for both experimants and simulations
    
    - tracks: A dictionary where the key is the filename and the value is a DataFrame with track data.
    - frame_interval: Time between frames (to convert to seconds).
    - save_dir: Directory where the plots and CSV files will be saved.
    - simulations_dir: Directory where the simulations tables (CSV files) will be saved.
    """
    
    track_list = []
    
    # Process each track and combine them
    for filename, track in tracks.items():
        print(f"---------")
        print(f"Processing {filename}...")

        # Extract condition info from the filename
        condition = extract_condition_from_filename(filename)

        # Add the 'file' and 'condition' columns to the track DataFrame
        track['file'] = filename
        track['condition'] = condition
        
        # Append the track DataFrame to the list
        track_list.append(track)
        
        # Save the individual plot for this track (you can call your plotting function here)
        plot_single_MSD(track, save_dir, filename, frame_interval)
        
        # OPTIONAL: Save the individual MSD DataFrame as a CSV
        #plot_filename = os.path.join(save_dir, f"{filename}_MSD_curves.png")
        #csv_filename = os.path.join(save_dir, f"{filename}_MSD.csv")
        #track.to_csv(csv_filename, index=False)
        #print(f"CSV for {filename} saved: {csv_filename}")
    
    # Now concatenate all the DataFrames in track_list into one DataFrame
    combined_df = pd.concat(track_list, ignore_index=True)
    
    # Now combine the MSD data by condition
    # Get unique conditions
    unique_conditions = combined_df['condition'].unique()
    
    for condition in unique_conditions:
        # Extract data for this condition
        condition_data = combined_df[combined_df['condition'] == condition]
        
        # Plot MSD curves for all individual tracks in this condition
        plot_condition_MSD(condition_data, save_dir, condition, frame_interval)
    
    # Call the function to plot all average MSD for all conditions
    plot_all_conditions_MSD_with_simus(combined_df, save_dir, frame_interval, simulations_dir)

    # Process MSD curves by surface coverage
    surface_coverages = combined_df['condition'].str.extract(r'_(\d+pc)').dropna()[0].unique()
    
    for surface_coverage in surface_coverages:
        surface_coverage_data = combined_df[combined_df['condition'].str.contains(surface_coverage)]
        plot_surface_coverage_MSD(surface_coverage_data, save_dir, surface_coverage, frame_interval)
    
    print("-------------------------")
    print("Finished processing and saving MSD plots and CSVs.")


def plot_all_conditions_MSD_with_simus(combined_df, save_dir, frame_interval, simulations_dir):
    """
    Plot the average MSD for all conditions together, in both linear and log-log scale, for experiments and simulations
    Convert x-axis to seconds.
    
    Parameters:
    - combined_df: DataFrame containing all track data.
    - save_dir: Directory to save the plots.
    - frame_interval: Time interval between consecutive frames in seconds.
    - simulations_dir: Directory where the simulations tables (CSV files) will be saved.
    """

    # Get all unique conditions
    conditions = combined_df['condition'].unique()

    # Define colors based on strain
    strain_colors = {
        "177": "magenta",    # WT
        "232": "deepskyblue",    # pilH
        "459": "darkorange",    # pilG cpdA
        "1047": "gray"  # Unknown
    }

    # Mapping for strain labels
    strain_labels = {
        '232': r'$\it{\Delta pilH}$', 
        '177': 'WT',
        '459': r'$\it{\Delta pilG}$', 
        '1047': 'Unknown'
    }

    fit_results = []  # Store fitting results

    ## --- First Plot: Average MSD all conditions (log-log, no fit) ---
    fig_no_fit, ax_no_fit = plt.subplots(figsize=(fig_x, fig_y))

    handles_no_fit = []
    new_labels_no_fit = []

    # Specify the desired order for the strains in the legend: pilH, WT, pilG
    ordered_strains = ['232', '177', '459']
    for strain_key in ordered_strains:
        condition = [cond for cond in conditions if strain_key in cond][0]

        color = strain_colors.get(strain_key, "black")
        label_base = strain_labels.get(strain_key, "Unknown")

        condition_data = combined_df[combined_df['condition'] == condition]
        average_msd = condition_data.groupby('MSD_LAG_TIME')['MSD'].mean()
        time_in_seconds = average_msd.index * frame_interval

        ax_no_fit.plot(time_in_seconds, average_msd, color=color, linewidth=5)

        handles_no_fit.append(plt.Line2D([], [], color=color, linewidth=5))
        new_labels_no_fit.append(label_base)

    ax_no_fit.set_xscale('log')
    ax_no_fit.set_yscale('log')
    ax_no_fit.set_xlabel('Lag time (sec)', fontsize=22, fontweight='bold', fontname='Arial')
    ax_no_fit.set_ylabel('MSD (µm²)', fontsize=22, fontweight='bold', fontname='Arial')
    ax_no_fit.set_xlim(left=1, right=1000)
    ax_no_fit.set_ylim(bottom=0.5, top=10000)
    ax_no_fit.tick_params(axis='both', which='minor', length=7, width=1, color='black')
    ax_no_fit.tick_params(axis='both', which='major', length=14, width=1, color='black')
    ax_no_fit.tick_params(axis='x', labelsize=20)
    ax_no_fit.tick_params(axis='y', labelsize=20)
    ax_no_fit.spines['top'].set_linewidth(1.25)
    ax_no_fit.spines['right'].set_linewidth(1.25)
    ax_no_fit.spines['left'].set_linewidth(1.25)
    ax_no_fit.spines['bottom'].set_linewidth(1.25)

    # Add legend with correct order
    ax_no_fit.legend(handles=handles_no_fit, labels=new_labels_no_fit, fontsize=13, loc='upper left')

    fig_no_fit.tight_layout()
    file_name_no_fit = os.path.join(save_dir, "average_MSD_no_fit.png")
    plt.savefig(file_name_no_fit, dpi=300, transparent=True)
    plt.close(fig_no_fit)  # Close the figure after saving it

    ## --- Second Plot: Average MSD with fit (log-log) ---
    fig_with_fit, ax_with_fit = plt.subplots(figsize=(fig_x, fig_y))

    handles_with_fit = []
    new_labels_with_fit = []

    # Specify the desired order for the strains in the legend: pilH, WT, pilG
    for strain_key in ordered_strains:
        condition = [cond for cond in conditions if strain_key in cond][0]

        color = strain_colors.get(strain_key, "black")
        label_base = strain_labels.get(strain_key, "Unknown")

        condition_data = combined_df[combined_df['condition'] == condition]
        average_msd = condition_data.groupby('MSD_LAG_TIME')['MSD'].mean()
        time_in_seconds = average_msd.index * frame_interval

        # Prepare full log-log data
        mask = (time_in_seconds >= 11) & (time_in_seconds <= 100)
        x_vals = time_in_seconds[mask]
        y_vals = average_msd[mask]

        if len(x_vals) == 0:
            continue

        log_x = np.log10(x_vals)
        log_y = np.log10(y_vals)

        slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
        r_squared = r_value ** 2

        # Plot: x < 10 s as dashed, x >= 10 s as solid
        solid_mask = time_in_seconds >= 10
        dashed_mask = time_in_seconds <= 10

        ax_with_fit.plot(time_in_seconds[solid_mask], average_msd[solid_mask], 
                      color=color, linewidth=5, linestyle='-')
        ax_with_fit.plot(time_in_seconds[dashed_mask], average_msd[dashed_mask], 
                      color=color, linewidth=5, linestyle=':')

        # Plot fit
        fit_line = 10**(slope * np.log10(x_vals) + intercept)
        ax_with_fit.plot(x_vals, fit_line, linestyle='--', linewidth=2, color='black', alpha=0.7)

        # Update legend
        handles_with_fit.append(plt.Line2D([], [], color=color, linewidth=5))
        new_labels_with_fit.append(f"{label_base}, a = {slope:.2f}") #R² = {r_squared:.0f}

        # Save fit results
        fit_results.append({
            "condition": condition,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared
        })


    # Add legend with correct order for the graph with fit
    ax_with_fit.legend(handles=handles_with_fit, labels=new_labels_with_fit, fontsize=14, loc='upper left')

    ax_with_fit.set_xscale('log')
    ax_with_fit.set_yscale('log')
    ax_with_fit.set_xlabel('Lag time (sec)', fontsize=22, fontweight='bold', fontname='Arial')
    ax_with_fit.set_ylabel('MSD (µm²)', fontsize=22, fontweight='bold', fontname='Arial')
    ax_with_fit.set_xlim(left=1, right=1000)
    ax_with_fit.set_ylim(bottom=0.5, top=10000)
    ax_with_fit.tick_params(axis='both', which='minor', length=7, width=1, color='black')
    ax_with_fit.tick_params(axis='both', which='major', length=14, width=1, color='black')
    ax_with_fit.tick_params(axis='x', labelsize=20)
    ax_with_fit.tick_params(axis='y', labelsize=20)
    ax_with_fit.spines['top'].set_linewidth(1.25)
    ax_with_fit.spines['right'].set_linewidth(1.25)
    ax_with_fit.spines['left'].set_linewidth(1.25)
    ax_with_fit.spines['bottom'].set_linewidth(1.25)

    fig_with_fit.tight_layout()
    file_name_with_fit = os.path.join(save_dir, "average_MSD_with_fit.png")
    plt.savefig(file_name_with_fit, dpi=300, transparent=True)
    plt.close(fig_with_fit)  # Close the figure after saving it
    
    ## --- Save fitting results ---
    fit_df = pd.DataFrame(fit_results)
    fit_df.to_csv(os.path.join(save_dir, "MSD_fit_parameters.csv"), index=False)
    
    ## --- Third Plot: Average normalized MSD with fit (log-log) ---

    fig_norm_with_fit, ax_norm_with_fit = plt.subplots(figsize=(4,4)) 
    
    handles_norm_with_fit = []
    new_labels_norm_with_fit = []
    
    characteristic_time = 4  # sec
    characteristic_length = 0.8  # µm
    
    fit_norm_results = []  # Initialize list to store fit results

    
    # Specify the desired order for the strains in the legend: pilH, WT, pilG
    for strain_key in ordered_strains:
        condition = [cond for cond in conditions if strain_key in cond][0]
    
        color = strain_colors.get(strain_key, "black")
        label_base = strain_labels.get(strain_key, "Unknown")
    
        condition_data = combined_df[combined_df['condition'] == condition]
        average_msd = condition_data.groupby('MSD_LAG_TIME')['MSD'].mean()
        time_in_seconds = average_msd.index * frame_interval
    
        # Normalize time and MSD
        norm_time = time_in_seconds / characteristic_time
        norm_msd = average_msd / (characteristic_length ** 2)
    
        # Prepare log-log fit in desired range (normalized time between ~2.75 and 25)
        mask = (norm_time >= 2.75) & (norm_time <= 25)
        x_vals_norm = norm_time[mask]
        y_vals_norm = norm_msd[mask]
    
        if len(x_vals_norm) == 0:
            continue
    
        normlog_x = np.log10(x_vals_norm)
        normlog_y = np.log10(y_vals_norm)
    
        norm_slope, norm_intercept, norm_r_value, norm_p_value, norm_std_err = linregress(normlog_x, normlog_y)
        norm_r_squared = norm_r_value ** 2
    
        # Plot: normalized time < 2.5 as dashed, >= 2.5 as solid
        solid_mask = norm_time >= 2.5
        dashed_mask = norm_time <= 2.5  
    
        ax_norm_with_fit.plot(norm_time[solid_mask], norm_msd[solid_mask],
                              color=color, linewidth=5, linestyle='-')
    
        # Plot fit line
        fit_norm_line = 10**(norm_slope * np.log10(x_vals_norm) + norm_intercept)
        ax_norm_with_fit.plot(x_vals_norm, fit_norm_line, linestyle='--', linewidth=2, color='black', alpha=0.7)
    
        # Update legend
        handles_norm_with_fit.append(plt.Line2D([], [], color=color, linewidth=5))
        new_labels_norm_with_fit.append(f"{label_base}, a = {norm_slope:.2f}")
    
        # Save fit results
        fit_norm_results.append({
            "condition": condition,
            "slope": norm_slope,
            "intercept": norm_intercept,
            "r_squared": norm_r_squared
        })
    
    # Add legend with correct order for the graph with fit
    ax_norm_with_fit.legend(handles=handles_norm_with_fit, labels=new_labels_norm_with_fit, fontsize=14, loc='upper left', frameon = False, bbox_to_anchor = (0, 1.05))
    
    ax_norm_with_fit.set_xscale('log')
    ax_norm_with_fit.set_yscale('log')
    ax_norm_with_fit.set_xlabel('Lag time', fontsize=22, fontweight='bold', fontname='Arial')
    ax_norm_with_fit.set_ylabel('MSD', fontsize=22, fontweight='bold', fontname='Arial')
    ax_norm_with_fit.set_xlim(left=0.8, right = 120)
    ax_norm_with_fit.set_ylim(bottom=1, top=10000)
    #ax_norm_with_fit.set_ylim(bottom=1, top=10000)
    ax_norm_with_fit.tick_params(axis='both', which='minor', length=7, width=1, color='black')
    ax_norm_with_fit.tick_params(axis='both', which='major', length=14, width=1, color='black')
    ax_norm_with_fit.tick_params(axis='x', labelsize=18)
    ax_norm_with_fit.tick_params(axis='y', labelsize=18)
    ax_norm_with_fit.spines['top'].set_visible(False)
    ax_norm_with_fit.spines['right'].set_visible(False)
    ax_norm_with_fit.spines['left'].set_linewidth(1.25)
    ax_norm_with_fit.spines['bottom'].set_linewidth(1.25)
    ax_norm_with_fit.set_box_aspect(1)
    
    fig_norm_with_fit.tight_layout()
    
    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)
    
    file_name_norm_with_fit = os.path.join(save_dir, "average_MSD_norm_with_fit.png")
    plt.savefig(file_name_norm_with_fit, dpi=300, transparent=True)
    plt.close(fig_norm_with_fit)


    ## --- Save fitting results ---
    fit_norm_df = pd.DataFrame(fit_norm_results)
    fit_norm_df.to_csv(os.path.join(save_dir, "MSD_norm_fit_parameters.csv"), index=False)

    
    # --- Fourth Plot: Average normalized MSD + simulations curves ---

    handles = []
    labels = []
    
    characteristic_time = 4  # sec
    characteristic_length = 0.8  # µm
    
    fig, ax = plt.subplots(figsize=(4,4)) 
    
    # Define zorder mapping to control plot order
    zorder_map = {
        '459_simulations': 1,
        '232_simulations': 2,
        '177_simulations': 3,
        '459_exp': 4,
        '232_exp': 5,
        '177_exp': 6,
    }
    
    # For legend ordering
    legend_order = ['232', '177', '459']
    legend_handles = {}
    legend_labels = {}
    
    for strain_key in ordered_strains:
        # Find the corresponding condition
        condition = [cond for cond in conditions if strain_key in cond]
        if not condition:
            print(f"No condition found for strain {strain_key}, skipping...")
            continue
        condition = condition[0]
    
        # Extract coverage (e.g. SC70pc)
        match_coverage = re.search(r'SC\d+pc', condition)
        if match_coverage:
            coverage_str = match_coverage.group(0)
        else:
            print(f"Could not extract coverage from condition '{condition}'")
            continue
    
        # Extract strain number
        match_strain = re.search(r'\d+', strain_key)
        if match_strain:
            strain_id = match_strain.group(0)
        else:
            print(f"Could not extract strain number from strain_key '{strain_key}'")
            continue
    
        # Filter data
        condition_data = combined_df[combined_df['condition'] == condition]
    
        # Color and label
        color = strain_colors.get(strain_key, "black")
        label_base = strain_labels.get(strain_key, "Unknown")
    
        # Load simulations curve
        simulations_filename = f"msd_{coverage_str}_{strain_id}.csv"
        simulations_filepath = os.path.join(simulations_dir, simulations_filename)
        print(f"Looking for simulations file: {simulations_filepath}")
    
        if os.path.isfile(simulations_filepath):
            simulations_df = pd.read_csv(simulations_filepath, header=None)
            simulations_time = simulations_df[0].values
            simulations_msd = simulations_df[1].values
            ax.plot(simulations_time, simulations_msd, color=color, linestyle='--', linewidth=3,
                    zorder=zorder_map.get(f'{strain_id}_simulations', 1))

        else:
            print(f"Simulations file NOT found: {simulations_filepath}")
    
        # Experimental MSD average
        average_msd = condition_data.groupby('MSD_LAG_TIME')['MSD'].mean()
        time_in_seconds = average_msd.index * frame_interval
    
        # Normalization
        norm_time = time_in_seconds / characteristic_time
        norm_msd = average_msd / (characteristic_length ** 2)
    
        # Fit log-log over [2.75, 25] normalized time
        fit_mask = (norm_time >= 2.75) & (norm_time <= 25)
        x_fit = norm_time[fit_mask]
        y_fit = norm_msd[fit_mask]
    
        if len(x_fit) > 0:
            log_x = np.log10(x_fit)
            log_y = np.log10(y_fit)
            slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
        else:
            slope = np.nan
    
        # Plot experimental curve for norm_time >= 2.5
        solid_mask = norm_time >= 2.5
        ax.plot(norm_time[solid_mask], norm_msd[solid_mask], color=color, linewidth=5, linestyle='-',
                zorder=zorder_map.get(f'{strain_id}_exp', 4))
    
        # Add to legend (only experiments)
        if not np.isnan(slope):
            legend_labels[strain_id] = f"{label_base}" #, a = {norm_slope:.2f}"
        else:
            legend_labels[strain_id] = label_base
    
        legend_handles[strain_id] = plt.Line2D([], [], color=color, linewidth=5)
    
    # Format legend
    handles = [legend_handles[sid] for sid in legend_order if sid in legend_handles]
    labels = [legend_labels[sid] for sid in legend_order if sid in legend_labels]
    ax.legend(handles=handles, labels=labels, fontsize=16, loc='upper left', frameon = False, bbox_to_anchor = (-0.01, 1.1))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Lag time', fontsize=22, fontweight='bold', fontname='Arial')
    ax.set_ylabel('MSD', fontsize=22, fontweight='bold', fontname='Arial')
    ax.set_xlim(left=0.8, right = 120)
    ax.set_ylim(bottom=1, top=10000)
    ax.tick_params(axis='both', which='minor', length=7, width=1, color='black')
    ax.tick_params(axis='both', which='major', length=14, width=1, color='black')
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.25)
    ax.spines['bottom'].set_linewidth(1.25)
    ax.set_box_aspect(1)
    
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    
    file_name = os.path.join(save_dir, "average_MSD_norm_with_simulations.png")
    plt.savefig(file_name, dpi=300, transparent=True)
    plt.close(fig)

