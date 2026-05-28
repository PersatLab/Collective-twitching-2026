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
from skimage.measure import label, regionprops
from skimage.io import imread
from scipy.stats import circmean
from code_modules import SegmentationAnalysis_02_utils as sau
import ast


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
                dataframes[filename] = pd.read_csv(filepath, encoding='utf-8')
                print(f"Loaded {filename} with utf-8 encoding")
            except UnicodeDecodeError:
                print(f"Failed to load {filename} with utf-8 encoding. Trying with latin1 encoding.")
                try:
                    # If utf-8 fails, try loading with latin1 encoding
                    dataframes[filename] = pd.read_csv(filepath, encoding='latin1')
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

    plt.show()




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
            

#-----------------------------
# Contact reversals analysis
#-----------------------------

def tracks_classification(spots_table, displacement_threshold):
    """
    Classify tracks based on displacement and create a subset of tracks with at least three consecutive "moving" frames.

    Parameters:
    - spots_table (pd.DataFrame): Dataframe containing tracks with ['POSITION_X', 'POSITION_Y', 'TRACK_ID', 'FRAME'].
    - displacement_threshold (float): Threshold in pixels to classify bacteria as "moving".

    Returns:
    - pd.DataFrame: Updated dataframe with "CLASS" column.
    - pd.DataFrame: Subset dataframe with tracks meeting the criteria.
    """
    # Initialize the CLASS column
    spots_table['CLASS'] = "not moving"

    # List to store valid TRACK_IDs
    valid_track_ids = []

    # Group by TRACK_ID to compute displacements
    for track_id, group in spots_table.groupby('TRACK_ID'):
        group = group.sort_values(by='FRAME')  # Ensure the group is sorted by FRAME

        # Compute displacements
        displacements = np.sqrt(
            np.diff(group['POSITION_X'])**2 + np.diff(group['POSITION_Y'])**2
        )

        # Assign "moving" if displacement > threshold
        moving_frames = group.iloc[:-1].index[displacements > displacement_threshold]
        spots_table.loc[moving_frames, 'CLASS'] = "moving"

        # Default last frame's CLASS to match the previous frame
        if len(group) > 1:
            spots_table.loc[group.index[-1], 'CLASS'] = spots_table.loc[group.index[-2], 'CLASS']

        # Check if there are at least three consecutive "moving" frames
        is_moving = spots_table.loc[group.index, 'CLASS'] == "moving"
        
        # Look for at least 3 consecutive True values in the "is_moving" series
        for i in range(len(is_moving) - 2):
            if is_moving.iloc[i] and is_moving.iloc[i + 1] and is_moving.iloc[i + 2]:
                valid_track_ids.append(track_id)
                break  # Found a valid track, no need to check further frames
        
    # Filter subtracks_table for only valid TRACK_IDs
    subtracks_table = spots_table[spots_table['TRACK_ID'].isin(valid_track_ids)]

    return spots_table, subtracks_table



def compute_angle(vec1, vec2):
    """
    Compute the angle between two vectors in degrees.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cos_angle = dot_product / (norm1 * norm2)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Ensure value is within range [-1, 1]
    return np.degrees(angle)



def compute_reversals(spots_table, excluded_track_ids=None):
    """
    Computes reversals based on relative angles, excluding specified track IDs and checking for movement status.

    Parameters:
    - spots_table (pd.DataFrame): DataFrame containing the trajectory data.
    - excluded_track_ids (list, optional): List of TRACK_IDs to exclude from the analysis.

    Returns:
    - pd.DataFrame: Updated spots_table with 'DISP_RELATIVE_ANGLE (deg)' and 'REVERSAL' columns.
    """
    # Remove rows where 'CLASS' is 'not moving'
    spots_table = spots_table[spots_table['CLASS'] != 'not moving'].copy()

    # Exclude specified track IDs if any
    if excluded_track_ids is not None:
        spots_table = spots_table[~spots_table['TRACK_ID'].isin(excluded_track_ids)]

    # Add new columns for relative angles and reversals
    spots_table['DISP_RELATIVE_ANGLE (deg)'] = np.nan
    spots_table['REVERSAL'] = 0  # Initialize the 'REVERSAL' column with 0

    # Iterate over each track_id
    track_ids = spots_table['TRACK_ID'].unique()

    for track_id in track_ids:
        track_data = spots_table[spots_table['TRACK_ID'] == track_id]
        positions = track_data[['POSITION_X', 'POSITION_Y']].values
        frames = track_data['FRAME'].values
        classes = track_data['CLASS'].values

        # Loop through frames, leaving enough frames for the 4-vector calculation
        for i in range(2, len(frames) - 2):
            # Ensure positions[i-2] and positions[i-1] are associated with "moving"
            if classes[i - 2] == "moving" and classes[i - 1] == "moving":
                # Define vectors
                vec1 = positions[i - 1] - positions[i - 2]  # Vector from frame i-2 to frame i-1
                vec2 = positions[i] - positions[i - 1]      # Vector from frame i-1 to frame i
                vec3 = positions[i + 1] - positions[i]      # Vector from frame i to frame i+1
                vec4 = positions[i + 2] - positions[i + 1]  # Vector from frame i+1 to frame i+2

                # Compute angles between consecutive vectors
                angle1 = compute_angle(vec1, vec2)
                angle2 = compute_angle(vec2, vec3)
                angle3 = compute_angle(vec3, vec4)

                # Save the angle between vec2 and vec3 in the corresponding frame (i)
                spots_table.loc[(spots_table['TRACK_ID'] == track_id) & 
                                (spots_table['FRAME'] == frames[i]), 'DISP_RELATIVE_ANGLE (deg)'] = angle2

                # Check for reversal condition
                if angle1 < 90 and angle2 > 110 and angle3 < 90:
                    # Mark the current frame as a reversal
                    spots_table.loc[(spots_table['TRACK_ID'] == track_id) & 
                                    (spots_table['FRAME'] == frames[i]), 'REVERSAL'] = 1

    return spots_table


def compute_total_tracked_time(spots_table, frame_interval):
    """
    Computes the total tracked time for all trajectories in the dataframe.

    Parameters:
    df (pd.DataFrame): DataFrame with columns ['TRACK_ID', 'FRAME']
    pixel_scale (float): Scale factor to convert frames to actual time (e.g., seconds).

    Returns:
    float: Total tracked time.
    """
    # Group by TRACK_ID and compute the duration for each trajectory
    #grouped = spots_table.groupby('TRACK_ID')['FRAME'].agg(['min', 'max'])
    #duration = (grouped['max'] - grouped['min'] + 1)  # Duration in frames for each trajectory
    #total_tracked_time = duration.sum() * frame_interval
    counts = spots_table.groupby('TRACK_ID')['FRAME'].count()
    duration_sec = (counts - 1) * frame_interval
    total_tracked_time = duration_sec.sum()
    
    
    return total_tracked_time


def is_within_half_circle(center, angle, radius, pixel_x, pixel_y, frame, segmented_image):
    """
    Check if a given pixel (pixel_x, pixel_y) is within a half-circle centered at 'center' with a given angle and radius.
    
    Parameters:
    - center (tuple): (x, y) coordinates of the center of the half-circle.
    - angle (float): The direction of the half-circle in radians (relative to the x-axis).
    - radius (int): The radius of the half-circle.
    - pixel_x, pixel_y (int): The coordinates of the pixel to check.
    - frame (int): The frame number in the segmented image.
    - segmented_image (np.ndarray): The segmented image (3D array).
    
    Returns:
    - bool: True if the pixel is within the half-circle, False otherwise.
    """
    # Compute the vector from the center to the pixel (pixel_x, pixel_y)
    dx = pixel_x - center[0]
    dy = pixel_y - center[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    # Check if the pixel is within the radius
    if distance > radius:
        return False
    
    # Compute the angle of the vector (center -> pixel) relative to the x-axis
    pixel_angle = np.arctan2(dy, dx)
    
    # Check if the pixel's angle is within the half-circle's angle range
    angle_range_start = angle - np.pi / 2  # Left boundary of the half-circle
    angle_range_end = angle + np.pi / 2    # Right boundary of the half-circle
    
    # Normalize the pixel_angle and angle_range to be between -pi and pi
    pixel_angle = (pixel_angle + np.pi) % (2 * np.pi) - np.pi
    angle_range_start = (angle_range_start + np.pi) % (2 * np.pi) - np.pi
    angle_range_end = (angle_range_end + np.pi) % (2 * np.pi) - np.pi
    
    # Check if the pixel's angle is within the range
    if angle_range_start <= pixel_angle <= angle_range_end:
        # If the pixel is within the radius and the angle range, check if it is part of the neighbors
        if 0 <= pixel_x < segmented_image.shape[2] and 0 <= pixel_y < segmented_image.shape[1]:
            pixel_value = segmented_image[frame, pixel_y, pixel_x]
            return pixel_value != 0  # Ensure the pixel is not background (0)
    
    return False



def find_contact_and_measure_angle(spots_dataframe, segmented_image):
    """
    Updates the spots_dataframe with contact information for each bacterium.

    Parameters:
    - spots_dataframe (pd.DataFrame): DataFrame with columns ['SPOT_LABEL', 'FRAME', 'NEIGHBOUR'].

    Returns:
    - pd.DataFrame: Updated spots_dataframe with new columns ['CONTACT', 'UPDATE_NEIGHBOUR'].
    """
    # Create new columns for contact and updated neighbors if not already present
    spots_dataframe['CONTACT'] = spots_dataframe.get('CONTACT', pd.Series(dtype='int'))
    spots_dataframe['UPDATE_NEIGHBOUR'] = spots_dataframe.get('UPDATE_NEIGHBOUR', pd.Series(dtype='object'))
    spots_dataframe['CONTACT_ANGLE (deg)'] = spots_dataframe.get('CONTACT_ANGLE (deg)')
    updated_neighbours = []
    
    num_frames, height, width = segmented_image.shape
    
    # Iterate through each row in the dataframe
    for idx, row in spots_dataframe.iterrows():
        spot_label = row['SPOT_LABEL']
        frame = row['FRAME']
        neighbours = row['NEIGHBOUR'] if isinstance(row['NEIGHBOUR'], list) else []
        updated_neighbours = list(neighbours) # Create a copy of the neighbours list
        contact = len(updated_neighbours)
        
        # Loop through each neighbour
        if len(neighbours)>0:
            for neighbour in neighbours:
                if spot_label > neighbour:
                    # Check if the neighbour has the current spot label in its neighbour list
                    neighbour_row = spots_dataframe[(spots_dataframe['SPOT_LABEL'] == neighbour) & (spots_dataframe['FRAME'] == frame)]
                    if not neighbour_row.empty and spot_label in neighbour_row['NEIGHBOUR']: #If this contact is already counted before, don't consider it and remove the corresponding neighbour.
                        updated_neighbours.remove(neighbour)
                        contact -= 1
                    else:
                        continue

        # Update the CONTACT and UPDATE_NEIGHBOUR columns
        spots_dataframe.at[idx, 'CONTACT'] = contact
        spots_dataframe.at[idx, 'UPDATE_NEIGHBOUR'] = updated_neighbours
        
        
        if len(updated_neighbours)>0:
            contact_angle_list = []
            segmented_frame = segmented_image[frame, :, :]
            cell_mask = segmented_frame == spot_label
            long_extremities = sau.find_extremities_of_cells_length(cell_mask)
            feature_values = sau.get_directions_degrees(cell_mask, long_extremities)
            
            if feature_values[1] < 0:
                feature_values[1] += 180
            if feature_values[1] >= 180:
                feature_values[1] -= 180
            
            for neighbour in updated_neighbours:
                neighbour_mask = segmented_frame == neighbour
                neighbour_long_extremities = sau.find_extremities_of_cells_length(neighbour_mask)
                neighbour_feature_values = sau.get_directions_degrees(neighbour_mask, neighbour_long_extremities)
                
                if neighbour_feature_values[1] < 0:
                    neighbour_feature_values[1] += 180
                if neighbour_feature_values[1] >= 180:
                    neighbour_feature_values[1] -= 180
                
                if neighbour_feature_values[1] > feature_values[1]:
                    contact_angle = neighbour_feature_values[1] - feature_values[1]
                else:
                    contact_angle = feature_values[1] - neighbour_feature_values[1]
                    
                contact_angle_list.append(contact_angle)
                
                
            spots_dataframe.at[idx, 'CONTACT_ANGLE (deg)'] = contact_angle_list

    return spots_dataframe


def find_contact_reversals(spots_dataframe):
    """
    Updates the spots_dataframe with contact reversal information for each bacterium.

    Parameters:
    - spots_dataframe (pd.DataFrame): DataFrame with columns ['CONTACT', 'REVERSAL'].

    Returns:
    - pd.DataFrame: Updated spots_dataframe with a new column ['CONTACT_REVERSAL'].
    """
    # Create a new column for contact reversal if not already present
    spots_dataframe['CONTACT_REVERSAL'] = spots_dataframe.get('CONTACT_REVERSAL', pd.Series(dtype='int'))

    # Iterate through each row in the dataframe
    for idx, row in spots_dataframe.iterrows():
        if row['CONTACT'] == 1 and row['REVERSAL'] == 1:
            spots_dataframe.at[idx, 'CONTACT_REVERSAL'] = 1
        else:
            spots_dataframe.at[idx, 'CONTACT_REVERSAL'] = 0

    return spots_dataframe


def save_contact_reversals_data(spots_dataframe, key, frame_interval):
    """
    Creates a summary dataframe with contact and reversal metrics.

    Parameters:
    - spots_dataframe (pd.DataFrame): DataFrame with columns ['CONTACT', 'REVERSAL', 'CONTACT_REVERSAL', 'TRACK_ID'].
    - key (str): Identifier for the file.
    - compute_total_tracked_time (function): Function to compute total tracked time.

    Returns:
    - pd.DataFrame: Summary dataframe.
    """
    # Calculate the total number of unique tracks
    num_tracks = spots_dataframe['TRACK_ID'].nunique()

    # Calculate the number of tracks with at least one contact reversal
    tracks_with_contact_reversal = spots_dataframe.loc[
        spots_dataframe['CONTACT_REVERSAL_CORRECTED'] > 0, 'TRACK_ID'
    ].nunique()

    if key == 'global':
        summary_data = {
            "Number of tracks": num_tracks,
            "Number of tracks with contact reversals": tracks_with_contact_reversal,
            "Number of contacts": spots_dataframe['CONTACT_CORRECTED'].sum(),
            "Number of reversals": spots_dataframe['REVERSAL'].sum(),
            "Number of contact reversals": spots_dataframe['CONTACT_REVERSAL_CORRECTED'].sum(),
            "Probability of contact reversals (%)": (spots_dataframe['CONTACT_REVERSAL_CORRECTED'].sum() / spots_dataframe['CONTACT_CORRECTED'].sum()) * 100 if spots_dataframe['CONTACT_CORRECTED'].sum() > 0 else 0,
            "Total tracked time (h)": compute_total_tracked_time(spots_dataframe, frame_interval / 3600),
            "Contacts frequency (/h)": spots_dataframe['CONTACT'].sum() / compute_total_tracked_time(spots_dataframe, frame_interval / 3600) if compute_total_tracked_time(spots_dataframe, frame_interval / 3600) > 0 else 0,
            "Reversals frequency (/h)": spots_dataframe['REVERSAL'].sum() / compute_total_tracked_time(spots_dataframe, frame_interval / 3600) if compute_total_tracked_time(spots_dataframe, frame_interval / 3600) > 0 else 0,
            "Contact reversals frequency (/h)": spots_dataframe['CONTACT_REVERSAL_CORRECTED'].sum() / compute_total_tracked_time(spots_dataframe, frame_interval / 3600) if compute_total_tracked_time(spots_dataframe, frame_interval / 3600) > 0 else 0
        }
    else:
        summary_data = {
            "File": key + ".tif",
            "Number of tracks": num_tracks,
            "Number of tracks with contact reversals": tracks_with_contact_reversal,
            "Number of contacts": spots_dataframe['CONTACT_CORRECTED'].sum(),
            "Number of reversals": spots_dataframe['REVERSAL'].sum(),
            "Number of contact reversals": spots_dataframe['CONTACT_REVERSAL_CORRECTED'].sum(),
            "Probability of contact reversals (%)": (spots_dataframe['CONTACT_REVERSAL_CORRECTED'].sum() / spots_dataframe['CONTACT_CORRECTED'].sum()) * 100 if spots_dataframe['CONTACT_CORRECTED'].sum() > 0 else 0,
            "Total tracked time (h)": compute_total_tracked_time(spots_dataframe, frame_interval / 3600),
            "Contacts frequency (/h)": spots_dataframe['CONTACT'].sum() / compute_total_tracked_time(spots_dataframe, frame_interval / 3600) if compute_total_tracked_time(spots_dataframe, frame_interval / 3600) > 0 else 0,
            "Reversals frequency (/h)": spots_dataframe['REVERSAL'].sum() / compute_total_tracked_time(spots_dataframe, frame_interval / 3600) if compute_total_tracked_time(spots_dataframe, frame_interval / 3600) > 0 else 0,
            "Contact reversals frequency (/h)": spots_dataframe['CONTACT_REVERSAL_CORRECTED'].sum() / compute_total_tracked_time(spots_dataframe, frame_interval / 3600) if compute_total_tracked_time(spots_dataframe, frame_interval / 3600) > 0 else 0
        }

    summary_df = pd.DataFrame([summary_data])
    
    if key == 'global':
        print("\n")
        print(f"  Number of tracks: {summary_data['Number of tracks']}")
        print(f"  Number of tracks with contact reversals: {summary_data['Number of tracks with contact reversals']}")
        print(f"  Number of contacts: {summary_data['Number of contacts']}")
        print(f"  Number of reversals: {summary_data['Number of reversals']}")
        print(f"  Number of contact reversals: {summary_data['Number of contact reversals']}")
        print(f"  Probability of contact reversals (%): {summary_data['Probability of contact reversals (%)']}")
        print(f"  Total tracked time (in hours): {summary_data['Total tracked time (h)']}")
        print(f"  Contacts frequency (/h): {summary_data['Contacts frequency (/h)']}")
        print(f"  Reversals frequency (/h): {summary_data['Reversals frequency (/h)']}")
        print(f"  Contact reversals frequency (/h): {summary_data['Contact reversals frequency (/h)']}")
        print("\n")
    else:
        print("\n")
        print(f"File: {summary_data['File']}")
        print(f"  Number of tracks: {summary_data['Number of tracks']}")
        print(f"  Number of tracks with contact reversals: {summary_data['Number of tracks with contact reversals']}")
        print(f"  Number of contacts: {summary_data['Number of contacts']}")
        print(f"  Number of reversals: {summary_data['Number of reversals']}")
        print(f"  Number of contact reversals: {summary_data['Number of contact reversals']}")
        print(f"  Probability of contact reversals (%): {summary_data['Probability of contact reversals (%)']}")
        print(f"  Total tracked time (in hours): {summary_data['Total tracked time (h)']}")
        print(f"  Contacts frequency (/h): {summary_data['Contacts frequency (/h)']}")
        print(f"  Reversals frequency (/h): {summary_data['Reversals frequency (/h)']}")
        print(f"  Contact reversals frequency (/h): {summary_data['Contact reversals frequency (/h)']}")
        print("\n")

    return summary_df




def plot_contact_reversals(spots_dataframe, key, save_dir, bin_size=10):
    """
    Plot histograms and polar plots of the probability of contact reversal.

    Args:
        spots_dataframe (pd.DataFrame): DataFrame containing the data with columns 
                                        ['CONTACT_CORRECTED', 'RESCALED_ANGLE (deg)', 'CONTACT_REVERSAL_CORRECTED'].
        bin_size (int): Bin size for the histogram (in degrees). Default is 10.
        save_dir (str): Path to save the figures.
    """
    # Explode the 'RESCALED_ANGLE (deg)' list into separate rows
    my_column = 'RESCALED_ANGLE (deg)'
    spots_dataframe = spots_dataframe.explode(my_column)

    # Convert 'RESCALED_ANGLE (deg)' values to float, setting NaN to -1 (or any placeholder)
    spots_dataframe[my_column] = pd.to_numeric(spots_dataframe[my_column], errors='coerce').fillna(-1)

    # Bin the contact angles, excluding rows with placeholder (-1)
    spots_dataframe['ANGLE_BIN'] = np.where(
        spots_dataframe[my_column] >= 0,
        (spots_dataframe[my_column] // bin_size * bin_size).astype(int),
        -1  # Assign -1 as the bin for NaN values
    )

    # Filter out rows with invalid bins (-1) for calculations
    valid_bins = spots_dataframe[spots_dataframe['ANGLE_BIN'] >= 0]

    # Calculate the total number of contacts and contact reversals in each valid bin
    grouped = valid_bins.groupby('ANGLE_BIN').agg(
        total_contacts=('CONTACT_CORRECTED', 'sum'),
        total_contact_reversals=('CONTACT_REVERSAL_CORRECTED', 'sum')
    ).reset_index()

    # Calculate global probability of contact reversals
    total_contacts_all = grouped['total_contacts'].sum()
    grouped['probability_reversal_global'] = grouped['total_contact_reversals'] / total_contacts_all
    grouped['probability_reversal_local'] = grouped['total_contact_reversals'] / grouped['total_contacts']

    # # Standard bar plot (Global Probability)
    # plt.figure(figsize=(10, 6))
    # plt.bar(grouped['ANGLE_BIN'], grouped['probability_reversal_global'], width=bin_size, align='edge', color='black', edgecolor='white')
    # plt.ylabel('Probability of Contact Reversal', fontsize=14)
    # plt.xlabel('Contact Angle (°)', fontsize=14)
    # plt.title('Probability of Contact Reversal vs. Contact Angle (0° to 90°)', fontsize=14)
    # plt.xticks(np.arange(0, 91, bin_size), fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.grid(axis='y', linestyle='--', alpha=0.5)
    # plt.xlim(0, 90)
    # plt.savefig(save_dir + '/' + key + '_contact_reversals_histogram_global.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # # Polar plot (Global Probability)
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='polar')
    # angles_rad = np.radians(grouped['ANGLE_BIN'] + bin_size / 2)
    # probabilities_global = grouped['probability_reversal_global']
    # ax.bar(angles_rad, probabilities_global, width=np.radians(bin_size), bottom=0.0, color='black', edgecolor='white', alpha=1)
    # ax.grid(alpha=0.5, linestyle='--')
    # ax.set_theta_zero_location('E')  # Set 0° to be at the right (East)
    # ax.set_thetamin(0) 
    # ax.set_thetamax(90) 
    # ax.set_xticks(np.radians(np.arange(0, 91, bin_size)))  
    # ax.set_xticklabels(np.arange(0, 91, bin_size), fontsize=12)
    # ax.tick_params(axis='y', labelsize=12) 
    # ax.set_title('Probability of Contact Reversal vs. Contact Angle (0° to 90°)', fontsize=14)
    # plt.savefig(save_dir + '/' + key + '_contact_reversals_polar_global.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # # Polar plot (Local Probability)
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='polar')
    # angles_rad = np.radians(grouped['ANGLE_BIN'] + bin_size / 2)
    # probabilities_local = grouped['probability_reversal_local']
    # ax.bar(angles_rad, probabilities_local, width=np.radians(bin_size), bottom=0.0, color='black', edgecolor='white', alpha=1)
    # ax.grid(alpha=0.5, linestyle='--')
    # ax.set_theta_zero_location('E')  # Set 0° to be at the right (East)
    # ax.set_thetamin(0)
    # ax.set_thetamax(90) 
    # ax.set_xticks(np.radians(np.arange(0, 91, bin_size)))  
    # ax.set_xticklabels(np.arange(0, 91, bin_size), fontsize=12)  
    # ax.tick_params(axis='y', labelsize=12) 
    # ax.set_title('Local Probability of Contact Reversal vs. Contact Angle (0° to 90°)', fontsize=14)
    # plt.savefig(save_dir + '/' + key + '_contact_reversals_polar_local.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # Distribution of 'RESCALED_ANGLE (deg)'
    # rescaled_column = 'RESCALED_ANGLE (deg)'
    # spots_dataframe[rescaled_column] = pd.to_numeric(spots_dataframe[rescaled_column], errors='coerce')

    # valid_rescaled_angles = spots_dataframe[
    #     spots_dataframe[rescaled_column].between(0, 90, inclusive='both')
    # ][rescaled_column]

    # if valid_rescaled_angles.empty:
    #     print(f"No valid data found in '{rescaled_column}' for the range [0, 90].")
    #else:
        # plt.figure(figsize=(10, 6))
        # plt.hist(valid_rescaled_angles, bins=np.arange(0, 91, bin_size), color='black', edgecolor='white', alpha=1)
        # plt.xlabel('Rescaled Angle (deg)', fontsize=14)
        # plt.ylabel('Frequency (/h)', fontsize=14)
        # plt.title('Distribution of Rescaled Angles (0° to 90°)', fontsize=14)
        # plt.xticks(np.arange(0, 91, bin_size), fontsize=12)
        # plt.yticks(fontsize=12)
        # plt.grid(axis='y', linestyle='--', alpha=0.5)
        # plt.xlim(0, 90)
        # plt.savefig(save_dir + '/' + key + '_rescaled_angle_distribution.png', dpi=300, bbox_inches='tight')
        # plt.show()


def load_corrected_files(folder_path):
    """
    Load all CSV files with '-corrected' in their name from a specified folder.
    Files are loaded into a dictionary with error handling.
    
    Parameters:
        folder_path (str): The path to the folder containing the CSV files.
    
    Returns:
        dict: A dictionary where keys are filenames and values are DataFrames.
    """
    # Find all CSV files with '-corrected' in their name
    corrected_csv_files = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if '-corrected' in filename and filename.endswith('.csv')
    ]

    # Load all matching CSV files into a dictionary with error handling
    spots_dataframes = {}
    for file in corrected_csv_files:
        try:
            # Change the delimiter to ';'
            spots_dataframes[os.path.basename(file)] = pd.read_csv(file, sep=';')
            print(f"Successfully loaded: {file}")
        except Exception as e:
            print(f"Failed to load {file} due to: {e}")

    # Check loaded dataframes
    #print("Loaded dataframes:", list(spots_dataframes.keys()))
    return spots_dataframes


def compute_corrected_contact_angle(spots_dataframe, segmented_image): # between 0° and 180°
    # Initialize the 'CONTACT_ANGLE_CORRECTED (deg)' column if it does not exist
    spots_dataframe['CONTACT_ANGLE_CORRECTED (deg)'] = spots_dataframe.get('CONTACT_ANGLE_CORRECTED (deg)')

    # Iterate over each row of the dataframe
    for idx, row in spots_dataframe.iterrows():
        spot_label = row['SPOT_LABEL']
        #print(idx)
        
        # Skip rows where 'UPDATE_NEIGHBOUR_CORRECTED' is NaN or not valid
        if pd.isna(row['UPDATE_NEIGHBOUR_CORRECTED']):
            continue
        
        try:
            updated_neighbours = ast.literal_eval(row['UPDATE_NEIGHBOUR_CORRECTED'])
        except (ValueError, SyntaxError):
            # Skip malformed data
            continue

        # Ensure the neighbours are in a list format (if not already)
        if len(updated_neighbours) > 0:
            contact_angle_list = []
            frame = int(row['FRAME'])
            segmented_frame = segmented_image[frame, :, :]
            cell_mask = segmented_frame == spot_label

            # Assuming sau.find_extremities_of_cells_length() and sau.get_directions_degrees() are defined somewhere
            long_extremities = sau.find_extremities_of_cells_length(cell_mask)
            feature_values = sau.get_directions_degrees(cell_mask, long_extremities)

            if feature_values[1] < 0:
                feature_values[1] += 180
            if feature_values[1] >= 180:
                feature_values[1] -= 180

            # Loop over each neighbour to calculate contact angles
            for neighbour in updated_neighbours:
                neighbour_mask = segmented_frame == neighbour
                neighbour_long_extremities = sau.find_extremities_of_cells_length(neighbour_mask)
                neighbour_feature_values = sau.get_directions_degrees(neighbour_mask, neighbour_long_extremities)
                
                if neighbour_feature_values[1] < 0:
                    neighbour_feature_values[1] += 180
                if neighbour_feature_values[1] >= 180:
                    neighbour_feature_values[1] -= 180

                # Compute contact angle based on the angle difference
                if neighbour_feature_values[1] > feature_values[1]:
                    contact_angle = neighbour_feature_values[1] - feature_values[1]
                else:
                    contact_angle = feature_values[1] - neighbour_feature_values[1]

                contact_angle_list.append(contact_angle)

            # Store the calculated contact angles in the dataframe
            spots_dataframe.at[idx, 'CONTACT_ANGLE_CORRECTED (deg)'] = contact_angle_list

    # Update 'CONTACT_REVERSAL_CORRECTED' column based on conditions
    spots_dataframe['CONTACT_REVERSAL_CORRECTED'] = spots_dataframe.apply(
        lambda row: 1 if row['CONTACT_CORRECTED'] == 1 and row['REVERSAL_CORRECTED'] == 1 else row['CONTACT_REVERSAL_CORRECTED'],
        axis=1
    )

    return spots_dataframe


def rescale_angle(df_with_corrected_angles): # between 0° and 90°
    """
    Add a new column ['RESCALED_ANGLE'] to the DataFrame with angles rescaled to the range 0° to 90°.
    If the angle in ['CONTACT_ANGLE_CORRECTED (deg)'] is a list, each element is processed.
    Angles greater than 90° are reflected (180 - angle). Angles less than or equal to 90° remain unchanged.
    Handles None or NaN values gracefully.
    
    Args:
        df_with_corrected_angles (pd.DataFrame): DataFrame containing the 'CONTACT_ANGLE_CORRECTED (deg)' column.
        
    Returns:
        pd.DataFrame: DataFrame with the added 'RESCALED_ANGLE' column.
    """
    def rescale(angle):
        """Rescale a single angle or a list of angles."""
        if angle is None:  # Handle NoneType
            return None
        if isinstance(angle, list):
            # Process each angle in the list, ignoring None values
            return [180 - a if a is not None and a > 90 else a for a in angle]
        elif isinstance(angle, (int, float)):  # Process single values
            return 180 - angle if angle > 90 else angle
        else:
            return None  # If the value is neither a list nor a number, return None

    # Apply the rescaling function to each row in the column
    df_with_corrected_angles['RESCALED_ANGLE (deg)'] = df_with_corrected_angles['CONTACT_ANGLE_CORRECTED (deg)'].apply(rescale)
    
    return df_with_corrected_angles



def find_leading_pole(spots_dataframe, segmented_image, spot_ID, frame, pixel_scale):
    """
    Identify the leading pole of a bacterium based on its displacement.
    """
    # Get pole coordinates of the bacterium in the given frame
    mask = segmented_image[frame, :, :] == spot_ID
    result_image = mask * segmented_image[frame, :, :]
    
    poles = sau.find_extremities_of_cells_length(result_image)
    
    pole_1 = poles[spot_ID][0]
    pole_2 = poles[spot_ID][1]

    # Filter the spots dataframe for the specific bacterium and frames i and i+1
    track_id_row = spots_dataframe[(spots_dataframe['SPOT_LABEL'] == spot_ID) & (spots_dataframe['FRAME']== frame)]
    track_id = track_id_row.iloc[0]['TRACK_ID']
    bacterium_data = spots_dataframe[
        (spots_dataframe['TRACK_ID'] == track_id) & 
        (spots_dataframe['FRAME'].isin([frame, frame + 1]))
    ]
    

    if bacterium_data.shape[0] != 2:
        return None  # Missing data for consecutive frames

    # Get the positions of the bacterium at frames i and i+1
    position_i = bacterium_data[bacterium_data['FRAME'] == frame][['POSITION_Y', 'POSITION_X']].iloc[0].values / pixel_scale
    position_i1 = bacterium_data[bacterium_data['FRAME'] == frame + 1][['POSITION_Y', 'POSITION_X']].iloc[0].values / pixel_scale

    # Compute displacement vector
    displacement = position_i1 - position_i
    #print(displacement)
    
    if np.all(displacement == 0):
        return None  # No movement detected

    # Compute vectors from position[i] to poles
    vector_pole_1 = np.array(pole_1) - position_i
    vector_pole_2 = np.array(pole_2) - position_i

    # Compute dot products
    dot_product_1 = np.dot(displacement, vector_pole_1)
    dot_product_2 = np.dot(displacement, vector_pole_2)
    
    # Determine the leading pole based on the dot product
    if dot_product_1 > 0:
        return pole_1
    elif dot_product_2 > 0:
        return pole_2
    else:
        return None  # No leading pole identified


def find_neighbours(spots_dataframe, segmented_images, pixel_scale):
    """
    Finds the neighbors of each bacterium based on the leading pole and stores the neighbor labels in a column.
    """
    # Get image dimensions
    num_frames, height, width = segmented_images.shape

    # Initialize new columns
    pd.options.mode.chained_assignment = None  # Disable the warning
    
    spots_dataframe['SPOT_LABEL'] = pd.Series(dtype='int')
    spots_dataframe['NEIGHBOUR'] = pd.Series(dtype='object')
    spots_dataframe['NEIGHBOUR_UNCERTAIN'] = pd.Series(dtype='object')
    

    # Vectorized mapping of SPOT_LABEL
    positions = spots_dataframe[['POSITION_X', 'POSITION_Y', 'FRAME']].values
    scaled_positions = (positions[:, :2] / pixel_scale).astype(int)
    frames = positions[:, 2].astype(int)

    labels = [
        segmented_images[frame, y, x] if 0 <= frame < num_frames and 0 <= y < height and 0 <= x < width else 0
        for (x, y), frame in zip(scaled_positions, frames)
    ]
    spots_dataframe['SPOT_LABEL'] = labels

    # Step 2: Find leading pole and neighbors
    for index, row in spots_dataframe.iterrows():
        spot_label = row['SPOT_LABEL']
        frame = int(row['FRAME'])

        # Ensure SPOT_LABEL is valid
        if pd.notna(spot_label) and spot_label != 0 and 0 <= frame < num_frames:
            # Find leading pole coordinates
            leading_pole = find_leading_pole(spots_dataframe, segmented_images, spot_label, frame, pixel_scale)
            
            if leading_pole is None:
                spots_dataframe.at[index, 'NEIGHBOUR'] = []
                spots_dataframe.at[index, 'NEIGHBOUR_UNCERTAIN'] = []
                continue  # Skip if leading pole is not found

            leading_pole_y, leading_pole_x = leading_pole

            # Create a circular mask around the leading pole
            y_indices, x_indices = np.ogrid[:height, :width]
            mask = (x_indices - leading_pole_x)**2 + (y_indices - leading_pole_y)**2 <= 3**2

            # Find labels within the circle (excluding the current SPOT_LABEL)
            neighboring_labels = segmented_images[frame][mask]
            neighboring_labels_updated = segmented_images[frame][mask]

            if np.count_nonzero(neighboring_labels_updated == 0)<5:
                neighboring_labels_updated = [] # meaning it is a sister cell
            else:
                neighboring_labels_updated = list(set(neighboring_labels_updated) - {0, spot_label})  # Exclude background and self
                
            neighboring_labels = list(set(neighboring_labels) - {0, spot_label})  # Exclude background and self

            # Save neighbors
            spots_dataframe.at[index, 'NEIGHBOUR'] = neighboring_labels
            spots_dataframe.at[index, 'NEIGHBOUR_UNCERTAIN'] = neighboring_labels_updated
            

    return spots_dataframe



def plot_probability_of_contact_reversals(directory):
    """
    Loads CSV files from a directory, extracts strain names and surface coverage from filenames,
    and plots the Probability of Contact Reversals as a scatter plot with specified colors and transparency.
    """
    # List to store data
    data_probability = []
    data_frequency_CR = []
    data_frequency_contact = []

    # Define strain colors
    strain_colors = {
        '177': 'magenta',
        '459': 'darkorange',
        '337': 'darksalmon',
        '232': 'deepskyblue',
        '436': 'mediumblue'
    }
    
    # Define surface coverage transparency levels
    sc_transparency = {
        'SC10pc': 1,
        'SC30pc': 1,
        'SC70pc': 1
    }

    # Load all CSV files from the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)

            # Extract strain name from filename
            #strain_match = re.search(r'_(177|232|459|337|436)', filename)
            strain_match = re.search(r'_(177|232|459)', filename)
            if strain_match:
                strain = strain_match.group(1)
                
                # Extract surface coverage from filename
                sc_match = re.search(r'_(SC10pc|SC30pc|SC70pc)', filename)
                surface_coverage = sc_match.group(1) if sc_match else 'Unknown'
                transparency = sc_transparency.get(surface_coverage, 1.0)  # Default to 1.0 if not found
                
                # Load the CSV file
                df = pd.read_csv(filepath)

                # Check if the required column exists
                if 'Probability of contact reversals (%)' in df.columns:
                    for value in df['Probability of contact reversals (%)']:
                        data_probability.append({'Strain': strain, 'Probability': value, 'Alpha': transparency, 'Surface Coverage': surface_coverage})

                if 'Contact reversals frequency (/h)' in df.columns:
                    for value in df['Contact reversals frequency (/h)']:
                        data_frequency_CR.append({'Strain': strain, 'Frequency': value, 'Alpha': transparency, 'Surface Coverage': surface_coverage})
                                        
                if 'Number of contacts' in df.columns:
                    for value in df['Contacts frequency (/h)']:
                        data_frequency_contact.append({'Strain': strain, 'Frequency': value, 'Alpha': transparency, 'Surface Coverage': surface_coverage})

    # Convert data to DataFrame
    plot_data_probability = pd.DataFrame(data_probability)
    plot_data_frequency_CR = pd.DataFrame(data_frequency_CR)
    plot_data_frequency_contact = pd.DataFrame(data_frequency_contact)

    # Plot Probability of Contact Reversals
    plt.figure(figsize=(2.5, 3.5))
    for strain in strain_colors.keys():
        subset = plot_data_probability[plot_data_probability['Strain'] == strain]
        if not subset.empty:
            plt.scatter(subset['Strain'], subset['Probability'], alpha=subset['Alpha'].values, edgecolor = 'black',
                        color=strain_colors[strain], s=70)
    
    plt.xlabel('Strain', fontsize=16)
    plt.ylabel('Probability of CR (%)', fontsize=18, fontweight='bold', fontname='Arial')
    plt.ylim(0, 27)
    plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['232', '177', '459', '337', '436'], fontsize=16)
    plt.gca().tick_params(axis='both', which='major', length=5, width=1, color='black')
    plt.yticks(fontsize=16)
    plt.gca().margins(x=0.1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.25)
    plt.gca().spines['bottom'].set_linewidth(1.25)
    plt.tight_layout()
    plt.savefig(directory + '/summary_CR_probability.png', dpi=300, bbox_inches='tight', transparent = True)
    plt.show()
    plt.close()

    # # Plot Frequency of Contact Reversals
    # plt.figure(figsize=(2.5, 3.5))
    # for strain in strain_colors.keys():
    #     subset = plot_data_frequency_CR[plot_data_frequency_CR['Strain'] == strain]
    #     if not subset.empty:
    #         plt.scatter(subset['Strain'], subset['Frequency'], alpha=subset['Alpha'].values, 
    #                     color=strain_colors[strain], s=70)
    
    # plt.ylabel('Frequency of CR (/h)', fontsize=18, fontweight='bold', fontname='Arial')
    # plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['232', '177', '459', '337', '436'], fontsize=16)
    # plt.yticks(fontsize=14)
    # plt.gca().margins(x=0.3)
    # plt.tight_layout()
    # plt.savefig(directory + '/summary_CR_frequency.png', dpi=300, bbox_inches='tight', transparent = True)
    # plt.show()
    # plt.close()
    
    # # Plot Frequency of Contacts
    # plt.figure(figsize=(2.5, 3.5))
    # print(plot_data_frequency_contact.head(5))
    # for strain in strain_colors.keys():
    #     subset = plot_data_frequency_contact[plot_data_frequency_contact['Strain'] == strain]
    #     if not subset.empty:
    #         plt.scatter(subset['Strain'], subset['Frequency'], alpha=subset['Alpha'].values, 
    #                     color=strain_colors[strain], s=70)
    
    # plt.ylabel('Frequency of Contact (/h)', fontsize=18, fontweight='bold', fontname='Arial')
    # plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['232', '177', '459', '337', '436'], fontsize=16)
    # plt.yticks(fontsize=14)
    # plt.gca().margins(x=0.3)
    # plt.tight_layout()
    # plt.savefig(directory + '/summary_Contact_frequency.png', dpi=300, bbox_inches='tight', transparent = True)
    # plt.show()
    # plt.close()


def extract_surface_coverage(filename):
    if "SC10pc" in filename:
        return 10
    elif "SC30pc" in filename:
        return 30
    elif "SC70pc" in filename:
        return 70
    else:
        return None  

def extract_strain(filename):
    strains = ['177', '232', '459', '1047', '1695', '337', '436']
    for strain in strains:
        if strain in filename:
            return strain
    return None 

def replace_legend_label(label):
    strain_labels = {
        '177': 'WT',
        '232': r'$\Delta\mathit{pilH}$',
        '459': r'$\Delta\mathit{pilG}$',
        '1047': r'$\Delta\mathit{pilA}$',
        '1695': r'$\Delta\mathit{pilA cpdA}$',
        '337': r'$\Delta\mathit{cpdA}$',
        '436': r'$\Delta\mathit{pilH cyaB}$',
    }
    return strain_labels.get(label, label)  # Return the mapped label, or the original if not found


def plot_probability_function_of_SC(directory):
    """
    Reads all CSV files in the given directory, combines them into a single dataframe,
    and returns the combined dataframe.
    
    Parameters:
        directory (str): Path to the directory containing the CSV files.
    
    Returns:
        pd.DataFrame: Combined dataframe containing all rows from the CSV files.
    """
    
    ### Create a combined dataframe
    all_data = []
    
    # Iterate over all files in the directory
    for file in os.listdir(directory):
        if file.endswith(".csv"):  # Only process CSV files
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            
            # Manually add filename-based metadata
            df['SC'] = extract_surface_coverage(file)  # Apply function to filename
            df['Strain'] = extract_strain(file)  # Apply function to filename
            
            all_data.append(df)
    
    # Combine all data into a single dataframe
    combined_df = pd.concat(all_data, ignore_index=True)
    
    ## Assign colormap
    color_map = {'177': 'magenta', '232': 'deepskyblue', '459': 'darkorange', '337': 'darksalmon', '436': 'mediumblue'}

    plt.figure(figsize=(4,4))

    for strain in ['232', '177', '459', '337', '436']:
        # Filter the data for the current strain
        strain_data = combined_df[combined_df['Strain'] == strain]
        
        # Update the label before adding to legend
        legend_label = replace_legend_label(strain)
    
        # Plot individual points with transparency
        plt.scatter(strain_data['SC'], strain_data['Probability of contact reversals (%)'], s = 90,
                color=color_map[strain], edgecolor = 'black', alpha=1, label=legend_label)
    
        # Group by binned surface coverage and calculate the mean
        #grouped = strain_data.groupby('SC_bin')['Probability of contact reversals (%)'].mean()

        # Plot the averages with a red contour
        #plt.scatter(bin_labels, grouped.values, color=color_map[strain], s=190, alpha=1, 
                #edgecolors='black', linewidths=1.5, label=legend_label)
    
    # Add legend and labels
    plt.xlabel('Surface coverage (%)', fontsize=22, fontweight='bold', fontname='Arial')
    plt.ylabel('Probability of CR (%)', fontsize=22, fontweight='bold', fontname='Arial')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xticks(ticks=[10, 30, 70]) 
    plt.xlim([0, 85])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.25)
    plt.gca().spines['bottom'].set_linewidth(1.25)
    plt.gca().yaxis.set_minor_locator(plt.FixedLocator([5, 15, 25]))
    plt.xticks(ticks=[10, 30, 70]) 
    plt.gca().tick_params(axis='both', which='major', length=7, width=1.25, color='black')
    plt.gca().tick_params(axis='both', which='minor', length=4, width=1.25, color='black')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    
    # Save the figure
    plt.savefig(directory + '/probability_of_CR_function_of_SC.png', dpi=300, bbox_inches='tight', transparent = True)
    plt.show()

    
def plot_frequency_function_of_SC(directory):
    """
    Reads all CSV files in the given directory, combines them into a single dataframe,
    and returns the combined dataframe.
    
    Parameters:
        directory (str): Path to the directory containing the CSV files.
    
    Returns:
        pd.DataFrame: Combined dataframe containing all rows from the CSV files.
    """
    
    ### Create a combined dataframe
    all_data = []
    
    # Iterate over all files in the directory
    for file in os.listdir(directory):
        if file.endswith(".csv"):  # Only process CSV files
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            print(file_path)
            
            # Manually add filename-based metadata
            df['SC'] = extract_surface_coverage(file)  # Apply function to filename
            df['Strain'] = extract_strain(file)  # Apply function to filename
            
            all_data.append(df)
    
    # Combine all data into a single dataframe
    combined_df = pd.concat(all_data, ignore_index=True)
    
    ## Assign colormap
    color_map = {'177': 'magenta', '232': 'deepskyblue', '459': 'darkorange', '337': 'darksalmon', '436': 'mediumblue'}

    plt.figure(figsize=(4,4))
    zorders = {'232': 1, '459': 2, '177': 3, '337': 4, '436': 5}
    alphas = {'232': 1, '459': 1, '177': 0.6, '337': 1, '436': 1}

    for strain in ['232', '177', '459']: 

        # Filter the data for the current strain
        strain_data = combined_df[combined_df['Strain'] == strain]
        
        # Update the label before adding to legend
        legend_label = replace_legend_label(strain)
    
        # Plot individual points with transparency
        plt.scatter(strain_data['SC'], strain_data['Contact reversals frequency (/h)'], s = 90,
                color=color_map[strain], edgecolor = 'black', alpha=alphas[strain], label=legend_label, zorder=zorders[strain])
    
        # Group by binned surface coverage and calculate the mean
        #grouped = strain_data.groupby('SC_bin')['Probability of contact reversals (%)'].mean()

        # Plot the averages with a red contour
        #plt.scatter(bin_labels, grouped.values, color=color_map[strain], s=190, alpha=1, 
                #edgecolors='black', linewidths=1.5, label=legend_label)
    
    # Add legend and labels
    plt.xlabel('Surface coverage (%)', fontsize=18, fontweight='bold', fontname='Arial')
    plt.ylabel('Frequency of CR (/h)', fontsize=18, fontweight='bold', fontname='Arial')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xlim([0, 90])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.25)
    plt.gca().spines['bottom'].set_linewidth(1.25)
    plt.xticks(ticks=[10, 30, 70]) 
    plt.yticks(ticks=[0, 20, 40, 60, 80])
    plt.gca().tick_params(axis='both', which='major', length=7, width=1.25, color='black')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    
    # Save the figure
    plt.savefig(directory + '/frequency_of_CR_function_of_SC.png', dpi=300, bbox_inches='tight', transparent = True)
    plt.show()   
    
    plt.figure(figsize=(4,4))

    for strain in ['177', '232', '459']: 
        # Filter the data for the current strain
        strain_data = combined_df[combined_df['Strain'] == strain]
        
        # Update the label before adding to legend
        legend_label = replace_legend_label(strain)
    
        # Plot individual points with transparency
        plt.scatter(strain_data['SC'], strain_data['Contacts frequency (/h)'], s = 90,
                color=color_map[strain], edgecolor = 'black', alpha=1, label=legend_label)
    
        # Group by binned surface coverage and calculate the mean
        #grouped = strain_data.groupby('SC_bin')['Probability of contact reversals (%)'].mean()

        # Plot the averages with a red contour
        #plt.scatter(bin_labels, grouped.values, color=color_map[strain], s=190, alpha=1, 
                #edgecolors='black', linewidths=1.5, label=legend_label)
    
    # Add legend and labels
    #plt.legend(loc='upper left', fontsize=16) #bbox_to_anchor=(1.05, 1), 
    plt.xlabel('Surface coverage (%)', fontsize=18, fontweight='bold', fontname='Arial')
    plt.ylabel('Frequency of Contact (/h)', fontsize=18, fontweight='bold', fontname='Arial')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xlim([0, 85])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.25)
    plt.gca().spines['bottom'].set_linewidth(1.25)
    plt.xticks(ticks=[10, 30, 70]) 
    plt.gca().tick_params(axis='both', which='major', length=7, width=1.25, color='black')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    
    # Save the figure
    plt.savefig(directory + '/frequency_of_contact_function_of_SC.png', dpi=300, bbox_inches='tight', transparent = True)
    plt.show()
    
    plt.figure(figsize=(4,4))
    zorders = {'232': 1, '459': 2, '177': 3, '337': 4, '436': 5}
    alphas = {'232': 1, '459': 1, '177': 1, '337': 1, '436': 1}

    for strain in ['232', '177', '459']: 
        # Filter the data for the current strain
        strain_data = combined_df[combined_df['Strain'] == strain]
        #print(strain_data.head(5))
        
        # Update the label before adding to legend
        legend_label = replace_legend_label(strain)
    
        # Plot individual points with transparency
        plt.scatter(strain_data['SC'], strain_data['Reversals frequency (/h)'], s = 90, color=color_map[strain], edgecolor = 'black', alpha=alphas[strain], label=legend_label, zorder = zorders[strain])
    
        # Group by binned surface coverage and calculate the mean
        #grouped = strain_data.groupby('SC_bin')['Probability of contact reversals (%)'].mean()

        # Plot the averages with a red contour
        #plt.scatter(bin_labels, grouped.values, color=color_map[strain], s=190, alpha=1, 
                #edgecolors='black', linewidths=1.5, label=legend_label)
    
    # Add legend and labels
    plt.xlabel('Surface coverage (%)', fontsize=18, fontweight='bold', fontname='Arial')
    plt.ylabel(r'Frequency of Reversals (/h)', fontsize=18, fontweight='bold', fontname='Arial')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xlim([0, 85])
    plt.ylim([0, 100])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.25)
    plt.gca().spines['bottom'].set_linewidth(1.25)
    plt.xticks(ticks=[10, 30, 70]) 
    plt.gca().tick_params(axis='both', which='major', length=7, width=1.25, color='black')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    
    # Save the figure
    plt.savefig(directory + '/frequency_of_reversals_function_of_SC.png', dpi=300, bbox_inches='tight', transparent = True)
    plt.show()
    