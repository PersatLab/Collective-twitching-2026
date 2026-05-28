"""
Utilities for 02_SegmentationAnalysis.ipynb
"""

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator
import matplotlib.patches as mpatches
from skimage.measure import label, regionprops
from scipy.stats import zscore
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from scipy import ndimage
from skimage.segmentation import find_boundaries
import pandas as pd
import cv2
import os
from skimage import io
import re
from skimage.draw import line
import math
import warnings
import seaborn as sns
import textwrap
from tqdm import tqdm
from scipy.optimize import curve_fit
from .helper import analysis_helper as ah
from PIL import Image
import time
from skimage.io import imsave
import tifffile as tiff



## Definitions:
# mask_frame: a 2D array representing a mask, where the background has a value of 0 and each cell has a different gray value.
# cell_extremities_length: a dictionary where the keys are the unique cell IDs and the values are lists containing two tuples. Each tuple represents the (y, x) coordinates of a length extremity of the corresponding cell.

##Information Retrieval Functions:


def get_cellIDs(mask_frame):
    """
    Find the gray values (ID) of each cell in a single mask.

    Parameters
    ----------
    mask_frame : numpy array
        A 2D array representing the mask, where the background has a value of 0 and each cell has a different gray value.

    Returns
    -------
    numpy array
        An array of the gray values of each cell.
    """
    return np.unique(mask_frame)[1:].astype(int).tolist()  # Exclude the background value of 0



def cell_coverage(mask_frame):
    """
    Calculate the cell coverage of a mask.

    Cell coverage is defined as the proportion of the frame that is covered by cells.

    Parameters
    ----------
    mask : numpy array
        A 2D array representing a mask, where the background has a value of 0 and each cell has a different gray value.

    Returns
    -------
    float
        The cell coverage of the mask.
    """
    return np.sum(mask_frame > 0) / np.size(mask_frame)

def find_extremities_of_cells_length (mask_frame):
    """
    Calculate the farthest extremities of each distinct cell in a mask, these are the long extremities.

    This function identifies each unique cell in the mask by its unique ID.
    For each cell, it determines the coordinates of all points within the cell and computes the pairwise distances between these points. 
    The two points that are furthest apart are considered the extremities of the cell.

    Parameters
    ----------
    mask_frame : numpy.ndarray
        A 2D array representing the mask, where the background is represented by 0 and each distinct cell is represented by a unique gray value.

    Returns
    -------
    dict
        A dictionary where the keys are the unique cell IDs and the values are lists containing two tuples. 
        Each tuple represents the (y, x) coordinates of an extremity of the corresponding cell.
    """
    cell_extremities_length = {}
    for ID in get_cellIDs(mask_frame):
        mask_of_1_cell = mask_frame == ID
        points = np.transpose(np.where(mask_of_1_cell))
        coords = [tuple(point) for point in points]
        distances = cdist(coords, coords)
        i, j = np.unravel_index(np.argmax(distances), distances.shape)
        cell_extremities_length[ID] = [coords[i], coords[j]]
    return cell_extremities_length



def import_masks_movies_outlines(dir_segments, dir_movies, dir_outlines):
    """
    Import all segmented TIF videos, corresponding raw movies, and their outlines from directories.

    This function reads all non-hidden TIF files from the specified directories. It will only load files from
    dir_segments that end with _segmented.tif, and will attempt to load corresponding raw movies
    and outlines based on matching filenames.

    Parameters
    ----------
    dir_segments : str
        The path to the directory from which to import segmented movies (masks).
    dir_movies : str
        The path to the directory from which to import corresponding raw movies.
    dir_outlines : str
        The path to the directory from which to import corresponding outlines.

    Returns
    -------
    all_masks, all_movies, all_outlines : dict
        Dictionaries where the keys are filenames and the values are lists of numpy arrays representing the frames of the videos/outlines.

    Raises
    ------
    FileNotFoundError
        If any specified directory does not exist, or if expected files are missing in the directories.
    ValueError
        If the mask movie, corresponding movie, or outlines do not have the same number of frames.
    """
    if not os.path.exists(dir_segments) or not os.path.exists(dir_movies) or not os.path.exists(dir_outlines):
        raise FileNotFoundError("One or more of the specified directories do not exist.")
    
    all_masks = {}
    all_movies = {}
    all_outlines = {}

    for file in os.listdir(dir_segments):
        if not file.startswith('.') and file.endswith('_segmented.tif'):
            mask_path = os.path.join(dir_segments, file)
            print(f"Loading segmented masks: {mask_path}")
            frames_masks = io.imread(mask_path)
            
            # Reshape masks if necessary
            if frames_masks.ndim == 3 and 'frames_' in file:
                frames_masks = np.transpose(frames_masks, (2, 0, 1))
            elif frames_masks.ndim == 2:
                frames_masks = np.expand_dims(frames_masks, axis=0)
                
            # Derive original movie and outline filenames
            base_name = re.sub(r'_frames_.*', '', file.replace('_segmented.tif', ''))
            movie_name = base_name + '.tif'
            outline_name = file.replace('_segmented.tif', '_segmented_outlines.tif')

            # Extract frame indices directly from the filename
            frames_str = re.search(r'_frames_([\d-]+(?:_[\d-]+)*)', file)
            if frames_str:
                frames_indices = []
                ranges = frames_str.group(1).split('_')
                for part in ranges:
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        frames_indices.extend(range(start, end + 1))
                    elif part.isdigit():
                        frames_indices.append(int(part))
                frame_indices_str = frames_str.group(1)
                key_name = f"{base_name}_frames_{frame_indices_str}"
            else:
                frames_indices = []
                key_name = base_name

            # Remove trailing underscore if present
            key_name = key_name.rstrip('_')

            # Load the corresponding raw movie
            raw_movie_path = os.path.join(dir_movies, movie_name)
            if not os.path.exists(raw_movie_path):
                # Attempt to find the parent name by removing frame indices
                parent_name = re.sub(r'_frames_.*', '', base_name)
                raw_movie_path = os.path.join(dir_movies, parent_name + '.tif')
                if not os.path.exists(raw_movie_path):
                    raise FileNotFoundError(f"Corresponding raw movie {parent_name}.tif does not exist in {dir_movies}")
            print(f"Loading raw movie: {raw_movie_path}")
            frames_movies = io.imread(raw_movie_path)

            if frames_movies.ndim == 2:
                frames_movies = np.expand_dims(frames_movies, axis=0)

            if frames_indices:
                # Ensure the frame indices are within the valid range for raw movies
                max_index = len(frames_movies) - 1
                frames_indices = [i for i in frames_indices if i <= max_index]
                if not frames_indices:
                    raise ValueError(f"No valid frame indices found for {file} within the range of the raw movie {movie_name}")
                frames_movies = frames_movies[frames_indices]
    
            # Load the corresponding outlines
            outlines_path = os.path.join(dir_outlines, outline_name)
            if not os.path.exists(outlines_path):
                raise FileNotFoundError(f"Corresponding outlines {outline_name} do not exist in {dir_outlines}")
            print(f"Loading outlines: {outlines_path}")
            frames_outlines = io.imread(outlines_path)
            if frames_outlines.ndim == 3:
                frames_outlines = np.expand_dims(frames_outlines, axis=0)

            # Validation for the same number of frames
            if not (len(frames_masks) == len(frames_movies) == len(frames_outlines)):
                print(f"Mismatch in number of frames for {file}:")
                print(f"Segmented masks: {len(frames_masks)} frames")
                print(f"Raw movies: {len(frames_movies)} frames")
                print(f"Outlines: {len(frames_outlines)} frames")
                raise ValueError(f"Files must have the same number of frames: {file}")
  
            all_masks[key_name] = frames_masks
            all_movies[key_name] = frames_movies
            all_outlines[key_name] = frames_outlines

    print(f"{len(all_masks)} masks, corresponding original movies, and outlines loaded.")
    return all_masks, all_movies, all_outlines
    

#OLD FUNCTION
def find_short_direction(mask_frame):
    """
    Calculate the short direction of each distinct cell in a mask.

    This function identifies each unique cell in the mask by its unique ID.
    For each cell, it determines the extremities and calculates the long vector between them.
    It then calculates a perpendicular vector to the long vector, which represents the short direction.
    The function then finds all points along this short direction that are within the cell.

    Parameters
    ----------
    mask_frame : numpy.ndarray
        A 2D array representing the mask, where the background is represented by 0 and each distinct cell is represented by a unique gray value.

    Returns
    -------
    dict
        A dictionary where the keys are the unique cell IDs and the values are lists of tuples. 
        Each tuple represents the (y, x) coordinates of a point along the short direction within the corresponding cell.
    """
    extremities = find_extremities_of_cells_length(mask_frame)
    vectorOnCell = {}
    for ID in get_cellIDs(mask_frame):
        long_vector = np.array(extremities[ID][0]) - np.array(extremities[ID][1])
        perpendicular_vector = np.array([-long_vector[1], long_vector[0]])
        pointsOfCell = ah.find_points_of_cell(mask_frame, ID)
        center_point = find_center_of_1cell(mask_frame, ID)
        short_vector_pixels = np.array(np.transpose(line(center_point[0] - perpendicular_vector[0], center_point[1] - perpendicular_vector[1], center_point[0] + perpendicular_vector[0], center_point[1] + perpendicular_vector[1])))
        #remove pixels that are not on the cell
        vectorOnCell[ID] = np.array([point for point in short_vector_pixels if ((pointsOfCell == point).all(axis=1).any())])
    return vectorOnCell

def find_extremities_of_cells_width(mask_frame, cell_extremities_length = None):
    """
    Calculate the short extremities of each distinct cell in a mask.

    This function requires the long extremities of the cells, 
    it's recommended that they are calculated earlier using "find_extremities_of_cells_length" and passed as an argument to avoid calculating them twice.

    This function identifies each unique cell in the mask by its unique ID.
    For each cell, it uses the long vector between the length extremities and calculates a perpendicular vector.
    It then finds all points along this perpendicular vector that are within the cell.
    The two points that are furthest apart along this vector are considered the short extremities of the cell.

    Parameters
    ----------
    mask_frame : numpy.ndarray
        A 2D array representing the mask, where the background is represented by 0 and each distinct cell is represented by a unique gray value.
    cell_extremities_length : dict
        A dictionary where the keys are the unique cell IDs and the values are lists containing two tuples. 
        Each tuple represents the (y, x) coordinates of a length extremity of the corresponding cell.

    Returns
    -------
    dict
        A dictionary where the keys are the unique cell IDs and the values are lists containing two tuples. 
        Each tuple represents the (y, x) coordinates of a width extremity of the corresponding cell.
    """

    if not isinstance(mask_frame, np.ndarray) or len(mask_frame.shape) != 2:
        raise ValueError("mask_frame must be a 2D numpy array")
    
    try: 
        if cell_extremities_length == None:
            raise ValueError("cell_extremities_length was not provided")
    except ValueError as e:
        print(f"{e}, it'll be calculated. It's recommended to provide this parameter (can be calculated with find_extremities_of_cells_length).")
        cell_extremities_length = find_extremities_of_cells_length(mask_frame)

    cell_extremities_width = {}

    for ID in get_cellIDs(mask_frame):
        long_vector = np.array(cell_extremities_length[ID][0]) - np.array(cell_extremities_length[ID][1])
        perpendicular_vector = np.array([-long_vector[1], long_vector[0]])
        pointsOfCell = ah.find_points_of_cell(mask_frame, ID)
        center_point = find_center_of_1cell(mask_frame, ID)
        short_vector_pixels = np.array(np.transpose(line(center_point[0] - perpendicular_vector[0], center_point[1] - perpendicular_vector[1], center_point[0] + perpendicular_vector[0], center_point[1] + perpendicular_vector[1])))
        #remove pixels that are not on the cell
        points_of_width = np.array([point for point in short_vector_pixels if ((pointsOfCell == point).all(axis=1).any())])

        coords = [tuple(point) for point in points_of_width]
        distances = cdist(coords, coords)
        i, j = np.unravel_index(np.argmax(distances), distances.shape)
        cell_extremities_width[ID] = [coords[i], coords[j]]
    return cell_extremities_width



def find_center_of_1cell(mask_frame, cellID):
    """
    Find the center of a specific cell in a mask.

    This function identifies the points in the mask that belong to the cell specified by the cellID.
    It then calculates the mean of these points, which represents the center of the cell.
    The coordinates of the center are returned as a 1D numpy array of integers.

    Parameters
    ----------
    mask_frame : numpy.ndarray
        A 2D array representing the mask, where the background is represented by 0 and each distinct cell is represented by a unique gray value.
    cellID : int
        The unique ID of the cell whose center is to be found.

    Returns
    -------
    numpy.ndarray
        A 1D array representing the (y, x) coordinates of the center of the cell.
    """
    if not isinstance(mask_frame, np.ndarray) or len(mask_frame.shape) != 2:
        raise ValueError("mask must be a 2D numpy array")
    if not isinstance(cellID, int) or cellID <= 0:
        raise ValueError("cellID must be non-zero positive integer")
    
    pointsOfCell = ah.find_points_of_cell(mask_frame, cellID)
    if pointsOfCell.size == 0:
        raise ValueError(f"No points found for cellID: {cellID}")
    
    return np.mean(pointsOfCell, axis=0).astype(int)

def find_cell_centers(mask_frame):
    """
    Find the centers of all cells in a mask.

    This function identifies each unique cell in the mask by its unique ID.
    For each cell, it calculates the center and stores it in a dictionary.
    The dictionary, where the keys are the cell IDs and the values are the centers of the cells, is returned.

    Parameters
    ----------
    mask_frame : numpy.ndarray
        A 2D array representing the mask, where the background is represented by 0 and each distinct cell is represented by a unique gray value.

    Returns
    -------
    dict
        A dictionary where the keys are the unique cell IDs and the values are 1D arrays representing the (y, x) coordinates of the centers of the cells.
    """

    if not isinstance(mask_frame, np.ndarray) or len(mask_frame.shape) != 2:
        raise ValueError("mask must be a 2D numpy array")

    IDs = get_cellIDs(mask_frame)

    centers = {}
    for ID in IDs:
        centers[ID] = find_center_of_1cell(mask_frame, ID)
    return centers


##
# SORTING FUNCTIONS
# For removing artifacts
##

#list all the bacteria values under the standard deviation
def list_bacteria_values_under_standard_deviation(images, standard_deviation):
    for i in images:
        surfaces = calculate_bacteria_surface_areas(images[i][0])
        
        average_surface_area = np.mean(list(surfaces.values()))        
        for ID in surfaces:
            if surfaces[ID] < standard_deviation:
                print(f"Cell {ID} has a surface area of {surfaces[ID]} square pixels, which is less than the standard deviation of {average_surface_area - standard_deviation} square pixels.")

#plot_z_values_vs_cell_area(images)
def identify_artifacts(mask, z_threshold):
    """
    Identify potential artifacts in a mask based on their size.

    Parameters
    ----------
    mask : numpy array
        A 2D array representing the mask, where the background has a value of 0 and each cell has a different gray value.
    z_threshold : float
        The Z-score threshold for considering a cell an artifact.

    Returns
    -------
    list
        A list of the gray values of the potential artifacts.
    """
    labeled_mask = label(mask)
    properties = regionprops(labeled_mask)
    areas = np.array([prop.area for prop in properties])
    z_scores = zscore(areas)
    artifact_values = [prop.label for prop, z_score in zip(properties, z_scores) if z_score < z_threshold]
    return artifact_values

#remove artifacts
def remove_artifacts(mask, artifact_values):
    """
    Remove artifacts from a mask.

    Parameters
    ----------
    mask : numpy array
        A 2D array representing the mask, where the background has a value of 0 and each cell has a different gray value.
    artifact_values : list
        A list of the gray values of the potential artifacts.

    Returns
    -------
    numpy array
        A 2D array representing the mask, where the background has a value of 0 and each cell has a different gray value.
    """
    for value in artifact_values:
        mask[mask == value] = 0
    return mask

#sort and save images
def sort_and_save_images(images, outputdir):
    """
    Sort and save the images.

    Parameters
    ----------
    images : dict
        A dictionary of the images.
    outputdir : str
        The path to the output directory.

    Returns
    -------
    dict
        A dictionary of the sorted images.
    """
    sorted_images = {}
    for i in images:
        min_z_value = -1.8
        sorted_images[i] = remove_artifacts(images[i], identify_artifacts(images[i], min_z_value))
    omnu.save_mask(sorted_images, outputdir, i ,suffix='.tif')
    return sorted_images

#
##Analysis Functions
#

def calculate_bacteria_surface_areas(mask_frame):
    """
    Calculate the surface area of each bacterium in a given frame.

    This function identifies each unique bacterium in the specified frame by its unique ID. 
    For each bacterium, it calculates the surface area.

    Parameters
    ----------
    mask_frame : numpy.ndarray
        A 2D array representing a mask frame, where the background is represented by 0 and each distinct cell is represented by a unique gray value.

    Returns
    -------
    dict
        A dictionary where the keys are the unique bacterium IDs 
        and the values are the surface areas of the corresponding bacteria.
    """
    surfaces = {}
    for cellID in get_cellIDs(mask_frame):
        surfaces[cellID] = np.sum(mask_frame == cellID)
    return surfaces

def get_cell_lengths(mask_frame, cell_extremities = None):
    """
    Calculate the lengths of all cells in a mask.

    This function identifies the extremities of each cell in the mask and calculates the Euclidean distance between them, which represents the length of the cell.
    The lengths are stored in a dictionary, where the keys are the cell IDs and the values are the lengths.

    Parameters
    ----------
    mask_frame : numpy.ndarray
        A 2D array representing the mask, where the background is represented by 0 and each distinct cell is represented by a unique gray value.

    Returns
    -------
    dict
        A dictionary where the keys are the unique cell IDs and the values are the lengths of the cells.

    Raises
    ------
    ValueError
        If mask_frame is not a 2D numpy array.
    """
    if not isinstance(mask_frame, np.ndarray) or len(mask_frame.shape) != 2:
        raise ValueError("mask_frame must be a 2D numpy array")

    try: 
        if cell_extremities == None:
            raise ValueError("cell_extremities was not provided")
    except ValueError as e:
        print(f"{e}, it'll be calculated. It's recommended to provide this parameter (can be calculated with find_extremities_of_cells_length).")
        cell_extremities = find_extremities_of_cells_length(mask_frame)

    lengths = {}

    for ID in cell_extremities:
        try:
            lengths[ID] = np.linalg.norm(np.array(cell_extremities[ID][0]) - np.array(cell_extremities[ID][1]))
        except Exception as e:
            print(f"An error occurred while calculating the length of cell {ID}: {e}")
            continue

    return lengths

def get_cell_widths(mask_frame, cell_extremities_length = None, cell_extremities_width = None):
    """
    Calculate the widths of all cells in a mask.

    This function identifies the extremities of each cell in the mask and calculates the Euclidean distance between them, which represents the length of the cell.
    The lengths are stored in a dictionary, where the keys are the cell IDs and the values are the lengths.

    Parameters
    ----------
    mask_frame : numpy.ndarray
        A 2D array representing the mask, where the background is represented by 0 and each distinct cell is represented by a unique gray value.
    cell_extremities_length : dict, optional
        A dictionary where the keys are the unique cell IDs and the values are lists containing two tuples. 
        Each tuple represents the (y, x) coordinates of a length extremity of the corresponding cell. 
        If not provided, the extremities will be calculated using the `find_extremities_of_cells_length` function.
    cell_extremities_width : dict, optional
        A dictionary where the keys are the unique cell IDs and the values are lists containing two tuples. 
        Each tuple represents the (y, x) coordinates of a width extremity of the corresponding cell. 
        If not provided, the extremities will be calculated using the `find_extremities_of_cells_width` function.

    Returns
    -------
    dict
        A dictionary where the keys are the unique cell IDs and the values are the widths of the cells.

    Raises
    ------
    ValueError
        If mask_frame is not a 2D numpy array.
    """
    if not isinstance(mask_frame, np.ndarray) or len(mask_frame.shape) != 2:
        raise ValueError("mask_frame must be a 2D numpy array")

    try: 
        if cell_extremities_length == None:
            raise ValueError("cell_extremities_length was not provided")
    except ValueError as e:
        print(f"{e}, it'll be calculated. It's recommended to provide this parameter (can be calculated with find_extremities_of_cells_length).")
        cell_extremities_length = find_extremities_of_cells_length(mask_frame)

    try: 
        if cell_extremities_width == None:
            raise ValueError("cell_extremities_width was not provided")
    except ValueError as e:
        print(f"{e}, it'll be calculated. It's recommended to provide this parameter (can be calculated with find_extremities_of_cells_width).")
        cell_extremities_width = find_extremities_of_cells_width(mask_frame, cell_extremities_length)
    
    widths = {}

    for ID in cell_extremities_width:
        try:
            widths[ID] = np.linalg.norm(np.array(cell_extremities_width[ID][0]) - np.array(cell_extremities_width[ID][1]))
        except Exception as e:
            print(f"An error occurred while calculating the width of cell {ID}: {e}")
            continue

    return widths

def get_directions_radians(mask_frame, cell_extremities = None):
    """
    Calculate the direction of each distinct cell in a mask.

    This function identifies each unique cell in the mask by its unique ID. 
    For each cell, it uses the `find_extremities_of_cells_length` function to determine the extremities of the cell if they are not provided as an argument. 
    It then calculates the direction of the cell from these extremities. 
    The direction is calculated as the angle between the vector from the first extremity to the second extremity and the x+ axis.

    Parameters
    ----------
    mask_frame : numpy.ndarray
        A 2D array representing a mask, where the background is represented by 0 and each distinct cell is represented by a unique gray value.
    cell_extremities : dict, optional
        A dictionary where the keys are the unique cell IDs and the values are lists containing two tuples. 
        Each tuple represents the (y, x) coordinates of an extremity of the corresponding cell. 
        If not provided, the extremities will be calculated using the `find_extremities_of_cells_length` function.

    Returns
    -------
    dict
        A dictionary where the keys are the unique cell IDs and the values are the directions of the corresponding cells in radians. 
        The direction is defined as the angle in radians between the vector from the first extremity to the second extremity of the cell and the x-axis.

    Raises
    ------
    ValueError
        If mask_frame is not a 2D numpy array.
    """
    if not isinstance(mask_frame, np.ndarray) or len(mask_frame.shape) != 2:
        raise ValueError("mask_frame must be a 2D numpy array")
    
    try: 
        if cell_extremities == None:
            raise ValueError("cell_extremities was not provided")
    except ValueError as e:
        print(f"{e}, it'll be calculated. It's recommended to provide this parameter (can be calculated with find_extremities_of_cells_length).")
        cell_extremities = find_extremities_of_cells_length(mask_frame)

    directions = {}

    for ID in cell_extremities:
        direction = np.array(cell_extremities[ID][1]) - np.array(cell_extremities[ID][0])
        angle = np.arctan2(direction[0], direction[1])
        directions[ID] = np.pi - angle

    return directions

def get_directions_degrees(mask_frame, cell_extremities=None):
    """
    Calculate the direction of each distinct cell in a mask in degrees from 0 to 360.

    This function leverages `get_directions_radians` to calculate the initial directions in radians,
    and then converts those angles to degrees, normalizing them to the range [0, 360).

    Parameters
    ----------
    mask_frame : numpy.ndarray
        A 2D array representing a mask, where the background is represented by 0 and each distinct cell is represented by a unique gray value.
    cell_extremities : dict, optional
        A dictionary where the keys are the unique cell IDs and the values are lists containing two tuples. 
        Each tuple represents the (y, x) coordinates of an extremity of the corresponding cell. 
        If not provided, the extremities will be calculated using the `find_extremities_of_cells_length` function.

    Returns
    -------
    dict
        A dictionary where the keys are the unique cell IDs and the values are the directions of the corresponding cells in degrees from 0 to 360.
    """
    # Get the directions in radians first by calling the previously defined function
    directions_radians = get_directions_radians(mask_frame, cell_extremities)

    # Convert radian values to degrees and normalize them to the range [0, 360)
    directions_degrees = {ID: (np.degrees(angle) % 360) for ID, angle in directions_radians.items()}

    return directions_degrees




def get_directions_unit_vector(mask_frame, cell_extremities = None):
    """
    Calculate the direction vector of each distinct cell in a mask.

    This function identifies each unique cell in the mask by its unique ID. 
    For each cell, it uses the `find_extremities_of_cells_length` function to determine the extremities of the cell if they are not provided as an argument. 
    It then calculates the direction of the cell from these extremities. 
    The direction is calculated as the unit vector from the first extremity to the second extremity.

    Parameters
    ----------
    mask_frame : numpy.ndarray
        A 2D array representing a mask, where the background is represented by 0 and each distinct cell is represented by a unique gray value.
    cell_extremities : dict, optional
        A dictionary where the keys are the unique cell IDs and the values are lists containing two tuples. 
        Each tuple represents the (y, x) coordinates of an extremity of the corresponding cell. 
        If not provided, the extremities will be calculated using the `find_extremities_of_cells_length` function.

    Returns
    -------
    dict
        A dictionary where the keys are the unique cell IDs and the values are the direction vectors of the corresponding cells. 
        The direction vector is defined as the unit vector from the first extremity to the second extremity of the cell.

    Raises
    ------
    ValueError
        If mask_frame is not a 2D numpy array or if cell_extremities was not provided.
    """
    if not isinstance(mask_frame, np.ndarray) or len(mask_frame.shape) != 2:
        raise ValueError("mask_frame must be a 2D numpy array")
    
    try: 
        if cell_extremities == None:
            raise ValueError("cell_extremities was not provided")
    except ValueError as e:
        print(f"{e}, it'll be calculated. It's recommended to provide this parameter (can be calculated with find_extremities_of_cells_length).")
        cell_extremities = find_extremities_of_cells_length(mask_frame)

    directions = {}

    for ID in cell_extremities:
        direction = (np.array(cell_extremities[ID][1]) - np.array(cell_extremities[ID][0]))/np.linalg.norm(np.array(cell_extremities[ID][1]) - np.array(cell_extremities[ID][0]))
        directions[ID] = direction

    return directions


#Hugo needs to buff up this function
def dot_product_local_order_parameters(mask_frame, r, return_n_close_cells = False):

    directions = get_directions_unit_vector(mask_frame, find_extremities_of_cells_length(mask_frame))
    
    centers = find_cell_centers(mask_frame)

    distance_matrix = cdist(list(centers.values()), list(centers.values()))
    
    IDs = get_cellIDs(mask_frame)
    order_parameters = {}
    n_Close_cells = {}

    for cell1 in enumerate(IDs):
        close_cells_directions = []
        for cell2 in enumerate(IDs):
            if cell1[1] != cell2[1] and distance_matrix[cell1[0]][cell2[0]] < r :
                close_cells_directions.append(directions[cell2[1]])
        if math.isnan(np.mean(close_cells_directions)):
            order_parameters[cell1[1]] = math.nan
        else:
            order_parameters[cell1[1]] = np.mean(ah.dot_products(directions[cell1[1]], np.array(close_cells_directions)))
        n_Close_cells[cell1[1]] = len(close_cells_directions)

    if return_n_close_cells:
        return order_parameters, n_Close_cells
    else:
        return order_parameters



def scalar_order_parameter(mask):
    """
    One frame of a mask video is used to calculate the nematic order parameter of the cells in the frame.
    the order parameter definition is based on the following formula:
    https://en.wikipedia.org/wiki/Liquid_crystal#Order_parameter
    """
    directions = np.array(list(get_directions_radians(mask).values()))
    avgDirection = np.mean(directions)
    s = (3*(np.cos((directions - avgDirection))**2)-1)/2
    return np.mean(s)

def cos_local_order_parameters(mask, r, return_n_close_cells = False):
    """
    Calculate the cosine local order parameter of the cells in a mask.
    
    One frame of a mask video is used to calculate the local nematic order parameter of the cells in the frame.
    the order parameter definition is based on the following formula:
    https://en.wikipedia.org/wiki/Liquid_crystal#Order_parameter

    As r becomes larger, the local order parameter becomes the scalar order parameter.
    """
    directions = get_directions_radians(mask, find_extremities_of_cells_length(mask))
    centers = find_cell_centers(mask)

    distance_matrix = cdist(list(centers.values()), list(centers.values()))
    
    IDs = get_cellIDs(mask)
    order_parameters = {}
    n_Close_cells = {}

    for cell1 in enumerate(IDs):
        close_cells_directions = []
        for cell2 in enumerate(IDs):
            if cell1[1] != cell2[1] and distance_matrix[cell1[0]][cell2[0]] < r :
                close_cells_directions.append(directions[cell2[1]])
        if math.isnan(np.mean(close_cells_directions)):
            order_parameters[cell1[1]] = math.nan
        else:
            order_parameters[cell1[1]] = ah.local_order_param_for1cell(directions[cell1[1]], close_cells_directions)
        n_Close_cells[cell1[1]] = len(close_cells_directions)

    if return_n_close_cells:
        return order_parameters, n_Close_cells
    else:
        return order_parameters


## Hugo needs to decide if to keep or not
def local_order_parameters_split_mask_into_quadrants(method, mask_quadrants, Radiuses, return_n_close_cells = False):
    ### method can be, dot_product_local_order_parameters or cos_local_order_parameters for the moment, it should be a _local_order_parameter function
    possible_methods = [function for function in all_functons_globals if function.endswith('cos_local_order_parameters')]
    
    if not isinstance(mask_quadrants, dict):
        raise ValueError("mask_quadrants must be a dictionary")
    
    if method.__name__ not in possible_methods:
        raise ValueError(f"The given method is {method.__name__} but it must be one of the following functions: {possible_methods}")
    
    dict_n_close_cells = {}
    for R in Radiuses:
        dict_n_close_cells[R] = 0
    
    significant_r = 0

    all_orders_np = np.zeros([len(mask_quadrants), len(Radiuses)])


    for quad_ID in enumerate(mask_quadrants):
        print(quad_ID)
        mask_quad = mask_quadrants[quad_ID[1]]

        dict_for_1Quad_n_close_cells = {}
        for R in enumerate(Radiuses):
            local_order_parameters_for_r, n_close_cells = method(mask_quad, int(R[1]), return_n_close_cells = True)
            dict_n_close_cells[R[1]] += np.mean(list(n_close_cells.values()))
            all_orders_np[quad_ID[0]][R[0]] = np.mean(list(local_order_parameters_for_r.values()))
            if any(np.array(list(local_order_parameters_for_r.values()))==0) and R[1] > significant_r:
                significant_r = R[1]
        
    #average the number of close cells for each R
    for R in dict_n_close_cells:
        dict_n_close_cells[R] = dict_n_close_cells[R]/len(mask_quadrants)

    #sum the orders of the differenct quadrants proportionnally to the number of cells in each quadrant
    number_cells_per_quad_list = []
    for quad_ID in mask_quadrants:
        frame = mask_quadrants[quad_ID]
        n_cells = len(get_cellIDs(frame))
        number_cells_per_quad_list.append(n_cells)
        print(quad_ID, '  ', n_cells)

    ratio_cells_list = number_cells_per_quad_list/np.sum(number_cells_per_quad_list)

    weighted_orders = np.reshape(np.array(ratio_cells_list), [len(ratio_cells_list), 1] ) * all_orders_np 

    sum_weighted_orders = np.sum(weighted_orders, axis=0)

    dict_weighted_orders = {}
    for R in enumerate(Radiuses):
        dict_weighted_orders[R[1]] = sum_weighted_orders[R[0]]
        
    return dict_weighted_orders, dict_n_close_cells, significant_r




#HUGO NEEDS TO CREATE SOME ERROR HANDLING HERE
def calculate_aspect_ratios(mask_frame, cell_extremities_length = None, cell_extremities_width = None):
    """
    Calculate the aspect ratios (length of cell/width of cell) of all cells in a mask frame.

    Args:
        mask_frame (numpy.ndarray): A 2D numpy array representing the mask frame, where the background is represented by 0 and each distinct cell is represented by a unique gray value.
        cell_extremities_length (dict, optional): A dictionary mapping cell IDs to their extremities' lengths. If not provided, it will be calculated.
        cell_extremities_width (dict, optional): A dictionary mapping cell IDs to their extremities' widths. If not provided, it will be calculated.

    Raises:
        ValueError: If mask_frame is not a 2D numpy array, or if cell_extremities_length or cell_extremities_width are not provided.

    Returns:
        dict: A dictionary mapping cell IDs to their aspect ratios.
    """
    if not isinstance(mask_frame, np.ndarray) or len(mask_frame.shape) != 2:
        raise ValueError("mask_frame must be a 2D numpy array")

    try: 
        if cell_extremities_length == None:
            raise ValueError("cell_extremities_length was not provided")
    except ValueError as e:
        print(f"{e}, it'll be calculated. It's recommended to provide this parameter (can be calculated with find_extremities_of_cells_length).")
        cell_extremities_length = find_extremities_of_cells_length(mask_frame)

    try: 
        if cell_extremities_width == None:
            raise ValueError("cell_extremities_width was not provided")
    except ValueError as e:
        print(f"{e}, it'll be calculated. It's recommended to provide this parameter (can be calculated with find_extremities_of_cells_width).")
        cell_extremities_width = find_extremities_of_cells_width(mask_frame, cell_extremities_length)
    
    aspectRatios = {}
    for ID in get_cellIDs(mask_frame):
        aspectRatios[ID] = np.linalg.norm(np.array(cell_extremities_length[ID][0]) - np.array(cell_extremities_length[ID][1])) / np.linalg.norm(np.array(cell_extremities_width[ID][0]) - np.array(cell_extremities_width[ID][1]))
    return aspectRatios

### This function only makes sense in the contexte of heterogenious images, as you'd only be comparing it to different densities within the same frame
def get_local_concentrations(mask_frame, radius):
    cell_centers = find_cell_centers(mask_frame)
    concentrations = {}
    for cell_ID in cell_centers:
        cell_center = cell_centers[cell_ID]
        # divide the number of pixels in the circle that are not equal to 0 by the total number of pixels in the circle
        # Create a circular mask
        h, w = mask_frame.shape
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - cell_center[0])**2 + (Y - cell_center[1])**2)
        mask = dist_from_center <= radius
        # Count the number of pixels in the circle that are not equal to 0
        num_pixels = np.count_nonzero(mask_frame[mask])
        # Divide by the total number of pixels in the circle
        concentrations[cell_ID] = num_pixels / (np.pi * radius**2)
    return concentrations

#not sure if this is necessarily an interesting calculation
def get_order_parameter_vs_concentration(mask_frame, radius, method,):
    order_parameters = method(mask_frame, radius)
    concentrations = get_local_concentrations(mask_frame, radius)

    order_vs_concentration = {}
    for ID in order_parameters:
        order_vs_concentration[ID] = order_parameters[ID]/concentrations[ID]
    
    return order_vs_concentration


def plot_order_parameters_vs_concentration(mask_frame, radius, method, pixel_scale):
    """
    Plots the order parameters versus the concentration of bacteria.

    This function is intended for use with heterogeneous density images, as it compares different densities within the same frame.

    Parameters:
    mask_frame (np.array): The mask frame to analyze.
    radius (float): The radius for local order parameter calculation in microns.
    method (function): The method for calculating local order parameters. This should be a function that ends with '_local_order_parameters'.
    pixel_scale (float): The scale of the pixels in the image.

    Raises:
    ValueError: If the provided method is not one of the expected '_local_order_parameters' functions.

    Returns:
    None: This function does not return a value; it generates plots.

    The function generates three plots:
    1. A scatter plot of order parameters vs concentration of bacteria.
    2. A heatmap of order parameters vs concentration of bacteria.
    3. A scatter plot of the average order parameter vs bacteria concentration in equidistant bins.
    """
    ### method can be, dot_product_local_order_parameters or cos_local_order_parameters for the moment, it should be a _local_order_parameter function
    possible_methods = [function for function in all_functons_globals if function.endswith('cos_local_order_parameters')]
    
    if method.__name__ not in possible_methods:
        raise ValueError(f"The given method is {method.__name__} but it must be one of the following functions: {possible_methods}")
    
    radius = radius/pixel_scale

    order_parameters = method(mask_frame, radius)
    concentrations = get_local_concentrations(mask_frame, radius)

    order_parameters_list = []
    concentrations_list = []
    for ID in order_parameters:
        order_parameters_list.append(order_parameters[ID])
        concentrations_list.append(concentrations[ID])
    
    # scatter plot of order parameters vs concentration of bacteria
    plt.scatter(concentrations_list, order_parameters_list)
    plt.xlabel('Concentration of bacteria')
    plt.ylabel('Order parameter')
    plt.title('Order parameter vs concentration of bacteria')
    plt.xlim(0, 1)
    plt.ylim(-1, 1)
    plt.show()

    #heatmap of order parameters vs concentration of bacteria
    plt.hist2d(concentrations_list, order_parameters_list, bins=50, range=[[0, 1], [-1, 1]])
    plt.colorbar()
    plt.xlabel('Concentration of bacteria')
    plt.ylabel('Order parameter')
    plt.title('Order parameter vs concentration of bacteria')
    plt.show()

    # Calculate the average order parameter for each bin
    num_bins = 20
    bins = np.linspace(min(concentrations_list), max(concentrations_list), num_bins+1)
    digitized = np.digitize(concentrations_list, bins)

    bin_means = [np.mean(np.array(order_parameters_list)[digitized == i]) for i in range(1, len(bins))]

    plt.scatter(bins[:-1], bin_means)

    #add legend above each point showing the number of cells in each bin
    for i in range(1, len(bins)):
        plt.text(bins[i-1], bin_means[i-1], str(len(np.array(order_parameters_list)[digitized == i])))

    plt.xlabel('Concentration of bacteria')
    plt.ylabel('Average order parameter')
    plt.title('Average order parameter vs bacteria concentration in ' + str(num_bins) + ' equiudistant bins')
    plt.show()



##Plotting functions
#docsting needs updating



def list_available_features():
    """
    Lists the available features that can be used for coloring the segmented masks.
    """
    available_features = [
        'Perimeter_PIX',
        'Surface_area_PIX^2',
        'Convex_hull_area_PIX^2',
        'Solidity',
        'Length_PIX',
        'Ellipse_major_axis_PIX',
        'Width_PIX',
        'Ellipse_minor_axis_PIX',
        'Aspect_ratio',
        'Ellipse_aspect_ratio',
        'Mean_intensity',
        'Center_position_x_PIX',
        'Ellipse_center_x_PIX',
        'Center_position_y_PIX',
        'Ellipse_center_y_PIX',
        'Angle_to_x+_axis_RAD',
        'Angle_to_x+_axis_DEG',
        'Ellipse_Angle_DEG',
        'Min_intensity',
        'Max_intensity',
        'Perimeter_MICRON',
        'Surface_area_MICRON^2',
        'Convex_hull_area_MICRON^2',
        'Length_MICRON',
        'Ellipse_major_axis_MICRON',
        'Width_MICRON',
        'Ellipse_minor_axis_MICRON',
        'Center_position_x_MICRON',
        'Ellipse_center_x_MICRON',
        'Center_position_y_MICRON',
        'Ellipse_center_y_MICRON'
    ]

    print("Available features for coloring the segmented masks:")
    for feature in available_features:
        print(f"- {feature}")
        

def calculate_feature_for_frame(mask_frame, raw_frame, feature_to_color_by, decimal_places, pixel_scale):
    """
    Calculate a specific cell feature for a given mask and raw frame.

    Parameters
    ----------
    mask_frame : numpy array
        A 2D array representing the segmented mask frame.
    raw_frame : numpy array
        A 2D array representing the corresponding raw frame.
    feature_to_color_by : str
        The feature to use for coloring the segmented masks.
    decimal_places : int
        The number of decimal places to keep in the results.
    pixel_scale : float, optional
        The scale to convert pixels to microns.

    Returns
    -------
    feature_dict : dict
        A dictionary where the keys are cell IDs and the values are the calculated feature values.
    """
    IDs = get_cellIDs(mask_frame)
    
    if feature_to_color_by == 'Perimeter_PIX':
        feature_values = calculate_perimeter(mask_frame)
    elif feature_to_color_by == 'Surface_area_PIX^2':
        feature_values = calculate_bacteria_surface_areas(mask_frame)
    elif feature_to_color_by == 'Convex_hull_area_PIX^2':
        feature_values = calculate_convex_hull_area(mask_frame)
    elif feature_to_color_by == 'Solidity':
        feature_values = calculate_solidity(mask_frame)
    elif feature_to_color_by == 'Length_PIX':
        long_extremities = find_extremities_of_cells_length(mask_frame)
        feature_values = get_cell_lengths(mask_frame, long_extremities)
    elif feature_to_color_by == 'Ellipse_major_axis_PIX':
        ellipses = fit_ellipse(mask_frame)
        feature_values = {ID: ellipses[ID]['Major_Axis'] if ID in ellipses else np.nan for ID in IDs}
    elif feature_to_color_by == 'Width_PIX':
        long_extremities = find_extremities_of_cells_length(mask_frame)
        short_extremities = find_extremities_of_cells_width(mask_frame, long_extremities)
        feature_values = get_cell_widths(mask_frame, long_extremities, short_extremities)
    elif feature_to_color_by == 'Ellipse_minor_axis_PIX':
        ellipses = fit_ellipse(mask_frame)
        feature_values = {ID: ellipses[ID]['Minor_Axis'] if ID in ellipses else np.nan for ID in IDs}
    elif feature_to_color_by == 'Aspect_ratio':
        long_extremities = find_extremities_of_cells_length(mask_frame)
        short_extremities = find_extremities_of_cells_width(mask_frame, long_extremities)
        feature_values = calculate_aspect_ratios(mask_frame, long_extremities, short_extremities)
    elif feature_to_color_by == 'Ellipse_aspect_ratio':
        ellipses = fit_ellipse(mask_frame)
        feature_values = {ID: ellipses[ID]['Aspect_Ratio'] if ID in ellipses else np.nan for ID in IDs}
    elif feature_to_color_by == 'Mean_intensity':
        intensities = calculate_intensities(mask_frame, raw_frame)
        feature_values = {ID: intensities[ID]['mean_intensity'] for ID in IDs}
    elif feature_to_color_by == 'Center_position_x_PIX':
        centers = find_cell_centers(mask_frame)
        feature_values = {ID: centers.get(ID, [np.nan])[1] for ID in IDs}
    elif feature_to_color_by == 'Ellipse_center_x_PIX':
        ellipses = fit_ellipse(mask_frame)
        feature_values = {ID: ellipses[ID]['Center_X'] if ID in ellipses else np.nan for ID in IDs}
    elif feature_to_color_by == 'Center_position_y_PIX':
        centers = find_cell_centers(mask_frame)
        feature_values = {ID: centers.get(ID, [np.nan])[0] for ID in IDs}
    elif feature_to_color_by == 'Ellipse_center_y_PIX':
        ellipses = fit_ellipse(mask_frame)
        feature_values = {ID: ellipses[ID]['Center_Y'] if ID in ellipses else np.nan for ID in IDs}
    elif feature_to_color_by == 'Angle_to_x+_axis_RAD':
        long_extremities = find_extremities_of_cells_length(mask_frame)
        feature_values = get_directions_radians(mask_frame, long_extremities)
    elif feature_to_color_by == 'Angle_to_x+_axis_DEG':
        long_extremities = find_extremities_of_cells_length(mask_frame)
        feature_values = get_directions_degrees(mask_frame, long_extremities)
    elif feature_to_color_by == 'Ellipse_Angle_DEG':
        ellipses = fit_ellipse(mask_frame)
        feature_values = {ID: ellipses[ID]['Angle'] if ID in ellipses else np.nan for ID in IDs}
    elif feature_to_color_by == 'Min_intensity':
        intensities = calculate_intensities(mask_frame, raw_frame)
        feature_values = {ID: intensities[ID]['min_intensity'] for ID in IDs}
    elif feature_to_color_by == 'Max_intensity':
        intensities = calculate_intensities(mask_frame, raw_frame)
        feature_values = {ID: intensities[ID]['max_intensity'] for ID in IDs}
    elif feature_to_color_by == 'Perimeter_MICRON':
        feature_values = calculate_perimeter(mask_frame)
        feature_values = {ID: value * pixel_scale for ID, value in feature_values.items()}
    elif feature_to_color_by == 'Surface_area_MICRON^2':
        feature_values = calculate_bacteria_surface_areas(mask_frame)
        feature_values = {ID: value * (pixel_scale ** 2) for ID, value in feature_values.items()}
    elif feature_to_color_by == 'Convex_hull_area_MICRON^2':
        feature_values = calculate_convex_hull_area(mask_frame)
        feature_values = {ID: value * (pixel_scale ** 2) for ID, value in feature_values.items()}
    elif feature_to_color_by == 'Length_MICRON':
        long_extremities = find_extremities_of_cells_length(mask_frame)
        feature_values = get_cell_lengths(mask_frame, long_extremities)
        feature_values = {ID: value * pixel_scale for ID, value in feature_values.items()}
    elif feature_to_color_by == 'Ellipse_major_axis_MICRON':
        ellipses = fit_ellipse(mask_frame)
        feature_values = {ID: ellipses[ID]['Major_Axis'] * pixel_scale if ID in ellipses else np.nan for ID in IDs}
    elif feature_to_color_by == 'Width_MICRON':
        long_extremities = find_extremities_of_cells_length(mask_frame)
        short_extremities = find_extremities_of_cells_width(mask_frame, long_extremities)
        feature_values = get_cell_widths(mask_frame, long_extremities, short_extremities)
        feature_values = {ID: value * pixel_scale for ID, value in feature_values.items()}
    elif feature_to_color_by == 'Ellipse_minor_axis_MICRON':
        ellipses = fit_ellipse(mask_frame)
        feature_values = {ID: ellipses[ID]['Minor_Axis'] * pixel_scale if ID in ellipses else np.nan for ID in IDs}
    elif feature_to_color_by == 'Center_position_x_MICRON':
        centers = find_cell_centers(mask_frame)
        feature_values = {ID: centers.get(ID, [np.nan])[1] * pixel_scale for ID in IDs}
    elif feature_to_color_by == 'Ellipse_center_x_MICRON':
        ellipses = fit_ellipse(mask_frame)
        feature_values = {ID: ellipses[ID]['Center_X'] * pixel_scale if ID in ellipses else np.nan for ID in IDs}
    elif feature_to_color_by == 'Center_position_y_MICRON':
        centers = find_cell_centers(mask_frame)
        feature_values = {ID: centers.get(ID, [np.nan])[0] * pixel_scale for ID in IDs}
    elif feature_to_color_by == 'Ellipse_center_y_MICRON':
        ellipses = fit_ellipse(mask_frame)
        feature_values = {ID: ellipses[ID]['Center_Y'] * pixel_scale if ID in ellipses else np.nan for ID in IDs}
    else:
        raise ValueError(f"Feature '{feature_to_color_by}' is not recognized.")

    feature_dict = {ID: round_values(feature_values.get(ID, np.nan), decimal_places) for ID in IDs}
    return feature_dict




def visualize_frames(movies_masks, movies_outline, frame_number='all', feature_to_color_by='Mean_intensity', decimal_places=2, save_images=False, output_dir='output', dpi=300, pixel_scale=0.1625):
    """
    Displays specified frames of segmented masks and corresponding outline movies side by side, with axis labels and a black background for better visibility. The segmented masks are colored based on a given feature.

    Parameters
    ----------
    movies_masks : dict
        A dictionary where the keys are video names and the values are lists of numpy arrays representing the frames of segmented masks.
    movies_outline : dict
        A dictionary where the keys are video names and the values are lists of numpy arrays representing the frames of outline movies.
    frame_number : int or 'all' (default = 'all')
        Frame number to display for each video, defaults to 'all' which processes all frames.
    feature_to_color_by : str (default = 'Mean_intensity')
        The feature to use for coloring the segmented masks. Use list_available_features() to see available features.
    decimal_places : int, optional
        The number of decimal places to keep in the results. Default is 2.
    save_images : bool, optional
        Whether to save the images. Default is False.
    output_dir : str, optional
        The directory to save the images to if save_images is True. Default is 'output'.
    dpi : int, optional
        The resolution in dots per inch (DPI) to use for saving the images. Default is 300.
    pixel_scale : float, optional
        The scale to convert pixels to microns. Default is 0.1625 microns/pixel.
    """
    start_time = time.time()

    angle_features = ['Angle_to_x+_axis_RAD', 'Angle_to_x+_axis_DEG', 'Ellipse_Angle_DEG']

    for key in movies_masks:
        if key in movies_outline:
            frames_range = range(len(movies_masks[key])) if frame_number == 'all' else [frame_number]

            images_for_movie = []

            for frame_num in tqdm(frames_range, desc=f"Processing frames for {key}"):
                # Check if the frame number is within the range for both masks and outline frames
                if frame_num >= len(movies_masks[key]) or frame_num >= len(movies_outline[key]):
                    print(f"Frame {frame_num} is out of range for {key}. Skipping this frame.")
                    continue

                mask_frame = movies_masks[key][frame_num]
                outline_frame = movies_outline[key][frame_num]

                # Calculate the feature for coloring
                feature_dict = calculate_feature_for_frame(mask_frame, outline_frame, feature_to_color_by, decimal_places, pixel_scale)
                feature_values = np.array([feature_dict.get(ID, np.nan) for ID in get_cellIDs(mask_frame)])

                norm = Normalize(vmin=np.nanmin(feature_values), vmax=np.nanmax(feature_values))
                
                if feature_to_color_by in angle_features:
                    cmap = cm.hsv  # Circular colormap for angles
                else:
                    #cmap = cm.viridis  # Non-circular colormap for other features
                    cmap = cm.plasma

                colored_mask = np.zeros((*mask_frame.shape, 3))
                for ID in np.unique(mask_frame):
                    if ID == 0:
                        continue
                    mask = mask_frame == ID
                    color = cmap(norm(feature_dict.get(ID, 0)))[:3]
                    colored_mask[mask] = color

                if frame_number == 'all':
                    # Convert the image to a PIL Image
                    img = Image.fromarray((colored_mask * 255).astype(np.uint8))
                    images_for_movie.append(img)
                else:
                    # Create figure for side-by-side display
                    fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor='black')

                    # Displaying outline movie frame
                    axes[0].imshow(outline_frame, cmap='gray')
                    axes[0].set_title(f'Outline Frame {frame_num} of {key}', color='white')
                    axes[0].axis('on')  # Show axis
                    axes[0].set_xlabel('Pixels', color='white')
                    axes[0].set_ylabel('Pixels', color='white')
                    axes[0].tick_params(colors='white', which='both', direction='out')  # Change tick color to white
                    axes[0].grid(False)

                    # Displaying segmented mask frame
                    axes[1].imshow(colored_mask)
                    axes[1].set_title(f'Segmented Mask Frame {frame_num} of {key}', color='white')
                    axes[1].axis('on')  # Show axis
                    axes[1].set_xlabel('Pixels', color='white')
                    axes[1].set_ylabel('Pixels', color='white')
                    axes[1].tick_params(colors='white', which='both', direction='out')  # Change tick color to white
                    axes[1].grid(False)

                    # Display color bar
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=axes[1], orientation='vertical')
                    cbar.set_label(f'{feature_to_color_by}', color='white')
                    cbar.ax.yaxis.set_tick_params(color='white')
                    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

                    plt.tight_layout()

                    if save_images:
                        os.makedirs(os.path.join(output_dir, key, 'visualization_by_feature'), exist_ok=True)
                        save_path_full = os.path.join(output_dir, key, 'visualization_by_feature', f"{key}_frame_{frame_num}_{feature_to_color_by}.tif")
                        plt.savefig(save_path_full, facecolor=fig.get_facecolor(), edgecolor='none', dpi=dpi)

                        # Save the segmented mask frame alone without any labels
                        fig_mask, ax_mask = plt.subplots(figsize=(6, 6), facecolor='black')
                        ax_mask.imshow(colored_mask)
                        ax_mask.axis('off')  # Turn off the axis
                        save_path_mask = os.path.join(output_dir, key, 'visualization_by_feature', f"{key}_frame_{frame_num}_{feature_to_color_by}_mask_only.tif")
                        plt.savefig(save_path_mask, facecolor=fig_mask.get_facecolor(), edgecolor='none', dpi=dpi)
                        plt.close(fig_mask)

                        plt.show()
                    plt.close(fig)

            if frame_number == 'all' and save_images and images_for_movie:
                movie_save_path = os.path.join(output_dir, key, 'visualization_by_feature', f"{key}_movie_{feature_to_color_by}.tif")
                os.makedirs(os.path.dirname(movie_save_path), exist_ok=True)
                images_for_movie[0].save(movie_save_path, save_all=True, append_images=images_for_movie[1:], duration=100, loop=0)
                print(f"Saved movie for {key}, feature {feature_to_color_by} at {movie_save_path}")

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")






def plot_polar_angles(movies_masks, movies_raw, frame_number='all', pixel_scale=0.1625, decimal_places=2, save_images=False, output_dir='output', dpi=300):
    """
    Plots polar histograms of cell orientation angles for specified frames of segmented masks.

    Parameters
    ----------
    movies_masks : dict
        A dictionary where the keys are video names and the values are lists of numpy arrays representing the frames of segmented masks.
    movies_raw : dict
        A dictionary where the keys are video names and the values are lists of numpy arrays representing the raw frames.
    frame_number : int or 'all' (default = 'all')
        Frame number to display for each video, defaults to 'all' which processes all frames.
    pixel_scale : float, optional
        The scale to convert pixels to microns. Default is 0.1625 microns/pixel.
    decimal_places : int, optional
        The number of decimal places to keep in the results. Default is 2.
    save_images : bool, optional
        Whether to save the images. Default is False.
    output_dir : str, optional
        The directory to save the images to if save_images is True. Default is 'output'.
    dpi : int, optional
        The resolution in dots per inch (DPI) to use for saving the images. Default is 300.
    """
    def plot_single_frame_polar(mask_frame, raw_frame, frame_num, key, pixel_scale, decimal_places, save_images, output_dir, dpi):
        long_extremities = find_extremities_of_cells_length(mask_frame)
        directions_deg = get_directions_degrees(mask_frame, long_extremities)
        angles = np.array(list(directions_deg.values()))

        total_cells = len(angles)

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_theta_direction(-1)  # Flip the image
        ax.set_theta_offset(np.pi)  # Adjust the offset to flip the image
        ax.set_thetamin(0)
        ax.set_thetamax(180)

        num_bins = 90
        bin_edges = np.linspace(0, 180, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        count, _ = np.histogram(angles, bins=bin_edges)
        count = count / total_cells  # Normalize counts
        width = np.deg2rad(np.diff(bin_edges))

        norm = Normalize(vmin=np.nanmin(bin_centers), vmax=np.nanmax(bin_centers))
        cmap = cm.hsv

        bars = ax.bar(np.deg2rad(bin_centers), count, width=width, color=cmap(norm(bin_centers)), edgecolor='black', alpha=0.7)

        ax.set_ylim(0, max(count))

        # Setting the radial ticks in percentages
        radial_ticks = np.linspace(0, max(count), num=5)
        radial_tick_labels = [f'{int(tick*100)}%' for tick in radial_ticks]
        ax.set_yticks(radial_ticks)
        ax.set_yticklabels(radial_tick_labels)

        # Make grid lines and contours thicker and blacker
        ax.grid(True, color='black', linewidth=0.5)
        for bar in bars:
            bar.set_edgecolor('black')
            bar.set_linewidth(0.5)

        # Create the color bar with adjusted settings
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', fraction=0.05, pad=0, aspect=30, shrink=0.7)
        cbar.set_label(f'Angle to x+ axis (degrees)')

        # Adjust layout to reduce spacing
        plt.subplots_adjust(bottom=0.15, top=0.85, hspace=0)

        if save_images:
            os.makedirs(os.path.join(output_dir, key, 'polar_plots'), exist_ok=True)
            save_path = os.path.join(output_dir, key, 'polar_plots', f"{key}_frame_{frame_num}_polar_plot.png")
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
        plt.show()

    start_time = time.time()

    for key in movies_masks:
        frames_range = range(len(movies_masks[key])) if frame_number == 'all' else [frame_number]

        images_for_movie = []

        for frame_num in tqdm(frames_range, desc=f"Processing frames for {key}"):
            if frame_num >= len(movies_masks[key]):
                print(f"Frame {frame_num} is out of range for {key}. Skipping this frame.")
                continue

            mask_frame = movies_masks[key][frame_num]
            raw_frame = movies_raw[key][frame_num]

            if frame_number == 'all':
                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
                long_extremities = find_extremities_of_cells_length(mask_frame)
                directions_deg = get_directions_degrees(mask_frame, long_extremities)
                angles = np.array(list(directions_deg.values()))

                total_cells = len(angles)

                num_bins = 60
                bin_edges = np.linspace(0, 180, num_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                count, _ = np.histogram(angles, bins=bin_edges)
                count = count / total_cells  # Normalize counts
                width = np.deg2rad(np.diff(bin_edges))

                norm = Normalize(vmin=np.nanmin(bin_centers), vmax=np.nanmax(bin_centers))
                cmap = cm.hsv

                bars = ax.bar(np.deg2rad(bin_centers), count, width=width, color=cmap(norm(bin_centers)), edgecolor='black', alpha=0.7)

                ax.set_theta_direction(-1)
                ax.set_theta_offset(np.pi)
                ax.set_ylim(0, np.max(count))
                ax.set_thetamin(0)
                ax.set_thetamax(180)

                # Setting the radial ticks in percentages
                radial_ticks = np.linspace(0, max(count), num=5)
                radial_tick_labels = [f'{int(tick*100)}%' for tick in radial_ticks]
                ax.set_yticks(radial_ticks)
                ax.set_yticklabels(radial_tick_labels)

                # Make grid lines and contours thicker and blacker
                ax.grid(True, color='black', linewidth=1.5)
                for bar in bars:
                    bar.set_edgecolor('black')
                    bar.set_linewidth(1.5)

                # Create the color bar with adjusted settings
                cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', fraction=0.05, pad=0, aspect=30, shrink=0.7)
                cbar.set_label(f'Angle to x+ axis (degrees)')

                # Adjust layout to reduce spacing
                plt.subplots_adjust(bottom=0.15, top=0.85, hspace=0)

                fig.canvas.draw()
                img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
                images_for_movie.append(img)
                plt.close(fig)
            else:
                plot_single_frame_polar(mask_frame, raw_frame, frame_num, key, pixel_scale, decimal_places, save_images, output_dir, dpi)

        if frame_number == 'all' and save_images and images_for_movie:
            movie_save_path = os.path.join(output_dir, key, 'polar_plots', f"{key}_polar_plots.tif")
            os.makedirs(os.path.dirname(movie_save_path), exist_ok=True)
            images_for_movie[0].save(movie_save_path, save_all=True, append_images=images_for_movie[1:], duration=100, loop=0)
            print(f"Saved polar plot movie for {key} at {movie_save_path}")

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")






## To see if we keep or not, we now have a function to plot any of the features, plot_distribution
def plot_cell_areas(mask_video, fileName, frameNumber = 0, bins=20, max_value = None, xlims=[], save=False, directory=''):
    """
    Plot the surface areas of the cells in a given frame of a mask video.

    This function removes the cells on the edge of the frame,
    Then it calculates the surface areas, which are then sorted by size and plotted.
    
    Parameters
    ----------
    mask_video : numpy.ndarray
        A 3D array representing the masks, where the first dimension is the frame number, 
        and the background of each frame is represented by 0 and each distinct cell is represented by a unique gray value.
    fileName : str
        The name of the file from which the mask was generated. This is used in the title of the plot.
    frameNumber : int
        The number of the frame in which to plot the cell areas.
    bins : int (default=20)
        Specify a bin size.
    xlims : list of int, format [lower limit, upper limit]
        specify the lower and upper x-axis limits --> same for all plots
    save : bool
        Not implemented yet, but should be part of any plot --> save as graphs.

    Returns
    -------
    None

    Example
    -------
    >>> folder_manager = dm.directory_manager()
    >>> images = omnu.import_images(folder_manager.segmentdir)
    >>> for i in images:
    >>>     plot_cell_areas(images[i],i)
    """

    singlemask = mask_video[frameNumber]
    cellsOnEdge = find_cells_on_edge(mask_video[frameNumber])
    #remove the cells on the edge
    for cellID in cellsOnEdge:
        singlemask[singlemask == cellID] = 0
    
    surfaces = calculate_bacteria_surface_areas(singlemask)

    #plot
    if max_value == None:
        plt.hist(surfaces.values(), bins)
    elif isinstance(max_value, (int, float)):
        plt.hist(surfaces.values(), bins, range=(0, max_value))
    plt.xlabel("Surface area (square pixels)")
    plt.ylabel("Number of cells")
    if xlims:
        plt.xlim(xlims)
    #plot mean and standard deviation
    average_surface_area = np.mean(list(surfaces.values()))
    standard_deviation = np.std(list(surfaces.values()))
    plt.axvline(average_surface_area, color='r', linestyle='dashed', linewidth=1, label='mean = ' + np.round(average_surface_area, 1).astype(str))
    plt.axvline(average_surface_area + standard_deviation, color='k', linestyle='dashed', linewidth=1, label='SD = ' + np.round(standard_deviation, 1).astype(str))
    plt.axvline(average_surface_area - standard_deviation, color='k', linestyle='dashed', linewidth=1)
    plt.text(average_surface_area, 0, np.round(average_surface_area, 1), color='r')
    plt.title(fileName + ' - area distribution')
    plt.plot([], [], ' ', label= 'n = ' + str(len(surfaces))) # number of cells in an empty spot on the legend
    plt.legend()

    if save: 
        plt.savefig(os.path.join(directory, 'cell_area_distribution_' + os.path.splitext(fileName)[0] + '.png'))
    
    plt.show()

    
    return surfaces


def plot_cell_lengths(mask_video, fileName, frameNumber = 0, bins=20, max_value = None, xlims=[], save=False, directory=''):
    """
    Plot the lengths of the cells in a given frame of a mask video.

    This function removes the cells on the edge of the frame,
    Then it calculates the lengths, which are then sorted by size and plotted.
    
    Parameters
    ----------
    mask_video : numpy.ndarray
        A 3D array representing the masks, where the first dimension is the frame number, 
        and the background of each frame is represented by 0 and each distinct cell is represented by a unique gray value.
    fileName : str
        The name of the file from which the mask was generated. This is used in the title of the plot.
    frameNumber : int
        The number of the frame in which to plot the cell areas.
    bins : int (default=20)
        Specify a bin size.
    xlims : list of int, format [lower limit, upper limit]
        specify the lower and upper x-axis limits --> same for all plots
    save : bool
        Not implemented yet, but should be part of any plot --> save as graphs.

    Returns
    -------
    None

    Example
    -------
    >>> folder_manager = dm.directory_manager()
    >>> images = omnu.import_images(folder_manager.segmentdir)
    >>> for i in images:
    >>>     plot_cell_lengths(images[i],i)
    """

    singlemask = mask_video[frameNumber]
    cellsOnEdge = find_cells_on_edge(mask_video[frameNumber])
    #remove the cells on the edge
    for cellID in cellsOnEdge:
        singlemask[singlemask == cellID] = 0
    
    cell_extremities_length = find_extremities_of_cells_length(singlemask)
    lengths = get_cell_lengths(singlemask, cell_extremities_length)

    #plot
    if max_value == None:
        plt.hist(lengths.values(), bins)
    elif isinstance(max_value, (int, float)):
        plt.hist(lengths.values(), bins, range=(0, max_value))
    plt.xlabel("Length (pixels)")
    plt.ylabel("Number of cells")
    if xlims:
        plt.xlim(xlims)
    #plot mean and standard deviation
    average_length = np.mean(list(lengths.values()))
    standard_deviation = np.std(list(lengths.values()))
    plt.axvline(average_length, color='r', linestyle='dashed', linewidth=1, label='mean = ' + np.round(average_length, 1).astype(str))
    plt.axvline(average_length + standard_deviation, color='k', linestyle='dashed', linewidth=1, label='SD = ' + np.round(standard_deviation, 1).astype(str))
    plt.axvline(average_length - standard_deviation, color='k', linestyle='dashed', linewidth=1)
    plt.text(average_length, 0, np.round(average_length, 1), color='r')
    plt.title(fileName + ' - length distribution')
    plt.plot([], [], ' ', label= 'n = ' + str(len(lengths))) # number of cells in an empty spot on the legend
    plt.legend()

    if save:
        plt.savefig(os.path.join(directory, 'cell_length_distribution_' + os.path.splitext(fileName)[0] + '.png'))

    plt.show()

    return lengths



# this function needs work it has issues with the calculate_aspect_ratios function
def plot_cell_length_width_ratio(mask_video, fileName, frameNumber = 0, bins=20, max_value = None, xlims=[], save=False, directory=''):
    """
    Plot the aspect ratios of the cells in a given frame of a mask video.

    This function removes the cells on the edge of the frame,
    Then it calculates the aspect ratios, which are then sorted by size and plotted.
    
    Parameters
    ----------
    mask_video : numpy.ndarray
        A 3D array representing the masks, where the first dimension is the frame number, 
        and the background of each frame is represented by 0 and each distinct cell is represented by a unique gray value.
    fileName : str
        The name of the file from which the mask was generated. This is used in the title of the plot.
    frameNumber : int
        The number of the frame in which to plot the cell areas.
    bins : int (default=20)
        Specify a bin size.
    xlims : list of int, format [lower limit, upper limit]
        specify the lower and upper x-axis limits --> same for all plots
    save : bool
        Not implemented yet, but should be part of any plot --> save as graphs.

    Returns
    -------
    None

    Example
    -------
    >>> folder_manager = dm.directory_manager()
    >>> images = omnu.import_images(folder_manager.segmentdir)
    >>> for i in images:
    >>>     plot_cell_length_width_ratio(images[i],i)
    """

    singlemask = mask_video[frameNumber]
    cellsOnEdge = find_cells_on_edge(mask_video[frameNumber])
    #I can try to remove the cells on the edge

    cell_extremities_length = find_extremities_of_cells_length(singlemask)
    cell_extremities_width = find_extremities_of_cells_width(singlemask, cell_extremities_length)
    aspectRatios = calculate_aspect_ratios(singlemask, cell_extremities_length, cell_extremities_width)

    #set inf values to the maximum value
    for ID in aspectRatios:
        if aspectRatios[ID] == np.inf:
            aspectRatios[ID] = 100

    #plot
    if max_value == None:
        plt.hist(aspectRatios.values(), bins)
    elif isinstance(max_value, (int, float)):
        plt.hist(aspectRatios.values(), bins, range=(0, max_value))
    plt.xlabel("Aspect ratio")
    plt.ylabel("Number of cells")
    if xlims:
        plt.xlim(xlims)
    #plot mean and standard deviation
    average_aspect_ratio = np.mean(list(aspectRatios.values()))
    standard_deviation = np.std(list(aspectRatios.values()))
    plt.axvline(average_aspect_ratio, color='r', linestyle='dashed', linewidth=1, label='mean = ' + np.round(average_aspect_ratio, 1).astype(str))
    plt.axvline(average_aspect_ratio + standard_deviation, color='k', linestyle='dashed', linewidth=1, label='SD = ' + np.round(standard_deviation, 1).astype(str))
    plt.axvline(average_aspect_ratio - standard_deviation, color='k', linestyle='dashed', linewidth=1)
    plt.text(average_aspect_ratio, 0, np.round(average_aspect_ratio, 1), color='r')
    plt.title(fileName + ' - aspect ratio distribution')
    plt.plot([], [], ' ', label= 'n = ' + str(len(aspectRatios))) # number of cells in an empty spot on the legend
    plt.legend()
    
    if save:
        plt.savefig(os.path.join(directory, 'cell_aspect_ratio_distribution_' + os.path.splitext(fileName)[0] + '.png'))
    
    plt.show()

    return aspectRatios


def plot_z_values_vs_cell_area(images):
    """
    calculatte and plot z values of cell area vs cell area
    """
    for i in images:
        cell_areas = list(calculate_bacteria_surface_areas(images[i][0]).values())
        average_surface_area = np.mean(cell_areas)
        standard_deviation = np.std(cell_areas)
        
        z_values = []
        for area in cell_areas:
            z_values.append((area - average_surface_area) / standard_deviation)
        plt.scatter(z_values, cell_areas)
        plt.xlabel("Z value")
        plt.ylabel("Cell area (square pixels)")
        plt.show()

def show_cell_directions(mask, arrow_length = 20):
    """
    Overlay the directions of the cells on a mask.

    This function works by first finding the directions of all cells using the `find_directions` function. Then it overlays these directions on the mask using arrows. The arrows are centered at the mean position of each cell and point in the direction of the cell.

    Parameters
    ----------
    mask : numpy array
        A 2D array representing the mask, where the background has a value of 0 and each cell has a different gray value.
    """
    directions = get_directions_radians(mask)
    plt.imshow(mask, cmap='gray')
    
    for ID in get_cellIDs(mask):
        y0, x0 = find_center_of_1cell(mask, ID)
        np.mean(np.transpose(np.where(mask == ID)), axis=0)
        orientation = directions[ID]
        dx = np.cos(orientation) * arrow_length
        dy = np.sin(orientation) * arrow_length
        plt.arrow(x0, y0, dx, dy, color='red', head_width=5)
    
    plt.show()

def overlay_extremities_segment(mask, extremities, otherExtremities = None):
    """
    Overlay the segment connecting the extremities of the cells on a mask.

    Parameters
    ----------
    mask : numpy array
        A 2D array representing the mask, where the background has a value of 0 and each cell has a different gray value.
    
    extremities : dict
        A dictionary where the keys are the cell IDs and the values are lists of two tuples representing the coordinates of the extremities of each cell.
    otherExtremities : dict, optional
        An optional second set of extremities to plot in a different color.
    ax : matplotlib Axes
        The axes on which to draw the overlay.
    video_name : str
        The name of the video being processed.
    frame_index : int
        The index of the frame being displayed.
    """
    ax.imshow(mask, cmap='gray')
    for i in extremities:
        ax.plot([extremities[i][0][1], extremities[i][1][1]], [extremities[i][0][0], extremities[i][1][0]], color='red', linewidth=2)
    if otherExtremities is not None:
        for i in otherExtremities:
            ax.plot([otherExtremities[i][0][1], otherExtremities[i][1][1]], [otherExtremities[i][0][0], otherExtremities[i][1][0]], color='blue', linewidth=2)

    ax.set_title(textwrap.fill(f'{video_name} - Frame {frame_index}', width=40), color='white')
    ax.axis('on')  # Show axis
    ax.set_xlabel('Pixels', color='white')
    ax.set_ylabel('Pixels', color='white')
    ax.tick_params(axis='both', colors='white', which='both')
    ax.grid(False)

    plt.tight_layout()
    plt.show()
    
    if otherExtremities != None:
        for i in otherExtremities:
            plt.plot([otherExtremities[i][0][1], otherExtremities[i][1][1]], [otherExtremities[i][0][0], otherExtremities[i][1][0]], color='red', linewidth=1)

    plt.show()

def show_cell_length_width(mask_frame):
    """
    Plot the length and width of all cells on the mask frame.

    This function calculates the long and short extremities of cells in the mask frame, and overlays these extremities on the frame.

    Args:
        mask_frame (numpy.ndarray): A 2D numpy array representing a mask frame, where the background is represented by 0 and each distinct cell is represented by a unique gray value.
    """

    long_extremities = find_extremities_of_cells_length(mask_frame)
    short_extremitiies = find_extremities_of_cells_width(mask_frame, long_extremities)
    overlay_extremities_segment(mask_frame, short_extremitiies, long_extremities)

def plot_short_vectors_on_mask(mask_frame):
    """
    Plot the short direction vectors on the first mask in the input list.

    This function takes a list of masks, calculates the short direction vectors for the first mask, and plots these vectors on the mask.

    Args:
        masks (list): A list of 2D numpy arrays representing masks frame, where the background is represented by 0 and each distinct cell is represented by a unique gray value.
    """
    vectorOnCell = find_short_direction(mask_frame)
    plt.imshow(mask_frame, cmap='gray')
    for ID in get_cellIDs(mask_frame):
        plt.plot(vectorOnCell[ID][:,1], vectorOnCell[ID][:,0], color='red', linewidth=1)
    plt.show()

def plot_asapect_ratios(mask_frame, imageName):
    """
    Plot the aspect ratios of cells in a given frame of a mask.

    This function calculates the aspect ratios of cells in a specified frame of the mask, plots a histogram of these aspect ratios, and overlays the mean and standard deviation on the plot.

    Args:
        mask_frame (numpy.ndarray): A 2D numpy array representing a mask frame, where the background is represented by 0 and each distinct cell is represented by a unique gray value.
        fileName (str): The name of the image to be used in the plot title.
    """
    cell_extremities_length = find_extremities_of_cells_length(mask_frame)
    cell_extremities_width = find_extremities_of_cells_width(mask_frame, cell_extremities_length)
    aspectRatios = calculate_aspect_ratios(mask_frame, cell_extremities_length, cell_extremities_width)
    plt.hist(list(aspectRatios.values()), bins=20)
    plt.xlabel("aspect ratio")
    plt.ylabel("Number of cells")
    #plot mean and standard deviation
    average_aspect_ratio = np.mean(list(aspectRatios.values()))
    standard_deviation = np.std(list(aspectRatios.values()))
    plt.axvline(average_aspect_ratio, color='r', linestyle='dashed', linewidth=1, label='mean')
    plt.axvline(average_aspect_ratio + standard_deviation, color='k', linestyle='dashed', linewidth=1, label='SD = ' + np.round(standard_deviation, 1).astype(str))
    plt.axvline(average_aspect_ratio - standard_deviation, color='k', linestyle='dashed', linewidth=1)
    plt.text(average_aspect_ratio, 0, np.round(average_aspect_ratio, 1), color='r')
    plt.title(imageName + ' - aspect ratio')
    plt.legend()
    plt.show()

# local Nematic order parameter over all of r
def plot_local_order_parameters(mask_frame, method, Radiuses, pixel_scale, output_directory, video_name, analysis_frame, plot=True, semilog=False, save=False, dpi=300, overwrite=False, return_total=False, return_average_and_total=False):
    """
    This function calculates and plots the local order parameters for a given method and a set of radii. 

    Parameters:
    mask_frame (numpy.ndarray): A 2D numpy array representing a mask frame, where the background is represented by 0 and each distinct cell is represented by a unique gray value.
    method (function): A function that ends with '_local_order_parameters'. It can be either dot_product_local_order_parameters or cos_local_order_parameters.
    Radiuses (list): A list of radii to calculate the local order parameters for.
    pixel_scale (float): The scale of the pixels in the mask frame (microns/pixel).
    video_name (str): The name of the video to be used in the plot title and save path.
    output_directory (str): The directory where the plot will be saved.
    semilog (bool): If True, the x-axis will be in log scale.
    save (bool): If True, the plot will be saved.
    overwrite (bool): If True, existing files with the same name will be overwritten.
    dpi (int): The resolution in dots per inch for the saved plot.
    plot (bool, optional): If False, the function will only calculate the values and not plot them. Defaults to True.

    Returns:
    dict_local_order_parameters (dict): A dictionary where the keys are the radii and the values are the mean local order parameters for that radius.
    significant_r (int): The smallest radius where every cell has one or more neighbors.
    dict_n_close_cells (dict): A dictionary where the keys are the radii and the values are the mean number of neighboring cells for that radius.
    """
    
    possible_methods = [function for function in all_functons_globals if function.endswith('cos_local_order_parameters')]

    if method.__name__ not in possible_methods:
        raise ValueError(f"The given method is {method.__name__} but it must be one of the following functions: {possible_methods}")
    
    dict_local_order_parameters = {}
    total_local_order_parameters = {}
    dict_n_close_cells = {}
    total_n_close_cells = {}
    significant_r = 0

    for r in Radiuses:
        local_order_parameters_for_r, n_close_cells = method(mask_frame, int(r), return_n_close_cells=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dict_local_order_parameters[r] = np.nanmean(list(local_order_parameters_for_r.values()))
            dict_n_close_cells[r] = np.mean(list(n_close_cells.values()))
        total_local_order_parameters[r] = local_order_parameters_for_r
        total_n_close_cells[r] = n_close_cells

        if np.isnan(np.array(list(local_order_parameters_for_r.values()))).any():
            significant_r = r

    if not plot:
        if return_total:
            if return_average_and_total:
                return dict_local_order_parameters, total_local_order_parameters, significant_r, dict_n_close_cells, total_n_close_cells
            return total_local_order_parameters, significant_r, total_n_close_cells
        return dict_local_order_parameters, significant_r, dict_n_close_cells

    X = Radiuses * pixel_scale
    Y = list(dict_local_order_parameters.values())
    n_cells = list(dict_n_close_cells.values())

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 6)
    plt.plot(X, Y, linestyle='-', marker='o', color='black', markersize=5, label='Local order parameter', alpha=0.5)
    plt.axvline(x=significant_r * pixel_scale, color='red', linestyle='--', label='all cells have >1 neighbor')
    plt.plot([], [], linestyle='', marker='o', color='blue', label='Mean number of neighbors')

    for x, y, n in zip(X, Y, n_cells):
        label = "{:.1f}".format(n)
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', color='blue', fontsize=8)

    plt.legend(loc='upper right')
    ax.set_ylim([0, 1])
    plt.xlabel('Radius of neighboring cells (µm)')
    plt.ylabel('Local nematic order')
    plt.title(method.__name__.removesuffix('_local_order_parameters').replace('_', ' ') + ' local order parameter, single frame - ' + video_name)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    save_path = os.path.join(output_directory, video_name, 'order_parameter')

    if semilog:
        ax.set_xscale('log')
        plt.title('semilog ' + method.__name__.removesuffix('_local_order_parameters').replace('_', ' ') + ' local order parameter, single frame - ' + video_name)
        save_file = os.path.join(save_path, f"{os.path.basename(video_name).replace('.csv', '')}_{method.__name__.removesuffix('_local_order_parameters')}_order_parameter_frame_{analysis_frame}_single_frame_semilog.png")
    else:
        save_file = os.path.join(save_path, f"{os.path.basename(video_name).replace('.csv', '')}_{method.__name__.removesuffix('_local_order_parameters')}_order_parameter_frame_{analysis_frame}_single_frame.png")

    if save:
        os.makedirs(save_path, exist_ok=True)
        if os.path.isfile(save_file) and not overwrite:
            print(f"""Warning: A file with the same name already exists: 
                  {save_file}
                  Change overwrite to True if you want to overwrite the file""")
            if return_total:
                if return_average_and_total:
                    return dict_local_order_parameters, total_local_order_parameters, significant_r, dict_n_close_cells, total_n_close_cells
                return total_local_order_parameters, significant_r, total_n_close_cells
            return dict_local_order_parameters, significant_r, dict_n_close_cells
        else:
            print(f"Saving file to {save_file}")
            plt.savefig(save_file, dpi=dpi)

    plt.show()

    if return_total:
        if return_average_and_total:
            return dict_local_order_parameters, total_local_order_parameters, significant_r, dict_n_close_cells, total_n_close_cells
        return total_local_order_parameters, significant_r, total_n_close_cells

    return dict_local_order_parameters, significant_r, dict_n_close_cells



def plot_multiple_frames_local_order_parameters(mask_video, method, Radiuses, pixel_scale, output_directory, video_name, number_of_analised_frames, semilog=False, save=False, dpi=300, overwrite=False):
    possible_methods = [function for function in all_functons_globals if function.endswith('cos_local_order_parameters')]

    if method.__name__ not in possible_methods:
        raise ValueError(f"The given method is {method.__name__} but it must be one of the following functions: {possible_methods}")
    
    if len(mask_video.shape) != 3:
        raise ValueError("The mask_video must be a 3D numpy array")
    
    if number_of_analised_frames < 0:
        raise ValueError(f"The number of frames to analyze must be a positive integer, it's currently {number_of_analised_frames}")
    
    if number_of_analised_frames > len(mask_video):
        max_number_of_frames = len(mask_video)
        raise ValueError(f"The number of frames to analyze ({number_of_analised_frames}) is greater than the number of frames in the video ({max_number_of_frames})")
    
    n_frames = len(mask_video)
    local_orders = {}
    significant_radii = {}

    X = Radiuses * pixel_scale

    if number_of_analised_frames == 1:
        step_size = n_frames
    else:
        step_size = n_frames / (number_of_analised_frames - 1)
        if step_size == int(step_size):
            step_size = int(step_size) - 1
        else:
            step_size = int(step_size)
  
    for frameID in range(number_of_analised_frames):
        frame_number = int(frameID * step_size)
        mask_frame = mask_video[frame_number]

        local_orders[frame_number], significant_radii[frame_number], n_close_cells = plot_local_order_parameters(
            mask_frame, 
            method, 
            Radiuses, 
            pixel_scale, 
            output_directory,  
            video_name, 
            analysis_frame=frame_number,  # Pass the frame index here
            plot=False, 
            save=False
        )

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 6)

    for frame_number in local_orders:
        one_local_order = local_orders[frame_number]
        Y_local_order = list(one_local_order.values())
        ax.plot(X, Y_local_order, linestyle='-', marker='o', markersize=5, label='frame ' + str(frame_number + 1), alpha=0.5)
    
    plt.axvline(x=(max(significant_radii.values()) * pixel_scale), color='red', linestyle='--', label='all cells have >1 neighbor')
    plt.legend(loc='upper right')
    ax.set_ylim([0, 1])
    plt.xlabel('Radius of neighboring cells (µm)')
    plt.ylabel('Local nematic order')
    plt.title(method.__name__.removesuffix('_local_order_parameters').replace('_', ' ') + ' local order parameter, ' + str(number_of_analised_frames) + ' frames - ' + video_name)

    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    save_path = os.path.join(output_directory, video_name)

    if semilog:
        ax.set_xscale('log')
        plt.title('semilog ' + method.__name__.removesuffix('_local_order_parameters').replace('_', ' ') + ' local order parameter, single frame - ' + video_name)
        save_file = os.path.join(save_path, 'order_parameter', f"{os.path.basename(video_name).replace('.csv', '')}_{method.__name__.removesuffix('_local_order_parameters')}_order_parameter_{number_of_analised_frames}_frames_semilog.png")
    else:
        save_file = os.path.join(save_path, 'order_parameter', f"{os.path.basename(video_name).replace('.csv', '')}_{method.__name__.removesuffix('_local_order_parameters')}_order_parameter_{number_of_analised_frames}_frames.png")

    if save:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        if os.path.isfile(save_file) and not overwrite:
            print(f"""Warning: A file with the same name already exists: 
              {save_file}
              Change overwrite to True if you want to overwrite the file""")
        else:
            print(f"Saving file to {save_file}")
            plt.savefig(save_file, dpi=dpi)
    else:
        print("Save flag is set to False, not saving the file.")

    plt.show()



def find_equally_spaced_frame_numbers (mask_video, number_of_equally_spaced_frames):
    # Validate input parameters
    # chech if mask_video is a 3D array
    if len(mask_video.shape) != 3:
        raise ValueError("The mask_video must be a 3D numpy array")
    if not isinstance(number_of_equally_spaced_frames, int):
        raise ValueError("number_of_equally_spaced_frames must be an integer")
    if number_of_equally_spaced_frames < 1:
        raise ValueError("number_of_equally_spaced_frames must be at least 1")
    if len(mask_video) < number_of_equally_spaced_frames:
        raise ValueError("number_of_equally_spaced_frames must be less than or equal to the number of frames in the video")
    
    frame_numbers = []
    n_frames = len(mask_video)
    if number_of_equally_spaced_frames == 1:
        step_size = n_frames
    else:
        step_size = n_frames / (number_of_equally_spaced_frames - 1)
        if step_size == int(step_size):
            step_size = int(step_size)-1
        else:    
            step_size = int(step_size)


    for frameID in range(number_of_equally_spaced_frames):
        frame_number = int(frameID*step_size)
        frame_numbers.append(frame_number)

    return frame_numbers
    

def plot_all_videos_local_order_parameters(movies_masks, method, Radiuses, pixel_scale, output_directory, number_of_analised_frames_to_average=1, model_fit=False, save=False, save_csv=False, csv_path=None, dpi=300, overwrite=False):
    possible_methods = [function for function in all_functons_globals if function.endswith('_local_order_parameters')]

    if method.__name__ not in possible_methods:
        raise ValueError(f"The given method is {method.__name__} but it must be one of the following functions: {possible_methods}")
    
    if not isinstance(movies_masks, dict):
        raise ValueError("movies_masks must be a dictionary of 3D numpy arrays")
    
    for key in movies_masks:
        if len(movies_masks[key].shape) != 3:
            raise ValueError("movies_masks must contain 3D numpy arrays")

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 6)

    dict_average_local_orders = {}
    dict_largest_significant_radii = {}
    analysed_frame_numbers = {}

    if model_fit:
        A_model = {}
        B_model = {}
        C_model = {}

    for video_name in movies_masks:
        # Define the path for CSV files within the video-specific folder
        if save_csv:
            csv_path_per_video = os.path.join(csv_path, video_name,'average_local_order_over_multiple_frames')
            os.makedirs(csv_path_per_video, exist_ok=True)

        mask_video = movies_masks[video_name]
        local_orders = {}
        total_local_orders = {}
        dict_largest_significant_radii[video_name] = {}
        total_n_close_cells = {}

        X = Radiuses * pixel_scale

        analysed_frame_numbers[video_name] = find_equally_spaced_frame_numbers(mask_video, number_of_analised_frames_to_average)

        for frameID in analysed_frame_numbers[video_name]:
            mask_frame = mask_video[frameID]
            local_orders[frameID], total_local_orders[frameID], dict_largest_significant_radii[video_name][frameID], n_close_cells, total_n_close_cells[frameID] = plot_local_order_parameters(
                mask_frame, 
                method, 
                Radiuses, 
                pixel_scale, 
                output_directory,  
                video_name, 
                analysis_frame=frameID,  # Pass the frame index here
                plot=False, 
                save=False, 
                return_total=True, 
                return_average_and_total=True
            )

        dict_average_local_orders[video_name] = {}

        for radius in Radiuses:
            dict_average_local_orders[video_name][radius] = np.nanmean([local_orders[frame][radius] for frame in local_orders])
        
        if model_fit:
            ax.plot(np.NaN, np.NaN, '-', color='none', label=video_name)



        one_video_local_order = dict_average_local_orders[video_name]
        Y_local_order = list(one_video_local_order.values())
        if model_fit:
            ax.plot(X, Y_local_order, linestyle='-', marker='o', markersize=5, label='local orders', alpha=0.5)
        else:
            ax.plot(X, Y_local_order, linestyle='-', marker='o', markersize=5, label=video_name, alpha=0.5)
        
        if model_fit:
            # Remove nan values from Y, and corresponding X values, for the fit
            X_filtered, Y_filtered = [], []
            for x, y in zip(X, Y_local_order):
                if not (math.isnan(y) or math.isinf(y)):
                    X_filtered.append(x)
                    Y_filtered.append(y)

            popt, pcov = curve_fit(model, X_filtered, Y_filtered)
            Y_fit = model(X_filtered, *popt)

            plt.plot(X_filtered, Y_fit, linestyle='--', label='model fit: a = ' + str(round(popt[0], 2)) + ', b = ' + str(round(popt[1], 2)) + ', c = ' + str(round(popt[2], 2)))

            plt.legend(loc='upper right')
            A_model[video_name] = popt[0]
            B_model[video_name] = popt[1]
            C_model[video_name] = popt[2]

        if save_csv:
            collumns_metadata = ['frame', 'significant_radius', 'pixel_scale']
            metadata_total = pd.DataFrame(columns=collumns_metadata)

            for frameID in analysed_frame_numbers[video_name]:
                local_order_parameter = total_local_orders[frameID]
                significant_radius = dict_largest_significant_radii[video_name][frameID]
                n_close_cells_frame = total_n_close_cells[frameID]

                collumns_per_frame = ['radius', 'cellID', 'local_order_parameter', 'n_close_cells']
                data_per_frame = pd.DataFrame(columns=collumns_per_frame)

                for radius_pixels in local_order_parameter:
                    radius_microns = radius_pixels * pixel_scale
                    for cellID in local_order_parameter[radius_pixels]:
                        data_per_frame = pd.concat([data_per_frame, pd.DataFrame({
                            'radius': radius_microns, 
                            'cellID': cellID, 
                            'local_order_parameter': local_order_parameter[radius_pixels][cellID], 
                            'n_close_cells': n_close_cells_frame[radius_pixels][cellID]
                        }, index=[0])], ignore_index=True)

                data_per_frame.to_csv(os.path.join(csv_path_per_video, video_name + '_local_order_parameters_data_frame_' + str(frameID) + '.csv'), index=False)

                metadata_total = pd.concat([metadata_total, pd.DataFrame({
                    'frame': [frameID], 
                    'significant_radius': [significant_radius * pixel_scale], 
                    'pixel_scale': [pixel_scale]
                }, index=[0])], ignore_index=True)   
            metadata_total.to_csv(os.path.join(csv_path_per_video, video_name + '_local_order_parameters_metadata_total.csv'), index=False)
        
            collumns_average = ['radius', 'average_local_order_parameter', 'average_n_close_cells']
            data_average = pd.DataFrame(columns=collumns_average)

            for radius_pixels in dict_average_local_orders[video_name]:
                radius_microns = radius_pixels * pixel_scale
                data_average = pd.concat([data_average, pd.DataFrame({
                    'radius': radius_microns, 
                    'average_local_order_parameter': dict_average_local_orders[video_name][radius_pixels], 
                    'average_n_close_cells': n_close_cells[radius_pixels]
                }, index=[0])], ignore_index=True)

            str_analysed_frame_numbers = ', '.join(str(x) for x in analysed_frame_numbers[video_name])

            largest_significant_radius = max(dict_largest_significant_radii[video_name].values()) * pixel_scale

            data_average.to_csv(os.path.join(csv_path_per_video, video_name + '_local_order_parameters_average_over_' + str(number_of_analised_frames_to_average) + '_frames.csv'), index=False)
            if not model_fit:
                metadata_average = pd.DataFrame({'frames': [str_analysed_frame_numbers], 'largest_significant_radius': largest_significant_radius, 'pixel_scale': [pixel_scale]})
            elif model_fit:
                metadata_average = pd.DataFrame({'frames': [str_analysed_frame_numbers], 'largest_significant_radius': largest_significant_radius, 'pixel_scale': [pixel_scale], 'a_model': [A_model[video_name]], 'b_model': [B_model[video_name]], 'c_model': [C_model[video_name]]})
            metadata_average.to_csv(os.path.join(csv_path_per_video, video_name + '_local_order_parameters_metadata_average_over_' + str(number_of_analised_frames_to_average) + '_frames.csv'), index=False)

    largest_significant_radius = max([max(dict_largest_significant_radii[video_name].values()) for video_name in dict_largest_significant_radii]) * pixel_scale
    plt.axvline(x=(largest_significant_radius), color='red', linestyle='--', label='all cells have >1 neighbor')
    plt.legend(loc='upper right')
    ax.set_ylim([0, 1])
    plt.xlabel('Radius of neighboring cells (µm)')
    plt.ylabel('Local nematic order')
    plt.title(method.__name__.removesuffix('_local_order_parameters').replace('_', ' ') + ' local order parameter, ' + str(number_of_analised_frames_to_average) + ' frame average - multiple videos')  
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    
    if save:
        save_path = os.path.join(output_directory, 'average_local_order_over_multiple_frames')
        os.makedirs(save_path, exist_ok=True)
        save_file_path = os.path.join(save_path, f"multiple_videos_{method.__name__.removesuffix('_local_order_parameters')}_order_parameter_averaged_over_{number_of_analised_frames_to_average}_frames.png")
        if os.path.isfile(save_file_path) and not overwrite:
            print(f"""Warning: A file with the same name already exists: 
                  {save_file_path}
                  change overwrite to True if you want to overwrite the file""")
            plt.show()
            return dict_average_local_orders, dict_largest_significant_radii, analysed_frame_numbers
        plt.savefig(save_file_path, dpi=dpi)
    
    plt.show()
    if model_fit:
        print(f'For the model, Local order = a * Radius^(-b) + c, in microns, the fit parameters are:')
        for video_name in A_model:
            print(f'{video_name}: a = {A_model[video_name]}, b = {B_model[video_name]}, c = {C_model[video_name]}')
        return dict_average_local_orders, dict_largest_significant_radii, analysed_frame_numbers, A_model, B_model, C_model
    return dict_average_local_orders, dict_largest_significant_radii, analysed_frame_numbers


def model(X, a, b, c):
    return a * X**(-b) + c

def fit_negative_exponential_order_parameter(mask_frame, method, Radiuses, pixel_scale, output_directory, video_name, analysis_frame, semilog=False, save=False, dpi=300, overwrite=False):
    
    total_orders, significant_radius, total_n_close_cells = plot_local_order_parameters(
        mask_frame, method, Radiuses, pixel_scale, output_directory, video_name, 
        analysis_frame, plot=False, save=False, return_total=True
    )

    orders = {}
    n_close_cells = {}

    for radius in total_orders:
        orders[radius] = np.nanmean(list(total_orders[radius].values()))
        n_close_cells[radius] = np.mean(list(total_n_close_cells[radius].values()))

    X = Radiuses * pixel_scale
    Y = list(orders.values())    

    # Remove nan values from Y, and corresponding X values, for the fit
    X_filtered, Y_filtered = [], []
    for x, y in zip(X, Y):
        if not (math.isnan(y) or math.isinf(y)):
            X_filtered.append(x)
            Y_filtered.append(y)
    X, Y = X_filtered, Y_filtered

    n_cells = list(n_close_cells.values())

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 6)
    plt.plot(X, Y, linestyle='-', marker='o', color='black', markersize=5, label='Local order parameter', alpha=0.5)
    plt.axvline(x=significant_radius * pixel_scale, color='red', linestyle='--', label='all cells have >1 neighbor')
    plt.plot([], [], linestyle='', marker='o', color='blue', label='Mean number of neighbors')  # for the legend
    
    # Add the average number of cells within the radius above each point
    for x, y, n in zip(X, Y, n_cells):
        label = "{:.1f}".format(n)
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', color='blue', fontsize=8) 

    # Legend
    plt.legend(loc='upper right')
    ax.set_ylim([0, 1])
    plt.xlabel('Radius of neighboring cells (µm)')
    plt.ylabel('Local nematic order')
    plt.title(method.__name__.removesuffix('_local_order_parameters').replace('_', ' ') + ' local order parameter, single frame - ' + video_name)  
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    # Fit a model that has the form Y = a * X^(-b) + c
    popt, pcov = curve_fit(model, X, Y)
    Y_fit = model(X, *popt)
    plt.plot(X, Y_fit, linestyle='--', color='black', label='Fit: a * Radius^(-b) + c, a = ' + str(round(popt[0], 2)) + ', b = ' + str(round(popt[1], 2)) + ', c = ' + str(round(popt[2], 2)))
    plt.legend(loc='upper right')
    a = popt[0]
    b = popt[1]
    c = popt[2]

    save_path = os.path.join(output_directory, video_name, 'order_parameter')
    if semilog:
        ax.set_xscale('log')
        plt.title('semilog ' + method.__name__.removesuffix('_local_order_parameters').replace('_', ' ') + ' local order parameter, single frame - ' + video_name)  
        save_file = os.path.join(save_path, f"{os.path.basename(video_name).replace('.csv', '')}_{method.__name__.removesuffix('_local_order_parameters')}_order_parameter_model_fit_semilog.png")
    else:
        save_file = os.path.join(save_path, f"{os.path.basename(video_name).replace('.csv', '')}_{method.__name__.removesuffix('_local_order_parameters')}_order_parameter_model_fit.png")

    if save:
        os.makedirs(save_path, exist_ok=True)
        if os.path.isfile(save_file) and not overwrite:
            print(f"""Warning: A file with the same name already exists: 
                  {save_file}
                  Change overwrite to True if you want to overwrite the file""")
        else:
            print(f"Saving file to {save_file}")
            plt.savefig(save_file, dpi=dpi)

    plt.show()
    
    print(f'For the model, Local order = a * Radius^(-b) + c, in microns, the fit parameters are a = {a}, b = {b}, c = {c}')
    
    return total_orders, significant_radius, total_n_close_cells, a, b, c

##Utility and Data functions


#this function was used for testing
def plot_cell_coords_on_mask(mask, cellID):
    """
    Plot all the coordinates of a cell on a mask.

    Parameters
    ----------
    mask : numpy array
        A 2D array representing the mask, where the background has a value of 0 and each cell has a different gray value.
    coords : numpy array
        A 2D array representing the coordinates of the cell.
    """
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    coords = [props[cellID].coords][0]
    x = coords[:, 0]
    y = coords[:, 1]

    plt.imshow(mask, cmap='gray')
    plt.scatter(y,x, color='red', s=1)
    plt.show()


#I need to update docstring
def get_and_save_info(images, outputDirectory, CSVname):
    """
    Calculate various metrics for each video in a collection of segmented videos, save the results to a CSV file, and return a DataFrame of the results.

    This function calculates the number of cells, 
    average  and standard deviation of cell surface area, 
    cell coverage, 
    average and standard deviation of cell length,
    mean standard deviation of aspect ratio (ie. long axis/short axis of cell)
    for each video. 
    
    For some of these values, only the first frame of the video is used.

    The results are saved to a CSV file and also returned as a pandas DataFrame.

    Parameters
    ----------
    images : dict
        A dictionary where the keys are video names and the values are 3D numpy arrays representing the videos.
    parameters : object
        An object that has an attribute `outputdir` which is a string representing the directory where the CSV file should be saved.
    CSVname : str
        The name of the CSV file to be created.

    Returns
    -------
    dataframe : pandas.DataFrame
        A DataFrame where each row represents a video and each column represents a calculated metric for that video.
    """
    data = []
    for video_name in images:
        video = images[video_name]
        first_frame = video[0]

        numberOfCells = len(get_cellIDs(first_frame))

        surfaces = calculate_bacteria_surface_areas(first_frame)
        average_surface_area = np.mean(list(surfaces.values()))
        SD_surface_area = np.std(list(surfaces.values()))

        cell_coverages = cell_coverage(first_frame)
        cell_lengths = get_cell_lengths(first_frame)
        average_length = np.mean(list(cell_lengths.values()))
        SD_length = np.std(list(cell_lengths.values()))
        aspectRatos = calculate_aspect_ratios(video, 0)
        mean_aspect_ratio = np.mean(list(aspectRatos.values()))
        SD_aspect_ratio = np.std(list(aspectRatos.values()))

        data.append([video_name, numberOfCells, average_surface_area, SD_surface_area, cell_coverages, average_length, SD_length, mean_aspect_ratio, SD_aspect_ratio])

    dataframe = pd.DataFrame(data, columns=['movie','numberOfCells','average_surface_area [pixels]', 'SD_surface_area', 'cell_coverages', 'average_length[pixels]', 'SD_length', 'mean_aspect_ratio', 'SD_aspect_ratio'])
    dataframe.to_csv(os.path.join(outputDirectory, CSVname) + '.csv', index=False)
    return dataframe





def calculate_intensities(mask_frame, raw_frame):
    """
    Calculate the minimum, maximum, and mean intensity for each unique cell ID in the mask_frame using the raw_frame.

    Parameters:
    - mask_frame: 2D numpy array with segmented mask of cells.
    - raw_frame: 2D numpy array representing the corresponding raw image data.

    Returns:
    - Dictionary of cell IDs to their min, max, and mean intensities.
    """
    unique_ids = np.unique(mask_frame)
    intensities = {}
    for cell_id in unique_ids:
        if cell_id == 0:  # Assuming ID 0 is background
            continue
        cell_pixels = raw_frame[mask_frame == cell_id]
        intensities[cell_id] = {
            'min_intensity': np.min(cell_pixels),
            'max_intensity': np.max(cell_pixels),
            'mean_intensity': np.mean(cell_pixels)
        }
    return intensities

def calculate_perimeter(labeled_mask):
    return {prop.label: prop.perimeter for prop in regionprops(labeled_mask)}

def calculate_convex_hull_area(labeled_mask):
    return {prop.label: prop.convex_area for prop in regionprops(labeled_mask)}

def calculate_solidity(labeled_mask):
    return {prop.label: prop.solidity for prop in regionprops(labeled_mask)}

def fit_ellipse(mask_frame):
    """
    Fits an ellipse to each segment in the mask frame and calculates the ellipse parameters.

    Parameters
    ----------
    mask_frame : numpy array
        A 2D array representing the mask, where the background has a value of 0 and each cell has a different gray value.

    Returns
    -------
    ellipses : dict
        A dictionary where the keys are the cell IDs and the values are dictionaries containing ellipse parameters:
            - 'Center_X': X-coordinate of the center of the ellipse.
            - 'Center_Y': Y-coordinate of the center of the ellipse.
            - 'Major_Axis': Length of the major axis of the ellipse.
            - 'Minor_Axis': Length of the minor axis of the ellipse.
            - 'Angle': Rotation angle of the ellipse in degrees, counter-clockwise from the x-axis.
            - 'Aspect_Ratio': Ratio of the lengths of the major axis to the minor axis.

    """
    ellipses = {}
    regions = regionprops(mask_frame)
    for region in regions:
        if region.area >= 5:  # Ignore very small regions that cannot fit an ellipse
            cell_id = region.label
            coords = region.coords
            if len(coords) >= 5:  # OpenCV requires at least 5 points to fit an ellipse
                ellipse = cv2.fitEllipse(np.array(coords))
                (center_y, center_x), (minor_axis, major_axis), angle = ellipse
                
                # Adjust angle to be within [0, 180)
                if angle < 0:
                    angle += 180
                if angle >= 180:
                    angle -= 180
                
                aspect_ratio = major_axis / minor_axis

                ellipses[cell_id] = {
                    'Center_X': center_x,
                    'Center_Y': center_y,
                    'Major_Axis': major_axis,
                    'Minor_Axis': minor_axis,
                    'Angle': angle,
                    'Aspect_Ratio': aspect_ratio
                }
    return ellipses

def round_values(value, decimal_places):
    if isinstance(value, (int, float)):
        return round(value, decimal_places)
    return value



def get_and_save_segments_info_and_stats(movies_masks, movies_raw, outputDirectory, frame_number="first", decimal_places=2):
    """
    Processes frames from a set of video masks and raw frames to extract cell information and save data to CSV files.

    This function processes frames from the provided dictionaries of masks and raw frames. By default, it processes
    the first frame of each video. It extracts various metrics related to cell properties and saves the data into CSV files
    in the specified output directory. It also calculates and saves summary statistics (mean, median, and standard deviation)
    for selected metrics.

    Parameters:
    ----------
    movies_masks : dict
        A dictionary where the keys are video names and the values are lists of 2D numpy arrays representing mask frames.
        Each mask frame is a 2D array where the background is represented by 0 and each distinct cell is represented by a unique gray value.
        
    movies_raw : dict
        A dictionary where the keys are video names and the values are lists of 2D numpy arrays representing raw frames corresponding to the mask frames.
        
    outputDirectory : str
        The directory where the resulting CSV files will be saved.
        
    frame_number : str or int or list, optional
        Option to select which frame(s) to process. Possible values are:
        - "first": Process the first frame (default).
        - "highest": Process the frame with the highest number of bacteria (cells).
        - "lowest": Process the frame with the lowest number of bacteria (cells).
        - "all": Process all frames in each movie.
        - An integer to specify a particular frame index.
        - A list of integers to specify particular frame indices.
        
    decimal_places : int, optional
        The number of decimal places to keep in the results. Default is 2.
        
    Saves:
    -----
    CSV files containing detailed cell information and summary statistics in the specified output directory.
    """
    stats_columns = ['Perimeter_PIX', 'Surface_area_PIX^2', 'Convex_hull_area_PIX^2', 
                     'Solidity', 'Length_PIX', 'Ellipse_major_axis_PIX', 'Width_PIX', 
                     'Ellipse_minor_axis_PIX', 'Aspect_ratio', 'Ellipse_aspect_ratio', 
                     'Mean_intensity']

    for video_name in movies_masks:
        if video_name not in movies_raw:
            print(f"Skipping {video_name} as it does not exist in raw data.")
            continue

        frames_to_process = []

        # Determine the frame(s) to process based on frame_number
        if frame_number == "highest":
            max_cell_count = 0
            selected_frame_index = 0
            for i, mask_frame in enumerate(movies_masks[video_name]):
                cell_count = len(np.unique(mask_frame)) - 1  # Subtract one for background
                if cell_count > max_cell_count:
                    max_cell_count = cell_count
                    selected_frame_index = i
            frames_to_process = [selected_frame_index]
        elif frame_number == "lowest":
            min_cell_count = float('inf')
            selected_frame_index = 0
            for i, mask_frame in enumerate(movies_masks[video_name]):
                cell_count = len(np.unique(mask_frame)) - 1  # Subtract one for background
                if cell_count < min_cell_count:
                    min_cell_count = cell_count
                    selected_frame_index = i
            frames_to_process = [selected_frame_index]
        elif frame_number == "first":
            frames_to_process = [0]
        elif frame_number == "all":
            frames_to_process = list(range(len(movies_masks[video_name])))
        elif isinstance(frame_number, int):
            if frame_number < len(movies_masks[video_name]):
                frames_to_process = [frame_number]
            else:
                print(f"Specified frame index {frame_number} for {video_name} is out of bounds. Skipping this frame.")
                continue
        elif isinstance(frame_number, list):
            frames_to_process = sorted([i for i in frame_number if i < len(movies_masks[video_name])])
            if not frames_to_process:
                print(f"Specified frame indices {frame_number} for {video_name} are out of bounds. Skipping this set of frames.")
                continue
        else:
            frames_to_process = [0]

        # Debugging: print frames to process
        print(f"Processing frames for {video_name}: {frames_to_process}")

        # Create a directory for the video if it doesn't exist
        video_dir = os.path.join(outputDirectory, video_name)
        os.makedirs(video_dir, exist_ok=True)

        # Determine the frame label based on the processed frames
        if len(frames_to_process) == 1:
            frame_label = f"{frames_to_process[0]}"
        else:
            ranges = []
            start = frames_to_process[0]
            end = frames_to_process[0]

            for i in range(1, len(frames_to_process)):
                if frames_to_process[i] == end + 1:
                    end = frames_to_process[i]
                else:
                    if start == end:
                        ranges.append(f"{start}")
                    else:
                        ranges.append(f"{start}-{end}")
                    start = frames_to_process[i]
                    end = frames_to_process[i]

            if start == end:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{end}")

            frame_label = "_".join(ranges)

        # Check if CSV file already exists
        _csv_filename = os.path.join(video_dir, f"{video_name.replace('.tif', '')}_frame_{frame_label}.csv")
        combined_stats_csv_filename = os.path.join(video_dir, f"{video_name.replace('.tif', '')}_frame_{frame_label}_stats.csv")
        
        if os.path.exists(_csv_filename) and os.path.exists(combined_stats_csv_filename):
            print(f"Skipping {video_name} as the CSV files already exist.")
            continue

        all_data = []

        # Add progress bar for frame processing
        for selected_frame_index in tqdm(frames_to_process, desc="Processing", unit="frame", ascii=False):
            mask_frame = movies_masks[video_name][selected_frame_index]
            raw_frame = movies_raw[video_name][selected_frame_index]

            # Compute various cell metrics
            IDs = get_cellIDs(mask_frame)
            centers = find_cell_centers(mask_frame)
            surfaces = calculate_bacteria_surface_areas(mask_frame)
            long_extremities = find_extremities_of_cells_length(mask_frame)
            short_extremities = find_extremities_of_cells_width(mask_frame, long_extremities)
            lengths = get_cell_lengths(mask_frame, long_extremities)
            widths = get_cell_widths(mask_frame, long_extremities, short_extremities)
            aspect_ratios = calculate_aspect_ratios(mask_frame, long_extremities, short_extremities)
            directions_rad = get_directions_radians(mask_frame, long_extremities)
            directions_deg = get_directions_degrees(mask_frame, long_extremities)
            intensities = calculate_intensities(mask_frame, raw_frame)
            perimeters = calculate_perimeter(mask_frame)
            convex_hull_areas = calculate_convex_hull_area(mask_frame)
            solidity_values = calculate_solidity(mask_frame)
            ellipses = fit_ellipse(mask_frame)

            # Creating DataFrame with pixel measurements
            data_per_cell = pd.DataFrame({
                'IDs': IDs,
                'Frame': selected_frame_index,
                'Center_position_x_PIX': [round_values(centers.get(ID, [np.nan])[1], decimal_places) for ID in IDs],
                'Ellipse_center_x_PIX': [round_values(ellipses[ID]['Center_X'], decimal_places) if ID in ellipses else np.nan for ID in IDs],
                'Center_position_y_PIX': [round_values(centers.get(ID, [np.nan])[0], decimal_places) for ID in IDs],
                'Ellipse_center_y_PIX': [round_values(ellipses[ID]['Center_Y'], decimal_places) if ID in ellipses else np.nan for ID in IDs],
                'Perimeter_PIX': [round_values(perimeters.get(ID, np.nan), decimal_places) for ID in IDs],
                'Surface_area_PIX^2': [round_values(surfaces.get(ID, np.nan), decimal_places) for ID in IDs],
                'Convex_hull_area_PIX^2': [round_values(convex_hull_areas.get(ID, np.nan), decimal_places) for ID in IDs],
                'Solidity': [round_values(solidity_values.get(ID, np.nan), decimal_places) for ID in IDs],
                'Length_PIX': [round_values(lengths.get(ID, np.nan), decimal_places) for ID in IDs],
                'Ellipse_major_axis_PIX': [round_values(ellipses[ID]['Major_Axis'], decimal_places) if ID in ellipses else np.nan for ID in IDs],
                'Width_PIX': [round_values(widths.get(ID, np.nan), decimal_places) for ID in IDs],
                'Ellipse_minor_axis_PIX': [round_values(ellipses[ID]['Minor_Axis'], decimal_places) if ID in ellipses else np.nan for ID in IDs],
                'Aspect_ratio': [round_values(aspect_ratios.get(ID, np.nan), decimal_places) for ID in IDs],
                'Ellipse_aspect_ratio': [round_values(ellipses[ID]['Aspect_Ratio'], decimal_places) if ID in ellipses else np.nan for ID in IDs],
                'Angle_to_x+_axis_RAD': [round_values(directions_rad.get(ID, np.nan), decimal_places) for ID in IDs],
                'Angle_to_x+_axis_DEG': [round_values(directions_deg.get(ID, np.nan), decimal_places) for ID in IDs],
                'Ellipse_Angle_DEG': [round_values(ellipses[ID]['Angle'], decimal_places) if ID in ellipses else np.nan for ID in IDs],
                'Min_intensity': [round_values(intensities[ID]['min_intensity'], decimal_places) for ID in IDs],
                'Max_intensity': [round_values(intensities[ID]['max_intensity'], decimal_places) for ID in IDs],
                'Mean_intensity': [round_values(intensities[ID]['mean_intensity'], decimal_places) for ID in IDs]
            })

            all_data.append(data_per_cell)

        # Combine data from all frames and save
        combined_data = pd.concat(all_data, ignore_index=True)

        combined_data.to_csv(_csv_filename, index=False)
        print(f"Processed and saved data for {video_name} in {_csv_filename}.")

        # Calculate and save summary statistics for selected columns
        summary_stats = combined_data[stats_columns].agg(['median', 'mean', 'std']).rename(index={'median': 'Median', 'mean': 'Mean', 'std': 'Standard Deviation'}).T
        summary_stats.insert(0, 'Feature', summary_stats.index)
        summary_stats = summary_stats.round(decimal_places)
        
        summary_stats.to_csv(combined_stats_csv_filename, index=False)
        print(f"Summary statistics processed and saved for {video_name} in {combined_stats_csv_filename}.")




def convert_pixels_to_microns(inputDirectory, outputDirectory, pixel_scale=0.1625, decimal_places=2):
    """
    Converts pixel measurements in CSV files to microns and saves the results in new CSV files.

    This function loads CSV files containing cell measurements in pixels from the specified input directory, converts the relevant
    measurements to microns using the provided pixel scale, and saves the converted data to new CSV files in the specified output directory.
    It also calculates and saves summary statistics (mean, median, and standard deviation) for the converted measurements.

    Parameters:
    ----------
    inputDirectory : str
        The directory containing the CSV files with pixel measurements.
        
    outputDirectory : str
        The directory where the resulting CSV files with micron measurements will be saved.
        
    pixel_scale : float, optional
        The scale to convert pixels to microns. Default is 0.1625 microns/pixel.
        
    decimal_places : int, optional
        The number of decimal places to keep in the results. Default is 2.
        
    Saves:
    -----
    CSV files containing detailed cell information and summary statistics with measurements in microns in the specified output directory.
    """
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    for subdir in os.listdir(inputDirectory):
        subdir_path = os.path.join(inputDirectory, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith(".csv") and not filename.endswith("_stats.csv") and not filename.endswith("_microns.csv"):
                    input_path = os.path.join(subdir_path, filename)

                    # Create the corresponding subdirectory in the output directory
                    video_dir = os.path.join(outputDirectory, subdir)
                    os.makedirs(video_dir, exist_ok=True)

                    # Extract the frame label from the filename
                    frame_label = filename.split('_frames_')[-1].replace('.csv', '')

                    csv_filename_microns = os.path.join(video_dir, filename.replace(".csv", "_microns.csv"))
                    stats_csv_filename_microns = os.path.join(video_dir, filename.replace(".csv", "_microns_stats.csv"))

                    if os.path.exists(csv_filename_microns) and os.path.exists(stats_csv_filename_microns):
                        print(f"Found existing micron data and stats files for {filename}. Skipping creation.")
                        continue

                    data_per_cell = pd.read_csv(input_path)

                    # Create a new DataFrame with micron measurements
                    data_per_cell_microns = data_per_cell.copy()
                    data_per_cell_microns.columns = data_per_cell_microns.columns.str.replace('_PIX', '_MICRON')
                    data_per_cell_microns['Center_position_x_MICRON'] = data_per_cell_microns['Center_position_x_MICRON'] * pixel_scale
                    data_per_cell_microns['Ellipse_center_x_MICRON'] = data_per_cell_microns['Ellipse_center_x_MICRON'] * pixel_scale
                    data_per_cell_microns['Center_position_y_MICRON'] = data_per_cell_microns['Center_position_y_MICRON'] * pixel_scale
                    data_per_cell_microns['Ellipse_center_y_MICRON'] = data_per_cell_microns['Ellipse_center_y_MICRON'] * pixel_scale
                    data_per_cell_microns['Perimeter_MICRON'] = data_per_cell_microns['Perimeter_MICRON'] * pixel_scale
                    data_per_cell_microns['Surface_area_MICRON^2'] = data_per_cell_microns['Surface_area_MICRON^2'] * (pixel_scale ** 2)
                    data_per_cell_microns['Convex_hull_area_MICRON^2'] = data_per_cell_microns['Convex_hull_area_MICRON^2'] * (pixel_scale ** 2)
                    data_per_cell_microns['Length_MICRON'] = data_per_cell_microns['Length_MICRON'] * pixel_scale
                    data_per_cell_microns['Ellipse_major_axis_MICRON'] = data_per_cell_microns['Ellipse_major_axis_MICRON'] * pixel_scale
                    data_per_cell_microns['Width_MICRON'] = data_per_cell_microns['Width_MICRON'] * pixel_scale
                    data_per_cell_microns['Ellipse_minor_axis_MICRON'] = data_per_cell_microns['Ellipse_minor_axis_MICRON'] * pixel_scale

                    # Ensure all values are rounded
                    data_per_cell_microns = data_per_cell_microns.apply(lambda col: col.map(lambda x: round_values(x, decimal_places)))

                    # Save micron data to CSV
                    data_per_cell_microns.to_csv(csv_filename_microns, index=False)
                    print(f"Converted and saved micron data for {filename} in {csv_filename_microns}.")

                    # Calculate and save summary statistics for micron data
                    stats_columns = ['Perimeter_MICRON', 'Surface_area_MICRON^2', 'Convex_hull_area_MICRON^2', 
                                     'Solidity', 'Length_MICRON', 'Ellipse_major_axis_MICRON', 'Width_MICRON', 'Ellipse_minor_axis_MICRON', 'Aspect_ratio', 'Ellipse_aspect_ratio', 'Mean_intensity']

                    summary_stats_microns = data_per_cell_microns[stats_columns].agg(['median', 'mean', 'std']).rename(index={'median': 'Median', 'mean': 'Mean', 'std': 'Standard Deviation'}).T
                    summary_stats_microns.insert(0, 'Feature', summary_stats_microns.index)
                    summary_stats_microns = summary_stats_microns.round(decimal_places)
                    summary_stats_microns.to_csv(stats_csv_filename_microns, index=False)
                    print(f"Summary statistics processed and saved for {filename} in {stats_csv_filename_microns}.")
                    
                    




def load_and_display_dataframes(outputDirectory, data_type="pixels"):
    """
    Loads and displays the content of CSV files and their corresponding statistics files from the specified directory
    and its immediate subdirectories for further analysis. It expects files named in the pattern "<video_name>_frame_<frame_index>.csv" 
    for data and "<video_name>_frame_<frame_index>_stats.csv" for statistics. Optionally, it can also load files with measurements in microns.
    
    Returns:
    - List of tuples, each containing the data frame, its corresponding statistics data frame, and the source filename.

    Parameters:
    - outputDirectory: Directory from which to load the CSV files.
    - data_type: Type of data to load, either "pixels" for pixels measurements or "microns" for microns measurements. Default is "pixels".
    """
    data_frames = []
    suffix = "_microns" if data_type == "microns" else ""
    
    # List all entries in the outputDirectory
    for root, dirs, files in os.walk(outputDirectory):
        # If we are in the root directory, process the subdirectories but don't go deeper
        if root == outputDirectory:
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                for filename in os.listdir(subdir_path):
                    if filename.endswith(".csv") and not filename.endswith("_stats.csv"):
                        if (data_type == "microns" and "_microns" in filename) or (data_type == "pixels" and "_microns" not in filename):
                            # Load and display the data file
                            data_file_path = os.path.join(subdir_path, filename)
                            data_df = pd.read_csv(data_file_path)
                            print(f"Data from {data_file_path}:")
                            display(data_df)  # Assuming use in a Jupyter notebook context or similar
                            
                            # Construct the stats file name and check if it exists
                            stats_file_name = filename.replace(".csv", "_stats.csv")
                            stats_file_path = os.path.join(subdir_path, stats_file_name)
                            if os.path.exists(stats_file_path):
                                stats_df = pd.read_csv(stats_file_path)
                                print(f"Statistics from {stats_file_path}:")
                                display(stats_df)  # Assuming use in a Jupyter notebook context or similar
                                data_frames.append((data_df, stats_df, filename))
                            else:
                                # Append data frame without statistics
                                data_frames.append((data_df, None, filename))
        # Skip deeper directories
        break
    
    return data_frames
    
    
def list_column_names(data_frames):
    """
    Lists the column names from the data frames.

    Parameters:
    - data_frames: List of tuples, each containing the data frame, its corresponding statistics data frame, and the source filename.

    Returns:
    - A list of column names from the data frames.
    """
    if not data_frames:
        print("No data frames loaded.")
        return []

    # Get the column names from the first data frame
    column_names = data_frames[0][0].columns.tolist()
    
    print("Available column names:")
    for column in column_names:
        print(column)
    
    return column_names


def plot_distribution(data_frames, column_name, save_path, frame_number='all', save_plots=False, dpi=300):
    """
    Plots the distribution of values in a specified column for each DataFrame provided.
    Includes provenance information in the plot, number of data points in the legend, and optionally saves the plot to a specified directory.
    Additionally, plots the mean, median, and standard deviation if stats DataFrame is available.

    Parameters:
    - data_frames: List of tuples (DataFrame, stats DataFrame, filename) as returned by `load_and_display_data`.
    - column_name: Name of the column to plot.
    - frame_number: The frame to plot for each DataFrame, or 'all' to plot the entire DataFrame. Defaults to 'all'.
    - save_plots: Boolean, if True saves the plots to the specified directory.
    - save_path: String, path to save the plots if save_plots is True.
    - dpi: Integer, the resolution in dots per inch of the saved plot. Default is 300.
    """
    if save_plots and not os.path.exists(save_path):
        os.makedirs(save_path)  # Create the directory if it does not exist

    for data_df, stats_df, source in data_frames:
        if data_df is not None:
            if frame_number == 'all':
                # Use the entire DataFrame
                filtered_df = data_df
            else:
                # Filter the DataFrame by the frame_number, default to the first frame if not provided
                if frame_number is None:
                    frame_number = data_df['Frame'].iloc[0]
                filtered_df = data_df[data_df['Frame'] == frame_number]
                if filtered_df.empty:
                    print(f"Skipping plot for {source}: frame {frame_number} not found.")
                    continue
            
            if column_name not in filtered_df.columns:
                print(f"Error: Column '{column_name}' not found in data from {source}.")
                continue

            # Extract the base name before the last "_frame"
            movie_base_name = source.rsplit('_frame', 1)[0]

            sns.set_style("whitegrid")
            plt.figure(figsize=(12, 6), dpi=dpi)
            data_count = filtered_df[column_name].count()  # Count non-NA/null entries.
            sns.histplot(data=filtered_df, x=column_name, kde=False, edgecolor='black', stat='density', label=f'Data points: {data_count}')

            legend_labels = [f'Data points: {data_count}']

            # Retrieve and plot statistics if available
            if stats_df is not None and column_name in stats_df['Feature'].values:
                stats_row = stats_df[stats_df['Feature'] == column_name].iloc[0]
                mean = stats_row['Mean']
                median = stats_row['Median']
                std_dev = stats_row['Standard Deviation']
                plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.3f}')
                plt.axvline(median, color='green', linestyle='-', label=f'Median: {median:.3f}')
                plt.axvline(mean - std_dev, color='purple', linestyle=':', label=f'Mean - Std Dev: {mean - std_dev:.3f}')
                plt.axvline(mean + std_dev, color='purple', linestyle=':', label=f'Mean + Std Dev: {mean + std_dev:.3f}')
                legend_labels.extend([f'Mean: {mean:.3f}', f'Median: {median:.3f}', f'Mean - Std Dev: {mean - std_dev:.3f}', f'Mean + Std Dev: {mean + std_dev:.3f}'])

            plt.title(f'Distribution of {column_name} - {source}', fontsize=16, fontweight='bold')
            plt.xlabel(column_name, fontsize=14)
            plt.ylabel('Density', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            # Only add legend if there are labels
            if legend_labels:
                plt.legend(fontsize='small', fancybox=True, framealpha=0.5)
            plt.tight_layout()

            # Save the plot if requested
            if save_plots:
                movie_save_path = os.path.join(save_path, movie_base_name, 'distributions')
                os.makedirs(movie_save_path, exist_ok=True)

                # Add the frame number to the plot's filename if provided
                if frame_number == 'all':
                    frame_suffix = "_all_frames"
                else:
                    frame_suffix = f"_frame_{frame_number}"
                plot_filename = os.path.join(movie_save_path, f"{os.path.basename(source).replace('.csv', '')}_{column_name}{frame_suffix}.png")
                plt.savefig(plot_filename, dpi=dpi)

            plt.show()



def get_long_axis_points(mask_frame, num_points=10):
    """
    Extracts points along the long axis of each cell in the mask, including the center of mass.

    Parameters:
    mask_frame (numpy array): Binary mask image with labeled regions.
    num_points (int): Total number of points to extract along the long axis (including extremities and center).

    Returns:
    numpy array: Array of all long axis points.
    dict: Dictionary mapping cell labels to their long axis points.
    """
    cell_extremities = find_extremities_of_cells_length(mask_frame)
    all_points = []
    points_per_cell = {}

    for cell_id, (ext1, ext2) in cell_extremities.items():
        # Get the cell mask
        cell_mask = (mask_frame == cell_id)
        
        # Calculate center of mass
        center = ndimage.center_of_mass(cell_mask)
        center = tuple(map(int, center))  # Convert to integer coordinates

        # Calculate the number of points for each segment
        points_per_segment = (num_points - 1) // 2  # -1 to account for the center point

        # Generate points along the line between ext1 and center
        line_points1 = np.linspace(ext1, center, num=points_per_segment + 1)[:-1]  # Exclude the last point (center)
        
        # Generate points along the line between center and ext2
        line_points2 = np.linspace(center, ext2, num=points_per_segment + 1)  # Include both center and ext2
        
        # Combine all points
        line_points = np.vstack((line_points1, line_points2))
        line_points = np.round(line_points).astype(int)

        all_points.extend(line_points)
        points_per_cell[cell_id] = line_points

    return np.array(all_points), points_per_cell

    

def add_border_points(all_points, shape, distance=10, num_points=10):
    """
    Adds border points around the image to the set of contour points.

    Parameters:
    all_points (numpy array): Array of contour points.
    shape (tuple): Shape of the image.
    distance (int): Distance from the image border to place the points.
    num_points (int): Number of border points per side.

    Returns:
    numpy array: Array of all points including border points.
    numpy array: Array of border points.
    """
    max_y, max_x = shape
    border_points = []

    x_points = np.linspace(-distance, max_x + distance, num_points)
    y_points = np.linspace(-distance, max_y + distance, num_points)

    border_points.extend([(-distance, x) for x in x_points])
    border_points.extend([(max_y + distance, x) for x in x_points])

    border_points.extend([(y, -distance) for y in y_points])
    border_points.extend([(y, max_x + distance) for y in y_points])

    border_points = np.array(border_points)
    return np.vstack([all_points, border_points]), border_points

def set_voronoi_tessellation_contours(mask_frame, num_points=10, distance=10, num_border_points=10):
    """
    Generates Voronoi tessellation from long axis points.

    Parameters:
    mask_frame (numpy array): Binary mask image with labeled regions.
    num_points (int): Number of points to extract per contour.
    distance (int): Distance from the image border to place the points.
    num_border_points (int): Number of border points per side.

    Returns:
    Voronoi: Voronoi tessellation 
    object.
    dict: Dictionary mapping cell labels to their contour points.
    numpy array: Array of border points.
    """
    all_points, points_per_cell = get_long_axis_points(mask_frame, num_points=num_points)
    all_points_flipped = np.flip(all_points, axis=1)
    all_points_flipped, border_points = add_border_points(all_points_flipped, mask_frame.shape, distance, num_border_points)
    vor = Voronoi(all_points_flipped)
    return vor, points_per_cell, border_points
    

def calculate_voronoi_areas(vor, points_per_cell, image_shape):
    """
    Calculates the areas of the Voronoi cells and identifies edge cells.

    Parameters:
    vor (Voronoi): Voronoi tessellation object.
    points_per_cell (dict): Dictionary mapping cell labels to their contour points.
    image_shape (tuple): Shape of the image (height, width).

    Returns:
    dict: Dictionary mapping cell labels to their areas.
    set: Set of cell IDs that are on the edge of the image.
    """
    cell_areas = {}
    edge_cells = set()
    point_to_region = {tuple(point): idx for idx, point in enumerate(vor.points)}

    for cell_id, points in points_per_cell.items():
        cell_area = 0
        is_edge_cell = False
        for point in points:
            flipped_point = tuple(point[::-1])
            if flipped_point in point_to_region:
                region_idx = vor.point_region[point_to_region[flipped_point]]
                region = vor.regions[region_idx]
                if -1 in region or any(vor.vertices[i][0] <= 0 or vor.vertices[i][0] >= image_shape[1] or 
                                       vor.vertices[i][1] <= 0 or vor.vertices[i][1] >= image_shape[0] 
                                       for i in region if i != -1):
                    is_edge_cell = True
                if not -1 in region and len(region) > 0:
                    polygon = [vor.vertices[i] for i in region]
                    polygon = np.array(polygon)
                    if len(polygon) > 2:
                        area = 0.5 * np.abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))
                        cell_area += area
        
        cell_areas[cell_id] = cell_area
        if is_edge_cell:
            edge_cells.add(cell_id)

    return cell_areas, edge_cells
    
def plot_voronoi_lines(ax, vor, points_per_cell, line_width=1, show_all_cells=True):
    """
    Plots Voronoi lines.

    Parameters:
    ax (matplotlib axis): Axis to plot on.
    vor (Voronoi): Voronoi tessellation object.
    points_per_cell (dict): Dictionary mapping cell labels to their contour points.
    line_width (int): Line width for the Voronoi lines.
    show_all_cells (bool): If True, show all cells. If False, show only non-edge cells.

    Returns:
    None
    """
    point_to_cell = {}
    for cell_id, points in points_per_cell.items():
        for point in points:
            flipped_point = point[::-1]
            point_to_cell[tuple(flipped_point)] = cell_id
    
    for ridge_points in vor.ridge_points:
        point1, point2 = tuple(vor.points[ridge_points[0]]), tuple(vor.points[ridge_points[1]])
        cell1, cell2 = point_to_cell.get(point1), point_to_cell.get(point2)
        if show_all_cells or (cell1 is not None and cell2 is not None):
            if cell1 != cell2:
                ridge_index = vor.ridge_points.tolist().index(ridge_points.tolist())
                line_points = vor.vertices[vor.ridge_vertices[ridge_index]]
                if -1 not in vor.ridge_vertices[ridge_index]:
                    ax.plot(line_points[:, 0], line_points[:, 1], 'r-', linewidth=line_width)
                else:
                    finite_points = [vor.vertices[i] for i in vor.ridge_vertices[ridge_index] if i != -1]
                    if len(finite_points) > 1:
                        finite_points = np.array(finite_points)
                        direction = finite_points[-1] - finite_points[0]
                        far_point = finite_points[-1] + direction * 1000
                        ax.plot([finite_points[0][0], far_point[0]], [finite_points[0][1], far_point[1]], 'r-', linewidth=line_width)
                        


def plot_voronoi_colored(ax, vor, points_per_cell, raw_frame, cell_areas, line_width=4, show_all_cells=True, custom_min=None, custom_max=None):
    """
    Plots Voronoi cells colored by their areas.
    Parameters:
    ax (matplotlib axis): Axis to plot on.
    vor (Voronoi): Voronoi tessellation object.
    points_per_cell (dict): Dictionary mapping cell labels to their contour points.
    raw_frame (numpy array): Original raw frame image.
    cell_areas (dict): Dictionary mapping cell labels to their areas.
    line_width (int): Line width for the Voronoi lines.
    show_all_cells (bool): If True, show all cells. If False, show only non-edge cells.
    custom_min (float): Custom minimum value for colorbar normalization.
    custom_max (float): Custom maximum value for colorbar normalization.
    Returns:
    None
    """
    from matplotlib.collections import PolyCollection
    # Normalize the cell areas for coloring
    if custom_min is not None and custom_max is not None:
        norm = Normalize(vmin=custom_min, vmax=custom_max)
    else:
        norm = Normalize(vmin=min(cell_areas.values()), vmax=max(cell_areas.values()))
    cmap = plt.get_cmap('plasma')
    
    # Display the raw frame in the background
    ax.imshow(raw_frame, cmap='gray')
    ax.grid(False)  # Turn off the grid
    # Dictionary mapping points to region indices
    point_to_region = {tuple(point): idx for idx, point in enumerate(vor.points)}
    # Prepare polygons for Voronoi cells
    polygons = []
    colors = []
    
    for cell_id, area in cell_areas.items():
        if show_all_cells or cell_id in cell_areas:
            for point in points_per_cell[cell_id]:
                flipped_point = tuple(point[::-1])
                if flipped_point in point_to_region:
                    region_idx = vor.point_region[point_to_region[flipped_point]]
                    region = vor.regions[region_idx]
                    if not -1 in region and len(region) > 0:
                        polygon = [vor.vertices[i] for i in region]
                        polygons.append(polygon)
                        colors.append(cmap(norm(area)))
    # Add colored Voronoi cells as a PolyCollection
    poly_collection = PolyCollection(polygons, facecolors=colors, edgecolors="none",linewidths=(0.0001,), alpha=0.5)
    ax.add_collection(poly_collection)
    
    # Call the function to draw Voronoi lines
    plot_voronoi_lines(ax, vor, points_per_cell, line_width=line_width, show_all_cells=show_all_cells)
    
    # Set axis limits and title
    ax.set_xlim(0, raw_frame.shape[1])
    ax.set_ylim(raw_frame.shape[0], 0)
    ax.set_title("Voronoi Tessellation Colored by Cell Areas")
    ax.axis('off')
    
    return norm, cmap  # Return these for use in creating the colorbar
    

        
def adjust_axes_to_unit(ax, scale_factor, unit='pixels'):
    """
    Adjusts axis tick labels to the specified unit.

    Parameters:
    ax (matplotlib axis): Axis to adjust.
    scale_factor (float): Scale factor for converting pixels to the given unit.
    unit (str): Unit of measurement ('pixels' or 'microns').

    Returns:
    None
    """
    if unit == 'microns':
        x_labels = ax.get_xticks()
        y_labels = ax.get_yticks()
        ax.xaxis.set_major_locator(FixedLocator(x_labels))
        ax.yaxis.set_major_locator(FixedLocator(y_labels))
        ax.set_xticklabels([f"{int(x * scale_factor)}" for x in x_labels])
        ax.set_yticklabels([f"{int(y * scale_factor)}" for y in y_labels])



def visualize_set_voronoi_contours(mask_frame, raw_frame, num_points=10, line_width=1, distance=10, num_border_points=10, pixel_scale=1, unit='pixels', save_path=None, dpi=300, show_all_cells=True, custom_min=None, custom_max=None):
    """
    Generates and visualizes Voronoi tessellation for cell segmentation analysis.

    This function creates four visualizations:
    1. Original raw image
    2. Voronoi tessellation with cell boundaries
    3. Probability distribution of Voronoi cell areas (excluding edge cells)
    4. Colored Voronoi tessellation based on cell areas

    It also produces a combined plot of all four visualizations.

    Parameters:
    mask_frame (numpy.ndarray): Binary mask of the segmented cells.
    raw_frame (numpy.ndarray): Original raw image.
    num_points (int, optional): Number of points to sample along each cell contour. Default is 10.
    line_width (int, optional): Line width for Voronoi boundaries. Default is 1.
    distance (int, optional): Distance between sampled points. Default is 10.
    num_border_points (int, optional): Number of border points to add. Default is 10.
    pixel_scale (float, optional): Scale factor to convert pixels to physical units. Default is 1.
    unit (str, optional): Unit of measurement ('pixels' or 'microns'). Default is 'pixels'.
    save_path (str, optional): Path to save the output images. If None, images are not saved. Default is None.
    dpi (int, optional): DPI for saved images. Default is 300.
    show_all_cells (bool, optional): If True, show all cells including edge cells in the Voronoi visualizations. If False, exclude edge cells. Default is True.
    custom_min (float, optional): Custom minimum value for colorbar range. If None, uses the minimum cell area. Default is None.
    custom_max (float, optional): Custom maximum value for colorbar range. If None, uses the maximum cell area. Default is None.

    Returns:
    None

    Saves (if save_path is provided):
    - Voronoi tessellation with cell boundaries (_boundaries.tif)
    - Probability distribution of cell areas (_distribution.tif)
    - Colored Voronoi tessellation (_colored.tif)
    - Combined plot of all visualizations (_combined.tif)

    Note:
    The probability distribution always excludes edge cells, regardless of the show_all_cells parameter.
    """
    
    vor, points_per_cell, border_points = set_voronoi_tessellation_contours(mask_frame, num_points=num_points, distance=distance, num_border_points=num_border_points)
    cell_areas, edge_cells = calculate_voronoi_areas(vor, points_per_cell, raw_frame.shape)
    
    if unit == 'microns':
        cell_areas = {k: v * (pixel_scale ** 2) for k, v in cell_areas.items()}
    
    # Always calculate non-edge areas for distribution plot
    non_edge_areas = [v for k, v in cell_areas.items() if k not in edge_cells]
    
    # Use all cells or only non-edge cells based on show_all_cells flag for other visualizations
    if not show_all_cells:
        cell_areas = {k: v for k, v in cell_areas.items() if k not in edge_cells}
    
    scaled_areas = list(cell_areas.values())
    

    # 2. Voronoi tessellation with cell boundaries
    fig2, ax2 = plt.subplots(figsize=(raw_frame.shape[1] / dpi, raw_frame.shape[0] / dpi))
    ax2.imshow(raw_frame, cmap='gray')
    plot_voronoi_lines(ax2, vor, points_per_cell, line_width=line_width, show_all_cells=show_all_cells)
    ax2.set_xlim(0, raw_frame.shape[1])
    ax2.set_ylim(raw_frame.shape[0], 0)
    ax2.set_title("Voronoi Tessellation with Cell Boundaries")
    ax2.set_xlabel(f"X [{unit}]")
    ax2.set_ylabel(f"Y [{unit}]")
    ax2.axis('on')
    ax2.grid(False)
    adjust_axes_to_unit(ax2, pixel_scale, unit)
    
    if save_path:
        fig2.set_size_inches(raw_frame.shape[1] / dpi, raw_frame.shape[0] / dpi)  
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax2.axis('off')
        ax2.set_title("")
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.savefig(
            save_path.replace('.png', '_boundaries.tif'),
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0
        )

    # 3. Probability distribution of Voronoi cell areas (always using non-edge areas)
    if save_path:
        distribution_save_path = save_path.replace('.png', '_distribution.tif')
    else:
        distribution_save_path = None

    fig3 = plot_distribution_voronoi(non_edge_areas, save_path=distribution_save_path, unit=unit, dpi=dpi)

    # 4. Colored Voronoi tessellation based on cell areas
    fig4, ax4 = plt.subplots(figsize=(10, 10)) 
    norm, cmap = plot_voronoi_colored(ax4, vor, points_per_cell, raw_frame, cell_areas, line_width=line_width, show_all_cells=show_all_cells, custom_min=custom_min, custom_max=custom_max)

    if save_path:
        fig4.set_size_inches(raw_frame.shape[1] / dpi, raw_frame.shape[0] / dpi, forward=True)
        ax4.set_title("")
        ax4.set_xlabel("")
        ax4.set_ylabel("")
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.axis('off')
        ax4.set_position([0, 0, 1, 1])
        plt.savefig(save_path.replace('.png', '_colored.tif'), format='tif', dpi=dpi, bbox_inches=None, pad_inches=0)

    # Create a new figure and axes for the combined plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # 1. Original Raw Image in the first subplot
    axs[0, 0].imshow(raw_frame, cmap='gray')
    axs[0, 0].set_title("Original Raw Image")
    axs[0, 0].set_xlabel(f"X [{unit}]")
    axs[0, 0].set_ylabel(f"Y [{unit}]")
    axs[0, 0].axis('on')
    axs[0, 0].grid(False)
    adjust_axes_to_unit(axs[0, 0], pixel_scale, unit)

    # 2. Insert the Voronoi Tessellation with Cell Boundaries (fig2)
    axs[0, 1].imshow(fig2.canvas.buffer_rgba())
    axs[0, 1].set_title("Voronoi Tessellation with Cell Boundaries")
    axs[0, 1].axis('off')

    # 3. Insert the Probability Distribution of Voronoi Cell Areas (fig3)
    axs[1, 0].imshow(fig3.canvas.buffer_rgba())
    axs[1, 0].axis('off')

    # 4. Insert the Colored Voronoi Tessellation (fig4)
    axs[1, 1].imshow(fig4.canvas.buffer_rgba())
    axs[1, 1].set_title("Voronoi Tessellation with Cell Areas")
    axs[1, 1].axis('off')

    # Add colorbar to the combined plot
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(mappable, ax=axs[1, 1], label=f'Voronoi Cell Areas [{unit}²]')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.replace('.png', '_combined.tif'), dpi=dpi)

    plt.close('all')
    


def convert_area_to_microns(area_pixels, pixel_scale):
    """
    Convert an area from square pixels to square microns.

    Parameters:
    area_pixels (float): Area in square pixels
    pixel_scale (float): Number of microns per pixel

    Returns:
    float: Area in square microns, rounded to 2 decimal places
    """
    if not isinstance(pixel_scale, (int, float)) or pixel_scale <= 0:
        raise ValueError("pixel_scale must be a positive number")
    
    area_microns = area_pixels * (pixel_scale ** 2)
    return round(area_microns, 2)

        
    if unit not in ['pixels', 'microns']:
        raise ValueError("unit must be either 'pixels' or 'microns'")

    if not isinstance(pixel_scale, (int, float)) or pixel_scale <= 0:
        raise ValueError("pixel_scale must be a positive number")

    if areas_save_path is None:
        raise ValueError("areas_save_path must be specified")

    for movie_name, mask_frames in movies_masks.items():
        print(f"Processing Voronoi Tessellation for {movie_name}")

        movie_areas_save_path = os.path.join(areas_save_path, movie_name, 'tessellation')
        os.makedirs(movie_areas_save_path, exist_ok=True)

        if frame_number == 'all':
            all_areas = []

            for i in tqdm(range(len(mask_frames)), desc=f"Processing frames for {movie_name}"):
                mask_frame = mask_frames[i]

                try:
                    vor, points_per_cell, border_points = set_voronoi_tessellation_contours(
                        mask_frame, num_points=num_points, distance=distance, num_border_points=num_border_points
                    )
                    cell_areas, edge_cells = calculate_voronoi_areas(vor, points_per_cell, mask_frame.shape)
                    
                    non_edge_areas = {cell_id: area for cell_id, area in cell_areas.items() if cell_id not in edge_cells}
                    
                    if non_edge_areas:
                        mean_area = sum(non_edge_areas.values()) / len(non_edge_areas)
                        
                        for cell_id, area in non_edge_areas.items():
                            if unit == 'microns':
                                area = convert_area_to_microns(area, pixel_scale)
                            normalized_area = area / mean_area
                            all_areas.append({'Frame': i, 'Cell ID': cell_id, f'Area [{unit}²]': area, 'Normalized Area': normalized_area})
                
                except Exception as e:
                    print(f"Error processing frame {i} of {movie_name}: {str(e)}")

            areas_save_file = os.path.join(movie_areas_save_path, f"movie_voronoi_areas_{unit}.csv")
            areas_df = pd.DataFrame(all_areas)
            areas_df.to_csv(areas_save_file, index=False)
            print(f"Voronoi areas for all frames (excluding edge cells) saved to {areas_save_file}")

        else:
            if not isinstance(frame_number, int) or frame_number < 0:
                raise ValueError("frame_number must be a non-negative integer or 'all'")

            if frame_number < len(mask_frames):
                mask_frame = mask_frames[frame_number]

                try:
                    vor, points_per_cell, border_points = set_voronoi_tessellation_contours(
                        mask_frame, num_points=num_points, distance=distance, num_border_points=num_border_points
                    )
                    cell_areas, edge_cells = calculate_voronoi_areas(vor, points_per_cell, mask_frame.shape)

                    non_edge_areas = {cell_id: area for cell_id, area in cell_areas.items() if cell_id not in edge_cells}
                    
                    if non_edge_areas:
                        mean_area = sum(non_edge_areas.values()) / len(non_edge_areas)
                        
                        single_frame_areas = []
                        for cell_id, area in non_edge_areas.items():
                            if unit == 'microns':
                                area = convert_area_to_microns(area, pixel_scale)
                            normalized_area = area / mean_area
                            single_frame_areas.append({'Cell ID': cell_id, f'Area [{unit}²]': area, 'Normalized Area': normalized_area})

                        areas_save_file = os.path.join(movie_areas_save_path, f"frame_{frame_number}_voronoi_areas_{unit}.csv")
                        areas_df = pd.DataFrame(single_frame_areas)
                        areas_df.to_csv(areas_save_file, index=False)
                        print(f"Voronoi areas for frame {frame_number} (excluding edge cells) saved to {areas_save_file}")

                except Exception as e:
                    print(f"Error processing frame {frame_number} of {movie_name}: {str(e)}")
            else:
                print(f"Frame index {frame_number} out of bounds for movie {movie_name}")
                
                
                


def save_voronoi_areas_to_csv(movies_masks, pixel_scale, num_points=10, distance=50, num_border_points=10, frame_number=0, unit='pixels', areas_save_path=None):
    """
    Processes all movies, generates Voronoi tessellation areas, and saves the areas to CSV files.

    Parameters:
    movies_masks (dict): Dictionary containing mask frames for each movie.
    pixel_scale (float): Scale factor for converting pixels to microns.
    num_points (int): Number of points to extract per contour.
    distance (int): Distance from the image border to place the points.
    num_border_points (int): Number of border points per side.
    frame_number (int or str): Index of the frame to process or 'all' to process all frames.
    unit (str): Unit of measurement ('pixels' or 'microns').
    areas_save_path (str): Path to save the Voronoi areas CSV file.

    Returns:
    None
    """
    if unit not in ['pixels', 'microns']:
        raise ValueError("unit must be either 'pixels' or 'microns'")

    if not isinstance(pixel_scale, (int, float)) or pixel_scale <= 0:
        raise ValueError("pixel_scale must be a positive number")

    if areas_save_path is None:
        raise ValueError("areas_save_path must be specified")

    def convert_area(area, to_microns=False):
        return area * (pixel_scale ** 2) if to_microns else area

    for movie_name, mask_frames in movies_masks.items():
        print(f"Processing Voronoi Tessellation for {movie_name}")

        movie_areas_save_path = os.path.join(areas_save_path, movie_name, 'tessellation')
        os.makedirs(movie_areas_save_path, exist_ok=True)

        if frame_number == 'all':
            all_areas = []

            for i in tqdm(range(len(mask_frames)), desc=f"Processing frames for {movie_name}"):
                mask_frame = mask_frames[i]

                try:
                    vor, points_per_cell, border_points = set_voronoi_tessellation_contours(
                        mask_frame, num_points=num_points, distance=distance, num_border_points=num_border_points
                    )
                    cell_areas, edge_cells = calculate_voronoi_areas(vor, points_per_cell, mask_frame.shape)
                    
                    non_edge_areas = {cell_id: area for cell_id, area in cell_areas.items() if cell_id not in edge_cells}
                    
                    if non_edge_areas:
                        # Convert areas to microns if needed
                        converted_areas = {cell_id: convert_area(area, to_microns=(unit == 'microns')) 
                                           for cell_id, area in non_edge_areas.items()}
                        
                        mean_area = sum(converted_areas.values()) / len(converted_areas)
                        
                        for cell_id, area in converted_areas.items():
                            normalized_area = area / mean_area  # This division is unit-independent
                            all_areas.append({
                                'Frame': i, 
                                'Cell ID': cell_id, 
                                f'Area [{unit}²]': area, 
                                f'Mean Area [{unit}²]': mean_area,
                                'Normalized Area': normalized_area
                            })
                
                except Exception as e:
                    print(f"Error processing frame {i} of {movie_name}: {str(e)}")

            areas_save_file = os.path.join(movie_areas_save_path, f"movie_voronoi_areas_{unit}.csv")
            areas_df = pd.DataFrame(all_areas)
            areas_df.to_csv(areas_save_file, index=False)
            print(f"Voronoi areas for all frames (excluding edge cells) saved to {areas_save_file}")

        else:
            if not isinstance(frame_number, int) or frame_number < 0:
                raise ValueError("frame_number must be a non-negative integer or 'all'")

            if frame_number < len(mask_frames):
                mask_frame = mask_frames[frame_number]

                try:
                    vor, points_per_cell, border_points = set_voronoi_tessellation_contours(
                        mask_frame, num_points=num_points, distance=distance, num_border_points=num_border_points
                    )
                    cell_areas, edge_cells = calculate_voronoi_areas(vor, points_per_cell, mask_frame.shape)

                    non_edge_areas = {cell_id: area for cell_id, area in cell_areas.items() if cell_id not in edge_cells}
                    
                    if non_edge_areas:
                        # Convert areas to microns if needed
                        converted_areas = {cell_id: convert_area(area, to_microns=(unit == 'microns')) 
                                           for cell_id, area in non_edge_areas.items()}
                        
                        mean_area = sum(converted_areas.values()) / len(converted_areas)
                        
                        single_frame_areas = []
                        for cell_id, area in converted_areas.items():
                            normalized_area = area / mean_area  # This division is unit-independent
                            single_frame_areas.append({
                                'Cell ID': cell_id, 
                                f'Area [{unit}²]': area, 
                                f'Mean Area [{unit}²]': mean_area,
                                'Normalized Area': normalized_area
                            })

                        areas_save_file = os.path.join(movie_areas_save_path, f"frame_{frame_number}_voronoi_areas_{unit}.csv")
                        areas_df = pd.DataFrame(single_frame_areas)
                        areas_df.to_csv(areas_save_file, index=False)
                        print(f"Voronoi areas for frame {frame_number} (excluding edge cells) saved to {areas_save_file}")

                except Exception as e:
                    print(f"Error processing frame {frame_number} of {movie_name}: {str(e)}")
            else:
                print(f"Frame index {frame_number} out of bounds for movie {movie_name}")








def voronoi_visualizations(movies_masks, movies_raw, pixel_scale, num_points=10, distance=50, num_border_points=10, frame_number=0, unit='pixels', save_path=None, dpi=300, show_all_cells=True, line_width=1, custom_min=None, custom_max=None):
    """
    Processes all movies, generates Voronoi tessellation data, and saves the visualizations using visualize_set_voronoi_contours.

    Parameters:
    movies_masks (dict): Dictionary containing mask frames for each movie.
    movies_raw (dict): Dictionary containing raw frames for each movie.
    pixel_scale (float): Scale factor for converting pixels to microns.
    num_points (int, optional): Number of points to extract per contour. Default is 10.
    distance (int, optional): Distance from the image border to place the points. Default is 50.
    num_border_points (int, optional): Number of border points per side. Default is 10.
    frame_number (int or str, optional): Index of the frame to process or 'all' to process all frames. Default is 0.
    unit (str, optional): Unit of measurement ('pixels' or 'microns'). Default is 'pixels'.
    save_path (str, optional): Path to save the generated plots. Default is None.
    dpi (int, optional): Dots per inch (DPI) for the saved image quality. Default is 300.
    show_all_cells (bool, optional): If True, show all cells in visualization. If False, show only non-edge cells. Default is True.
    line_width (float, optional): Width of the lines in the Voronoi tessellation. Default is 1.
    custom_min (float, optional): Custom minimum value for colorbar range. If None, uses the minimum cell area. Default is None.
    custom_max (float, optional): Custom maximum value for colorbar range. If None, uses the maximum cell area. Default is None.

    Returns:
    None
    """
    for movie_name in movies_masks:
        print(f"Processing Voronoi Tessellation for {movie_name}")
        mask_frames = movies_masks[movie_name]
        raw_frames = movies_raw[movie_name]

        movie_save_path = os.path.join(save_path, movie_name, 'tessellation')
        os.makedirs(movie_save_path, exist_ok=True)

        if frame_number == 'all':
            all_areas = []

            for i in tqdm(range(len(mask_frames)), desc=f"Processing frames for {movie_name}"):
                mask_frame = mask_frames[i]
                raw_frame = raw_frames[i]

                vor, points_per_cell, border_points = set_voronoi_tessellation_contours(mask_frame, num_points=num_points, distance=distance, num_border_points=num_border_points)
                cell_areas, edge_cells = calculate_voronoi_areas(vor, points_per_cell, raw_frame.shape)
                
                # Always calculate non-edge areas for distribution
                if unit == 'microns':
                    non_edge_areas = [v * (pixel_scale ** 2) for k, v in cell_areas.items() if k not in edge_cells]
                else:
                    non_edge_areas = [v for k, v in cell_areas.items() if k not in edge_cells]
                
                all_areas.extend(non_edge_areas)

                save_file = os.path.join(movie_save_path, f"{os.path.basename(movie_name).replace('.csv', '')}_tessellation_frame_{i}{'_all_cells' if show_all_cells else ''}.png")
                visualize_set_voronoi_contours(mask_frame, raw_frame, num_points=num_points, distance=distance, num_border_points=num_border_points, pixel_scale=pixel_scale, unit=unit, save_path=save_file, dpi=dpi, line_width=line_width, show_all_cells=show_all_cells, custom_min=custom_min, custom_max=custom_max)

        else:
            if frame_number < len(mask_frames):
                mask_frame = mask_frames[frame_number]
                raw_frame = raw_frames[frame_number]

                vor, points_per_cell, border_points = set_voronoi_tessellation_contours(mask_frame, num_points=num_points, distance=distance, num_border_points=num_border_points)
                cell_areas, edge_cells = calculate_voronoi_areas(vor, points_per_cell, raw_frame.shape)
                
                # Always calculate non-edge areas for distribution
                if unit == 'microns':
                    non_edge_areas = [v * (pixel_scale ** 2) for k, v in cell_areas.items() if k not in edge_cells]
                else:
                    non_edge_areas = [v for k, v in cell_areas.items() if k not in edge_cells]

                save_file = os.path.join(movie_save_path, f"{os.path.basename(movie_name).replace('.csv', '')}_tessellation_frame_{frame_number}{'_all_cells' if show_all_cells else ''}.png")
                visualize_set_voronoi_contours(mask_frame, raw_frame, num_points=num_points, distance=distance, num_border_points=num_border_points, pixel_scale=pixel_scale, unit=unit, save_path=save_file, dpi=dpi, line_width=line_width, show_all_cells=show_all_cells, custom_min=custom_min, custom_max=custom_max)

            else:
                print(f"Frame index {frame_number} out of bounds for movie {movie_name}")





def plot_distribution_voronoi(area_values, save_path=None, unit='pixels', dpi=300):
    """
    Plots the distribution of Voronoi cell areas.

    Parameters:
    area_values (list): List of Voronoi cell area values.
    save_path (str, optional): Path to save the plot. Defaults to None.
    unit (str, optional): Unit of measurement ('pixels' or 'microns'). Defaults to 'pixels'.
    dpi (int, optional): Dots per inch (DPI) for the saved image quality. Defaults to 300.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """
    
    # Create a new figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the histogram of the area values
    ax.hist(area_values, bins=50, density=True, alpha=0.75, edgecolor='black')
    
    # Label the axes and set the title
    ax.set_xlabel(f"Area [{unit}²]")
    ax.set_ylabel("Probability")
    ax.set_title("Probability Distribution of Voronoi Cell Areas")
    
    # Remove grid lines from the plot
    ax.grid(False)
    
    # Calculate and plot the mean and median lines
    mean_area = np.mean(area_values)
    median_area = np.median(area_values)
    ax.axvline(mean_area, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_area:.2f}')
    ax.axvline(median_area, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_area:.2f}')
    
    # Add a legend to the plot
    ax.legend()

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    
    # Return the figure object
    return fig


def calculate_surface_coverage_and_density(movies_masks, save_path, pixel_scale):
    """
    Calculates the surface coverage and cell density of cells in each frame of the movies.

    Parameters:
    movies_masks (dict): Dictionary containing mask frames for each movie.
    save_path (str): Path to save the CSV files with coverage and density data.
    pixel_scale (float): Scale to convert pixel measurements to microns.

    Returns:
    None
    """
    for movie_name in movies_masks:
        print(f"Calculating surface coverage and cell density for {movie_name}")
        mask_frames = movies_masks[movie_name]

        movie_coverage_save_path = os.path.join(save_path, movie_name, 'surface_coverage')
        os.makedirs(movie_coverage_save_path, exist_ok=True)

        coverage_data = []

        for i in tqdm(range(len(mask_frames)), desc=f"Processing frames for {movie_name}"):
            mask_frame = mask_frames[i]

            # Calculate the percentage of the frame covered by cells
            total_pixels = mask_frame.size
            cell_pixels = np.count_nonzero(mask_frame)
            coverage_percentage = (cell_pixels / total_pixels) * 100

            # Calculate the cell density by counting unique labels
            unique_labels = np.unique(mask_frame)
            # Remove the background label (assumed to be 0)
            unique_labels = unique_labels[unique_labels != 0]
            num_bacteria = len(unique_labels)
            cell_density_pixels = num_bacteria / total_pixels

            # Calculate area in microns^2
            total_area_microns = total_pixels * (pixel_scale ** 2)
            cell_density_microns = num_bacteria / total_area_microns

            coverage_data.append({
                'Frame': i,
                'Coverage [%]': coverage_percentage,
                'Cell Density [bacteria/pixel^2]': cell_density_pixels,
                'Cell Density [bacteria/micron^2]': cell_density_microns
            })

        # Save the coverage and density data to a CSV file
        coverage_save_file = os.path.join(movie_coverage_save_path, f"surface_coverage_and_density.csv")
        coverage_df = pd.DataFrame(coverage_data)
        coverage_df.to_csv(coverage_save_file, index=False)

    print("Surface coverage and cell density calculation completed.")








def periodic_distance(p1, p2, box_size):
    """
    Calculate the distance between two points with periodic boundary conditions.

    Parameters
    ----------
    p1 : numpy.ndarray
        A 1D array representing the (y, x) coordinates of the first point.
    p2 : numpy.ndarray
        A 1D array representing the (y, x) coordinates of the second point.
    box_size : tuple
        The size of the box as (height, width).

    Returns
    -------
    float
        The distance between the two points considering periodic boundaries.
    """
    delta = np.abs(p1 - p2)
    delta = np.where(delta > 0.5 * np.array(box_size), np.array(box_size) - delta, delta)
    return np.sqrt((delta**2).sum(axis=-1))




def radial_distribution_function_pbc(centers, max_radius, bin_width, box_size):
    """
    Calculate the Radial Distribution Function (RDF) for a set of particle centers with periodic boundary conditions.

    Parameters
    ----------
    centers : numpy.ndarray
        An array of shape (N, 2) representing the (y, x) coordinates of the particle centers.
    max_radius : float
        The maximum radius to consider for the RDF, in pixels.
    bin_width : float
        The width of the distance bins, in pixels.
    box_size : tuple
        The size of the box as (height, width) in pixels.

    Returns
    -------
    r : numpy.ndarray
        An array of bin centers (radii) in pixels.
    rdf : numpy.ndarray
        The RDF values for each bin.
    """
    num_bins = int(max_radius / bin_width)
    r = np.linspace(bin_width / 2, max_radius - bin_width / 2, num_bins)
    rdf = np.zeros(num_bins)
    
    N = centers.shape[0]
    area_density = N / (box_size[0] * box_size[1])
    
    for i in range(N):
        # Calculate distances from the i-th cell to all other cells
        distances = periodic_distance(centers, centers[i], box_size)
        # Exclude the zero distance (self-distance)
        distances = distances[distances > 0]
        # Histogram the distances
        hist, _ = np.histogram(distances, bins=num_bins, range=(0, max_radius))
        # Accumulate the histogram results
        rdf += hist

    # Average the RDF by the number of cells
    rdf /= N
    # Normalize by the shell area
    shell_area = np.pi * (np.square(r + bin_width / 2) - np.square(r - bin_width / 2))
    rdf /= (shell_area * area_density)
    
    return r, rdf




def plot_rdf_for_frame_pbc(movies_masks, frame_number='all', pixel_scale=1.0, bin_width=1, unit='pixels', output_directory='.', csv_output_directory='.'):
    """
    Plot the Radial Distribution Function (RDF) for a specific frame or average over all frames using Periodic Boundary Conditions (PBC).

    Parameters
    ----------
    movies_masks : dict
        A dictionary where keys are movie names and values are lists of mask frames.
    frame_number : int or str
        The frame number to analyze or 'all' to average over all frames.
    bin_width : float
        The width of the distance bins, in pixels.
    pixel_scale : float
        Scale factor to convert pixels to physical units (e.g., microns per pixel).
    unit : str
        Unit of distance for the plot ('pixels' or 'microns').
    output_directory : str
        Directory to save the plot images.
    csv_output_directory : str
        Directory to save the RDF data as CSV files.

    Raises
    ------
    ValueError
        If the frame number is out of range.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not os.path.exists(csv_output_directory):
        os.makedirs(csv_output_directory)
    
    for movie_name, mask_frames in movies_masks.items():
        if frame_number != 'all':
            if frame_number < 0 or frame_number >= len(mask_frames):
                raise ValueError(f"Frame number {frame_number} is out of range for movie {movie_name}.")
            frames_to_analyze = [mask_frames[frame_number]]
        else:
            frames_to_analyze = mask_frames

        image_shape = frames_to_analyze[0].shape
        max_radius = max(image_shape[0], image_shape[1]) / 2
        if unit == 'microns':
            max_radius_microns = max_radius * pixel_scale
        else:
            max_radius_microns = max_radius
        
        total_rdf = None
        for mask_frame in frames_to_analyze:
            centers_dict = find_cell_centers(mask_frame)
            centers = np.array(list(centers_dict.values()))
            
            r, rdf = radial_distribution_function_pbc(centers, max_radius, bin_width, image_shape)
            
            if total_rdf is None:
                total_rdf = rdf
            else:
                total_rdf += rdf
        
        avg_rdf = total_rdf / len(frames_to_analyze)
        
        if unit == 'microns':
            r = r * pixel_scale
        
        plt.figure(figsize=(10, 6))
        plt.plot(r, avg_rdf, label='Radial Distribution Function')
        plt.xlabel(f'Distance (r) [{unit}]')
        plt.ylabel('g(r)')
        plt.title(f'RDF for {movie_name} - Frame {frame_number}\nMax radius: {max_radius_microns:.2f} {unit}')
        plt.axhline(1, color='gray', linestyle='--')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(output_directory, movie_name, 'radial_distribution_function')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = f"{movie_name}_frame_{frame_number}_radial_distribution_function.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath)
        plt.show()
        plt.close()
        
        # Save the RDF data to CSV
        csv_save_path = os.path.join(csv_output_directory, movie_name, 'radial_distribution_function')
        if not os.path.exists(csv_save_path):
            os.makedirs(csv_save_path)
        csv_filename = f"{movie_name}_frame_{frame_number}_rdf.csv"
        csv_filepath = os.path.join(csv_save_path, csv_filename)
        rdf_data = pd.DataFrame({'r': r, 'g(r)': avg_rdf})
        rdf_data.to_csv(csv_filepath, index=False)
        
        print(f"Plot saved to {filepath}")
        print(f"RDF data saved to {csv_filepath}")
        




def cellID_test(IDs):
    """
    Test function to check if the cell IDs are being correctly identified.
    """
    i = 0
    pass_test = True
    for cellID in IDs:
        i += 1
        if cellID != i:
            pass_test = False
    return pass_test



def display_loaded_movies_info(movies_masks, movies_raw, movies_outlines):
    """
    Displays the information about the loaded segmented, raw, and outlined movies.
    
    Parameters:
    - movies_masks: Dictionary of segmented movie masks.
    - movies_raw: Dictionary of raw movies.
    - movies_outlines: Dictionary of outlined movies.
    """
    # Calculate the number of items in each category
    num_segmented_movies = len(movies_masks)
    num_raw_movies = len(movies_raw)
    num_outlined_movies = len(movies_outlines)

    # Check for consistency in the number of movies across different categories
    if num_segmented_movies == num_raw_movies == num_outlined_movies:
        print(f"{num_segmented_movies} segmented, raw, and outlined movies loaded successfully.")
    else:
        print("Warning: The number of segmented, raw, and outlined movies does not match:")
        print(f"Segmented Movies: {num_segmented_movies}")
        print(f"Raw Movies: {num_raw_movies}")
        print(f"Outlined Movies: {num_outlined_movies}")

    # Display the keys for segmented movies
    print("\nSegmented Movies Loaded:")
    for key in movies_masks:
        print(key)

    # Display the keys for raw movies
    print("\nRaw Movies Loaded:")
    for key in movies_raw:
        print(key)

    # Display the keys for outlined movies
    print("\nOutlined Movies Loaded:")
    for key in movies_outlines:
        print(key)
        


all_functons_globals = dir()