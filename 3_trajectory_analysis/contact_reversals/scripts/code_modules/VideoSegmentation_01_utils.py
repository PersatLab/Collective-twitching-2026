"""
Utilities for 01_VideoSegmentation.ipynb
"""

from PIL import Image
import os
from cellpose_omni import io, utils, transforms, plot
import time
from omnipose.utils import normalize99
from numpy import nonzero, array
import numpy as np
from PIL import Image
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import re
import textwrap

# Defines function that imports all tif images and videos from a directory
def import_image_or_video(directory):
    """
    Import all non-hidden .tif images or movies from a specified directory, without sub-directories. 

    Each video is imported into a list of numpy arrays representing the frames of the video,
    and stored in a dictionary, where the key is the filename and the value is the list of frames.

    Parameters
    ----------
    directory : str
        The path to the directory from which to import .tif files.

    Returns
    -------
    all_tifs : dict
        A dictionary where the keys are the file names and the values are the imported images.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist or if there are no .tif files in the directory.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"No access to directory or doesn't exist: {directory}. Try opening directory through file explorer and retry, or reconnect network drive.")

    all_tifs = {}
    for file in os.listdir(directory):
        #remove hidden files
        if not file.startswith('.') and (file.endswith('.tif') or file.endswith('.tiff')): # hidden file removal by excluding files that start with '.' doesn't work on windows!:
            image = io.imread(os.path.join(directory, file))
            all_tifs[file]=image

    if not all_tifs:
        raise FileNotFoundError(f"There are no compatible tif files in the specified directory: {directory}")
    return all_tifs


# Defines function to print info about the imported movies and/or images
def print_info_files(movies):
    """
    Prints information about all imported movies or images, such as number of frames and frame size.  

    Parameters
    ----------
    movies: dict
        Raw input movies or images as tif, use import_image_or_video function to load. 

    Warns
    ------
    Short video or multichannel image
        If file is detected as movie with 2-4 frames, prints warning that file could also be multichannel image. Doesn't warn for videos with > 4 frames. 
    
    Raises
    ------
    ValueError
        If the imported file is neither an image with 2 dimensions (pixel_x,pixel_y), nor a video with 3 dimensions (frames,pixel_x,pixel_y).
    """
    print('There are {} movies/images to be segmented'.format(len(movies)))
    print()
    for i in movies:
        movie = movies[i]
        if len(movie.shape)==3:
            first_frame = movie[0]
            print("Movie:",i)
            print("file shape:", str(np.shape(movie)))
            number_frames = np.shape(movie)[0]
            print("frames:",str(number_frames))
            print('image size (pixel):',first_frame.shape)
            print('data type:',first_frame.dtype)
            if number_frames in [2,3,4]:
                print(f"Warning: This file could either be a video with {number_frames} frames or a single frame with {number_frames} channels. The segmentation function will work, but doesn't make sense for mixed phase contrast + fluorescence images. If file is a single frame image --> split channels before importing.")
            print()      
        elif len(movie.shape)==2:
            first_frame = movie
            print("Image:",i)
            print("file shape:", str(np.shape(movie)))
            print("frames: 1")
            print('image size (pixel):',first_frame.shape)
            print('data type:',first_frame.dtype)
            print()  
        else:
            raise ValueError(f"movie.shape must be (frames,pixel_x,pixel_y) for videos, or (pixel_x,pixel_y) for images, but is {movie.shape}. This will cause issues with the function segment_save_movies")


# Defines function to import segmented videos
def import_segments(directory):
    """
    Import all .tif segments from a specified directory.

    This function scans the specified directory for .tif files with names ending in '_segmented.tif', 
    and returns a dictionary where the keys are the file names and the values are the imported images.

    Parameters
    ----------
    directory : str
        The path to the directory from which to import .tif files.

    Returns
    -------
    all_segments : dict
        A dictionary where the keys are the file names and the values are the imported images.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist or if there are no .tif files ending with '_segmented.tif' in the directory.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The specified directory does not exist: {directory}")

    all_segments = {}
    for file in os.listdir(directory):
        #remove hidden files
        if not file.startswith('.') and file.endswith('_segmented.tif'): # hidden file removal by excluding files that start with '.' doesn't work on windows!:
            image = io.imread(os.path.join(directory, file))
            all_segments[file]=image

    if not all_segments:
        raise FileNotFoundError(f"There are no compatible tif files in the specified directory: {directory}")
    return all_segments




# Defines function to import segmented videos and corresponding raw movies
def import_masks_movies(dir_segments, dir_movies):
    """
    Import all segmented TIF videos from a directory.

    This function reads all non-hidden files from the specified directory 
    and assumes they are TIF videos. The function will only load files that
    end with _segmented.tif, i.e. the labelled masks.
    Each video or image is read into a list of numpy arrays 
    representing the frames of the video, and stored in a dictionary, 
    where the key is the filename and the value is the list of frames.

    Parameters
    ----------
    dir_segments, dir_movies : str
        The path to the directory from which to import segmented movies and corresponding raw movies, respectively.

    Returns
    -------
    all_masks and all_movies : dict
        A dictionary where the keys are filenames and the values are lists of numpy arrays representing the frames of the videos.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist or there are no tif movies.
    FileNotFoundError
        If the original movie corresponding to the mask movie does not exist in the input directory.
    ValueError
        If the mask movie and corresponding movie do not have the same number of frames. 
    """
    if not os.path.exists(dir_segments):
        raise FileNotFoundError(f"The specified directory does not exist: {dir_segments}")
    
    all_masks = {}
    all_frames = {}
    # Iterate through files in the specified directory
    for file in os.listdir(dir_segments):
        if not file.startswith('.') and file.endswith('_segmented.tif'):
            frames_masks = io.imread(os.path.join(dir_segments, file))
            if frames_masks.ndim == 2:  # If the loaded file is an image, adjust its shape
                frames_masks = np.expand_dims(frames_masks, axis=0)
            frames_per_vid = []
            for frame in frames_masks:
                frames_per_vid.append(frame)
            ori_name = file.replace('_segmented.tif','.tif')
            all_masks[ori_name] = frames_per_vid

            if not os.path.exists(os.path.join(dir_movies, ori_name)):
                raise FileNotFoundError(f"The corresponding original movie {ori_name} does not exist in the input directory: {dir_movies}")
            
            frames_ori = io.imread(os.path.join(dir_movies, ori_name))
            if frames_ori.ndim == 2:  # If the original movie is actually an image, adjust its shape
                frames_ori = np.expand_dims(frames_ori, axis=0)
            if len(frames_ori) != len(frames_masks):
                raise ValueError(f"Mask movies and original movies must have the same number of frames. File: {ori_name}")
            frames_per_vid = []
            for frame in frames_ori:
                frames_per_vid.append(frame)
            all_frames[ori_name] = frames_per_vid

    print(len(all_masks), "masks and corresponding original movies loaded")
            
    return all_masks, all_frames
    
# Defines function to import segmented videos, corresponding raw movies and outlines

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
            if frames_masks.ndim == 2:
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

# Defines function to reassemble video from individual masks
def reassemble_mask_video(masks):
    """
    Reassembles a video from its masks.

    This function takes a list of masks, where each mask is a 2D numpy array 
    representing a frame of the video. It ensures all masks have the same dimensions, 
    converts each mask to a PIL Image object, and appends it to the video.

    Parameters
    ----------
    masks : list
        A list of 2D numpy arrays representing the masks of the video frames.

    Returns
    -------
    video : list
        A list of PIL Image objects representing the reassembled video.

    Raises
    ------
    ValueError
        If not all masks have the same dimensions.
    """
    # Ensure all images have the same dimensions
    image_width, image_height = masks[0].shape
        
    video = []
    for frame in masks:
        if frame.shape != (image_width, image_height):
            raise ValueError("All images must have the same dimensions.")
        
        # Convert each mask to 16-bit before creating the image
        frame_16bit = np.uint16(frame)
        image = Image.fromarray(frame_16bit)
        video.append(image)
    return video


# function to convert edited masks to outlines and print into original movie
def generate_outlines(mask,ori_image):
    """
    Imprint outlines generated from segments into original image.

    Parameters
    ----------
    mask: np array
        numpy array representing the mask of the frame.

    imori_image: np array
        numpy array representing the original raw image.

    Returns
    -------
    img_outline: PIL Image
        Single frame of original image with imprinted segment outlines. 
    """
    
    outlines = utils.masks_to_outlines(mask) # transform mask to outlines (logical array)
    outX, outY = nonzero(outlines) # indexes of x and y coordinates of all outlines

    if ori_image.ndim < 3:
        ori_image = plot.image_to_rgb(ori_image, channels=[0,0]) # transforms image into rgb
        img_outline = ori_image.copy()
    img_outline[outX, outY] = array([255,0,255]) # changes the pixels corresponding to a cell outline to pure magenta

    img_outline = Image.fromarray(img_outline) # transforms into saveable image

    return img_outline
                

# Defines function to imprint outlines into images and reassemble video from mutliple images
def reassemble_video_outlines(masks,images,n): 
    """
    Imprint outlines generated from masks into images and reassembles video.

    This function takes a list of masks, where each mask is a 2D numpy array 
    representing a frame of the video, and takes corresponding raw images. It ensures all masks and images have the same dimensions, 
    generates the semgent outline based on the masks, prints the segment outlines in the raw image, and appends it to the video.

    Parameters
    ----------
    masks : list
        A list of 2D numpy arrays representing the masks of the video frames.

    images : list
        A list of 2D numpy arrays representing the raw image of the video frames.

    n : integer
        the number frames in the movie

    Returns
    -------
    video : list
        A list of PIL Image objects representing the reassembled video.

    Raises
    ------
    ValueError
        If not all masks or original images have the same dimensions.
    """

    # Ensure all images have the same dimensions
    image_width, image_height = masks[0].shape # check the dimension of the first frame in masks
    
    for tm in masks: # check if all frames of the masks have the same dimensions
        if tm.shape != (image_width, image_height):
            raise ValueError("All mask images must have the same dimensions.")
    
    for ti in images: # check if all frames of the images have the same dimensions and the same dimensions as the masks
        if ti.shape != (image_width, image_height):
            raise ValueError("All images must have the same dimensions and the same dimensions as the masks.")
    
    video = []
    for idx,i in enumerate(n): # by using enumare(n) here, this goes through all masks and corresponding images; only required if n is a sub-list of all images
        maski = masks[idx] # get masks for current image
        imgi = images[i] # get current image
        imgi = transforms.normalize99(imgi) # normalized version of current image
        
        img_outl = generate_outlines(maski,imgi) # function to return image with imprinted segment outlines
        
        # outlinei = utils.masks_to_outlines(maski) # transform mask to outlines (logical array)
        # outX, outY = nonzero(outlinei) # indexes of x and y coordinates of all outlines
    
        # if imgi.ndim < 3:
        #     imgi = plot.image_to_rgb(imgi, channels=[0,0]) # transforms image into rgb if dimsnsions include less than 3, meaning only x, y and no rgb channel
        # img_outl = imgi.copy()
        # img_outl[outX, outY] = array([255,0,255]) # changes the pixels corresponding to a cell outline to pure magenta

        # img_outl = Image.fromarray(img_outl) # transforms into saveable image

        
        video.append(img_outl) # adds all images with printed in outlines into video list
    return video


# Defines function to save image or video
def save_image_or_video(image_or_video, outpath, outputname, suffix='.tif' , overwrite=False):
    """
    Save a sequence of images as a TIF file or a single image as a TIF file.

    This function takes either a list of PIL Image objects or a single PIL Image object. 
    If a list is provided, it is treated as a sequence of frames that are saved as a TIF video. 
    If a single image is provided, it is saved as a TIF image. The output file is saved in the specified directory with the specified name and a suffix. 
    If a file with the same name already exists in the directory, the behavior depends on the `overwrite` parameter.

    Parameters
    ----------
    image_or_video : list of PIL.Image.Image or PIL.Image.Image
        The image or video to save. If a list of PIL Image objects is provided, they are treated as frames of a video. 
        If a single PIL Image object is provided, it is treated as a single image.
    outpath : str
        The path to the directory where the image or video will be saved.
    outputname : str
        The base name of the output file (without extension).
    suffix : str, optional
        The suffix to append to the output file name. Default is '_segmented.tif'.
    overwrite : bool, optional
        Whether to overwrite an existing file with the same name. If True, an existing file with the same name will be overwritten. 
        If False, a warning will be printed and the file will not be saved if a file with the same name already exists in the output directory. 
        Default is False.

    Raises
    ------
    FileNotFoundError
        If the parent directory of `outpath` does not exist.

    Returns
    -------
    None
    """
    ensure_directory_exists(outpath)
    outputname = outputname + suffix
    output_file_path = os.path.join(outpath, outputname)
    
    if os.path.isfile(output_file_path) and not overwrite:
        print(f"Warning: A file with the name {outputname} already exists in the directory {outpath}.")
        return False
    
    # When saving, check if the image_or_video is a list (for a video) or a single frame
    if isinstance(image_or_video, list):
        if isinstance(image_or_video[0], Image.Image):
            image_or_video[0].save(output_file_path, save_all=True, append_images=image_or_video[1:], compression='tiff_deflate', depth=16)
        else:
            raise TypeError("Expected PIL images in the list for saving video.")
    else:
        if not isinstance(image_or_video, Image.Image):
            image_or_video = Image.fromarray(image_or_video)
        image_or_video.save(output_file_path, compression='tiff_deflate', depth=16)
    
    return True


# Defines function to create a directory if it does not exist
def ensure_directory_exists(directory_path):
    """
    Ensure that the directory at the specified path exists.

    If the directory does not exist, it is created.

    Parameters
    ----------
    directory_path (str):
        The path to the directory.
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
        except FileNotFoundError:
            print(f"Error: The parent directory of {directory_path} does not exist.")


def frame_range(start_frame, end_frame):
    """Transform a range of numbers into useful frame numbers."""
    return range(start_frame, end_frame + 1)

def frame_ranges(frames):
    """Convert a list of frames to a string representation with ranges."""
    if not frames:
        return ""
    frames = sorted(set(frames))
    ranges = []
    start = prev = frames[0]
    for frame in frames[1:]:
        if frame == prev + 1:
            prev = frame
        else:
            if start == prev:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{prev}")
            start = prev = frame
    if start == prev:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{prev}")
    return "_frames_" + "_".join(ranges)



def segment_save_movies(movies,
                        model,
                        outputdir,
                        frames=[],
                        suffix='_segmented',
                        overwrite=True,
                        mask_threshold=0.4,
                        verbose=0,
                        transparency=True,
                        rescale=None,
                        omni=True,
                        flow_threshold=0,
                        niter=None,
                        resample=True,
                        cluster=True,
                        augment=False,
                        tile=False,
                        affinity_seg=0):
    """
    Runs omnipose segmentation and saves a video with the identified segments as labelled image, i.e. every cell segment gets a unique grey value.
    Also saves corresponding raw input movie with cell segment outlines based on the segmented video.
    Masks can optionally be used directly from the returned outputs.
    Prints warnings if specific movies or frames couldn't be segmented and saved.

    Parameters
    ----------
    movies: dict
        raw input movies as tif, use import_image_or_video function to load.
    model: ?
        model used by omnipose chosen by CellposeModel function in the cellpose_omni.models module MODEL_NAMES; typically model_type='bact_phase_omni'.
    outputdir: str
        directory where to save the segmented movies; typically defined in the Params class (e.g. segmentdir).
    frames: list of integers or omnu.frame_range (optional)
        specify a list or range of individual frames to segment, e.g. [1,2,42] or omnu.frame_range(1,42)
        leave empty [] or remove argument for complete movie.
        the first frame of the video has the frame value 1, not 0.
    omnipose-specific options:
        choose as required to get good segmentation; given options are defaults.

    Raises
    ------
    ValueError
        If the imported movie is neither an image with 2 dimensions (pixel_x, pixel_y), nor a video with 3 dimensions (frames, pixel_x, pixel_y).
    TypeError
        If the imported movie is neither detected as image nor as video.
    """

    outputs = {'masks': {}, 'flows': {}, 'styles': {}}
    print('Segmenting {} movies/images'.format(len(movies)))

    failed_segmentation = []
    failed_save_masks = []
    failed_save_outlines = []

    tic = time.time()
    for movie_name in movies:
        movie = movies[movie_name]

        # Determine if the input is an image or a video
        is_image = (len(movie.shape) == 2)
        is_video = (len(movie.shape) == 3)

        # Handle frame validation differently for images and videos
        if is_image:
            if frames:
                print(f"Warning: Frames specified for an image {movie_name}. Processing only the first frame.")
            valid_frames = [0]
        elif is_video:
            if not frames:  # If no frames specified, segment all frames
                valid_frames = list(range(len(movie)))
            else:
                # Validate frames for each movie
                valid_frames = [f for f in frames if 0 <= f < len(movie)]
                invalid_frames = [f for f in frames if f < 0 or f >= len(movie)]
                for invalid_frame in invalid_frames:
                    print(f"Warning: Frame {invalid_frame} is out of bounds for movie {movie_name}.")
                if not valid_frames:
                    print(f"Warning: None of the specified frames are valid for movie {movie_name}. Skipping this movie.")
                    continue
                print(f"Using valid frames only: {valid_frames}")
        else:
            raise ValueError(f"movie.shape must be (frames, pixel_x, pixel_y) for videos, or (pixel_x, pixel_y) for images, but is {movie.shape}. This function can only process images or videos with max 1 channel (e.g. phase contrast).")

        # Ensure frames are in ascending order
        valid_frames = sorted(valid_frames)

        # Pre-check if we can overwrite or if the file already exists
        frame_str = frame_ranges(valid_frames) if is_video and frames else ""
        base_output_name = os.path.splitext(movie_name)[0] + frame_str + suffix + '.tif'
        output_file_path = os.path.join(outputdir, base_output_name)

        if not overwrite and os.path.isfile(output_file_path):
            print(f"Warning: A file with the name {base_output_name} already exists in {outputdir}. " \
                  "The file will not be overwritten. Modify 'overwrite' parameter to True to overwrite the file.")
            continue  # Skip to the next movie

        # Categorize imported files into images or videos with 1 channel.
        if is_image:
            number_frames = range(1)
            movie = [movie]
            print(f"{movie_name} (image)")
        elif is_video:
            movie = movie[valid_frames]
            number_frames = range(len(movie))
            print(f"{movie_name} (video), frames: {valid_frames}")
        else:
            raise ValueError(f"movie.shape must be (frames, pixel_x, pixel_y) for videos, or (pixel_x, pixel_y) for images, but is {movie.shape}. This function can only process images or videos with max 1 channel (e.g. phase contrast).")

        chans = [0, 0]  # this means segment based on first channel, no second channel

        try:
            if is_video:
                outputs['masks'][movie_name], outputs['flows'][movie_name], outputs['styles'][movie_name] = model.eval(
                    [movie[ii] for ii in number_frames],
                    channels=chans,
                    rescale=rescale,
                    mask_threshold=mask_threshold,
                    transparency=transparency,
                    flow_threshold=flow_threshold,
                    niter=niter,
                    omni=omni,
                    cluster=cluster,
                    resample=resample,
                    verbose=verbose,
                    affinity_seg=affinity_seg,
                    tile=tile,
                    augment=augment)
            elif is_image:
                outputs['masks'][movie_name], outputs['flows'][movie_name], outputs['styles'][movie_name] = model.eval(
                    movie,
                    channels=chans,
                    rescale=rescale,
                    mask_threshold=mask_threshold,
                    transparency=transparency,
                    flow_threshold=flow_threshold,
                    niter=niter,
                    omni=omni,
                    cluster=cluster,
                    resample=resample,
                    verbose=verbose,
                    affinity_seg=affinity_seg,
                    tile=tile,
                    augment=augment)
            else:
                raise TypeError(f"Imported files must be either detected as image or video, but is_image={is_image} and is_video={is_video}.")
        except Exception as e:
            print(f'Failed to segment {movie_name} due to {str(e)}')
            print('Moving on to the next movie')
            failed_segmentation.append(movie_name)
            continue

        video_masks = reassemble_mask_video(outputs['masks'][movie_name])
        if not save_image_or_video(video_masks, outputdir, os.path.splitext(movie_name)[0] + frame_str, suffix=suffix + '.tif', overwrite=overwrite):
            failed_save_masks.append(movie_name)
        video_outlines = reassemble_video_outlines(outputs['masks'][movie_name], movie, number_frames)
        if not save_image_or_video(video_outlines, outputdir, os.path.splitext(movie_name)[0] + frame_str, suffix=suffix + '_outlines.tif', overwrite=overwrite):
            failed_save_outlines.append(movie_name)

    net_time = time.time() - tic
    print('Segmentation complete')
    print('Total segmentation time: {}s'.format(net_time))

    print('{} videos/images failed to segment'.format(len(failed_segmentation)))
    if len(failed_segmentation) > 0:
        print('The failed movies/images are: {}'.format(failed_segmentation))

    print('{} videos/images failed to save masks'.format(len(failed_save_masks)))
    if len(failed_save_masks) > 0:
        print('The failed movies/images are: {}'.format(failed_save_masks))

    print('{} videos/images failed to save outlines'.format(len(failed_save_outlines)))
    if len(failed_save_outlines) > 0:
        print('The failed movies/images are: {}'.format(failed_save_outlines))




# Functions to filter segments (so far only small objects, segments on the edge, segments that in the raw movie/image have an intensity similar to the background, can be expanded in the future, e.g. to remove weird shapes, etc.)


# Function for small segments

# Small Segments Identification and Visualization

def identify_small_segments(frame, min_size=50):
    """
    Identifies small segments in a given frame and lists their unique IDs based on the specified minimum size.

    Parameters
    ----------
    frame : numpy.ndarray
        A 2D array representing the segmented frame, where each unique grayscale value identifies a segment.
    min_size : int
        Minimum segment area in pixels to be considered valid.

    Returns
    -------
    numpy.ndarray
        A binary mask where True represents pixels belonging to small segments.
    list
        A list of unique IDs corresponding to the small segments.
    """
    labeled_frame = frame
    small_segments_ids = []
    properties = measure.regionprops(labeled_frame)
    small_segments_mask = np.zeros_like(frame, dtype=bool)
    
    for prop in properties:
        if prop.area < min_size:
            small_segments_mask[labeled_frame == prop.label] = True
            small_segments_ids.append(prop.label)
    
    return small_segments_mask, small_segments_ids

def visualize_small_segments(movies_masks, movies_outlines, min_size=50, frame_number=0):
    """
    Visualizes cells below the specified size threshold in red for each video in movies_masks, with outlines displayed on the left.

    Parameters
    ----------
    movies_masks : dict
        A dictionary where the keys are filenames and the values are lists of numpy arrays representing the frames of the segmented videos.
    movies_outlines : dict
        A dictionary where the keys are filenames and the values are lists of numpy arrays representing the corresponding outlines.
    min_size : int (default = 50)
        Minimum segment area in pixels to remove. To adjust if needed.
    frame_number : int (default = 0)
        Frame number to display for each video. Frame 0 is the first frame.

    Returns:
    - min_size: int
        The minimum segment area in pixels which used as threshold to filter the segments.
    """
    for video_name, movie in movies_masks.items():
        outline_frames = movies_outlines[video_name]
        
        if frame_number >= len(movie) or frame_number >= len(outline_frames) or frame_number < 0:
            print(f"Warning: Frame number {frame_number} is out of bounds for video {video_name}. Using the first frame.")
            selected_frame_number = 0
        else:
            selected_frame_number = frame_number

        frame = movie[selected_frame_number]
        outline_frame = outline_frames[selected_frame_number]

        small_segments_mask, small_segments_ids = identify_small_segments(frame, min_size)
        properties = measure.regionprops(frame)
        areas = [prop.area for prop in properties]

        plt.figure(figsize=(10, 5))
        plt.hist(areas, bins=30, edgecolor='black')
        plt.axvline(min_size, color='red', linestyle='dashed', linewidth=1)
        plt.title(f"Cell Area Distribution for {video_name} (Frame {selected_frame_number})")
        plt.xlabel('Area (pixels)')
        plt.ylabel('Frequency')
        plt.show()

        rgb_image = np.zeros((*frame.shape, 3), dtype=np.uint8)
        rgb_image[frame > 0] = [255, 255, 255]
        rgb_image[small_segments_mask] = [255, 0, 0]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].imshow(outline_frame)
        axs[0].set_title(f"Original outlines:\n {video_name} (Frame {selected_frame_number})")
        axs[0].set_xlabel('Pixels')
        axs[0].set_ylabel('Pixels')
        axs[0].axis('on') 

        axs[1].imshow(rgb_image)
        axs[1].set_title(f"Cell below size threshold (in red):\n {video_name} (Frame {selected_frame_number})")
        axs[1].set_xlabel('Pixels')
        axs[1].set_ylabel('Pixels')
        axs[1].axis('on')

        plt.show()

    return min_size

# Intensity-based Segments Identification and Visualization

def identify_segments_by_intensity(mask_frame, raw_frame, threshold_coefficient=0.95, custom_threshold=None):
    """
    Identify cells in a frame based on their mean intensity, provide masks for cells above the intensity threshold,
    and list IDs of cells above the intensity threshold.

    Parameters:
    - mask_frame: 2D numpy array representing the segmented mask of cells, with unique IDs.
    - raw_frame: 2D numpy array representing the raw image data corresponding to the mask_frame.
    - threshold_coefficient: Optional; the coefficient used to calculate the default threshold based on background intensity. Default is 0.95.
    - custom_threshold: Optional; if provided, it overrides the background-based threshold.
    
    Returns:
    - A binary mask where cells above the threshold are marked as True.
    - A list of IDs of cells that are above the threshold.
    """
    unique_ids = np.unique(mask_frame)
    mean_intensities = {cell_id: np.mean(raw_frame[mask_frame == cell_id]) for cell_id in unique_ids}
    
    if custom_threshold is not None:
        threshold = custom_threshold
    else:
        background_intensity = mean_intensities.get(0, 0)
        threshold = threshold_coefficient * background_intensity

    above_threshold_mask = np.zeros_like(mask_frame, dtype=bool)
    above_threshold_ids = []

    for cell_id, intensity in mean_intensities.items():
        if cell_id == 0:
            continue
        if intensity > threshold:
            above_threshold_mask[mask_frame == cell_id] = True
            above_threshold_ids.append(cell_id)

    return above_threshold_mask, above_threshold_ids, mean_intensities, threshold

def visualize_segments_by_intensity(movies_masks, movies_raw, movies_outlines, frame_number=0, threshold_coefficient=0.95, custom_threshold=None):
    """
    Visualizes cells based on an intensity threshold, coloring cells above the threshold in red, with the outlines plotted nearby.
    This function displays the outlines and uses the raw data for threshold calculation.

    Parameters:
    - movies_masks : dict
        A dictionary where the keys are filenames and the values are lists of numpy arrays representing the frames of the segmented videos.
    - movies_raw : dict
        A dictionary where the keys are filenames and the values are lists of numpy arrays representing the frames of the original raw movies.
    - movies_outlines : dict
        A dictionary where the keys are filenames and the values are lists of numpy arrays representing the corresponding outlines.
    - frame_number : int (default=0)
        Frame number to display for each video. Frame 0 is the first frame.
    - threshold_coefficient: Optional; 
        The coefficient used to calculate the default threshold based on background intensity. Default is 0.95.
    - custom_threshold: Optional
        If provided, it overrides the background-based threshold calculated using threshold_coefficient.
        
    Returns:
    - threshold_coefficient: float
        The coefficient used for calculating the default intensity threshold.
    - custom_threshold: float or None
        The custom intensity threshold provided by the user, or None if not provided. This return allows the user to confirm the threshold settings used during the visualization.
    """

    for video_name, mask_frames in movies_masks.items():
        raw_frames = movies_raw[video_name]
        outline_frames = movies_outlines[video_name]
        
        if frame_number >= len(mask_frames) or frame_number >= len(outline_frames) or frame_number < 0:
            print(f"Warning: Frame number {frame_number} is out of bounds for video {video_name}. Using the first frame.")
            selected_frame_number = 0
        else:
            selected_frame_number = frame_number

        mask_frame = mask_frames[selected_frame_number]
        raw_frame = raw_frames[selected_frame_number]
        outline_frame = outline_frames[selected_frame_number]
        
        above_threshold_mask, above_threshold_ids, mean_intensities, threshold = identify_segments_by_intensity(mask_frame, raw_frame, threshold_coefficient, custom_threshold)
        properties = measure.regionprops(mask_frame)
        intensities = [np.mean(raw_frame[mask_frame == prop.label]) for prop in properties]

        plt.figure(figsize=(10, 5))
        plt.hist(intensities, bins=30, edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='dashed', linewidth=1)
        plt.title(f"Intensity Distribution for {video_name} (Frame {selected_frame_number})")
        plt.xlabel('Mean Intensity')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.show()
        
        rgb_image = np.zeros((*mask_frame.shape, 3), dtype=np.uint8)
        rgb_image[mask_frame > 0] = [255, 255, 255]
        rgb_image[above_threshold_mask] = [255, 0, 0]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].imshow(outline_frame)
        axs[0].set_title(f"Original outlines:\n {video_name} (Frame {selected_frame_number})")
        axs[0].set_xlabel('Pixels')
        axs[0].set_ylabel('Pixels')
        axs[0].axis('on')

        axs[1].imshow(rgb_image)
        axs[1].set_title(f"Cells above intensity threshold (in red)\nThreshold: {custom_threshold if custom_threshold is not None else threshold:.2f} (Frame {selected_frame_number})")
        axs[1].set_xlabel('Pixels')
        axs[1].set_ylabel('Pixels')
        axs[1].axis('on')
        
        plt.show()

    return threshold_coefficient, custom_threshold

# Edge Cells Identification and Visualization

def identify_edge_cells(frame):
    """
    Identifies cells on the edge of a given frame and lists their unique IDs.

    Parameters
    ----------
    frame : numpy.ndarray
        A single frame represented as a numpy array, where each unique grayscale value identifies a segment.

    Returns
    -------
    numpy.ndarray
        A binary mask where True represents pixels belonging to edge segments.
    list
        A list of the unique IDs of the cells that are on the edge of the frame.
    """
    labeled_frame = frame
    properties = regionprops(labeled_frame)
    edge_cells_ids = []
    edge_cells_mask = np.zeros_like(frame, dtype=bool)
    
    for prop in properties:
        if prop.bbox[0] == 0 or prop.bbox[1] == 0 or prop.bbox[2] == frame.shape[0] or prop.bbox[3] == frame.shape[1]:
            edge_cells_mask[labeled_frame == prop.label] = True
            edge_cells_ids.append(prop.label)
    
    return edge_cells_mask, edge_cells_ids

def visualize_edge_cells(movies_masks, movies_outlines, frame_number=0):
    """
    Visualizes cells on the edge of the frame by highlighting edge cells in red for each video in movies_masks, with outlines displayed on the left.

    Parameters
    ----------
    movies_masks : dict
        A dictionary where the keys are filenames and the values are lists of numpy arrays representing the frames of the segmented videos.
    movies_outlines : dict
        A dictionary where the keys are filenames and the values are lists of numpy arrays representing the corresponding outlines.
    frame_number : int (default = 0)
        Frame number to display for each video. Frame 0 is the first frame.
    """
    for video_name, movie in movies_masks.items():
        outline_frames = movies_outlines[video_name]

        if frame_number >= len(movie) or frame_number >= len(outline_frames) or frame_number < 0:
            print(f"Warning: Frame number {frame_number} is out of bounds for video {video_name}. Using the first frame.")
            selected_frame_number = 0
        else:
            selected_frame_number = frame_number

        frame = movie[selected_frame_number]
        outline_frame = outline_frames[selected_frame_number]

        edge_cells_mask, _ = identify_edge_cells(frame)

        rgb_image = np.zeros((*frame.shape, 3), dtype=np.uint8)
        rgb_image[frame > 0] = [255, 255, 255]
        rgb_image[edge_cells_mask] = [255, 0, 0]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].imshow(outline_frame)
        axs[0].set_title(f"Original outlines:\n {video_name} (Frame {selected_frame_number})")
        axs[0].set_xlabel('Pixels')
        axs[0].set_ylabel('Pixels')
        axs[0].axis('on')

        axs[1].imshow(rgb_image)
        axs[1].set_title(f"Cells on the edge (in red):\n {video_name} (Frame {selected_frame_number})")
        axs[1].set_xlabel('Pixels')
        axs[1].set_ylabel('Pixels')
        axs[1].axis('on')

        plt.show()


# Functions for filtering, saving and visualization.


def apply_all_filters_and_save(movies_masks, movies_raw, dir_segments, save=True, apply_size_filter=True, min_size=42, apply_intensity_filter=True, threshold_coefficient=0.95, custom_threshold=None, apply_edge_filter=True):
    """
    Applies selected filters (size, intensity, edge) to each frame of the segmented movies and compiles IDs of segments to remove based on filters.
    Optionally, saves the processed frames and their outlines as multi-page TIF files to a specified directory.

    Parameters:
    - movies_masks (dict): A dictionary where the keys are filenames and the values are lists of numpy arrays representing the frames of the segmented videos. Use function `import_masks_movies_outlines` to load `movies_masks` and corresponding raw movies.
    - movies_raw (dict): A dictionary where the keys are filenames and the values are lists of numpy arrays representing the frames of the original raw videos. Use function `import_masks_movies_outlines` to load `movies_masks` and corresponding raw movies.
    - dir_segments (str): The path to the directory where segmented movies are stored. If saving is enabled, edited segments and their outlines will be saved in a subdirectory named 'edited'.
    - save (bool, optional): Specifies whether to save the edited segments and their outlines. Defaults to False. If True, edited segments are saved in the 'edited' subdirectory within `dir_segments`.
    - apply_size_filter (bool, optional): Determines whether to apply size-based filtering to remove small segments. Defaults to True.
    - min_size (int, optional): Specifies the minimum segment area in pixels to retain. Segments smaller than this size are removed. Adjustable as needed. Defaults to 42.
    - apply_intensity_filter (bool, optional): Specifies whether to apply intensity-based filtering to remove segments based on their mean intensity compared to a calculated threshold. Defaults to True.
    - threshold_coefficient (float, optional): The coefficient used to calculate the default intensity threshold as a multiplier of the background intensity. Ignored if `custom_threshold` is provided. Defaults to 0.95.
    - custom_threshold (float, optional): Specifies a custom intensity threshold. If provided, it overrides the default threshold calculated using `threshold_coefficient`. Defaults to None.
    - apply_edge_filter (bool, optional): Determines whether to remove segments touching the edges of the frame. Defaults to True.
    """
 
    outputfolder = os.path.join(dir_segments, 'edited')
    os.makedirs(outputfolder, exist_ok=True)

    for video_name, mask_frames in movies_masks.items():
        raw_frames = movies_raw.get(video_name, [])
        print(f"Processing {video_name}...")

        processed_frames = []  # For storing PIL.Image objects of processed frames
        outlines = []  # For storing PIL.Image objects of outlines

        for frame_index, mask_frame in enumerate(mask_frames):
            raw_frame = raw_frames[frame_index] if frame_index < len(raw_frames) else None
            remove_ids = set()  # IDs to remove based on filters

            # Apply size filter
            if apply_size_filter:
                _, small_ids = identify_small_segments(mask_frame, min_size)
                remove_ids.update(small_ids)

            # Apply intensity filter
            if apply_intensity_filter and raw_frame is not None:
                _, above_threshold_ids, _, _ = identify_segments_by_intensity(mask_frame, raw_frame, threshold_coefficient, custom_threshold)
                remove_ids.update(above_threshold_ids)

            # Apply edge filter
            if apply_edge_filter:
                _, edge_ids = identify_edge_cells(mask_frame)
                remove_ids.update(edge_ids)

            # Removing segments based on compiled remove_ids
            for remove_id in remove_ids:
                mask_frame[mask_frame == remove_id] = 0  # Set to background

            # Convert processed mask to Image for saving
            processed_image = Image.fromarray(mask_frame.astype(np.uint16))
            processed_frames.append(processed_image)

            if raw_frame is not None:
                # Assuming generate_outlines function returns a PIL.Image object
                outline_image = generate_outlines(mask_frame, raw_frame)
                outlines.append(outline_image)

        # Saving the processed masks and outlines as multi-page TIF files
        if save:
            processed_tif_path = os.path.join(outputfolder, f"{video_name}_segmented.tif")
            save_image_or_video(processed_frames, outputfolder, f"{os.path.splitext(video_name)[0]}_segmented", suffix='.tif', overwrite=True)

            # Save outlines as a TIF, if any
            if outlines:
                outlines_tif_path = os.path.join(outputfolder, f"{os.path.splitext(video_name)[0]}_segmented_outlines.tif")
                save_image_or_video(outlines, outputfolder, f"{os.path.splitext(video_name)[0]}_segmented_outlines", suffix='.tif', overwrite=True)

        if save:
            print(f"All processed masks and outlines for {video_name} saved in {outputfolder} as TIF files.")


def compare_outlines(segment_dir, edited_dir, frame_number=0):
    """
    Compares the original outlines with the new outlines for a specified frame across all movies.
    It automatically loads outlines from the original 'segments' directory and the 'edited' subdirectory
    where filtered outlines are stored. This allows for a visual comparison to see the effects of
    applied filters on each movie's frame.

    Parameters:
    - segment_dir (str): The directory containing the original 'segments' folder with outlines.
                         This directory should contain outline files named as '<movie_name>_outlines.tif'.
    - edited_dir (str): The directory containing the 'edited' outlines after filtering. It is expected
                        to be a subdirectory within 'segment_dir' named 'edited', containing files with
                        the same naming convention as in 'segment_dir'.
    - frame_number (int, optional): The specific frame number to compare across all movies. Defaults to 0.
                                    If the specified frame number exceeds the number of frames in an outline,
                                    the function adjusts to handle it appropriately.

    Note:
    The function assumes that the outline files are named consistently and end with '_outlines.tif' or
    '_outlines.tiff'. It also assumes that each movie's original and edited outlines share the same
    file name, facilitating direct comparison.

    The function displays a two-panel figure for each movie found in the directories: the left panel shows
    the original outlines, and the right panel shows the new, edited outlines after filtering. This side-by-side
    comparison helps in visually assessing the impact of the filtering process on the outlines.
    """
    
    movies = [f for f in os.listdir(segment_dir) if f.endswith('_outlines.tif') or f.endswith('_outlines.tiff')]
    
    for movie in movies:
        original_outlines_path = os.path.join(segment_dir, movie)
        new_outlines_path = os.path.join(edited_dir, movie)
        
        if not os.path.exists(new_outlines_path):
            print(f"New outlines file for {movie} not found in {edited_dir}.")
            continue
        
        original_outlines = io.imread(original_outlines_path)
        new_outlines = io.imread(new_outlines_path)
        
        # Expand dimensions if necessary
        if original_outlines.ndim == 3:
            original_outlines = np.expand_dims(original_outlines, axis=0)
        if new_outlines.ndim == 3:
            new_outlines = np.expand_dims(new_outlines, axis=0)
        
        # Ensure the frame number is within valid range
        max_frame_index = min(original_outlines.shape[0], new_outlines.shape[0]) - 1
        actual_frame_index = frame_number if frame_number <= max_frame_index else 0
        
        # Debugging: Print frame index information
        print(f"Frame number requested: {frame_number}, Actual frame index used: {actual_frame_index}")
        
        original_frame = original_outlines[actual_frame_index]
        new_frame = new_outlines[actual_frame_index]
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Split the movie name into multiple lines if it's too long
        movie_title = "\n".join(textwrap.wrap(movie, width=40))
        fig.suptitle(movie_title, fontsize=16)
        
        axs[0].imshow(original_frame, cmap='gray')
        axs[0].set_title(f"Original Outlines (Frame {actual_frame_index})")
        axs[0].set_xlabel('Pixels')
        axs[0].set_ylabel('Pixels')
        axs[0].axis('on')
        
        axs[1].imshow(new_frame, cmap='gray')
        axs[1].set_title(f"New Outlines (Frame {actual_frame_index})")
        axs[1].set_xlabel('Pixels')
        axs[1].set_ylabel('Pixels')
        axs[1].axis('on')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()