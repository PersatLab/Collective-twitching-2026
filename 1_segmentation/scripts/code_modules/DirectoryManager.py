import sys
sys.path.append('..')
import os
from conf import basedir

class directory_manager():
    def __init__(self):
        """
        Defines the structure of the folders in the working directory depending on your operating system and needs.

        For Windows, using backslash "\" as path separator, the path string must have the format r"path" (r = raw string), because the single backslas "\" is an escape character in Python and has to be converted to a double backslash "\\". 
        For Mac, a normal slash "/" works as path separator. 

        Parameters
        ----------
        basedir : str
            The base directory where all the data is stored.
        moviedir : str
            The directory where the raw tif movies are stored to be processed (e.g. for segmentation)
        segmentdir : str
            The directory where the segments from omnipose are saved
        editeddir : str
            The directory where the edited segments after filtering are saved
        graphdir : str
            The directory where the final graphs are saved
        tracksdir : str
            The directory where the tracks should be saved from TrackMate

        Raises
        ----------
        FileNotFoundError
            If the defined base directory doesn't exist or there is no access to it. If directory is a network drive, try to open the folder with file explorer or reconnect. Doesn't check the downstream folders.
        """

        self.basedir = basedir

        self.moviedir = os.path.join(self.basedir, 'movies') # directory where the raw tif movies are stored to be processed (e.g. for segmentation)
        self.segmentdir  = os.path.join(self.basedir, 'segments') # directory where the segments from omnipose are saved
        self.segmentsanalysisdir = os.path.join(self.basedir , 'segments_analysis') # directory where the csv from the segments analysis are saved
        self.editeddir = os.path.join(self.segmentdir, 'edited') # directory where the edited segments after filtering are saved
        self.graphdir = os.path.join(self.basedir, 'graphs') # directory where the final graphs are saved
        self.tracksdir = os.path.join(self.basedir, 'tracks') # directory where the tracks should be saved from TrackMate
        self.tracksanalysisdir = os.path.join(self.basedir , 'tracks_analysis') # directory where the csv from the segments analysis are saved

        if not os.access(self.basedir,os.R_OK):
            raise FileNotFoundError(f"No access to directory or doesn't exist: {self.basedir}. Try opening directory through file explorer and retry, or reconnect network drive. Modify basedir in conf.py.")
