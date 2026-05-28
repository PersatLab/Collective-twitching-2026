# Collective-twitching-2026

This repository contains the codes and datasets needed to reproduce the experimental findings reported in our paper: **Bacteria Tune Collective Navigation by Mechanosensing Collisions**

> **DOI:** 10.5281/zenodo.20341025

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Guide](#installation-guide)
3. [Repository Structure](#repository-structure)
4. [Demo](#demo)
5. [Instructions for Use](#instructions-for-use)
6. [Reproduction Instructions](#reproduction-instructions)

---

## System Requirements

### Software Dependencies
- **Language:** Python 3.9.0, Fiji 2.16.0
- **Key packages/libraries:** see `environment.yml` for the full list with version numbers
- **Operating System:** Tested on macOS Sonoma 14.1; expected to work on Linux and Windows

### Non-standard Hardware
-  A GPU is recommended for the segmentation process (`1_segmentation/`); 
  all other analyses run on standard CPU

---

## Installation Guide

### Instructions
   To use this repository, follow these steps:

1. **Set up your git**: Depending on your system, set up git on your computer.
   Use the Git Bash command line. Navigate to the directory where you want to set up the 
   local copy of this repository:
   ```
   cd "git directory"
   ```
   
2. **Clone the Repository**: Clone this GitHub repository to your local machine using the 
   following command:
   ```
   git clone https://github.com/PersatLab/Collective-twitching-2026
   ```
   
3. **Set up Environment**: The environment contains conda channels and the required python 
   version. 
   Use Anaconda command line. Move to the directory where the cloned repository is using 
   the cd command:
   ```
   cd directory/Collective-twitching-2026
   ```
   Create and start a new conda environment using:
   ```
   conda env create -f environment.yml
   ```
   Then:
   ```
   conda activate collective-twitching
   ```
  
4. **Install Requirements**: 
   If you have GPU support on your computer, install compatible torch according to 
   [Latest Version](https://pytorch.org/get-started/locally/).
 
   First uninstall torch and torchvision that don't have cuda support:
   ```
   pip uninstall torch torchvision
   ```
   Then install torch+cuda:
   ```
   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
   ```
   
5. **Regular startup**: Use Anaconda command lines:
   ```
   cd "directory of local repository"
   conda activate collective-twitching
   jupyter lab
   ```

### Typical Install Time
~15 minutes on a standard computer. 

---

## Repository Structure
This repository is organized into 6 main folders (numbered 1–6), each corresponding 
to a specific analysis described in the paper.

For Python-based analyses, each folder contains the following subfolders:
- `scripts/`: Python scripts used for the analysis
- `dataset/` or `movies/`: input data
- `results/` or `figures/`: output files

For Fiji-based analyses, each folder contains the following subfolders:
- `dataset/` or `movies/`: input data
- one subfolder per intermediate processing step (see folder tree below)

├── 1_segmentation/ 
│   ├── scripts/
│   └── working-dir/
│    		└── movies/
│    		└── segments/
│ 
├── 2_collective_organization/ 
│   ├── calculations_and_colormaps/ 
│   │		└── scripts/
│   │		└── working-dir/
│   │ 				└── movies/
│   │ 				└── segments/
│   │ 				└── segments_analysis/
│   │ 				└── graphes/
│   │
│   ├── tessellation_postprocessing/
│   │		└── script/
│   │		└── dataset/
│   │		└── figures/
│   │
│   ├── nematic_correlation_plot/ 
│   │		└── script/
│   │		└── dataset/
│   │		└── figures/
│   │
│   └── nematic_correlation_decay_length/ 
│   		└── script/
│   		└── dataset/
│   		└── figures/
│ 
├── 3_trajectory_analysis/ 
│   ├── trajectories_reconstruction_/ #in Fiji
│   │		└── TrackMate_CSV_files/ #Output from TrackMate used for subsequent analysis in Python
│   │
│   ├── contact_reversals/ 
│   │		└── scripts/
│   │		└── working-dir/
│   │
│   ├── MSD/ 
│   │		└── scripts/
│   │		└── working-dir/
│   │
│   └── persistence/ 
│   		└── script/
│   		└── dataset/
│   		└── results/
│   		└── graphes/
│ 
├── 4_polarization_analysis/ 
│   └── script/
│   └── dataset/
│   └── results/
│   └── graphes/
│ 
├── 5_competition_assay/ 
│   └── data_preprocessing/
│   └── script/
│   └── dataset/
│   └── figures/
│ 
├── 6_maze_exploration/ 
│   └── spreading/ #in Fiji
│   		└── dataset/
│   		└── dataset_thresholded/
│   		└── spreading_values_/ #Output used for subsequent analysis in Python.
│ 								   #Surface coverage in mazes (parameter %Area for each 
│								   #image stored in the subfolder 'dataset_thresholded') 
│								   #Rows 1-3: WT, rows 4-6: pilH-
│ 
│   └── probability_of_escape/
│   		└── script/
│   		└── dataset/
│   		└── figures/
│ 
├── environment.yml
└── README.md

---

## Demo

All input data and expected outputs are provided in each analysis folder. To verify that 
the codes run correctly on your system, run the scripts on the provided input data and 
compare the results with the expected outputs stored in `results/` or `figures/`.

> **Note:** Running the scripts will overwrite the output files in `results/` or `figures/`. 
> We recommend duplicating the folder before running if you want to preserve the original 
> outputs for comparison.

---

## Instructions for Use

For Python-based analyses:

1. **Set the folder paths**: each script contains a clearly marked `## INPUT USER` 
   section at the top where the input and output folder paths must be updated to 
   match your local setup. For analyses that include a `working-dir/` subfolder, 
   the main path is centralized in a `conf.py` file stored in `scripts/` — 
   this file must also be updated.

2. **Input data**: sample input data are provided in each analysis folder (`dataset/` 
   or `movies/`).

3. **Run the script(s)**: each analysis folder contains a single script to execute. 
   For `5_competition_assay/`, multiple scripts are provided and must be executed 
   in the order indicated by their filename numbering (see `Reproduction Instructions` below).

4. **Output**: results will be automatically saved in the `results/` or `figures/` 
   subfolder of the corresponding analysis folder.

For Fiji-based analyses, no scripts are provided as the processing was performed 
manually. The detailed step-by-step procedure is described in the Methods 
section of the paper. Input data, intermediate steps, and final outputs are 
provided in the corresponding analysis folder for reference.

---

## Reproduction Instructions

All input data, including preprocessed images and segmentation files, are provided 
in each analysis folder. The following instructions describe how to run the analysis 
scripts to reproduce the figures shown in the paper.

> **Note:** Segmented images were generated using `1_segmentation/scripts/01_VideoSegmentation.ipynb`
> and are already provided in the corresponding `working-dir/segments` folders. Re-running 
> the segmentation is not required to reproduce the figures.
 
 
### Analysis 1 — Voronoi tessellations *(Figure 1.B–D and Figure 2.E)*

1. Run `2_collective_organization/calculations_and_colormaps/scripts/02_SegmentationAnalysis.ipynb` 
to reproduce Figure 1.B. >>> This script has a long run time.
2. Run `/2_collective_organization/tessellation_postprocessing/script/03_Tessellation_postprocessing.ipynb` 
to reproduce Figures 1.C, D and Figure 2.E (left)

Expected run time: ~5 hours.


### Analysis 2 — Collective ordering *(Figure 1.E-G and Figure 2.E)*

1. Run `2_collective_organization/calculations_and_colormaps/scripts/02_SegmentationAnalysis.ipynb`
to reproduce Figure 1.E
2. Run `2_collective_organization/nematic_correlation_plot/script/04_nematic_correlation_function_processing.ipynb`
to reproduce Figure 1.F
3. Run `2_collective_organization/nematic_correlation_decay_length/script/05_decay_length_plot.ipynb`
to reproduce Figure 1.G and Figure 2.E (right)

Expected run time: ~30 min.


### Analysis 3 — Single-cell trajectories analysis *(Figure 3.C, E and Supplementary Figure 2.A)*

Single-cell trajectories were reconstructed using the TrackMate plugin in Fiji, and the 
resulting "spots" CSV files were used as input for subsequent analysis in Python.
1. Run `3_trajectory_analysis/persistence/scripts/persistence_analysis.py` to reproduce Figure 3.C
2. Run `3_trajectory_analysis/MSD/scripts/03_TrackAnalysis.ipynb` to reproduce Figure 3.E
3. Run `3_trajectory_analysis/contact_reversals/scripts/03_TrackAnalysis.ipynb` to reproduce Supplementary Figure 2.A

Expected run time: ~10 min.


### Analysis 4 — PilG polarization *(Figure 3.I)*

1. Run `4_polarization_analysis/polarization_analysis.ipynb` to reproduce Figure 3.I

Expected run time: ~5 min.


### Analysis 5 — Competition assay *(Figure 3.K)*

1. Run `5_competition_assay/data_preprocessing/scripts/1_file_preparation.py` to threshold the images. 
Thresholded images are already provided in `5_competition_assay/data_preprocessing/2_Thresholded/`
2. Rotate the thresholded images manually in Fiji so that the colony core is on the left and the uncolonized 
space is on the right, then save them in `5_competition_assay/data_preprocessing/3_Rotated_for_use/`.
Rotated images are already provided.
3. Run `5_competition_assay/data_preprocessing/scripts/competition_assay_main.py` to generate CSV 
files of bacterial counts and relative abundances. Output CSV files are already provided in 
`5_competition_assay/data_preprocessing/output/`.
4. Run `5_competition_assay/scripts/3_competition_plot.ipynb` to reproduce Figure 3.K

Expected run time: ~10 min.


### Analysis 6 — Bacterial exploration in mazes *(Figure 4.E-F)*

1. Run `6_maze_exploration/probability_of_escape/script/probability_escape.ipynb` to reproduce Figures 4.E and F

> **Note — Surface coverage:** Movies of bacterial motility in the mazes were segmented, 
> binarized, and SUM time-projected. The resulting projections are provided in `6_maze_exploration/spreading/dataset/`, 
> along with the maze outline as an ROI selection. In Fiji, images were thresholded (all pixels > 1 
> set to 255) and the percentage of area covered by bacteria was measured using the *%Area* 
> parameter (see `6_maze_exploration/spreading/dataset_thresholded/` and `/spreading_values/`). 
> These values are provided as input to the Python script.

Expected run time: ~5 min.

---

## License
BSD 3-Clause "New" or "Revised" License
