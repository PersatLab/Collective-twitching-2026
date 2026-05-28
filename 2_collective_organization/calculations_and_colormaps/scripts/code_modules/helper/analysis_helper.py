import numpy as np

def find_points_of_cell(mask_frame, cellID):
    """
    Find the points that belong to a specific cell in a mask.

    This function identifies the points in the mask that belong to the cell specified by the cellID.
    It returns these points as a 2D numpy array, where each row is a point and the columns are the y and x coordinates of the frame.

    Parameters
    ----------
    mask_frame : numpy.ndarray
        A 2D array representing the mask, where the background is represented by 0 and each distinct cell is represented by a unique gray value.
    cellID : int
        The unique ID of the cell whose points are to be found.

    Returns
    -------
    numpy.ndarray
        A 2D array where each row represents a point (y, x) that belongs to the cell.
    """
    if not isinstance(mask_frame, np.ndarray) or len(mask_frame.shape) != 2:
        raise ValueError("mask_frame must be a 2D numpy array")
    if not isinstance(cellID, int) or cellID <= 0:
        raise ValueError("cellID must be non-zero positive integer")

    try:
        points = np.array(np.transpose(np.where(mask_frame == cellID)))
        if points.size == 0:
            raise ValueError(f"No points found for cellID: {cellID}")
    except ValueError as e:
        print(e)
    
    return points


def dot_products(cellDirection, close_cells_directions):
    """
    Compares the direction of a cell to the directions of all its close cells, using the mean of the dot products.

    Args:
        cellDirection (numpy.ndarray): A 1D numpy array of size 2 representing the direction of the cell.
        close_cells_directions (numpy.ndarray): A 2D numpy array where each row is the direction of a close cell.
    Returns:
        float: The mean dot product.
    """
    dot_products = np.dot(cellDirection, close_cells_directions.T)
    return np.mean(dot_products)

def local_order_param_for1cell(cellDirection, close_cells_directions):
    """
    Function used in cosine_local_order_parameters to calculate the local order parameter of one cell.
    """
    avgDirection = np.mean(close_cells_directions)
    return (3*(np.cos(cellDirection - avgDirection)**2)-1)/2
