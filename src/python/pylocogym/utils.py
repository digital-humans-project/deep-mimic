import numpy as np
from scipy.spatial import distance

def find_closest_frames(motion1, motion2):
    '''
    Function to find the closest frame between two motions
    Input:
        motion1: q values of first motion
        motion2: q values of second motion
    Returns:
        frame_idx: tuple containing index of (motion1, motion2) which are the closest
    '''

    #calculating euclidean distance between all the pairs of frames of both motions
    dist_arr = distance.cdist(motion1, motion2, 'euclidean')
    frame_idx = np.unravel_index(np.argmin(dist_arr, axis=None), dist_arr.shape)
    return frame_idx

