import numpy as np
from scipy.spatial import distance

def find_closest_frames(motion1, motion2):
    dist_arr = distance.cdist(motion1, motion2, 'euclidean')
    frame_idx = np.unravel_index(np.argmin(dist_arr, axis=None), dist_arr.shape)
    #returns (frame_idx_motion1, frame_idx_motion2)
    return frame_idx

