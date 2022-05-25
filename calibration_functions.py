from turtle import left
import matplotlib.pyplot as plt 
import numpy as np
from IPython import embed
from etrack import MarkerTask, ImageMarker


def mark_crop_positions(self):
        task = MarkerTask("crop area", ["bottom left corner", "top left corner", "top right corner", "bottom right corner"], "Mark crop area")
        im = ImageMarker([task])
        
        marker_positions = im.mark_movie(file_name, frame_number)
        print(marker_positions)

        np.save('marker_positions', marker_positions)
        plt.close() # perhaps not necessary
        return marker_positions


def assign_marker_positions(marker_positions):
    bottom_left_x = marker_positions[0]['bottom left corner'][0]
    bottom_left_y = marker_positions[0]['bottom left corner'][1]
    bottom_right_x = marker_positions[0]['bottom right corner'][0]
    bottom_right_y = marker_positions[0]['bottom right corner'][1]
    top_left_x = marker_positions[0]['top left corner'][0]
    top_left_y = marker_positions[0]['top left corner'][1]
    top_right_x = marker_positions[0]['top right corner'][0]
    top_right_y = marker_positions[0]['top right corner'][1]
    return bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y, top_left_x, top_left_y, top_right_x, top_right_y


def assign_checkerboard_positions(checkerboard_marker_positions):
    checkerboard_top_right = checkerboard_marker_positions[0]['top right corner']
    checkerboard_top_left = checkerboard_marker_positions[0]['top left corner']
    checkerboard_bottom_right = checkerboard_marker_positions[0]['bottom right corner']
    checkerboard_bottom_left = checkerboard_marker_positions[0]['bottom left corner']
    return checkerboard_top_right, checkerboard_top_left, checkerboard_bottom_right, checkerboard_bottom_left


def crop_frame(frame, marker_positions):

    # load the four marker positions 
    bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y, top_left_x, top_left_y, top_right_x, top_right_y = assign_marker_positions(marker_positions)

    # define boundaries of frame, taken by average of points on same line but slightly different pixel values
    left_bound = int(np.mean([bottom_left_x, top_left_x]))
    right_bound = int(np.mean([bottom_right_x, top_right_x]))
    top_bound = int(np.mean([top_left_y, top_right_y]))
    bottom_bound = int(np.mean([bottom_left_y, bottom_right_y]))
    
    # crop the frame by boundary values
    cropped_frame = frame[top_bound:bottom_bound, left_bound:right_bound]
    cropped_frame = np.mean(cropped_frame, axis=2)    # mean over 3rd dimension (RGB/color values)

    # mean over short or long side of the frame corresponding to x or y axis of picture
    frame_width = np.mean(cropped_frame,axis=0)
    frame_height = np.mean(cropped_frame,axis=1)

    # differences of color values lying next to each other --> derivation 
    diff_width = np.diff(frame_width)
    diff_height = np.diff(frame_height)

    # two x vectors for better plotting
    x_width = np.arange(0, len(diff_width), 1)
    x_height = np.arange(0, len(diff_height), 1)

    return cropped_frame, frame_width, frame_height, diff_width, diff_height, x_width, x_height

def rotation_angle():
        pass


def threshold_crossings(data, threshold_factor):
    # upper and lower threshold 
    median_data = np.median(data)
    median_lower = median_data + np.min(data)
    median_upper = np.max(data) - median_data
    lower_threshold = median_lower / threshold_factor
    upper_threshold = median_upper / threshold_factor
    
    # array with values if data >/< than threshold = True or not
    lower_crossings = np.diff(data < lower_threshold, prepend=False)    # prepend: point after crossing
    upper_crossings = np.diff(data > upper_threshold, append=False)     # append: point before crossing
    
    # indices where crossings are
    lower_crossings_indices = np.argwhere(lower_crossings)
    upper_crossings_indices = np.argwhere(upper_crossings)
    
    # sort out several crossings of same edge of checkerboard (due to noise)
    half_window_size = 10
    lower_peaks = []
    upper_peaks = []
    for lower_idx in lower_crossings_indices:   # for every lower crossing..
        if lower_idx < half_window_size:        # ..if indice smaller than window size near indice 0
            half_window_size = lower_idx
        lower_window = data[lower_idx[0] - int(half_window_size):lower_idx[0] + int(half_window_size)]    # create data window from -window_size to +window_size 
        min_window = np.min(lower_window)     # take minimum of window
        min_idx = np.where(data == min_window)  # find indice where minimum is 
        
        lower_peaks.append(min_idx)     # append to list
    for upper_idx in upper_crossings_indices:   # same for upper crossings with max of window
        if upper_idx < half_window_size:
            half_window_size = upper_idx
        upper_window = data[upper_idx[0] -  int(half_window_size) : upper_idx[0] + int(half_window_size)]
        
        max_window = np.max(upper_window)
        max_idx = np.where(data == max_window)
        upper_peaks.append(max_idx)    

    # if several crossings create same peaks due to overlapping windows, only one (unique) will be taken  
    lower_peaks = np.unique(lower_peaks)
    upper_peaks = np.unique(upper_peaks)
    
    return lower_peaks, upper_peaks


def checkerboard_position(lower_crossings_indices, upper_crossings_indices):
    """Take crossing positions to generate a characteristic sequence for a corresponding position of the checkerboard inside the frame. 
    Positional description has to be interpreted depending on the input data.

    Args:
        lower_crossings_indices: Indices where lower threshold was crossed by derivation data.
        upper_crossings_indices: Indices where upper threshold was crossed by derivation data

    Returns:
        checkerboard_position: General position where the checkerboard lays inside the frame along the axis of the input data.
    """
    
    # create zipped list with both indices
    zip_list = []
    for zl in lower_crossings_indices:
        zip_list.append(zl)
    for zu in upper_crossings_indices:
        zip_list.append(zu)
    
    zip_list = np.sort(zip_list)    # order by indice

    # compare and assign zipped list to original indices lists and corresponding direction (to upper or lower threshold) 
    sequence = []
    for z in zip_list:
        if z in lower_crossings_indices:
            sequence.append('down')
        else:
            sequence.append('up')

    # depending on order of crossings through upper or lower treshold, we get a characteristic sequence for a position of the checkerboard in the frame
    if sequence == ['up', 'down', 'up', 'down']:    # first down, second up are edges of checkerboard
        checkerboard_position = 'middle'
        left_checkerboard_edge = zip_list[1]
        right_checkerboard_edge = zip_list[2]
    elif sequence == ['up', 'up', 'down']:  # first and second up are edges of checkerboard
        checkerboard_position = 'left'
        left_checkerboard_edge = zip_list[0]
        right_checkerboard_edge = zip_list[1]
    else:   # first and second down are edges of checkerboard
        checkerboard_position = 'right'
        left_checkerboard_edge = zip_list[1]
        right_checkerboard_edge = zip_list[2]
    
    return checkerboard_position, left_checkerboard_edge, right_checkerboard_edge    # position of checkerboard then will be returned


def filter_data(data, n):
    """Filter/smooth data with kernel of length n.

    Args:
        data: Raw data.
        n: Number of datapoints the mean gets computed over.

    Returns:
        filtered_data: Filtered data.
    """        
    new_data = np.zeros(len(data))  # empty vector where data will be put in in the following steps
    for k in np.arange(0, len(data) - n):
        kk = int(k)
        f = np.mean(data[kk:kk+n])  # mean over data over window from kk to kk+n
        kkk = int(kk+n / 2) # position where mean datapoint will be placed (so to say)
        if k == 0:
            new_data[:kkk] = f
        new_data[kkk] = f   # assignment of value to datapoint
    new_data[kkk:] = f
    for nd in new_data[0:n-1]:  # correction of left boundary effects (boundaries up to length of n were same number)
        nd_idx = np.argwhere(nd)
        new_data[nd_idx] = data[nd_idx]
    for nd in new_data[-1 - (n-1):-1]:  # same as above, correction of right boundary effect
        nd_idx = np.argwhere(nd)
        new_data[nd_idx] = data[nd_idx]
    
    return new_data