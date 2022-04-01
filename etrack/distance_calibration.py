from multiprocessing import allow_connection_pickling
from turtle import left
from cv2 import MARKER_TRIANGLE_UP, calibrationMatrixValues, threshold
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import os
import sys
from IPython import embed
from etrack import MarkerTask, ImageMarker


class DistanceCalibration():

    def __init__(self, file_name, frame_number, x_0=154, y_0=1318, cam_dist=1.36, width=1.35, height=0.805, width_pixel=1900, height_pixel=200, checkerboard_width=0.24, checkerboard_height=0.18, 
    checkerboard_width_pixel=500, checkerboard_height_pixel=350, rectangle_width=0.024, rectangle_height=0.0225, rectangle_width_pixel=100, rectangle_height_pixel=90, 
    rectangle_count_width=9, rectangle_count_height=7) -> None: 
        super().__init__()
        # aktualisieren
        """Calibration of the dimensions of the tank. Conversion of pixel into meter. Width refers to the "x-axis", height to the "y-axis" of the tank.
        
        Args:
            file_name (_type_): _description_
            x_0 (int, optional): X-value of the "origin" of the tank. Defaults to 0.
            y_0 (int, optional): Y-value of the "origin" of the tank. Defaults to 0.
            cam_dist (int, optional): Distance of camera lense to tank floor. Defaults to 1.36.
            width (int, optional): Width in meter from one lightened corner of the tank to the other. Defaults to 1.35.
            height (int, optional): Height in meter from one lightened corner of the tank to the other. Defaults to 1.35.
            width_pixel (int, optional): Width in pixel from one lightened corner of the tank to the other. Defaults to 1975.
            height_pixel (int, optional): Height in pixel from one lightened corner of the tank to the other.  Defaults to 1375.
            rectangle_width (float, optional): Width of one black or corresponding white rectangle of the checkerboard. Defaults to 0.024.
            rectangle_height (float, optional): Height of one black or corresponding white rectangle of the checkerboard. Defaults to 0.0225.
            rectangle_count_width (int, optional): Number of black rectangles over the width of the whole checkerboard. Defaults to 9.
            rectangle_count_height (int, optional): Number of black rectangles over the height of the whole checkerboard. Defaults to 7.
        """
        
        self._file_name = file_name
        self._x_0 = x_0
        self._y_0 = y_0
        self._width_pix = width_pixel
        self._height_pix = height_pixel
        self._cam_dist = cam_dist
        self._width = width
        self._height = height
        self._cb_width = checkerboard_width
        self._cb_height = checkerboard_height
        self._cb_width_pix = checkerboard_width_pixel
        self._cb_height_pix = checkerboard_height_pixel
        self._rect_width = rectangle_width
        self._rect_height = rectangle_height
        self._x_factor = self.width / self.width_pix   # m/pix
        self._y_factor = self.height / self.height_pix # m/pix

        self.mark_crop_positions
        self.threshold_crossings

    @property
    def x_0(self):
        return self._x_0
    
    @x_0.setter
    def x_0(self, value):
        self._x_0 = value

    @property
    def y_0(self):
        return self._y_0

    @y_0.setter
    def y_0(self, value):
        self._y_0 = value

    @property
    def cam_dist(self):
        return self._cam_dist
    
    @property
    def width(self):
        return self._width
    
    @width.setter
    def width(self, value):
        self._width = value
    
    @property
    def height(self):
        return self._height
    
    @height.setter
    def height(self, value):
        self._height = value
    
    @property
    def width_pix(self):
        return self._width_pix

    @width_pix.setter
    def width_pix(self, value):
        self._width_pix = value

    @property
    def height_pix(self):
        return self._height_pix
    
    @height_pix.setter
    def height_pix(self, value):
        self._height_pix_ = value

    @property
    def cb_width(self):
        return self._cb_width
    
    @cb_width.setter
    def cb_width(self, value):
        self._cb_width = value
    
    @property
    def cb_height(self):
        return self._cb_height
    
    @cb_height.setter
    def cb_height(self, value):
        self._cb_height = value

    @property
    def x_factor(self):
        return self._x_factor

    @property
    def y_factor(self):
        return self._y_factor       


    def mark_crop_positions(self):
        task = MarkerTask("crop area", ["bottom left corner", "top left corner", "top right corner", "bottom right corner"], "Mark crop area")
        im = ImageMarker([task])
        
        marker_positions = im.mark_movie(file_name, frame_number)
        print(marker_positions)

        np.save('marker_positions', marker_positions)

        return marker_positions


    def crop_frame(self, frame, marker_positions):

        bottom_left = marker_positions[0]['bottom left corner']
        bottom_right = marker_positions[0]['bottom right corner']
        top_left = marker_positions[0]['top left corner']
        top_right = marker_positions[0]['top right corner']

        left_bound = int(np.mean([bottom_left[0], top_left[0]]))
        right_bound = int(np.mean([bottom_right[0], top_right[0]]))
        top_bound = int(np.mean([top_left[1], top_right[1]]))
        bottom_bound = int(np.mean([bottom_left[1], bottom_right[1]]))
        
        crop_frame = frame[top_bound:bottom_bound, left_bound:right_bound]
        crop_frame = np.mean(crop_frame, axis=2)

        frame_width = np.mean(crop_frame,axis=0)
        frame_height = np.mean(crop_frame,axis=1)

        diff_width = np.diff(frame_width)
        diff_height = np.diff(frame_height)

        x_width = np.arange(0, len(diff_width), 1)
        x_height = np.arange(0, len(diff_height), 1)
        return frame_width, frame_height, diff_width, diff_height, x_width, x_height


    def rotation_angle():
        pass


    def threshold_crossings(self, data, threshold_factor):
        lower_threshold = np.min(data) / threshold_factor
        upper_threshold = np.max(data) / threshold_factor
        
        lower_crossings = np.diff(data < lower_threshold, prepend=False)
        upper_crossings = np.diff(data > upper_threshold, append=False)
        
        lower_crossings_indices = np.argwhere(lower_crossings)
        upper_crossings_indices = np.argwhere(upper_crossings)
        
        half_window_size = 10
        lower_peaks = []
        upper_peaks = []
        for lower_idx in lower_crossings_indices:
            if lower_idx < half_window_size:
                half_window_size = lower_idx
            window = data[lower_idx[0] - int(half_window_size):lower_idx[0] + int(half_window_size)]
            min_window = np.min(window)
            min_idx = np.where(data == min_window)
            lower_peaks.append(min_idx)

        for upper_idx in upper_crossings_indices:
            if upper_idx < half_window_size:
                half_window_size = upper_idx
            window = data[upper_idx[0] -  int(half_window_size) : upper_idx[0] + int(half_window_size)]
            
            max_window = np.max(window)
            max_idx = np.where(data == max_window)
            upper_peaks.append(max_idx)    
        
        lower_peaks = np.unique(lower_peaks)
        upper_peaks = np.unique(upper_peaks)
               
        return lower_peaks, upper_peaks


    def detect_checkerboard(self, filename, frame_number, marker_positions):
        
        if not os.path.exists(filename):
            raise IOError("file %s does not exist!" % filename)
        video = cv2.VideoCapture()
        video.open(filename)
        frame_counter = 0
        success = True
        frame = None
        
        x_0 = self._x_0
        y_0 = self._y_0
        width_pix = self._width_pix
        height_pix = self._height_pix

        while success and frame_counter <= frame_number:    # iterating until frame_counter == frame_number --> success (True)
            print("Reading frame: %i" % frame_counter, end="\r")
            success, frame = video.read()
            frame_counter += 1

        marker_positions = np.load('marker_positions.npy', allow_pickle=True)

        frame_width, frame_height, diff_width, diff_height, _, _ = dc.crop_frame(frame, marker_positions)
        
        # y-axis is inverted..        
        
        thresh_fact = 7
        lci_width, uci_width = dc.threshold_crossings(diff_width, threshold_factor=thresh_fact)
        lci_height, uci_height = dc.threshold_crossings(diff_height, threshold_factor=thresh_fact)
        
        print('lower crossings:', lci_width)
        print('upper crossings:', uci_width)

        # make function for this
        zip_list = []
        for zl in lci_width:
            zip_list.append(zl)
        for zu in uci_width:
            zip_list.append(zu)
        
        zip_list = np.sort(zip_list)

        sequence = []
        for z in zip_list:
            if z in lci_width:
                sequence.append('down')
            else:
                sequence.append('up')
        print('sequence:', sequence)

        if sequence == ['up', 'down', 'up', 'down']:
            print('in middle')
            # first down, second up are edges of checkerboard
        elif sequence == ['up', 'up', 'down']:
            print('at left')
            # first and second up are edges of checkerboard
        else:
            print('at right')
            # first and second down are edges of checkerboard
        
        # find mistake in threshold detection (_7.mp4) where two detections at side (by thresh factor)
        # find which indices (=pixels) represent edges of checkerboard by the corresponding sequence of ups and downs
        # both for width and height
        # assign x and y positions for the checkerboard corners

        # pixel to meter factor for default position with checkerboard in center of tank underneath camera
        
        plt.plot(diff_width)
        plt.axhline(np.min(diff_width) / thresh_fact)
        plt.axhline(np.max(diff_width) / thresh_fact)
        for l in lci_width:
            plt.axvline(l, color='yellow')
        for u in uci_width:
            plt.axvline(u, color='green')
        # plt.plot(frame_height)
        plt.plot(frame_width)
        plt.show()
        embed()
        quit()
        
        # rotation angle
        
      
if __name__ == "__main__":    
    file_name = "/home/efish/etrack/videos/2022.03.28_7.mp4"
    frame_number = 10
    dc = DistanceCalibration(file_name=file_name, frame_number=frame_number)
    
    # marker_positions = dc.mark_crop_positions()
    
    dc.detect_checkerboard(file_name, frame_number=frame_number, marker_positions=np.load('marker_positions.npy', allow_pickle=True))
    
    # print(sys.argv[0])
    # print (sys.argv[1])
    # vid1 = sys.argv[1]
    
    # embed()