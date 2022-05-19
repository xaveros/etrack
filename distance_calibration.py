from multiprocessing import allow_connection_pickling
from turtle import left
from xml.dom.expatbuilder import FILTER_ACCEPT
from cv2 import MARKER_TRIANGLE_UP, calibrationMatrixValues, mean, threshold
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import os
import sys
import glob
from IPython import embed
from calibration_functions import *



class DistanceCalibration():

    def __init__(self, file_name, frame_number, x_0=154, y_0=1318, cam_dist=1.36, tank_width=1.35, tank_height=0.805, width_pixel=1900, height_pixel=200, 
    checkerboard_width=0.24, checkerboard_height=0.18, checkerboard_width_pixel=500, checkerboard_height_pixel=350) -> None: 
        super().__init__()
                
        self._file_name = file_name
        self._x_0 = x_0
        self._y_0 = y_0
        self._width_pix = width_pixel
        self._height_pix = height_pixel
        self._cam_dist = cam_dist
        self._tank_width = tank_width
        self._tank_height = tank_height
        self._cb_width = checkerboard_width
        self._cb_height = checkerboard_height
        self._cb_width_pix = checkerboard_width_pixel
        self._cb_height_pix = checkerboard_height_pixel
        self._x_factor = tank_width / width_pixel   # m/pix
        self._y_factor = tank_height / height_pixel # m/pix

        self.distance_factor_calculation
        self.mark_crop_positions

    # if needed include setter: @y_0.setter   def y_0(self, value):   self._y_0 = value
    @property
    def x_0(self):
        return self._x_0

    @property
    def y_0(self):
        return self._y_0

    @property
    def cam_dist(self):
        return self._cam_dist
    
    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height
    
    @property
    def width_pix(self):
        return self._width_pix

    @property
    def height_pix(self):
        return self._height_pix
    
    @property
    def cb_width(self):
        return self._cb_width

    @property
    def cb_height(self):
        return self._cb_height
    
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


    def detect_checkerboard(self, filename, frame_number, marker_positions):
        # load frame
        if not os.path.exists(filename):
            raise IOError("file %s does not exist!" % filename)
        video = cv2.VideoCapture()
        video.open(filename)
        frame_counter = 0
        success = True
        frame = None

        while success and frame_counter <= frame_number:    # iterating until frame_counter == frame_number --> success (True)
            print("Reading frame: %i" % frame_counter, end="\r")
            success, frame = video.read()
            frame_counter += 1

        marker_positions = np.load('marker_positions.npy', allow_pickle=True)   # load saved numpy marker positions file
        
        # care: y-axis is inverted, top values are low, bottom values are high

        cropped_frame, frame_width, frame_height, diff_width, diff_height, _, _ = crop_frame(frame, marker_positions)   # crop frame to given marker positions

        bottom_left_x = 0
        bottom_left_y = np.shape(cropped_frame)[0]
        bottom_right_x = np.shape(cropped_frame)[1]
        bottom_right_y = np.shape(cropped_frame)[0]
        top_left_x = 0
        top_left_y = 0
        top_right_x = np.shape(cropped_frame)[1]
        top_right_y = 0

        cropped_marker_positions = [{'bottom left corner': (bottom_left_x, bottom_left_y), 'top left corner': (top_left_x, top_left_y), 
        'top right corner': (top_right_x, top_right_y), 'bottom right corner': (bottom_right_x, bottom_right_y)}]

        thresh_fact = 7    # factor by which the min/max is divided to calculate the upper and lower thresholds 
        
        # filtering/smoothing of data using kernel with n datapoints
        kernel = 4
        diff_width = filter_data(diff_width, n=kernel)  # for widht (x-axis)
        diff_height = filter_data(diff_height, n=kernel)    # for height (y-axis)

        # input data is derivation of color values of frame
        lci_width, uci_width = threshold_crossings(diff_width, threshold_factor=thresh_fact)     # threshold crossings (=edges of checkerboard) for width (x-axis)
        lci_height, uci_height = threshold_crossings(diff_height, threshold_factor=thresh_fact)  # ..for height (y-axis)

        # position of checkerboard in width
        width_position, left_width_position, right_width_position = checkerboard_position(lci_width, uci_width)
        
        # position of checkerboard in height
        height_position, left_height_position, right_height_position = checkerboard_position(lci_height, uci_height)  # left height refers to top, right height to bottom
        
        # structure of frame is different from (x,y), it is (row,column) where column represents the width (x-axis) of the frame
        width_window_size = 300
        height_window_size = 250
        if width_position == 'left' and height_position == 'left':
            checkerboard_position_tank = 'top left'
        elif width_position == 'left' and height_position == 'right':
            checkerboard_position_tank = 'bottom left'
            
            bottom_left_window = cropped_frame[bottom_left_y - height_window_size:bottom_left_y, bottom_left_x:bottom_left_x + width_window_size]
            embed()
            # y_line = bottom_left_window[:, 10]
            # x_line = bottom_left_window[10, :]
        elif width_position == 'right' and height_position == 'right':
            checkerboard_position_tank = 'bottom right'
        elif width_position == 'right' and height_position == 'left':
            checkerboard_position_tank = 'top right'
            
            top_right_window = cropped_frame[top_right_y:top_right_y + height_window_size, top_right_x - width_window_size: top_right_x]
            
            height_line = top_right_window[:, 200]
            width_line = top_right_window[50, :]

            for idx, wl in enumerate(width_line):
                if idx == 0:
                    continue
                if wl[idx] > wl[idx-1]:
                    # checkerboard corner = corner of cropped frame

            # kacke, zu fein und unzuverlässlich. Nächster Ansatz: frame binärisieren mit 0 oder 1 für bestimmte Farbwerte 
            # damit Kontrast quasi größer ist und an Rändern Detektion genauer wird (hoffentlich.)


            embed()
            quit()
        else: 
            checkerboard_position_tank = 'middle'

        print(checkerboard_position_tank)   

        # final corner positions of checkerboard
        checkerboard_marker_positions = [{'bottom left corner': (left_width_position, right_height_position), 'top left corner': (left_width_position, left_height_position), 
        'top right corner': (right_width_position, left_height_position), 'bottom right corner': (right_width_position, right_height_position)}]

        print('checkerboard_marker_positions:', checkerboard_marker_positions)
        
        checkerboard_top_right, checkerboard_top_left, checkerboard_bottom_right, checkerboard_bottom_left = assign_checkerboard_positions(checkerboard_marker_positions)

        fig, ax = plt.subplots()
        ax.imshow(cropped_frame)
        for p in checkerboard_top_left, checkerboard_top_right, checkerboard_bottom_left, checkerboard_bottom_right:
            ax.scatter(p[0], p[1])
        ax.scatter(bottom_left_x, bottom_left_y)
        ax.scatter(bottom_right_x, bottom_right_y)
        ax.scatter(top_left_x, top_left_y)
        ax.scatter(top_right_x, top_right_y)
        plt.show()

       
        return checkerboard_marker_positions, cropped_marker_positions, checkerboard_position_tank

    
    def distance_factor_calculation(self, checkerboard_marker_positions, marker_positions):
        
        checkerboard_top_right, checkerboard_top_left, checkerboard_bottom_right, checkerboard_bottom_left = assign_checkerboard_positions(checkerboard_marker_positions)

        checkerboard_width = 0.24
        checkerboard_height = 0.18
        
        checkerboard_width_pixel = checkerboard_top_right[0] - checkerboard_top_left[0]
        checkerboard_height_pixel = checkerboard_bottom_right[1] - checkerboard_top_right[1]

        x_factor = checkerboard_width / checkerboard_width_pixel
        y_factor = checkerboard_height / checkerboard_height_pixel

        bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y, top_left_x, top_left_y, top_right_x, top_right_y = assign_marker_positions(marker_positions)

        tank_width_pixel = np.mean([bottom_right_x - bottom_left_x, top_right_x - top_left_x]) 
        tank_height_pixel = np.mean([bottom_left_y - top_left_y, bottom_right_y - top_right_y])

        tank_width = tank_width_pixel * x_factor
        tank_height = tank_height_pixel * y_factor
        
        print(tank_width, tank_height)    
     
        return x_factor, y_factor


    def distance_factor_interpolation(x_factors, y_factors):
        pass


if __name__ == "__main__":    
    
    all_x_factor = []
    all_y_factor = []
    all_checkerboard_position_tank = []
    for file_name in glob.glob("/home/efish/etrack/videos/*"): 
        # file_name = "/home/efish/etrack/videos/2022.03.28_4.mp4"
        frame_number = 10
        dc = DistanceCalibration(file_name=file_name, frame_number=frame_number)
        
        # dc.mark_crop_positions()
        
        checkerboard_marker_positions, cropped_marker_positions, checkerboard_position_tank = dc.detect_checkerboard(file_name, frame_number=frame_number, marker_positions=np.load('marker_positions.npy', allow_pickle=True))
        
        x_factor, y_factor = dc.distance_factor_calculation(checkerboard_marker_positions, marker_positions=cropped_marker_positions)
            
        all_x_factor.append(x_factor)
        all_y_factor.append(y_factor)
        all_checkerboard_position_tank.append(checkerboard_position_tank)

    x_factors = np.load('x_factors.npy')
    y_factors = np.load('y_factors.npy')
    all_checkerboard_position_tank = np.load('all_checkerboard_position_tank.npy')
    print(x_factors)
    print(y_factors)
    

    # embed()
    # quit()
        
    # next up: distance calculation with angle
        # is this needed or are current videos enough?: 
            # laying checkerboard at position directly above and below / left and right to centered checkerboard near edge of tank
        # calculating x and y factor for centered checkerboard, then for the ones at the edge 
        # --> afterwards interpolate between them to have continuous factors for whole tank
        # maybe smaller object in tank to have more accurate factor

    # make function to refine checkerboard detection at edges of tank by saying if no lower color values appears near edge --> checkerboard position then == corner of tank? 
    #  
    # mark_crop_positions why failing plot at end?
    # with rectangles of checkerboard?

    # embed()