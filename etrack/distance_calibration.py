from cv2 import calibrationMatrixValues
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import os
import sys
from IPython import embed


class DistanceCalibration:

    def __init__(self, file_name, x_0=95, y_0=185, cam_dist=1.36, width=1.35, height=0.805, width_pixel=1975, height_pixel=1375, checkerboard_width=0.24, checkerboard_height=0.18, 
    checkerboard_width_pixel=500, checkerboard_height_pixel=350, rectangle_width=0.024, rectangle_height=0.0225, rectangle_width_pixel=100, rectangle_height_pixel=90, 
    rectangle_count_width=9, rectangle_count_height=7) -> None:
        # aktualisieren
        """Calibration of the dimensions of the tank. Conversion of pixel into meter. Width refers to the "x-axis", height to the "y-axis" of the tank.
        
        Args:
            file_name (_type_): _description_
            x_0 (int, optional): X-value of the "origin" of the tank. Defaults to 0.
            y_0 (int, optional): Y-value of the "origin" of the tank. Defaults to 0.
            cam_dist (int, optional): Distance of camera lense to tank floor. Defaults to 1.36.
            width (int, optional): Width in meter from one lightened corner of the tank to the other. Defaults to 1.35.
            heigth (int, optional): Height in meter from one lightened corner of the tank to the other. Defaults to 1.35.
            width_pixel (int, optional): Width in pixel from one lightened corner of the tank to the other. Defaults to 1975.
            height_pixel (int, optional): Heigth in pixel from one lightened corner of the tank to the other.  Defaults to 1375.
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

        # properties
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
    def x_factor(self):
        return self._x_factor

    @property
    def y_factor(self):
        return self._y_factor


    def crop_movie():
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
        if success:
           self._fig.gca().imshow(frame)    # plot wanted frame of video
        else:
           print("Could not read frame number %i either failed to open movie or beyond maximum frame number!" % frame_number)
           return []
        plt.ion()   # turn on interactive mode
        plt.show(block=False)   # block=False allows to continue interact in terminal while the figure is open
        
        self._task_index = -1
        if len(self._tasks) > 0:
            self._next_task()
        
        while not self._tasks_done:
            plt.pause(0.250)
            if self._interrupt:
                return []

        self._fig.gca().set_title("All set and done!\n Window will close in 2s")
        self._fig.canvas.draw()
        plt.pause(2.0)
        plt.close()
        return [t.marker_positions for t in self._tasks]


    def mark_checkerboard(self, filename, frame_number=10):
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
        width_mean = np.mean(frame,axis=1)
        crop_width_mean = width_mean[x_0:width_pix]
        
        height_mean = np.mean(frame,axis=0)
        crop_height_mean = height_mean[y_0:height_pix]
# HELLO, here you at

        embed()
        quit()
        if success:
           self._fig.gca().imshow(frame)    # plot wanted frame of video
        else:
           print("Could not read frame number %i either failed to open movie or beyond maximum frame number!" % frame_number)
           return []
        plt.ion()   # turn on interactive mode
        plt.show(block=False)   # block=False allows to continue interact in terminal while the figure is open
        
        self._task_index = -1
        if len(self._tasks) > 0:
            self._next_task()
        
        while not self._tasks_done:
            plt.pause(0.250)
            if self._interrupt:
                return []

        self._fig.gca().set_title("All set and done!\n Window will close in 2s")
        self._fig.canvas.draw()
        plt.pause(2.0)
        plt.close()
        return [t.marker_positions for t in self._tasks]    
        

if __name__ == "__main__":    
    vid2 = "/home/efish/etrack/videos/2022.03.28_3.mp4"
    calibration_task = DistanceCalibration(vid2)
    dc = DistanceCalibration(calibration_task)
    dc.mark_checkerboard(vid2, 10)

    # print(sys.argv[0])
    # print (sys.argv[1])
    # vid1 = sys.argv[1]
    
    embed()