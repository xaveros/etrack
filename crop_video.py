from csv import QUOTE_NONNUMERIC
from multiprocessing import allow_connection_pickling
from re import M
from turtle import left
from xml.dom.expatbuilder import FILTER_ACCEPT
from cv2 import MARKER_TRIANGLE_UP, calibrationMatrixValues, mean, threshold
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import os
import sys
import glob
import os
import uuid
from ffmpy import FFmpeg
from IPython import embed
from calibration_functions import *



class CropVideo():
    
    def __init__(self, file_name) -> None: 
        super().__init__()

        self._file_name = file_name
        self.cut_out_video
        self.mark_crop_positions


    def mark_crop_positions(self, file_name, frame_number):
        
        task = MarkerTask("crop area", ["bottom left corner", "top left corner", "top right corner", "bottom right corner"], "Mark crop area")
        im = ImageMarker([task])
        
        marker_positions = im.mark_movie(file_name, frame_number)
        print(marker_positions)

        np.save('marker_positions', marker_positions)
        plt.close()
        return marker_positions


    def cut_out_video(self, video_path: str, output_dir: str, start_pix: tuple, size: tuple):
        ext = os.path.basename(video_path).strip().split('.')[-1]
        print(ext)
        output_path = video_path.split('/')[-1].split('.')[:-1]
        output_path = '.'.join(output_path)
        if ext not in ['mp4', 'avi', 'flv']:
            raise Exception('format error')
        original_result = os.path.join(output_dir, '{}_original.{}'.format(output_path, ext))
        
        print(output_path)
        ff = FFmpeg(inputs={video_path: None},
                    outputs={
                        result: '-filter:v crop={}:{}:{}:{} {}_out.mp4'.format(size[0] - start_pix[0], start_pix[1] - size[1], start_pix[0], size[1], original_result)})
        
        ff.run()
        return result
'''
start_pix                                                               
Out[1]: (167, 1281)

In [2]: size                                                                    
Out[2]: (1887, 221)

ffmpeg -i in.mp4 -filter:v "crop=out_w:out_h:x:y" out.mp4

Where the options are as follows:

    out_w is the width of the output rectangle
    out_h is the height of the output rectangle
    x and y specify the top left corner of the output rectangle

'''

if __name__ == "__main__":    
    
    for file_name in glob.glob("/home/efish/etrack/videos/*"): 
        frame_number = 10
        cv = CropVideo(file_name=file_name)
        
        destination_folder = "/home/efish/etrack/cropped_videos/"
        

        marker_positions = cv.mark_crop_positions(file_name=file_name, frame_number=frame_number)
        print('marker positions done')
        first1 = int(marker_positions[0]['bottom left corner'][0]) 
        first2 = int(marker_positions[0]['bottom left corner'][1])
        second1 = int(marker_positions[0]['top right corner'][0])
        second2 = int(marker_positions[0]['top right corner'][1])
        result = cv.cut_out_video(file_name, destination_folder, (first1, first2), (second1, second2))
        print(result)
    