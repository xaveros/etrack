import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import numbers as nb
import os 

"""
x_0 = 0
width = 1230
y_0 = 0 
height = 1100
x_factor = 0.81/width  # Einheit m/px 
y_factor = 0.81/height  # Einheit m/px 
center = (np.round(x_0 + width/2), np.round(y_0 + height/2))
center_meter = ((center[0] - x_0) * x_factor, (center[1] - y_0) * y_factor)
"""

class TrackingResult(object):
    
    def __init__(self, results_file, x_0=0, y_0= 0, width_pixel=1975, height_pixel=1375, width_meter=0.81, height_meter=0.81) -> None:
        super().__init__()
        """Width refers to the "x-axis" of the tank, height to the "y-axis" of it. 

        Args:
            results_file (_type_): Results file of the before done animal tracking.
            x_0 (int, optional): . Defaults to 95.
            y_0 (int, optional): _description_. Defaults to 185.
            width_pixel (int, optional): Width from one lightened corner of the tank to the other. Defaults to 1975.
            height_pixel (int, optional): Heigth from one lightened corner of the tank to the other.  Defaults to 1375.
            width_meter (float, optional): Width of the tank in meter. Defaults to 0.81.
            height_meter (float, optional): Height of the tank in meter. Defaults to 0.81.
        """
        if not os.path.exists(results_file):
            raise ValueError("File %s does not exist!" % results_file)
        self._file_name = results_file
        self.x_0 = x_0
        self.y_0 = y_0
        self.width_pix = width_pixel
        self.width_m = width_meter
        self.height_pix = height_pixel
        self.height_m = height_meter
        self.x_factor = self.width_m / self.width_pix   # m/pix
        self.y_factor = self.height_m / self.height_pix # m/pix

        self.center = (np.round(self.x_0 + self.width_pix/2), np.round(self.y_0 + self.height_pix/2))   # middle of width and height --> center
        self.center_meter = ((self.center[0] - self.x_0) * self.x_factor, (self.center[1] - self.y_0) * self.y_factor)  # center in meter by multipling with factor

        self._data_frame = pd.read_hdf(results_file)    # read dataframe of scorer
        self._level_shape = self._data_frame.columns.levshape   # shape of dataframe (?)
        self._scorer = self._data_frame.columns.levels[0].values    # scorer of dataset
        self._bodyparts = self._data_frame.columns.levels[1].values if self._level_shape[1] > 0 else [] # tracked body parts 
        self._positions = self._data_frame.columns.levels[2].values if self._level_shape[2] > 0 else [] # position in x and y values and the likelihood of it

    def angle_to_center(self, bodypart=0, twopi=True, inversed_yaxis=False, min_likelihood=0.95):
        """Angel of animal position in relation to the center of the tank.

        Args:
            bodypart (int, optional): Bodypart of the animal. Defaults to 0.
            twopi (bool, optional): _description_. Defaults to True.
            inversed_yaxis (bool, optional): Inversed y-axis = True when 0 is at the top of axis. Defaults to False.
            min_likelihood (float, optional): The likelihood of the position estimation. Defaults to 0.95.

        Raises:
            ValueError: No valid x-position values. 

        Returns:
            phi: Angle of animal in relation to center.
        """
        if  isinstance(bodypart, nb.Number):    # check if the instance bodypart of this class is a number
            bp = self._bodyparts[bodypart]  
        elif isinstance(bodypart, str) and bodypart in self._bodyparts: # or if bodypart is a string
            bp = bodypart
        else:
            raise ValueError("Bodypart %s is not in dataframe!" % bodypart) # or if it is existing
        _, x, y, _, _ = self.position_values(bodypart=bp, min_likelihood=min_likelihood)    # set x and y values, already in meter from position_values
        if x is None:
            print("Error: no valid angles for %s" % self._file_name)
            return []
        x_to_center = x - self.center_meter[0]  # 
        y_to_center = y - self.center_meter[1]
        if inversed_yaxis == True:
            y_to_center *= -1
        phi = np.arctan2(y_to_center, x_to_center) * 180 / np.pi
        if twopi:
            phi[phi < 0] = 360 + phi[phi < 0]
        
        return phi

    def coordinate_transformation(self, position):
        x = (position[0] - self.x_0) * self.x_factor
        y = (position[1] - self.y_0) * self.y_factor
        return (x, y) #in m 

    @property
    def filename(self):
        return self._file_name

    @property
    def dataframe(self):
        return self._data_frame

    @property
    def scorer(self):
        return self._scorer

    @property
    def bodyparts(self):
        return self._bodyparts

    @property
    def positions(self):
        return self._positions

    def position_values(self, scorer=0, bodypart=0, framerate=25, interpolate=True, min_likelihood=0.95):
        """Returns the x and y positions of a bodypart over time and the likelihood of it.

        Args:
            scorer (int, optional): Scorer of dataset. Defaults to 0.
            bodypart (int, optional): Bodypart of the animal. Can be seen in etrack.TrackingResults.bodyparts. Defaults to 0.
            framerate (int, optional): Framerate of the video. Defaults to 25.

        Raises:
            ValueError: Scorer not existing in dataframe.
            ValueError: Bodypart not existing in dataframe. 

        Returns:
            time [np.array]: The time axis.
            x [np.array]: x-position in meter.
            y [np.array]: y-position in meter.
            l [np.array]: The likelihood of the position estimation. Originating from animal tracking done before. 
            bp string: The body part of the animal.
            [type]: [description]
        """

        if  isinstance(scorer, nb.Number):
            sc = self._scorer[scorer]
        elif isinstance(scorer, str) and scorer in self._scorer:
            sc = scorer
        else:
            raise ValueError("Scorer %s is not in dataframe!" % scorer)
        if  isinstance(bodypart, nb.Number):
            bp = self._bodyparts[bodypart]
        elif isinstance(bodypart, str) and bodypart in self._bodyparts:
            bp = bodypart
        else:
            raise ValueError("Bodypart %s is not in dataframe!" % bodypart)

        x = self._data_frame[sc][bp]["x"] if "x" in self._positions else []
        x = (np.asarray(x) - self.x_0) * self.x_factor
        y = self._data_frame[sc][bp]["y"] if "y" in self._positions else []
        y = (np.asarray(y) - self.y_0) * self.y_factor
        l = self._data_frame[sc][bp]["likelihood"] if "likelihood" in self._positions else []

        time = np.arange(len(self._data_frame))/framerate
        time2 = time[l > min_likelihood]
        if len(l[l > min_likelihood]) < 100:
            print("%s has not datapoints with likelihood larger than %.2f" % (self._file_name, min_likelihood) )
            return None, None, None, None, None
        x2 = x[l > min_likelihood]
        y2 = y[l > min_likelihood]
        x3 = np.interp(time, time2, x2)
        y3 = np.interp(time, time2, y2)
        return time, x3, y3, l, bp


    def plot(self, scorer=0, bodypart=0, threshold=0.9, framerate=25):
        """Plot the position of a bodypart in the tank over time.

        Args:
            scorer (int, optional): Scorer of dataset. Defaults to 0.
            bodypart (int, optional): Given bodypart to plot. Defaults to 0.
            threshold (float, optional): Threshold below which the likelihood has to be. Defaults to 0.9.
            framerate (int, optional): Framerate of the video. Defaults to 25.
        """
        t, x, y, l, name  = self.position_values(scorer=scorer, bodypart=bodypart, framerate=framerate)
        plt.scatter(x[l > threshold], y[l > threshold], c=t[l > threshold], label=name)
        plt.scatter(self.center_meter[0], self.center_meter[1], marker="*")
        plt.plot(x[l > threshold], y[l > threshold])
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.gca().invert_yaxis()
        bar = plt.colorbar()
        bar.set_label("time [s]")
        plt.legend()
        plt.show()
        
        pass

if __name__ == '__main__':
    from IPython import embed
    filename = "/2022.01.12_3DLC_resnet50_efish_tracking3Mar21shuffle1_300000.h5"
    path = "/home/efish/efish_tracking/efish_tracking3-Xaver-2022-03-21/videos"

    tr = TrackingResult(path+filename)  # usage of class with given file
    time, x, y, l, bp = tr.position_values(bodypart=2)  # time, x and y values, likelihood of position estimation, tracked bodypart
    phi = tr.angle_to_center(0, True, False, 0.95)

    thresh = 0.95
    time2 = time[l>thresh]  # time values where likelihood of position estimation > threshold
    x2 = x[l>thresh]    # x values with likelihood > threshold
    y2 = y[l>thresh]    # y values -"-
    x3 = np.interp(time, time2, x2) # x value interpolation at points where likelihood has been under threshold
    y3 = np.interp(time, time2, y2) # y value -"- 


    fig, axes = plt.subplots(3,1, sharex=True)                                                                                                                                                  
    axes[0].plot(time, x)
    axes[0].plot(time, x3)                                                                                                                                                                   
    axes[0].set_ylabel('x-position')
    axes[1].plot(time, y)
    axes[1].plot(time, y3)                                                                                                                                                                       
    axes[1].set_ylabel('y-position')
    axes[2].plot(time, l)
    axes[2].set_xlabel('time [s]')    
    axes[2].set_ylabel('likelihood')                            
    plt.show()

    embed()