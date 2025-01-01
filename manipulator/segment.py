import numpy as np
import math



class Segment():
    def __init__(self, r :int, position :list[float, float]=[0.0, 0.0], orientation :float=np.pi, k :float=0, color :str='-b', width :int=2):
        #style
        self.color = color
        self.width = width
        
        self.r = r # radius
        self.x0 = -self.r # origin x coordinate
        self.y0 = 0  # origin y coordinate
        self.k = 1/self.r # curvature
        self.L = self.r *np.pi # initial lenght

        delta_theta = self.L / self.r 
        theta = np.linspace(0, delta_theta, 100) 
        self.arc_x = self.x0 + self.r*np.cos(theta)
        self.arc_y = self.y0 + self.r*np.sin(theta)

        self.position = position #init position
        self.orientation = orientation  # init orientation in radians
        
        self.transformed_x, self.transformed_y = self.transform_arc(self.position, self.orientation)

        self.x_end = 0
        self.y_end = 0
        self.orientation_end = 0

        self.update(k)
        self.k = k



    def transform_arc(self, position :list[float, float], orientation :float) -> list[float, float]:
        rotated_x = self.arc_x * np.cos(orientation) - self.arc_y * np.sin(orientation)
        rotated_y = self.arc_x * np.sin(orientation) + self.arc_y * np.cos(orientation)

        transformed_x = rotated_x + position[0]
        transformed_y = rotated_y + position[1]

        return transformed_x, transformed_y
    


    def update(self, dk:float, position :list[float,float]=None, orientation:float=None):

        if position is not None and orientation is not None:
            self.position = position
            self.orientation = orientation
            
        self.k += dk

        K_LIM = 0.2
        self.k = max(-K_LIM, min(self.k, K_LIM))
        if self.k == 0:
            self.k = 1e-6  # Use a small non-zero curvature
        
        self.r = 1/self.k
        self.x0 = -self.r

        delta_theta = self.L/self.r
        theta = np.linspace(0, delta_theta, 100)
        self.arc_x = self.x0 + self.r*np.cos(theta)
        self.arc_y = self.y0 + self.r*np.sin(theta)
        
        self.transformed_x, self.transformed_y = self.transform_arc(self.position,self.orientation) #aplay transformation


        # preturn the position and orientation of the end effector
        x_last = self.transformed_x[-1]
        y_last = self.transformed_y[-1]
        x_last_but_one = self.transformed_x[-2]
        y_last_but_one = self.transformed_y[-2]
        angle = math.atan2(y_last - y_last_but_one, x_last - x_last_but_one) - np.pi/2

        # print(f"angle : {angle}")

        self.x_end = x_last
        self.y_end = y_last
        self.orientation_end = angle

        
