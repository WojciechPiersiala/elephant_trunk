from manipulator.segment import Segment
import numpy as np
import matplotlib.pyplot as plt
from pynput.keyboard import Key, Listener


class Manipulator():
    def __init__(self, manual_mode :bool=True, target_cor :list[float,float]=[0.0, 0.0], render_mode :str = "human"):
        self.render_modes = ["human", "rgb"]

        self.render_mode = render_mode
        assert render_mode in self.render_modes, f"tender mode incorrect. Correct render modes: {self.render_modes}"
        if manual_mode:
            assert self.render_mode == "human", "manual mode possible only when render_mode is set to human"
        
        # plots
        self.fig, self.ax = plt.subplots()
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-30,30)
        self.ax.set_ylim(-40,0)

        self.segments = [] 

        for i in range(4):
            if i == 0:
                segment = Segment(r=3-0.1*i, k=-0.0, orientation=np.pi)
                self.segments.append(segment)
            else:
                segment = Segment(r=3-0.1*i, 
                                  k=-0.0, 
                                  position=[self.segments[i-1].x_end,self.segments[i-1].y_end], 
                                  orientation=self.segments[i-1].orientation_end)
                self.segments.append(segment)

        self.arc0, = self.ax.plot(self.segments[0].transformed_x, self.segments[0].transformed_y, '-b', lw=2) 
        self.arc1, = self.ax.plot(self.segments[1].transformed_x, self.segments[1].transformed_y, '-r', lw=2)
        self.arc2, = self.ax.plot(self.segments[2].transformed_x, self.segments[2].transformed_y, '-g', lw=2)
        self.arc3, = self.ax.plot(self.segments[3].transformed_x, self.segments[3].transformed_y, '-m', lw=2)

        #end effector
        self.end_effector_x = self.segments[3].x_end
        self.end_effector_y = self.segments[3].y_end

        # rarget
        self.target_cor = target_cor
        self.target = self.ax.scatter(self.target_cor[0], self.target_cor[1], color='black', label='Target')
        self.dist_to_target = 100
        self.on_target = False

        # manual mode
        self.manual_mode = manual_mode
        self.seg_idx = 0
        self.seg_move = [0,0,0,0]
        self.dk = 0
        self.move = False

        # normal mode
        self.action_space = [0, 0, 0, 0] # each element in this vector specifies the curvature update k of a single segment
        self.state_space = [[0, 0], [0, 0]] # first elemnt contains the coordinates of the end effector and secod the coordinates of the target



    def run(self, action_space :list[float,float,float,float]=[0,0,0,0]):
        self.track_target()
        if self.manual_mode:
            self.segments[0].update(self.move*self.seg_move[0]*self.dk)
            # print(f"k: {self.segments[0].k}")
            self.arc0.set_data(self.segments[0].transformed_x, self.segments[0].transformed_y)

            self.segments[1].update(self.move*self.seg_move[1]*self.dk,
                                    [self.segments[0].x_end, self.segments[0].y_end],
                                    self.segments[0].orientation_end)
            self.arc1.set_data(self.segments[1].transformed_x, self.segments[1].transformed_y)

            self.segments[2].update(self.move*self.seg_move[2]*self.dk,
                                    [self.segments[1].x_end, self.segments[1].y_end],self.segments[1].orientation_end)
            self.arc2.set_data(self.segments[2].transformed_x, self.segments[2].transformed_y)

            self.segments[3].update(self.move*self.seg_move[3]*self.dk,
                                    [self.segments[2].x_end, self.segments[2].y_end],self.segments[2].orientation_end)
            self.arc3.set_data(self.segments[3].transformed_x, self.segments[3].transformed_y)
            self.move = False
            # print(f"move: {self.seg_move}, dk: {self.dk} \n")

        else:
            self.action_space = action_space
            self.segments[0].update(self.action_space[0])
            # print(f"k: {self.segments[0].k}")
            self.arc0.set_data(self.segments[0].transformed_x, self.segments[0].transformed_y)

            self.segments[1].update(self.action_space[1],
                                    [self.segments[0].x_end, self.segments[0].y_end],
                                    self.segments[0].orientation_end)
            self.arc1.set_data(self.segments[1].transformed_x, self.segments[1].transformed_y)

            self.segments[2].update(self.action_space[2],
                                    [self.segments[1].x_end, self.segments[1].y_end],self.segments[1].orientation_end)
            self.arc2.set_data(self.segments[2].transformed_x, self.segments[2].transformed_y)

            self.segments[3].update(self.action_space[3],
                                    [self.segments[2].x_end, self.segments[2].y_end],self.segments[2].orientation_end)
            self.arc3.set_data(self.segments[3].transformed_x, self.segments[3].transformed_y)


    def track_target(self):
        self.end_effector_x = self.segments[3].x_end
        self.end_effector_y = self.segments[3].y_end
        self.dist_to_target = np.sqrt((self.end_effector_x - self.target_cor[0])**2 +  (self.end_effector_y - self.target_cor[1])**2)
        # print(f"end effector: [{self.end_effector_x}, {self.end_effector_y}], \t target: {self.target_cor}, \t dist: {self.dist_to_target}")

        self.state_space = [[self.end_effector_x, self.end_effector_y],self.target_cor]
        DIST_TO_TARGET_OK = 0.5
        if self.dist_to_target < DIST_TO_TARGET_OK:
            self.on_target = True



    def manual_move(self,key):
        # print(f"Key {key} pressed")
        SK_STEP = 0.003
        # change the index
        if key == Key.up:
            self.seg_idx += 1
        elif key == Key.down:
            self.seg_idx -= 1

        self.seg_idx = max(0, min(self.seg_idx, 3))
        for i in range(len(self.seg_move)):
            self.seg_move[i] = 0
        self.seg_move[self.seg_idx] = 1

        # move the selected segment
        if key == Key.left:
            self.dk =+ SK_STEP 
            self.move = True
        elif key == Key.right:
            self.dk =- SK_STEP
            self.move = True
