#!/home/wp/Studia/soft_robotics/elephant_trunk/trunk/bin/python
import matplotlib.pyplot as plt
import numpy as np
import pygame
from manipulator.segment import Segment
from pynput.keyboard import Key, Listener


running = True

# segment_states = 0
# segment_dk = 0
# segments_symbols = [False, False, False]
segment_idx = 1
move = 0
step = 0.0
def on_press(key):
    global running, move, step

    print(f"Key {key} pressed")
    if key == Key.up:
        segment_idx += 1
    elif key == Key.down:
        segment_idx += 1
    
    if key == Key.left:
        move = True
        step = -0.01
    elif key == Key.right:
        move = True
        step = 0.01
    else:
        move = False
    # if segment_states < 0:
    #     segment_states = 0
    # if segment_states > 3:
    #     segment_states = 3
    # for seg in segments_symbols:
    #     seg = False

    # if key == Key.esc:  # Exit on pressing 'Esc'
    #     running = False
    #     return False  # Stop the listener

    # if key == Key.up:
    #     segment_dk = 0
    #     segment_states += 1
    #     print('up')
    # elif key == Key.down:
    #     segment_dk = 0
    #     segment_states -= 1

    # if key == Key.left:
    #     segment_dk = 0.1
    #     segments_symbols[segment_states] = True
    # elif key == Key.right:
    #     segment_dk =- 0.1
    #     segments_symbols[segment_states] = True
    # else: 
    #     segment_dk = 0






def main():
    segment1 = Segment(r=3, k=-0.21, orientation=0.0)
    segment2 = Segment(r=3, k=-0.31, position=[segment1.x_end,segment1.y_end], orientation=segment1.orientation_end)
    segment3 = Segment(r=3, k=-0.31, position=[segment2.x_end,segment2.y_end], orientation=segment2.orientation_end)
    segments = [segment1, segment2, segment3]

    # create the plot
    fig, ax = plt.subplots()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)

    arc1, = ax.plot(segment1.transformed_x, segment1.transformed_y, '-b', lw=2) 
    arc2, = ax.plot(segment2.transformed_x, segment2.transformed_y, '-r', lw=2)
    arc3, = ax.plot(segment3.transformed_x, segment3.transformed_y, '-g', lw=2)

    with Listener(on_press=on_press) as listener:
        while running:
            # print((segments_symbols))
            segments[segment_idx].update()
            # segment1.update(segments_symbols[0]*segment_dk)
            arc1.set_data(segment1.transformed_x, segment1.transformed_y)
            # segment2.update(segments_symbols[1]*segment_dk,[segment1.x_end, segment1.y_end],segment1.orientation_end)
            arc2.set_data(segment2.transformed_x, segment2.transformed_y)
            # segment3.update(segments_symbols[2]*segment_dk,[segment2.x_end, segment2.y_end],segment2.orientation_end)
            arc3.set_data(segment3.transformed_x, segment3.transformed_y)
            plt.pause(0.01)
    plt.show()

if __name__ == "__main__":
    main()

