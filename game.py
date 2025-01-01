#!/home/wp/Studia/soft_robotics/elephant_trunk/trunk/bin/python
import matplotlib.pyplot as plt
from manipulator.manipulator import Manipulator
from pynput.keyboard import Key, Listener
import random


target_x = random.uniform(-20.0, 20.0)
target_y = random.uniform(-30, 0)
running = True  # Global flag to control program execution
manipulator = Manipulator(manual_mode=True, target_cor=[target_x, target_y])


def on_press(key):
    """Handle key press events."""
    global running
    manipulator.manual_move(key)
    if key == Key.esc:  # Exit on pressing 'Esc'
        running = False
        return False


def main():
    global running

    # Set up the keyboard listener
    with Listener(on_press=on_press) as listener:
        while running:
            manipulator.run()  # Update the manipulator's state
            if manipulator.on_target:
                running = False
            plt.pause(0.001)  # Allow matplotlib to render updates
        listener.join()

    plt.show()  # Show the plot after the program ends


if __name__ == "__main__":
    main()
