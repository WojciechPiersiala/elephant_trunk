#!/home/wp/Studia/soft_robotics/elephant_trunk/trunk/bin/python
#register the project
from manipulator.trunk_environment import TrunkEnv  # Import TrunkEnv

register = True
if register:
    print("registering..")
    env_reg = TrunkEnv(target=[10,-30])
    env_reg.close()