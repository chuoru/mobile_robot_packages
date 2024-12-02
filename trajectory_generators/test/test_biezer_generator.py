
# Standard library
import os
import sys
# import math
# import numpy as np
# from matplotlib import pyplot as plt

# Internal library
sys.path.append(os.path.join("..", ".."))
from models.differential_drive import DifferentialDrive  # noqa
from trajectory_generators.biezer_generator import BiezerGenerator  # noqa


if __name__ == "__main__":
    model = DifferentialDrive(wheel_base=0.53)

    trajectory = BiezerGenerator()

    
