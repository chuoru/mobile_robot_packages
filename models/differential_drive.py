#!/usr/bin/env python3
##
# @file differential_drive.py
#
# @brief Provide implementation of the differential drive model.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 27/08/2024.
#
# Copyright (c) 2024 System Engineering Laboratory.  All rights reserved.

# Standard library
import math

# External library
import numpy as np


class DifferentialDrive:
    """! Differential drive model

    The class provides implementation of the differential drive model.
    @note The model is used to simulate the vehicle motion.
    @variable nx: The number of states. The states are [x, y, theta]
    @variable nu: The number of inputs. The inputs are [v, w]
    """
    nx = 3

    nu = 2

    def __init__(self, wheel_base):
        """! Constructor
        @param wheel_base<float>: The wheel base of the vehicle
        @note The wheel base is the distance between the front and rear axles.
        """
        self.wheel_base = wheel_base

        self.velocity_max = 1.0

    def function(self, state, input, dt):
        """! Compute the next state
        @param state<list>: The current state of the vehicle
        @param input<list>: The input of the vehicle
        @param dt<float>: The time step
        @return next_state<list>: The next state of the vehicle
        """
        v = input[0]

        w = input[1]

        dfdt = np.array([math.cos(state[2]) * v, math.sin(state[2]) * v, w])

        next_state = state + dfdt * dt

        return next_state
