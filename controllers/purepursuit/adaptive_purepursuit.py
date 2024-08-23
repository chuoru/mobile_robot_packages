#!/usr/bin/env python3
##
# @file adaptive_purepursuit.py
#
# @brief Provide implementation of adaptive pure pursuit controller for
# autonomous driving.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 07/29/2024.
#
# Copyright (c) 2024 System Engineering Laboratory.  All rights reserved.

# Standard library
import math
import numpy as np


class AdaptivePurePursuit:
    """! Adaptive Pure Pursuit controller

    The class provides implementation of adaptive pure pursuit controller for
    autonomous driving.
    """

    lookahead_distance = 2.0

    lookahead_gain = 0.1

    k = 1

    # ==================================================================================================
    # PUBLIC METHODS
    # ==================================================================================================
    def __init__(self, model, trajectory):
        """! Constructor
        @param model<instance>: The vehicle model
        @param trajectory<instance>: The trajectory
        """
        self.model = model

        self.trajectory = trajectory

        self.old_nearest_point_index = None

    def initialize(self):
        """! Initialize the controller"""
        pass

    def execute(self, state, input, previous_index):
        """! Execute the controller
        @param state<list>: The state of the vehicle
        @param input<list>: The input of the vehicle
        @param previous_index<int>: The previous index
        @return<tuple>: The status and control
        """
        status = True

        index, lookahead_distance = self._search_target_index(state, input)

        alpha = (
            math.atan2(
                self.trajectory.x[index, 1] - state[1],
                self.trajectory.x[index, 0] - state[0],
            )
            - state[2]
        )

        v = 5.0

        w = v * 2.0 * math.sin(alpha) / lookahead_distance

        # v_r = v + w * self.model.wheel_base / 2.0

        # v_l = v - w * self.model.wheel_base / 2.0

        return status, [v, w]

    # ==================================================================================================
    # PRIVATE METHODS
    # ==================================================================================================
    def _apply_proportional_control(k_p, target, current):
        """! Apply proportional control
        @param k_p<float>: The proportional gain
        @param target<list>: The target
        @param current<list>: The current
        @return<float>: The control
        """
        target_v = (target[0] + target[1]) / 2

        v = (current[0] + current[1]) / 2

        a = k_p * (target_v - v)

        return a

    def _search_target_index(self, state, input):
        if self.old_nearest_point_index is None:
            all_distance = self._calculate_distance(self.trajectory.x, state)

            index = np.argmin(all_distance)

        else:
            index = self.old_nearest_point_index

            this_distance = self._calculate_distance(
                self.trajectory.x[index], state)

            while True:
                next_distance = self._calculate_distance(
                    self.trajectory.x[index + 1], state
                )

                if this_distance < next_distance:
                    break

                if (index + 1) < len(self.trajectory.x):
                    index += 1

                this_distance = next_distance

            self.old_nearest_point_index = index

        v = (input[0] + input[1]) / 2

        lookahead_distance = (
            AdaptivePurePursuit.lookahead_gain * v
            + AdaptivePurePursuit.lookahead_distance
        )

        distance = self._calculate_distance(self.trajectory.x[index], state)

        while lookahead_distance > distance:
            if index + 1 >= len(self.trajectory.x):
                break

            index += 1

            distance = self._calculate_distance(
                self.trajectory.x[index], state)

        return index, lookahead_distance

    # ==================================================================================================
    # STATIC METHODS
    # ==================================================================================================
    @staticmethod
    def _calculate_distance(reference_x, current_x):
        distance = current_x - reference_x

        x = distance[:, 0] if distance.ndim == 2 else distance[0]

        y = distance[:, 1] if distance.ndim == 2 else distance[1]

        return np.hypot(x, y)
