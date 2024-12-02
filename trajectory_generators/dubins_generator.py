#!/usr/bin/env python3
##
# @file dubins_generator.py
#
# @brief Provide a class to generate a trajectory based on dubins curves.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2024/11/13
#
# Copyright (c) 2024 System Engineering Laboratory.  All rights reserved.

# Standard library
import math
import numpy as np


class DubinsGenerator:
    """! Dubins trajectory generator."""

    # ========================================================================
    # PUBLIC METHODS
    # ========================================================================
    def __init__(self, sampling_time=0.05):
        """! Constructor of the class.
        @param sampling_time<float>: The time interval between two consecutive
        points on the trajectory.
        """
        self._sampling_time = sampling_time

        self._turning_radius = 0.5

        self._max_velocity = 1.0

        self.x = [[0.0, 0.0, 0.0]]

        self.u = [[0.0, 0.0]]

        self.t = [0.0]

    def generate(self, waypoints):
        """! Generate a trajectory based on the waypoints.
        @param waypoints<list>: A list of waypoints. Each waypoint is a tuple
        (x, y, yaw).
        @return trajectory<list>: A list of points on the trajectory.
        """
        waypoints = self._generate_intermediate_waypoints(waypoints)

        start_x, start_y, start_yaw = waypoints[0]

        for i in range(1, len(waypoints)):
            end_x, end_y, end_yaw = waypoints[i]

            px, py, pyaw, mode, clen = DubinsGenerator.dubins_path_planning(
                start_x, start_y, start_yaw,
                end_x, end_y, end_yaw, self._turning_radius)

            self.x.extend([x, y, yaw] for x, y, yaw in zip(px, py, pyaw))

            self._generate_velocity_profile(px, py, pyaw, clen)

            start_x, start_y, start_yaw = end_x, end_y, end_yaw

        self.x = np.array(self.x)

        self.u = np.array(self.u).T

        self.t = np.array(self.t)

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================
    def _generate_velocity_profile(self, px, py, pyaw, clen):
        """! Generate the velocity profile based on the path.
        @param px<list>: A list of x coordinates.
        @param py<list>: A list of y coordinates.
        @param pyaw<list>: A list of yaw angles.
        @param clen<float>: The length of the path.
        @return None
        """
        previous_x, previous_y, previous_angle = px[0], py[0], 0.0

        for x, y in zip(px[1:], py[1:]):
            distance = math.sqrt((x - previous_x)**2 + (y - previous_y)**2)

            angle = DubinsGenerator._calculate_angle_from_vectors(
                np.array([previous_x, previous_y]), np.array([x, y]))

            omega = DubinsGenerator.pi_2_pi(
                angle - previous_angle) / self._sampling_time

            self.u.append([distance / self._sampling_time, omega])

            self.t.append(self.t[-1] + self._sampling_time)

            previous_x, previous_y = x, y

    def _generate_intermediate_waypoints(self, waypoints):
        """! Generate intermediate waypoints between two consecutive waypoints.
        @param waypoints<list>: A list of waypoints. Each waypoint is a tuple
        (x, y, yaw).
        @return intermediate_waypoints<list>: A list of intermediate waypoints.
        """
        if len(waypoints) < 2:
            return waypoints

        number_of_corner = len(waypoints) - 2

        intermediate_waypoints = [waypoints[0]]

        for index in range(number_of_corner):
            start = waypoints[index]

            corner = waypoints[index + 1]

            theta_start = DubinsGenerator._calculate_angle_from_vectors(
                np.array(start[:2]), np.array(corner[:2]))

            corner_x, corner_y, _ = corner

            entry_x = corner_x - self._turning_radius * math.cos(theta_start)

            entry_y = corner_y - self._turning_radius * math.sin(theta_start)

            entry_yaw = theta_start

            intermediate_waypoints.append((entry_x, entry_y, entry_yaw))

        intermediate_waypoints.append(waypoints[-1])

        return intermediate_waypoints

    # ========================================================================
    # STATIC METHODS
    # ========================================================================
    @staticmethod
    def _calculate_unit_vector(start, end):
        """! Calculate the unit vector
        @param start<np.array>: The start point
        @param end<np.array>: The end point
        @return The unit vector
        """
        delta = end - start

        return delta / np.linalg.norm(delta)

    @staticmethod
    def _calculate_angle_from_vectors(start, end):
        """! Calculate the angle
        @param start<np.array>: The start point
        @param end<np.array>: The end point
        @return The angle
        """
        unit_vector = DubinsGenerator._calculate_unit_vector(start, end)

        return DubinsGenerator._calculate_angle(unit_vector)

    @staticmethod
    def _calculate_angle(unit_array):
        """! Calculate the angle
        @param unit_array<np.array>: The unit array
        @return The angle
        """
        angle = 0

        unit_array_x = abs(unit_array[0])

        if unit_array[1] >= 0:
            if DubinsGenerator._is_same(unit_array[0], 0):
                angle = 0.5 * math.pi

            else:
                angle = math.acos(unit_array[0])

        else:
            if DubinsGenerator._is_same(unit_array[0], 0):
                angle = -0.5 * math.pi

            elif unit_array[0] < 0:
                angle = -math.pi + math.acos(unit_array_x)

            else:
                angle = -math.acos(unit_array_x)

        return angle

    @staticmethod
    def _is_same(lhs, rhs):
        """! Check if two values are the same
        @param lhs<float>: The left hand side value
        @param rhs<float>: The right hand side value
        @return True if two values are the same, False otherwise
        """
        return abs(lhs - rhs) < 1e-9

    @staticmethod
    def mod2pi(theta):
        return theta - 2.0 * math.pi * math.floor(theta / 2.0 / math.pi)

    @staticmethod
    def pi_2_pi(angle):
        while (angle >= math.pi):
            angle = angle - 2.0 * math.pi

        while (angle <= -math.pi):
            angle = angle + 2.0 * math.pi

        return angle

    @staticmethod
    def LSL(alpha, beta, d):
        sa = math.sin(alpha)
        sb = math.sin(beta)
        ca = math.cos(alpha)
        cb = math.cos(beta)
        c_ab = math.cos(alpha - beta)

        tmp0 = d + sa - sb

        mode = ["L", "S", "L"]
        p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sa - sb))
        if p_squared < 0:
            return None, None, None, mode
        tmp1 = math.atan2((cb - ca), tmp0)
        t = DubinsGenerator.mod2pi(-alpha + tmp1)
        p = math.sqrt(p_squared)
        q = DubinsGenerator.mod2pi(beta - tmp1)
        #  print(math.degrees(t), p, math.degrees(q))

        return t, p, q, mode

    @staticmethod
    def RSR(alpha, beta, d):
        sa = math.sin(alpha)
        sb = math.sin(beta)
        ca = math.cos(alpha)
        cb = math.cos(beta)
        c_ab = math.cos(alpha - beta)

        tmp0 = d - sa + sb
        mode = ["R", "S", "R"]
        p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sb - sa))
        if p_squared < 0:
            return None, None, None, mode
        tmp1 = math.atan2((ca - cb), tmp0)
        t = DubinsGenerator.mod2pi(alpha - tmp1)
        p = math.sqrt(p_squared)
        q = DubinsGenerator.mod2pi(-beta + tmp1)

        return t, p, q, mode

    @staticmethod
    def LSR(alpha, beta, d):
        sa = math.sin(alpha)
        sb = math.sin(beta)
        ca = math.cos(alpha)
        cb = math.cos(beta)
        c_ab = math.cos(alpha - beta)

        p_squared = -2 + (d * d) + (2 * c_ab) + (2 * d * (sa + sb))
        mode = ["L", "S", "R"]
        if p_squared < 0:
            return None, None, None, mode
        p = math.sqrt(p_squared)
        tmp2 = math.atan2((-ca - cb), (d + sa + sb)) - math.atan2(-2.0, p)
        t = DubinsGenerator.mod2pi(-alpha + tmp2)
        q = DubinsGenerator.mod2pi(-DubinsGenerator.mod2pi(beta) + tmp2)

        return t, p, q, mode

    @staticmethod
    def RSL(alpha, beta, d):
        sa = math.sin(alpha)
        sb = math.sin(beta)
        ca = math.cos(alpha)
        cb = math.cos(beta)
        c_ab = math.cos(alpha - beta)

        p_squared = (d * d) - 2 + (2 * c_ab) - (2 * d * (sa + sb))
        mode = ["R", "S", "L"]
        if p_squared < 0:
            return None, None, None, mode
        p = math.sqrt(p_squared)
        tmp2 = math.atan2((ca + cb), (d - sa - sb)) - math.atan2(2.0, p)
        t = DubinsGenerator.mod2pi(alpha - tmp2)
        q = DubinsGenerator.mod2pi(beta - tmp2)

        return t, p, q, mode

    @staticmethod
    def RLR(alpha, beta, d):
        sa = math.sin(alpha)
        sb = math.sin(beta)
        ca = math.cos(alpha)
        cb = math.cos(beta)
        c_ab = math.cos(alpha - beta)

        mode = ["R", "L", "R"]
        tmp_rlr = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
        if abs(tmp_rlr) > 1.0:
            return None, None, None, mode

        p = DubinsGenerator.mod2pi(2 * math.pi - math.acos(tmp_rlr))
        t = DubinsGenerator.mod2pi(
            alpha - math.atan2(ca - cb, d - sa + sb) + DubinsGenerator.mod2pi(
                p / 2.0))
        q = DubinsGenerator.mod2pi(
            alpha - beta - t + DubinsGenerator.mod2pi(p))
        return t, p, q, mode

    @staticmethod
    def LRL(alpha, beta, d):
        sa = math.sin(alpha)
        sb = math.sin(beta)
        ca = math.cos(alpha)
        cb = math.cos(beta)
        c_ab = math.cos(alpha - beta)

        mode = ["L", "R", "L"]
        tmp_lrl = (6. - d * d + 2 * c_ab + 2 * d * (- sa + sb)) / 8.
        if abs(tmp_lrl) > 1:
            return None, None, None, mode
        p = DubinsGenerator.mod2pi(2 * math.pi - math.acos(tmp_lrl))
        t = DubinsGenerator.mod2pi(
            -alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.)
        q = DubinsGenerator.mod2pi(
            DubinsGenerator.mod2pi(beta) - alpha - t +
            DubinsGenerator.mod2pi(p))

        return t, p, q, mode

    @staticmethod
    def dubins_path_planning_from_origin(ex, ey, eyaw, c):
        # nomalize
        dx = ex
        dy = ey
        D = math.sqrt(dx ** 2.0 + dy ** 2.0)
        d = D / c
        #  print(dx, dy, D, d)

        theta = DubinsGenerator.mod2pi(math.atan2(dy, dx))
        alpha = DubinsGenerator.mod2pi(- theta)
        beta = DubinsGenerator.mod2pi(eyaw - theta)
        #  print(theta, alpha, beta, d)

        planners = [DubinsGenerator.LSL,
                    DubinsGenerator.RSR,
                    DubinsGenerator.LSR,
                    DubinsGenerator.RSL,
                    DubinsGenerator.RLR,
                    DubinsGenerator.LRL]

        bcost = float("inf")
        bt, bp, bq, bmode = None, None, None, None

        for planner in planners:
            t, p, q, mode = planner(alpha, beta, d)
            if t is None:
                #  print("".join(mode) + " cannot generate path")
                continue

            cost = (abs(t) + abs(p) + abs(q))
            if bcost > cost:
                bt, bp, bq, bmode = t, p, q, mode
                bcost = cost

        #  print(bmode)
        px, py, pyaw = DubinsGenerator.generate_course([bt, bp, bq], bmode, c)

        return px, py, pyaw, bmode, bcost

    @staticmethod
    def dubins_path_planning(sx, sy, syaw, ex, ey, eyaw, c):
        """
        Dubins path plannner
        input:
            sx x position of start point [m]
            sy y position of start point [m]
            syaw yaw angle of start point [rad]
            ex x position of end point [m]
            ey y position of end point [m]
            eyaw yaw angle of end point [rad]
            c curvature [1/m]
        output:
            px
            py
            pyaw
            mode
        """

        ex = ex - sx
        ey = ey - sy

        lex = math.cos(syaw) * ex + math.sin(syaw) * ey
        ley = - math.sin(syaw) * ex + math.cos(syaw) * ey
        leyaw = eyaw - syaw

        lpx, lpy, lpyaw, mode, clen = DubinsGenerator. \
            dubins_path_planning_from_origin(
                lex, ley, leyaw, c)

        px = [math.cos(-syaw) * x + math.sin(-syaw) *
              y + sx for x, y in zip(lpx, lpy)]
        py = [- math.sin(-syaw) * x + math.cos(-syaw) *
              y + sy for x, y in zip(lpx, lpy)]
        pyaw = [DubinsGenerator.pi_2_pi(iyaw + syaw) for iyaw in lpyaw]
        #  print(syaw)
        #  pyaw = lpyaw

        #  plt.plot(pyaw, "-r")
        #  plt.plot(lpyaw, "-b")
        #  plt.plot(eyaw, "*r")
        #  plt.plot(syaw, "*b")
        #  plt.show()

        return px, py, pyaw, mode, clen

    @staticmethod
    def generate_course(length, mode, c):

        px = [0.0]
        py = [0.0]
        pyaw = [0.0]

        v = 0.5  # [m/s]

        sampling_time = 0.05  # [s]

        for m, l in zip(mode, length):
            pd = 0.0
            if m == "S":
                d = v * sampling_time / c
            else:  # turning couse
                d = math.radians(3.0)

            while pd < abs(l - d):
                #  print(pd, l)
                px.append(px[-1] + d * c * math.cos(pyaw[-1]))
                py.append(py[-1] + d * c * math.sin(pyaw[-1]))

                if m == "L":  # left turn
                    pyaw.append(pyaw[-1] + d)
                elif m == "S":  # Straight
                    pyaw.append(pyaw[-1])
                elif m == "R":  # right turn
                    pyaw.append(pyaw[-1] - d)
                pd += d
            else:
                d = l - pd
                px.append(px[-1] + d * c * math.cos(pyaw[-1]))
                py.append(py[-1] + d * c * math.sin(pyaw[-1]))

                if m == "L":  # left turn
                    pyaw.append(pyaw[-1] + d)
                elif m == "S":  # Straight
                    pyaw.append(pyaw[-1])
                elif m == "R":  # right turn
                    pyaw.append(pyaw[-1] - d)
                pd += d

        return px, py, pyaw

    @staticmethod
    def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
        u"""
        Plot arrow
        """
        import matplotlib.pyplot as plt

        if not isinstance(x, float):
            for (ix, iy, iyaw) in zip(x, y, yaw):
                DubinsGenerator.plot_arrow(ix, iy, iyaw)
        else:
            plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                      fc=fc, ec=ec, head_width=width, head_length=width)
            plt.plot(x, y)

    # if __name__ == '__main__':
    #     print("Dubins path planner sample start!!")
    #     import matplotlib.pyplot as plt

    #     start_x = 1.0  # [m]
    #     start_y = 1.0  # [m]
    #     start_yaw = math.radians(45.0)  # [rad]

    #     end_x = -3.0  # [m]
    #     end_y = -3.0  # [m]
    #     end_yaw = math.radians(-45.0)  # [rad]

    #     curvature = 1.0

    #     px, py, pyaw, mode, clen = dubins_path_planning(
    #         start_x, start_y, start_yaw,
    #         end_x, end_y, end_yaw, curvature)

    #     plt.plot(px, py, label="final course " + "".join(mode))

    #     # plotting
    #     plot_arrow(start_x, start_y, start_yaw)
    #     plot_arrow(end_x, end_y, end_yaw)

    #     #  for (ix, iy, iyaw) in zip(px, py, pyaw):
    #     #  plot_arrow(ix, iy, iyaw, fc="b")

    #     plt.legend()
    #     plt.grid(True)
    #     plt.axis("equal")
    #     plt.show()
