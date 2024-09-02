#!/usr/bin/env python3
##
# @file dwa_with_eight_curve.py
#
# @brief Provide example for DWA tracking eight curve trajectory.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 27/08/2024.

# Standard library
import sys


# Internal library
sys.path.append('..')
from visualizers.plotter import Plotter  # noqa
from controllers.dwa import DynamicWindowApproach  # noqa
from simulators.time_stepping import TimeStepping  # noqa
from models.differential_drive import DifferentialDrive  # noqa
from trajectory_generators.simple_generator import SimpleGenerator  # noqa


if __name__ == "__main__":
    wheel_base = 0.53

    model = DifferentialDrive(wheel_base)

    trajectory = SimpleGenerator(model)

    trajectory.generate("eight_curve.csv", nx=3, nu=2,
                        is_derivative=False)

    controller = DynamicWindowApproach(model, trajectory)

    simulator = TimeStepping(model, trajectory, controller, None, t_max=600)

    plotter = Plotter(simulator, trajectory)

    simulator.run(0.0)

    plotter.plot()