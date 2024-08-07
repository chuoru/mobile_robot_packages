# Standard library
import os
import numpy as np

# Internal library


class SimpleGenerator:
    """! Simple trajectory generator."""

    def __init__(self, environment):
        """! Constructor."""
        self.x = None

        self.u = None

        self.t = None

        current_directory = os.path.dirname(os.path.abspath(__file__))

        self._data_folder = os.path.join(current_directory, "data")

    def generate(self, file_name, nx, nu, is_derivative=False):
        """! Generate a simple trajectory.
        @param file_name<string>: The file name to save the
        generated trajectory
        @param nx<int>: The number of states
        @param nu<int>: The number of inputs
        @param is_derivative<bool>: The flag to indicate if the
        generated trajectory is a derivative
        @return None
        """
        data = np.genfromtxt(os.path.join(
            self._data_folder, file_name), delimiter=",")

        initial_index = 0

        if np.isnan(np.nan):
            initial_index = 1

        self.x = np.array(data[initial_index:, 1: 1 + nx])

        if len(data) > 1 + nx:
            self.u = np.array(data[initial_index:, 1 + nx: 1 + nx + nu])

        self.t = np.array(data[initial_index:, 0])

        self.sampling_time = self.t[1] - self.t[0]

    # ==================================================================
    # PRIVATE METHODS
    # ==================================================================
    def _generate_derivative(self):
        """! Generate a derivative trajectory."""
        pass