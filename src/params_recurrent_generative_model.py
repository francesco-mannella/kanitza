"""Hyperparameters of the FORCE-trained recurrent generative model."""
import sys

import numpy as np

from params import Parameters


class ParamsFORCE:
    """
    Parameters of RecurrentGenerativeModel.

    Attributes:
        target_shape_lenght (int): timesteps of a goal presentation.
        num_input_units (int): input units (= maps_output_size).
        num_recurrent_units (int): reservoir units.
        num_output_units (int): readout units (= maps_output_size), one per
            map unit.
        RNN_tau (float): membrane time constant.
        p_rec_connections (float): recurrent connection probability.
        rec_gain (float): recurrent weight gain g.
        phi (float): feedback weight gain.
        uniform_dist (float): range of the uniform weight initializations.
        normal_dist_mean, normal_dist_sd (float): recurrent weight
            distribution before gain.
        dt (float): integration step.
        alpha (float): initial inverse-correlation scale (P = I / alpha).
        teacher_noise (float): uniform noise added to the teacher signal.
        param_types (dict): converters used by string_to_params.

    maps_output_size is read from ./loaded_params if present.
    """

    def __init__(self):
        """
        Input-output parameters
        """
        
        params = Parameters()
        try:
            params.load("loaded_params")
        except FileNotFoundError:
            print("no local parameters")

        self.target_shape_lenght = 100
        self.num_input_units = params.maps_output_size  # 100
        self.num_recurrent_units = 1000
        self.num_output_units = params.maps_output_size  # 100

        """
        Network parameters
        """
        self.RNN_tau = 10
        self.p_rec_connections = 0.1
        self.rec_gain = 1.5
        self.phi = 1
        self.uniform_dist = 1
        self.normal_dist_mean = 0
        self.normal_dist_sd = np.sqrt(
            1 / (self.p_rec_connections * self.num_recurrent_units)
        )
        self.dt = 1

        """
        Training parameters
        """
        self.alpha = 1
        self.teacher_noise = 0.05

        self.param_types = {
            "target_shape_lenght": int,
            "num_input_units": int,
            "num_output_units": int,
            "RNN_tau": int,
            "p_rec_connections": float,
            "rec_gain": float,
            "uniform_dist": float,
            "normal_dist_mean": float,
            "normal_dist_sd": float,
            "dt": int,
            "alpha": int,
            "teacher_noise": float,
        }

    def string_to_params(self, param_list):
        """
        Read parameter values from a semicolon-separated string of key-value
        pairs.

        Parameters:
            - param_list (str): A semicolon-separated string where each element
              is a key-value pair in the format 'key=value'.

        Example:
            If `self` has attributes `a` and `b`, calling:

            self.string_to_params("a=1;b=2")

            will set:
            self.a = 1
            self.b = 2
        """
        if not param_list:
            return

        param_dict = dict(
            item.split("=", 1) for item in param_list.split(";") if "=" in item
        )

        for key, value in param_dict.items():
            if key in dir(self):
                converter = self.param_types[key]
                if converter is not bool:
                    setattr(self, key, converter(value))
                else:
                    setattr(self, key, value == "True")

            else:
                print(f"There's no parameter named {key}")
                sys.exit(1)

    def save(self, filepath):
        """Write each parameter as a "key=value" line to filepath."""
        with open(filepath, "w") as file:
            for key in self.__dict__:
                if key != "param_types":
                    file.write(f"{key}={getattr(self, key)}\n")

    def load(self, filepath):
        """Read "key=value" lines from filepath into the attributes."""
        with open(filepath, "r") as file:
            param_list = "".join([line.strip() + ";" for line in file])
        self.string_to_params(param_list)

    def __hash__(self):
        """Hash of all public, non-callable attributes and their values."""
        # Using a tuple comprehension to collect all non-callable and
        # non-private attributes (those not starting with "_") into a tuple
        attr_values = tuple(
            (attr, self._make_hashable(getattr(self, attr)))
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("_")
        )
        hashid = hash(attr_values)
        # Create a unique string from the tuple and return its hash
        return hashid

    def _make_hashable(self, value):
        """Recursively convert dicts, lists and sets to hashable types."""
        if isinstance(value, dict):
            # Convert dictionary to a frozenset of its items (key-value pairs)
            return frozenset(
                (key, self._make_hashable(v)) for key, v in value.items()
            )
        elif isinstance(value, list):
            # Convert list to a tuple of its elements
            return tuple(self._make_hashable(v) for v in value)
        elif isinstance(value, set):
            # Convert set to a frozenset of its elements
            return frozenset(self._make_hashable(v) for v in value)
        # Add other types like list, set, etc., if needed
        return value
