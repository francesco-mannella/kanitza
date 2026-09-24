"""Base class for parameter containers with string/file (de)serialization."""
import ast
import json
import sys

import numpy as np


class ParameterManager:
    """
    Base class for parameter containers.

    Subclasses set their parameters as instance attributes and then call
    `super().__init__()`, which records the type of each default in
    `param_types`. Parameters can be updated from "k1=v1;k2=v2" strings or
    JSON, and saved/loaded as "key = value" text files. In "k=v" strings
    each value is parsed as a Python literal (numbers, True/False, quoted
    strings, lists); unparsable values are kept as plain strings. Values of
    known parameters must match the type of their default (int and float
    are interchangeable).
    """

    def __init__(self):
        """Record the types of the attributes set so far in `param_types`."""
        self._set_param_types()

    def __getstate__(self):
        return self._params_to_dict()

    def __setstate__(self, state):

        for key, value in state.items():
            setattr(self, key, value)

    def _set_param_types(self):
        self.param_types = {
            k: type(v)
            for k, v in self.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }

    @staticmethod
    def _parse_value(value):
        """Parse a Python literal, or return the stripped string if invalid."""
        value = value.strip()
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    def _string_to_json(self, param_string, mode="user"):
        """Converts a parameter string to a dictionary.

        Args:
            param_string (str): String representing the params.
            mode (str): Format of the string; "user" for
              semicolon-separated key=value pairs, "json" for JSON string.

        Returns:
            dict: Dictionary of parsed parameters.

        Raises:
            ValueError: If mode is neither "user" nor "json".
        """
        if mode not in ("user", "json"):
            raise ValueError(f"Invalid mode {mode!r}. Use 'user' or 'json'.")

        if not param_string:
            return {}

        if mode == "json":
            return json.loads(param_string)

        pairs = (s.split("=", 1) for s in param_string.split(";") if "=" in s)
        return {k.strip(): self._parse_value(v) for k, v in pairs}

    def _check_type(self, key, value):
        """Return value checked against the type of the key's default.

        Unknown keys are accepted as they are. int and float are
        interchangeable (ints are converted to float where the default is a
        float); any other mismatch raises TypeError.
        """
        expected = self.param_types.get(key)
        if expected is None:
            return value
        is_number = isinstance(value, (int, float)) and not isinstance(
            value, bool
        )
        if expected in (int, float) and is_number:
            return float(value) if expected is float else value
        if isinstance(value, expected) and (
            expected is bool or not isinstance(value, bool)
        ):
            return value
        raise TypeError(
            f"Parameter {key!r} expects {expected.__name__}, "
            f"got {type(value).__name__} ({value!r})"
        )

    def _json_to_params(self, param_dict):
        """Set attributes from a dictionary, checking their types.

        Args:
            param_dict (dict): Dictionary of parameters.

        Raises:
            TypeError: If a known parameter gets a value of the wrong type.
        """
        for key, value in param_dict.items():
            setattr(self, key, self._check_type(key, value))

    def _params_to_dict(self):
        """Return all non-callable attributes except `param_types` as a dict."""
        params = {
            key: value
            for key, value in self.__dict__.items()
            if key != "param_types" and not callable(value)
        }
        return params

    def __repr__(self):
        params = self._params_to_dict()
        return json.dumps(params)

    def update(self, param_string, mode="user"):
        """Update parameters based on the input string.

        Args:
            param_string (str): Input string containing parameters.
            mode (str, optional): "user" for "k1=v1;k2=v2" strings, "json"
                for a JSON object. Defaults to "user".

        Raises:
            ValueError: If mode is neither "user" nor "json".
            TypeError: If a known parameter gets a value of the wrong type.
        """
        param_dict = self._string_to_json(param_string, mode=mode)
        self._json_to_params(param_dict)

    def save(self, filepath, mode="user"):
        """Saves parameters to a file.

        Args:
            filepath (str): The path to the file.
            mode (str, optional): Specifies the saving mode.
              "user": Saves parameters in a human-readable format.
              Each parameter is written as 'key = value' on a new line.
              "json": Saves parameters in JSON format.
              Defaults to "user".

        Raises:
            ValueError: If mode is neither "user" nor "json".
        """
        if mode not in ("user", "json"):
            raise ValueError(f"Invalid mode {mode!r}. Use 'user' or 'json'.")
        with open(filepath, "w") as file:
            if mode == "user":
                for key, value in self._params_to_dict().items():
                    if isinstance(value, str):
                        value = f'"{value}"'
                    file.write(f"{key} = {value}\n")
            elif mode == "json":
                params = self._params_to_dict()
                json.dump(params, file, indent=4)

    def load(self, filepath, mode="user"):
        """Loads parameters from a file.

        Args:
            filepath (str): The path to the file.
            mode (str, optional): Specifies the loading mode.
              "user": Loads parameters from a human-readable format.
              Expects each parameter to be in the format 'key = value'.
              "json": Loads parameters from JSON format.
              Defaults to "user".

        Raises:
            ValueError: If mode is neither "user" nor "json".
            TypeError: If a known parameter gets a value of the wrong type.
        """
        if mode not in ("user", "json"):
            raise ValueError(f"Invalid mode {mode!r}. Use 'user' or 'json'.")
        with open(filepath, "r") as file:
            if mode == "user":
                param_list = "".join([line.strip() + ";" for line in file])
                self.update(param_list)
            elif mode == "json":
                self._json_to_params(json.load(file))

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

