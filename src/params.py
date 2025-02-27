class Parameters:
    def __init__(
        self,
        project_name="eye-simulation",
        entity_name="francesco-mannella",
        init_name="offline controller tester",
        env_name="EyeSim/EyeSim-v0",
        episodes=20,
        epochs=400,
        saccade_num=10,
        saccade_time=10,
        plot_sim=False,
        plot_maps=True,
        plotting_epochs_interval=50,
        agent_sampling_threshold=0.001,
        maps_output_size=100,
        action_size=2,
        attention_size=2,
        maps_learning_rate=0.1,
        saccade_threshold=14,
        neighborhood_modulation=10,
        neighborhood_modulation_baseline=0.8,
        learnigrate_modulation=0.8,
        learnigrate_modulation_baseline=0.02,
        match_std=4.0,
        decaying_speed=5.0,
        local_decaying_speed=1.0,
    ):
        self.project_name = project_name
        self.entity_name = entity_name
        self.init_name = init_name
        self.env_name = env_name
        self.episodes = episodes
        self.epochs = epochs
        self.saccade_num = saccade_num
        self.saccade_time = saccade_time
        self.plot_sim = plot_sim
        self.plot_maps = plot_maps
        self.plotting_epochs_interval = plotting_epochs_interval
        self.agent_sampling_threshold = agent_sampling_threshold
        self.maps_output_size = maps_output_size
        self.action_size = action_size
        self.attention_size = attention_size
        self.maps_learning_rate = maps_learning_rate
        self.saccade_threshold = saccade_threshold
        self.learnigrate_modulation = learnigrate_modulation
        self.neighborhood_modulation = neighborhood_modulation
        self.learnigrate_modulation_baseline = learnigrate_modulation_baseline
        self.neighborhood_modulation_baseline = (
            neighborhood_modulation_baseline
        )
        self.match_std = match_std
        self.decaying_speed = decaying_speed
        self.local_decaying_speed = local_decaying_speed

        self.param_types = {
            "project_name": str,
            "entity_name": str,
            "init_name": str,
            "env_name": str,
            "episodes": int,
            "epochs": int,
            "saccade_num": int,
            "saccade_time": int,
            "plot_sim": bool,
            "plot_maps": bool,
            "plotting_epochs_interval": int,
            "agent_sampling_threshold": float,
            "maps_output_size": int,
            "action_size": int,
            "attention_size": int,
            "maps_learning_rate": float,
            "saccade_threshold": int,
            "neighborhood_modulation": int,
            "neighborhood_modulation_baseline": float,
            "learnigrate_modulation": float,
            "learnigrate_modulation_baseline": float,
            "match_std": float,
            "decaying_speed": float,
            "local_decaying_speed": float,
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

            self.string_to_dict("a=1;b=2")

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
            if hasattr(self, key):
                converter = self.param_types.get(
                    key, str
                )  # Default to str if type is not specified
                setattr(self, key, converter(value))
            else:
                print(f"There's no parameter named {key}")

    def save(self, filepath):
        with open(filepath, "w") as file:
            for key in self.__dict__:
                if key != "param_types":
                    file.write(f"{key}={getattr(self, key)}\n")

    def load(self, filepath):
        with open(filepath, "r") as file:
            param_list = "".join([line.strip() + ";" for line in file])
        self.string_to_params(param_list)

    def __hash__(self):
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
