from parameter_manager import ParameterManager


class Parameters(ParameterManager):
    def __init__(
        self,
        project_name="eye-simulation",
        entity_name="francesco-mannella",
        init_name="offline controller tester",
        env_name="EyeSim/EyeSim-v0",
        episodes=20,
        epochs=500,
        saccade_num=10,
        saccade_time=10,
        plot_sim=False,
        plot_maps=True,
        plotting_epochs_interval=50,
        agent_sampling_precision=0.999,
        maps_output_size=100,
        action_size=2,
        attention_size=2,
        maps_learning_rate=0.1,
        predictor_learning_rate=0.01,
        saccade_threshold=12,
        neighborhood_modulation=20.0,
        neighborhood_modulation_baseline=0.8,
        learningrate_modulation=10.0,
        learningrate_modulation_baseline=0.02,
        match_std=8.0,
        anchor_std=2.0,
        decaying_speed=4.0,
        local_decaying_speed=1.0,
        triangles_percent=50,
        colors=True,
        attention_max_variance=2,
        attention_fixed_variance_prop=0.5,
        attention_center_distance_variance_prop=0.5,
        attention_center_distance_slope=2,
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
        self.agent_sampling_precision = agent_sampling_precision
        self.maps_output_size = maps_output_size
        self.action_size = action_size
        self.attention_size = attention_size
        self.maps_learning_rate = maps_learning_rate
        self.predictor_learning_rate = predictor_learning_rate
        self.saccade_threshold = saccade_threshold
        self.learningrate_modulation = learningrate_modulation
        self.neighborhood_modulation = neighborhood_modulation
        self.learningrate_modulation_baseline = learningrate_modulation_baseline
        self.neighborhood_modulation_baseline = neighborhood_modulation_baseline
        self.match_std = match_std
        self.anchor_std = anchor_std
        self.decaying_speed = decaying_speed
        self.local_decaying_speed = local_decaying_speed
        self.triangles_percent = triangles_percent
        self.colors = colors
        self.attention_max_variance = attention_max_variance
        self.attention_fixed_variance_prop = attention_fixed_variance_prop
        self.attention_center_distance_variance_prop = (
            attention_center_distance_variance_prop
        )
        self.attention_center_distance_slope = attention_center_distance_slope

        super(Parameters, self).__init__()
