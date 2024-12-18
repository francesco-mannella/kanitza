import numpy as np

class Parameters:
    def __init__(
        self,
        project_name="eye-simulation",
        entity_name="francesco-mannella",
        init_name="offline controller tester",
        env_name="EyeSim-v0",
        episodes=2,
        epochs=800,
        focus_num=10,
        focus_time=10,
        plot_sim=False,
        plot_maps=True,
        plotting_epochs_interval=50,
        agent_sampling_threshold=0.001,
        maps_output_size=100,
        action_size=2,
        attention_size=2,
        maps_learning_rate=0.1,
        saccade_threshold=14,
        neighborhood_modulation=4,
        neighborhood_modulation_baseline=.8,
        learnigrate_modulation=0.1,
        learnigrate_modulation_baseline=0.02,
        decaying_speed=5,
        local_decaying_speed=1,
    ):
        self.project_name = project_name
        self.entity_name = entity_name
        self.init_name = init_name
        self.env_name = env_name
        self.episodes = episodes
        self.epochs = epochs
        self.focus_num = focus_num
        self.focus_time = focus_time
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
        self.neighborhood_modulation_baseline = neighborhood_modulation_baseline
        self.decaying_speed = decaying_speed
        self.local_decaying_speed = local_decaying_speed
