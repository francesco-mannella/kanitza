from parameter_manager import ParameterManager


class Parameters(ParameterManager):
    """
    Default hyperparameters of the simulation.

    Every keyword is stored as an attribute of the same name and can be
    overridden with `update("name=value;...")` or `load("loaded_params")`.

    Args:
        project_name (str): wandb project.
        entity_name (str): wandb entity.
        init_name (str): wandb run name (overwritten by the entry scripts).
        env_name (str): gymnasium id of the environment.
        taskspace_xlim, taskspace_ylim (list): [min, max] of the scene, in
            task-space units.
        retina_scale (list): [w, h] task-space window seen by the retina.
        retina_size (list): [h, w] retina image size in pixels.
        fovea_scale (list): [h, w] central retina region, in retina pixels,
            that the agent crops from its color-saliency map and resizes to
            fovea_size to form the visual input of the maps (also drawn by
            the plotter).
        fovea_size (list): [h, w] fovea image size in pixels (also the size
            of the env FOVEA observation, a central crop of the retina).
        episodes (int): episodes per epoch.
        epochs (int): total epochs; a resumed run stops at this epoch.
        saccade_num (int): saccades per episode.
        saccade_time (int): timesteps per saccade; the controller proposes
            an attention target at step saccade_time // 2.
        plot_sim (bool): save a fovea/saliency gif of the last episode of
            plotting epochs.
        plot_maps (bool): save maps gifs/pngs.
        plotting_epochs_interval (int): epochs between saved plots.
        agent_sampling_precision (float): saliency sampling threshold as a
            fraction of the maximum, in [0, 1) (see agent.sampling).
        maps_output_size (int): units per topological map; must be a
            perfect square (side = sqrt).
        action_size (int): action dimensions (2, retina displacement).
        attention_size (int): attention dimensions (2, normalized center).
        maps_learning_rate (float): Adam learning rate of the maps.
        predictor_learning_rate (float): Adam learning rate of the
            competence predictor.
        saccade_threshold (float): minimum action norm (task-space units)
            for a timestep to count as a saccade.
        neighborhood_modulation (float): gain of the competence-dependent
            part of the maps' neighborhood std.
        neighborhood_modulation_baseline (float): minimum neighborhood std;
            also the std of the grid representations.
        learningrate_modulation (float): gain of the competence-dependent
            part of the maps' loss modulation.
        learningrate_modulation_baseline (float): minimum loss modulation.
        match_std_baseline (float): unused.
        match_std (float): std (map units) of the Gaussian converting goal
            distances into match scores.
        anchor_std (float): neighborhood std around the goal anchors in the
            STM update.
        decaying_speed (float): slope of tanh(decaying_speed * competence)
            giving the global decay of plasticity.
        local_decaying_speed (float): slope of the same function for the
            per-sample (local) competence.
        triangles_percent (float): epochs with epoch % 100 below this value
            use world 0 (triangle), the others world 1 (square).
        colors (bool): unused (the env always uses the colored objects).
        magnitude_decay (float): unused.
        attention_max_variance (float): attentional Gaussian variance as a
            fraction of the retina size.
        attention_fixed_variance_prop (float): fraction of the variance
            independent of eccentricity.
        attention_center_distance_variance_prop (float): fraction of the
            variance that shrinks with eccentricity.
        attention_center_distance_slope (float): tanh slope of that
            shrinkage.
        gabor_scales (list): sigmas of the Gabor kernels (one bank per
            scale).
        gabor_orientation_bins (int): the orientations are
            linspace(0, 360, bins)[:-1] degrees, i.e. bins - 1 of them.
        gabor_frequency (float): spatial frequency (cycles per pixel).
        gabor_phase_offset (float): phase of the carrier, in radians.
        gabor_kernel_size (int): side of the square kernels, in pixels.
        gabor_sigma_y_multiplier (float): sigma_y = sigma * multiplier
            (kernel elongation).
        gabor_rgb_prop (float): weight of the color-opponent channels in
            the adjusted saliency.
        gabor_bright_prop (float): weight of the brightness channel in the
            adjusted saliency.
        test_fovea (bool): if True the maps get the raw FOVEA observation
            instead of the color-saliency fovea.
        use_wandb (bool): upload the fovea simulation gifs to wandb.
    """

    def __init__(
        self,
        project_name="eye-simulation",
        entity_name="francesco-mannella",
        init_name="offline controller tester",
        env_name="EyeSim/EyeSim-v0",
        taskspace_xlim=[0, 80],
        taskspace_ylim=[0, 80],
        retina_scale=[80, 80],
        retina_size=[80, 80],
        fovea_scale=[50, 50],
        fovea_size=[16, 16],
        episodes=20,
        epochs=400,
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
        saccade_threshold=12.0,
        neighborhood_modulation=10.0,
        neighborhood_modulation_baseline=0.8,
        learningrate_modulation=0.8,
        learningrate_modulation_baseline=0.02,
        match_std_baseline=0.5,
        match_std=4.0,
        anchor_std=8.0,
        decaying_speed=3.0,
        local_decaying_speed=1.0,
        triangles_percent=50.0,
        colors=True,
        magnitude_decay=1e-10,
        attention_max_variance=6.0,
        attention_fixed_variance_prop=1.0,
        attention_center_distance_variance_prop=0.0,
        attention_center_distance_slope=5.0,
        gabor_scales=[8.0],
        gabor_orientation_bins=10,
        gabor_frequency=0.09,
        gabor_phase_offset=-3.141592653589793 * (0.5 - 8e-2),
        gabor_kernel_size=9,
        gabor_sigma_y_multiplier=6.0,
        gabor_rgb_prop=1.0,
        gabor_bright_prop=1.0,
        test_fovea=False,
        use_wandb=False,
    ):
        self.project_name = project_name
        self.entity_name = entity_name
        self.init_name = init_name
        self.env_name = env_name
        self.taskspace_xlim = taskspace_xlim
        self.taskspace_ylim = taskspace_ylim
        self.retina_scale = retina_scale
        self.retina_size = retina_size
        self.fovea_scale = fovea_scale
        self.fovea_size = fovea_size
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
        self.match_std_baseline = match_std_baseline
        self.match_std = match_std
        self.anchor_std = anchor_std
        self.decaying_speed = decaying_speed
        self.local_decaying_speed = local_decaying_speed
        self.triangles_percent = triangles_percent
        self.colors = colors
        self.magnitude_decay = magnitude_decay
        self.attention_max_variance = attention_max_variance
        self.attention_fixed_variance_prop = attention_fixed_variance_prop
        self.attention_center_distance_variance_prop = (
            attention_center_distance_variance_prop
        )
        self.attention_center_distance_slope = attention_center_distance_slope
        self.gabor_scales = gabor_scales
        self.gabor_orientation_bins = gabor_orientation_bins
        self.gabor_frequency = gabor_frequency
        self.gabor_phase_offset = gabor_phase_offset
        self.gabor_kernel_size = gabor_kernel_size
        self.gabor_sigma_y_multiplier = gabor_sigma_y_multiplier
        self.gabor_rgb_prop = gabor_rgb_prop
        self.gabor_bright_prop = gabor_bright_prop
        self.test_fovea = test_fovea
        self.use_wandb = use_wandb

        super(Parameters, self).__init__()
