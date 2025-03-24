import numpy as np
import torch

from model.topological_maps import TopologicalMap, Updater


class OfflineController:
    """
    Manages the update of topological maps using sensory inputs from an
    environment.
    """

    def __init__(self, env, params, seed=None):
        """
        Initialize the OfflineController with the given environment,
        parameters, and optional random seed.

        Parameters:
        - env: The environment interface providing sensory inputs.
        - params: Parameters detailing map sizes, learning rates, and other
                  configuration variables.
        - seed: Optional random seed for reproducibility.
        """
        # Environment and parameters setup
        self.env = env
        self.params = params

        # Param inits
        self.epoch = 0
        self.competence = torch.tensor(0.0)
        self.competences = torch.tensor(0.0)
        self.local_incompetence = torch.tensor(0.0)
        self.match_std = self.params.match_std

        # Random generator initialization
        self.seed = seed if seed is not None else 0
        self.rng = np.random.RandomState(self.seed)

        # Initialize maps for sensory processing
        self._init_maps()

        # Preallocate storage for states
        self._init_storage()

        # Set initial hyperparameters
        self.set_hyperparams()

    def _init_maps(self):
        """Initialize the topological maps and their updaters."""
        input_size = np.prod(
            self.env.observation_space["FOVEA"].sample().shape
        )
        output_size = self.params.maps_output_size

        # Visual conditions map
        self.visual_conditions_map = TopologicalMap(input_size, output_size)
        self.visual_conditions_updater = Updater(
            self.visual_conditions_map,
            self.params.maps_learning_rate,
            "stm",
        )

        # Visual effects map
        self.visual_effects_map = TopologicalMap(input_size, output_size)
        self.visual_effects_updater = Updater(
            self.visual_effects_map,
            self.params.maps_learning_rate,
            "stm",
        )

        # Attention map
        input_size = self.params.attention_size
        self.attention_map = TopologicalMap(input_size, output_size)
        self.attention_updater = Updater(
            self.attention_map,
            self.params.maps_learning_rate,
            "stm",
        )

    def _init_storage(self):
        """Initialize storage for visual, action, and attention states."""
        visual_size = np.prod(
            self.env.observation_space["FOVEA"].sample().shape
        )
        self.params.visual_size = visual_size

        self.visual_states = np.zeros(
            (
                self.params.episodes,
                self.params.saccade_num,
                self.params.saccade_time,
                visual_size,
            )
        )
        self.action_states = np.zeros(
            (
                self.params.episodes,
                self.params.saccade_num,
                self.params.saccade_time,
                self.params.action_size,
            )
        )
        self.attention_states = np.zeros(
            (
                self.params.episodes,
                self.params.saccade_num,
                self.params.saccade_time,
                self.params.attention_size,
            )
        )
        self.world_states = np.zeros(
            (
                self.params.episodes,
                self.params.saccade_num,
                self.params.saccade_time,
                1,
            )
        )

    def reset_states(self):
        """Reset the stored states for the current episode."""
        self.visual_states.fill(0)
        self.action_states.fill(0)
        self.attention_states.fill(0)
        self.world_states.fill(99)

    def set_hyperparams(self):
        """Set the controller's hyperparameters based on current competence."""
        decay = torch.tanh(self.params.decaying_speed * self.competence)
        local_decay = torch.tanh(
            self.params.local_decaying_speed * self.competences
        )

        self.incompetence = 1 - decay
        self.local_incompetence = (1 - local_decay).reshape(-1, 1)

        lm_baseline = self.params.learningrate_modulation_baseline
        self.learningrate_modulation = lm_baseline + (
            self.params.learningrate_modulation
            * self.local_incompetence
            * self.incompetence
        )

        nm_baseline = self.params.neighborhood_modulation_baseline
        self.neighborhood_modulation = nm_baseline + (
            self.params.neighborhood_modulation
            * self.incompetence
            * self.local_incompetence
        )

        self.neighborhood_modulation = self.neighborhood_modulation.reshape(
            -1, 1
        )

        self.match_std = self.params.match_std

    def generate_attentional_input(self, saccade_num):
        """
        Generate random attentional input.

        Parameters:
        - saccade_num: Number of saccades to generate input for.

        Returns:
        - 2D array of attentional inputs with random values.
        """
        # if not hasattr(self, "saccade_samples"):
        #     self.saccade_samples = 0.9 * np.array(
        #         [
        #             [np.cos(a), np.sin(a)]
        #             for a in np.linspace(0, (9 / 10) * 2 * np.pi, 9)
        #         ]
        #     )
        #
        # saccade_idcs = self.rng.randint(0, 9, saccade_num)
        # res = self.saccade_samples[saccade_idcs].copy()
        res = self.rng.rand(saccade_num, 2)

        return res

    def record_states(self, episode, saccade, ts, state):
        """
        Record the visual, action, and attention states for a given timestep.

        Parameters:
        - episode: The current episode number.
        - saccade: The current saccade number.
        - ts: The current timestep within a saccade.
        - state: The state dictionary containing 'vision', 'action', and
                 'attention' keys.
        """
        self.visual_states[episode, saccade, ts] = (
            state["vision"].ravel() / 255.0
        )
        self.action_states[episode, saccade, ts] = state["action"]
        self.attention_states[episode, saccade, ts] = state["attention"]
        self.world_states[episode, saccade, ts] = state["world"]

    def filter_salient_states(self):
        """
        Filter out states based on the magnitude of the saccades.
        States where the magnitude exceeds the configured threshold
        are considered salient.
        """
        magnitudes = np.linalg.norm(self.action_states, axis=-1)
        is_salient = magnitudes > self.params.saccade_threshold
        self.filtered_idcs = np.stack(np.where(is_salient))

    def update(self):
        """
        Update all sensory maps based on stored states and current competence.
        """
        offset = 2
        # Only consider saccades that are not on the time-limits edges
        idcs = self.filtered_idcs
        ts_cond = (offset < idcs[2]) & (
            idcs[2] < (self.params.saccade_time - offset)
        )
        idcs = idcs[:, ts_cond]

        # Get states for attention, visual conditions, and visual effects
        def get_state_data(states, offset):
            item_size = states.shape[-1]
            return torch.tensor(
                states[idcs[0], idcs[1], idcs[2] + offset],
                dtype=torch.float32,
            ).reshape(-1, item_size)

        attention_states = get_state_data(self.attention_states, offset=offset)
        visual_conditions = get_state_data(self.visual_states, offset=-offset)
        visual_effects = get_state_data(self.visual_states, offset=offset)

        # np.save(
        #     "visuals",
        #     torch.stack([visual_conditions, visual_effects])
        #     .cpu()
        #     .detach()
        #     .numpy(),
        # )

        # Retrieve representations
        representations = self.get_representations(
            attention_states, visual_conditions, visual_effects
        )
        self.representations = representations

        # Compute competence
        self.competences = self._compute_matches()
        self.competence = self.competences.mean()

        # Update hyperparameters
        self.set_hyperparams()

        # Spread map outputs for the update graph
        attention_output = self.attention_map(attention_states)
        visual_conditions_output = self.visual_conditions_map(
            visual_conditions
        )
        visual_effects_output = self.visual_effects_map(visual_effects)

        self._update_maps(
            attention_output, visual_conditions_output, visual_effects_output
        )

        self.weight_change = self.compute_weight_change()

    def get_map_representations(self, map, norms, std_baseline, grid=True):
        """
        Compute point and grid representations for a map.

        Parameters:
        - map: The map object to compute representations for.
        - norms: Norms to be used in computing the representation.
        - std_baseline: Baseline standard deviation for modulation.
        - grid (bool, optional): Indicates whether to include grid
          representation; defaults to True.

        Returns:
        - A dictionary with point and grid representations.
        """
        return {
            "point": map.get_representation(norms, rtype="point"),
            "grid": map.get_representation(
                norms, rtype="grid", neighborhood_std=std_baseline
            ),
        }

    def get_representations(
        self, attention_states, visual_conditions, visual_effects
    ):
        """
        Compute representations of states in the maps.

        Parameters:
        - attention_states: Current states of attention inputs.
        - visual_conditions: Visual condition states.
        - visual_effects: Visual effect states.

        Returns:
        - A dictionary with point and grid representations for attention,
          visual conditions, and visual effects.
        """
        std_baseline = self.params.neighborhood_modulation_baseline

        representations = {
            "pa": self.get_map_representations(
                self.attention_map,
                self.attention_map(attention_states),
                std_baseline,
            ),
            "pvc": self.get_map_representations(
                self.visual_conditions_map,
                self.visual_conditions_map(visual_conditions),
                std_baseline,
            ),
            "pve": self.get_map_representations(
                self.visual_effects_map,
                self.visual_effects_map(visual_effects),
                std_baseline,
            ),
        }
        return representations

    def _update_maps(
        self, attention_output, visual_conditions_output, visual_effects_output
    ):
        """
        Update topological maps using their respective updaters and current
        representations.

        Parameters:
        - attention_output: Output from the attention map processing.
        - visual_conditions_output: Output from the visual conditions map.
        - visual_effects_output: Output from the visual effects map.
        """
        point_attention_representations = self.representations["pa"]["point"]
        neighborhood_modulation = self.neighborhood_modulation
        learningrate_modulation = self.learningrate_modulation

        self.visual_conditions_updater(
            output=visual_conditions_output,
            neighborhood_std=neighborhood_modulation,
            anchors=point_attention_representations,
            learning_modulation=learningrate_modulation,
            neighborhood_std_anchors=self.params.anchor_std,
        )

        self.visual_effects_updater(
            output=visual_effects_output,
            neighborhood_std=neighborhood_modulation,
            anchors=point_attention_representations,
            learning_modulation=learningrate_modulation,
            neighborhood_std_anchors=self.params.anchor_std,
        )

        self.attention_updater(
            output=attention_output,
            neighborhood_std=neighborhood_modulation,
            anchors=point_attention_representations,
            learning_modulation=learningrate_modulation,
            neighborhood_std_anchors=self.params.anchor_std,
        )

    def _compute_matches(self):
        """
        Compute the matching scores between visual effects and attention
        states.

        Returns:
        - A tensor of match scores based on the Euclidean distance between
          points.
        """
        # Determine the positional difference between the "pve" and "pa"
        # representations
        pve_pa_difference = (
            self.representations["pve"]["point"]
            - self.representations["pa"]["point"]
        )
        # Compute the Euclidean norm of the above difference, resulting in a
        # distance measure
        norm_pve_pa = torch.norm(pve_pa_difference, dim=-1)

        # Determine the positional difference between the "pvc" and "pa"
        # representations
        pvc_pa_difference = (
            self.representations["pvc"]["point"]
            - self.representations["pa"]["point"]
        )
        # Compute the Euclidean norm of the  above difference for distance
        # measurement
        norm_pvc_pa = torch.norm(pvc_pa_difference, dim=-1)

        # Stack the calculated norms to form a distances tensor
        dists = torch.stack([norm_pve_pa, norm_pvc_pa])

        # Convert distances to similarity scores using a Gaussian-like decay
        # based on match_std
        matches = torch.exp(-((self.match_std**-2) * dists**2))
        # Calculate the average of the scores for a comprehensive similarity
        # measure
        matches = matches.mean(0)

        # Return the computed average similarity scores
        return matches

    def compute_weight_change(self):
        """
        Compute the change in weights for each map.

        The method compares the current weights with previously stored
        weights for specified keys (e.g., 'visual_conditions',
        'visual_effects', 'attention'). It calculates the norm (magnitude
        of difference) between the current and previous weights for each
        key, and then updates the stored weights with the current ones.

        Returns:
            - norms (dict): A dictionary containing the norms of the weight
              differences for each specified map:
              ["visual_conditions", "visual_effects", "attention"].
        """
        keys = ["visual_conditions", "visual_effects", "attention"]

        # Initialize weight_dict if it doesn't exist
        if not hasattr(self, "weight_dict"):
            self.weight_dict = {
                key: torch.zeros_like(getattr(self, f"{key}_map").weights)
                for key in keys
            }

        weight_dict_curr = {
            key: getattr(self, f"{key}_map").weights.clone() for key in keys
        }

        # Calculate norms and update weight_dict in a single loop
        norms = {}
        for key in keys:
            norms[key] = torch.norm(
                self.weight_dict[key] - weight_dict_curr[key]
            )
            self.weight_dict[key] = weight_dict_curr[key]

        return norms

    def get_action_from_condition(self, condition):
        """
        Retrieve the action representation given a visual condition.

        Parameters:
        - condition: A visual state to obtain corresponding action.

        Returns:
        - A tuple containing the focus point and representation.
        """
        condition_tensor = (
            torch.tensor(condition.ravel().reshape(1, -1), dtype=torch.float32)
            / 255.0
        )
        norm = self.visual_conditions_map(condition_tensor)

        representation = self.visual_conditions_map.get_representation(
            norm,
            rtype="point",
            neighborhood_std=self.params.neighborhood_modulation_baseline,
        )

        focus = self.attention_map.backward(
            representation, self.params.neighborhood_modulation_baseline
        )
        return (
            focus.cpu().detach().numpy(),
            representation.cpu().detach().numpy(),
        )

    def save(self, file_path):
        """
        Serialize and save the state of the OfflineController to a file.

        Parameters:
        - file_path: Path to the file where the state should be saved.
        """
        state = {
            "epoch": self.epoch,
            "competence": self.competence,
            "competences": self.competences,
            "local_incompetence": self.local_incompetence,
            "visual_conditions_map_state_dict": (
                self.visual_conditions_map.state_dict()
            ),
            "visual_conditions_updater_optimizer_state_dict": (
                self.visual_conditions_updater.optimizer.state_dict()
            ),
            "visual_effects_map_state_dict": (
                self.visual_effects_map.state_dict()
            ),
            "visual_effects_updater_optimizer_state_dict": (
                self.visual_effects_updater.optimizer.state_dict()
            ),
            "attention_map_state_dict": (self.attention_map.state_dict()),
            "attention_updater_optimizer_state_dict": (
                self.attention_updater.optimizer.state_dict()
            ),
            "rng_state": self.rng.get_state(),
        }

        torch.save(state, file_path)

    @staticmethod
    def load(file_path, env, params, seed=None):
        """
        Load and restart an OfflineController from a saved state.

        Parameters:
        - file_path: Path to the file from which to load the state.
        - env: Environment to be associated with the loaded controller.
        - params: Parameter settings to apply.
        - seed: Optional seed for random number generators.

        Returns:
        - An OfflineController instance with the loaded state.
        """
        state = torch.load(file_path, weights_only=False)

        # Create a new OfflineController instance
        offline_controller = OfflineController(env, params, seed)

        # Restore saved state
        offline_controller.epoch = state["epoch"] + 1
        offline_controller.competence = state["competence"]
        offline_controller.competences = state["competences"]
        offline_controller.local_incompetence = state["local_incompetence"]
        offline_controller.visual_conditions_map.load_state_dict(
            state["visual_conditions_map_state_dict"]
        )
        offline_controller.visual_conditions_updater.optimizer.load_state_dict(
            state["visual_conditions_updater_optimizer_state_dict"]
        )
        offline_controller.visual_effects_map.load_state_dict(
            state["visual_effects_map_state_dict"]
        )
        offline_controller.visual_effects_updater.optimizer.load_state_dict(
            state["visual_effects_updater_optimizer_state_dict"]
        )
        offline_controller.attention_map.load_state_dict(
            state["attention_map_state_dict"]
        )
        offline_controller.attention_updater.optimizer.load_state_dict(
            state["attention_updater_optimizer_state_dict"]
        )
        offline_controller.rng.set_state(state["rng_state"])

        return offline_controller
