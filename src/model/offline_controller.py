import numpy as np
import torch

from model.topological_maps import STMUpdater, TopologicalMap


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
        self.epoch = 0
        self.competence = torch.tensor(0.0)
        self.competences = torch.tensor(0.0)
        self.local_incompetence = torch.tensor(0.0)

        # Environment and parameters setup
        self.env = env
        self.params = params

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
        self.visual_conditions_updater = STMUpdater(
            self.visual_conditions_map, self.params.maps_learning_rate
        )

        # Visual effects map
        self.visual_effects_map = TopologicalMap(input_size, output_size)
        self.visual_effects_updater = STMUpdater(
            self.visual_effects_map, self.params.maps_learning_rate
        )

        # Attention map
        input_size = self.params.attention_size
        self.attention_map = TopologicalMap(input_size, output_size)
        self.attention_updater = STMUpdater(
            self.attention_map, self.params.maps_learning_rate
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

    def reset_states(self):
        """Reset the stored states for the current episode."""
        self.visual_states.fill(0)
        self.action_states.fill(0)
        self.attention_states.fill(0)

    def set_hyperparams(self):
        """Set the controller's hyperparameters based on current competence."""
        decay = torch.tanh(self.params.decaying_speed * self.competence)
        local_decay = torch.tanh(
            self.params.local_decaying_speed * self.competences
        )

        self.incompetence = 1 - decay
        self.local_incompetence = (1 - local_decay).reshape(-1, 1)

        lm_baseline = self.params.learnigrate_modulation_baseline
        self.learnigrate_modulation = lm_baseline + (
            self.params.learnigrate_modulation
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

    def generate_attentional_input(self, saccade_num):
        """
        Generate random attentional input.

        Parameters:
        - saccade_num: Number of saccades to generate input for.

        Returns:
        - 2D array of attentional inputs with random values.
        """
        return self.rng.rand(saccade_num, 2)

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

    def filter_salient_states(self):
        """
        Filter out states based on the magnitude of the saccades.
        States where the magnitude exceeds the configured threshold
        are considered salient.
        """
        magnitudes = np.linalg.norm(self.action_states, axis=-1)
        is_salient = magnitudes > self.params.saccade_threshold
        self.filtered_idcs = np.stack(np.where(is_salient))

    def update_maps(self):
        """
        Update all sensory maps based on stored states and current competence.
        """
        # Only consider states that are not on the time edges
        idcs = self.filtered_idcs
        ts_cond = (2 < idcs[2]) & (idcs[2] < (self.params.saccade_time - 2))
        idcs = idcs[:, ts_cond]

        # Get states for attention, visual conditions, and visual effects
        def get_state_data(offset):
            return torch.tensor(
                self.visual_states[idcs[0], idcs[1], idcs[2] + offset],
                dtype=torch.float32,
            ).reshape(-1, self.params.visual_size)

        attention_states = torch.tensor(
            self.attention_states[idcs[0], idcs[1], idcs[2] + 2]
        ).reshape(-1, self.params.attention_size)

        visual_conditions = get_state_data(-2)
        visual_effects = get_state_data(2)

        # Retrieve representations
        representations = self._get_representations(
            attention_states, visual_conditions, visual_effects
        )
        self.representations = representations

        # Compute competence
        self.matches = self._compute_matches()
        self.competences = torch.exp(
            -((self.params.match_std**-2) * self.matches**2)
        )
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

    def _get_representations(
        self, attention_states, visual_conditions, visual_effects
    ):
        """
        Compute representations of states in the maps.

        Parameters:
        - attention_states: Current states of attention inputs.
        - visual_conditions: Visual condition representations.
        - visual_effects: Visual effect representations.

        Returns:
        - A dictionary with point and grid representations for attention,
          visual conditions, and visual effects.
        """

        def get_map_representations(map, norms, std_baseline):
            return {
                "point": map.get_representation(norms, rtype="point"),
                "grid": map.get_representation(
                    norms, rtype="grid", std=std_baseline
                ),
            }

        std_baseline = self.params.neighborhood_modulation_baseline

        representations = {
            "pa": get_map_representations(
                self.attention_map,
                self.attention_map(attention_states),
                std_baseline,
            ),
            "pvc": get_map_representations(
                self.visual_conditions_map,
                self.visual_conditions_map(visual_conditions),
                std_baseline,
            ),
            "pve": get_map_representations(
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
        learnigrate_modulation = self.learnigrate_modulation

        self.visual_conditions_updater(
            output=visual_conditions_output,
            std=neighborhood_modulation,
            target=point_attention_representations,
            learning_modulation=learnigrate_modulation,
            target_std=1,
        )

        self.visual_effects_updater(
            output=visual_effects_output,
            std=neighborhood_modulation,
            target=point_attention_representations,
            learning_modulation=learnigrate_modulation,
            target_std=1,
        )

        self.attention_updater(
            output=attention_output,
            std=neighborhood_modulation,
            target=point_attention_representations,
            learning_modulation=learnigrate_modulation,
            target_std=1,
        )

    def _compute_matches(self):
        """
        Compute the matching scores between visual effects and attention
        states.

        Returns:
        - A tensor of match scores based on the Euclidean distance between
          points.
        """
        difference = (
            self.representations["pve"]["point"]
            - self.representations["pa"]["point"]
        )
        return torch.norm(difference, dim=-1)

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
            std=self.params.neighborhood_modulation_baseline,
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
