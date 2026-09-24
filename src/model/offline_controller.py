"""Offline controller: three topological maps plus a competence predictor.

visual_conditions_map (fovea before a saccade), visual_effects_map (fovea
after) and attention_map (normalized attention center) share a 2D grid of
maps_output_size units. States are recorded online and the maps are
trained offline at the end of each epoch with an STM loss anchored on the
visual-conditions winner (the goal); a logistic predictor learns how close
the attention and visual-effects winners fall to the goal (competence),
which in turn modulates plasticity.
"""
import numpy as np
import torch

from model.predict import Predictor, PredictorUpdater
from model.topological_maps import FilteredTopologicalMap as TopologicalMap
from model.topological_maps import Updater


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
        self.competence = 0.0
        self.competences = torch.tensor(0.0)
        self.local_incompetence = torch.tensor(0.0)
        self.match_std = self.params.match_std

        # Random generator initialization
        self.seed = seed if seed is not None else 0
        self.rng = np.random.RandomState(self.seed)

        self._init_internal_space_prototypes()

        # Initialize maps for sensory processing
        self._init_maps()

        # Initialize prediction component
        self._init_predictor()

        # Preallocate storage for states
        self._init_storage()

        # Set initial hyperparameters
        self.set_hyperparams()

    def _init_internal_space_prototypes(self):
        """Build a Gaussian bump centered on every map unit.

        Returns:
            torch.Tensor: (maps_output_size, maps_output_size) bumps with
            std neighborhood_modulation_baseline, used to average the
            predictor over the whole map (global competence).
        """
        output_size = self.params.maps_output_size
        output_side = int(output_size**0.5)

        side_idcs = torch.arange(output_side).float()
        grid_x, grid_y = torch.meshgrid(side_idcs, side_idcs, indexing="ij")
        grid = torch.stack([grid_x.ravel(), grid_y.ravel()], dim=-1)

        self.prototype_grid_reps = torch.stack(
            [
                torch.exp(
                    -(self.params.neighborhood_modulation_baseline**-2)
                    * torch.norm(grid - m, dim=-1) ** 2
                )
                for m in grid
            ]
        )

        return self.prototype_grid_reps

    def _init_maps(self):
        """Initialize the topological maps and their updaters."""
        input_size = np.prod(self.env.observation_space["FOVEA"].sample().shape)
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

    def _init_predictor(self):
        """Create the competence predictor and its Adam updater."""
        output_size = self.params.maps_output_size
        lr = self.params.predictor_learning_rate
        self.predictor = Predictor(output_size)
        self.predictor_updater = PredictorUpdater(self.predictor, learning_rate=lr)

    def _init_storage(self):
        """Initialize storage for visual, action, and attention states."""
        visual_size = np.prod(self.env.observation_space["FOVEA"].sample().shape)
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
        self.timestep_competences = np.zeros(
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
        self.timestep_competences.fill(None)

    def set_hyperparams(self):
        """Set the controller's hyperparameters based on current competence.

        incompetence = 1 - tanh(decaying_speed * competence) (scalar) and
        local_incompetence = 1 - tanh(local_decaying_speed * competences)
        (N, 1). Learning-rate and neighborhood modulations are
        baseline + gain * incompetence * local_incompetence, shape (N, 1).
        """
        decay = np.tanh(self.params.decaying_speed * self.competence)
        local_decay = torch.tanh(self.params.local_decaying_speed * self.competences)

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

        self.neighborhood_modulation = self.neighborhood_modulation.reshape(-1, 1)

        self.match_std = self.params.match_std

    def get_local_competence(self, representation):
        """Predicted competence for one grid representation.

        Args:
            representation (torch.Tensor): (1, maps_output_size) bump.

        Returns:
            float: predictor output rescaled from [0.5, 1] to [0, 1]
            (values below 0.5 map to 0).
        """
        comp = self.predictor(representation.reshape(1, -1))[0]
        comp = comp.tolist()[0]

        # convert values so that [0.5, 1.0] maps into [0, 1]
        comp = 2 * (np.maximum(0.5, comp) - 0.5)

        return comp

    def get_global_competence(self):
        """Mean predicted competence over all map units, rescaled as in
        get_local_competence.

        Returns:
            float: global competence in [0, 1].
        """
        grid_comps = self.predictor(self.prototype_grid_reps)
        comp = grid_comps.mean().tolist()

        # convert values so that [0.5, 1.0] maps into [0, 1]
        comp = 2 * (np.maximum(0.5, comp) - 0.5)

        return comp

    def generate_saccade(self, visual_input):
        """Propose an attention center for the current fovea image.

        With probability equal to the local competence of the fovea's
        winner on the visual-conditions map, returns the attention-map
        weight at that winner; otherwise a random point at distance
        0.3-0.6 from the retina center.

        Args:
            visual_input (np.ndarray): (16, 16, 3) uint8 fovea image.

        Returns:
            tuple: (saccade, competence). saccade is a length-2 list or
            array of normalized retina coordinates (may fall slightly
            outside [0, 1] when random); competence is a float in [0, 1].
        """

        torch_visual_input = torch.tensor(visual_input).reshape(1, -1) / 255.0
        visual_map_output = self.visual_conditions_map(torch_visual_input)
        reps = self.get_map_representations(
            self.visual_conditions_map,
            visual_map_output,
            self.params.neighborhood_modulation_baseline,
        )

        competence = self.get_local_competence(reps["grid"])
        coin = self.rng.rand() > (1 - competence)

        if coin:
            saccade = self.attention_map.backward(
                reps["point"],
                self.params.neighborhood_modulation_baseline,
            )
            saccade = saccade.flatten().tolist()
        else:
            r = 0.3 + 0.3 * self.rng.rand()
            a = 2 * np.pi * self.rng.rand()
            saccade = [0.5, 0.5] + r * np.array([np.cos(a), np.sin(a)])

        return saccade, competence

    def record_states(self, episode, saccade, ts, state):
        """
        Record the visual, action, and attention states for a given timestep.

        Parameters:
        - episode: The current episode number.
        - saccade: The current saccade number.
        - ts: The current timestep within a saccade.
        - state: The state dictionary containing 'vision', 'action',
          'attention' and 'competence' keys.
        """

        self.visual_states[episode, saccade, ts] = state["vision"].ravel() / 255.0
        self.action_states[episode, saccade, ts] = state["action"]
        self.attention_states[episode, saccade, ts] = state["attention"]
        self.world_states[episode, saccade, ts] = state["world"]
        self.timestep_competences[episode, saccade, ts] = state["competence"]

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

        For each salient timestep ts with offset < ts < saccade_time - offset
        (offset = 2) uses the fovea at ts - 2 as visual condition, the fovea
        at ts + 2 as visual effect and the attention/competence at ts + 2;
        samples whose attention or competence at ts + 2 was not recorded yet
        (NaN) are skipped.
        Computes matches, global competence and hyperparameters, then
        updates the three maps and the predictor, and stores the weight
        change norms in self.weight_change.
        """
        offset = 2
        # Only consider saccades that are not on the time-limits edges
        idcs = self.filtered_idcs
        ts_cond = (offset < idcs[2]) & (idcs[2] < (self.params.saccade_time - offset))
        idcs = idcs[:, ts_cond]
        # Drop samples whose attention or competence was not yet set
        recorded = ~(
            np.isnan(self.attention_states[idcs[0], idcs[1], idcs[2] + offset])
            .any(-1)
            | np.isnan(
                self.timestep_competences[idcs[0], idcs[1], idcs[2] + offset]
            ).any(-1)
        )
        idcs = idcs[:, recorded]

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
        competences = get_state_data(self.timestep_competences, offset=offset)

        # Retrieve representations
        representations = self.get_representations(
            attention_states, visual_conditions, visual_effects
        )
        self.representations = representations

        # Compute competence
        self.matches = self._compute_matches()
        self.competences = competences
        self.competence = self.get_global_competence()

        # hyperparameters
        self.set_hyperparams()

        # Updates
        self._update_maps(
            attention_states,
            visual_conditions,
            visual_effects,
        )
        self._update_predictor()

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

    def get_representations(self, attention_states, visual_conditions, visual_effects):
        """
        Compute representations of states in the maps.

        Parameters:
        - attention_states: Current states of attention inputs.
        - visual_conditions: Visual condition states.
        - visual_effects: Visual effect states.

        Returns:
        - A dictionary with point and grid representations for attention,
          visual conditions, and visual effects and goals.
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
        # The goal is the visual-conditions winner
        representations["pg"] = representations["pvc"]
        return representations

    def _update_maps(
        self,
        attention_states,
        visual_conditions,
        visual_effects,
    ):
        """
        Update topological maps using their respective updaters and current
        representations.

        Parameters:
        - attention_states: attentional templetes i.e. saccades.
        - visual_conditions: fovea vision before saccades.
        - visual_effects: fovea vision after saccades.
        """

        # Spread map outputs for the update graph
        attention_output = self.attention_map(attention_states)
        visual_conditions_output = self.visual_conditions_map(visual_conditions)
        visual_effects_output = self.visual_effects_map(visual_effects)

        # define common vars
        point_goal_representations = self.representations["pg"]["point"]
        neighborhood_modulation = self.neighborhood_modulation
        learningrate_modulation = self.learningrate_modulation

        # Update maps
        self.visual_conditions_updater(
            output=visual_conditions_output,
            neighborhood_std=neighborhood_modulation,
            anchors=point_goal_representations,
            learning_modulation=learningrate_modulation,
            neighborhood_std_anchors=self.params.anchor_std,
        )

        self.visual_effects_updater(
            output=visual_effects_output,
            neighborhood_std=neighborhood_modulation,
            anchors=point_goal_representations,
            learning_modulation=learningrate_modulation,
            neighborhood_std_anchors=self.params.anchor_std,
        )

        self.attention_updater(
            output=attention_output,
            neighborhood_std=neighborhood_modulation,
            anchors=point_goal_representations,
            learning_modulation=learningrate_modulation,
            neighborhood_std_anchors=self.params.anchor_std,
        )

    def _update_predictor(self):
        point_goal_representations = self.representations["pg"]["grid"]
        outputs = self.predictor(point_goal_representations)
        self.predictor_updater(outputs, self.matches.reshape(-1, 1), 1.0)

    def _compute_matches(self):
        """
        Compute the matching scores between the goal (visual-conditions
        winner) and the attention and visual-effects winners.

        Returns:
        - A (N,) tensor of match scores in [0, 1]: the mean over the two
          winners of exp(-(d / match_std)**2), d being the grid distance
          from the goal.
        """
        goal = self.representations["pg"]["point"]
        norm_pa_pg = torch.norm(
            self.representations["pa"]["point"] - goal, dim=-1
        )
        norm_pve_pg = torch.norm(
            self.representations["pve"]["point"] - goal, dim=-1
        )
        dists = torch.stack([norm_pa_pg, norm_pve_pg])

        matches = torch.exp(-((self.match_std**-2) * dists**2))
        return matches.mean(0)

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
                key: torch.zeros_like(getattr(self, f"{key}_map").weights) for key in keys
            }

        weight_dict_curr = {
            key: getattr(self, f"{key}_map").weights.clone() for key in keys
        }

        # Calculate norms and update weight_dict in a single loop
        norms = {}
        for key in keys:
            norms[key] = torch.norm(self.weight_dict[key] - weight_dict_curr[key])
            self.weight_dict[key] = weight_dict_curr[key]

        return norms

    def get_representation_from_condition(self, condition):
        """Converts a visual condition to a goal representation.

        Args:
            condition (array): The visual condition to be converted, an array
                of FOVEA activation.

        Returns:
            array: The representation derived from the condition.
        """
        # Convert condition to a normalized tensor
        condition_tensor = (
            torch.tensor(condition.ravel().reshape(1, -1), dtype=torch.float32) / 255.0
        )
        # Get normalized output from the visual conditions map
        som_output_norm = self.visual_conditions_map(condition_tensor)

        # Retrieve the representation using the normalized output
        representation = self.visual_conditions_map.get_representation(
            som_output_norm,
            rtype="point",
            neighborhood_std=self.params.neighborhood_modulation_baseline,
        )

        return representation

    def apply_filter_to_map(self, map_object, target_goal, filter_strength):
        """Applies a filter to the map object based on the target goal.

        Args:
            map_object: The map object to which the filter will be applied.
            target_goal: The goal used to filter the map. If None, no filter
                will be applied.
            filter_strength: The strength of the filter to be applied.
        """
        if target_goal is not None:
            map_object.set_filter(
                [target_goal],
                magnitude=filter_strength,
            )

    def get_saccade_from_representation(self, representation):
        """Calculate the attentional focus from a given representation.

        This function uses the attention map to compute the attentional focus
        based on the provided representation and neighborhood modulation
        baseline.

        Args:
            representation: The input representation to process.

        Returns:
            numpy.ndarray: The computed attentional focus as a NumPy array.
        """
        attentional_focus = self.attention_map.backward(
            representation, self.params.neighborhood_modulation_baseline
        )
        return attentional_focus.cpu().detach().numpy()

    def get_action_from_condition(
        self,
        condition,
        condition_filter=None,
        filter_magnitude=0,
    ):
        """
        Retrieve the action representation given a visual condition.

        Parameters:
        - condition: A visual state to obtain corresponding action.
        - condition_filter: A predicted point representation.
        - filter_magnitude: The proportion of influence of the filter

        Returns:
        - A tuple containing the focus point and representation.
        """
        condition_tensor = (
            torch.tensor(condition.ravel().reshape(1, -1), dtype=torch.float32) / 255.0
        )

        if condition_filter is not None:
            self.visual_conditions_map.set_filter(
                [condition_filter],
                magnitude=filter_magnitude,
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
            "visual_conditions_map_state_dict": (self.visual_conditions_map.state_dict()),
            "visual_conditions_updater_optimizer_state_dict": (
                self.visual_conditions_updater.optimizer.state_dict()
            ),
            "visual_effects_map_state_dict": (self.visual_effects_map.state_dict()),
            "visual_effects_updater_optimizer_state_dict": (
                self.visual_effects_updater.optimizer.state_dict()
            ),
            "attention_map_state_dict": (self.attention_map.state_dict()),
            "attention_updater_optimizer_state_dict": (
                self.attention_updater.optimizer.state_dict()
            ),
            "rng_state": self.rng.get_state(),
            "predictor_state_dict": (self.predictor.state_dict()),
            "predictor_updater_optimizer_state_dict": (
                self.predictor_updater.optimizer.state_dict()
            ),
        }
        if hasattr(self, "weight_dict"):
            state["weight_dict"] = self.weight_dict

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
        offline_controller.predictor.load_state_dict(state["predictor_state_dict"])
        offline_controller.predictor_updater.optimizer.load_state_dict(
            state["predictor_updater_optimizer_state_dict"]
        )
        offline_controller.rng.set_state(state["rng_state"])
        if "weight_dict" in state:
            offline_controller.weight_dict = state["weight_dict"]

        return offline_controller
