```
# Main function - entry point of the program
FUNCTION Main():
    # Load configuration parameters from file/settings
    params = LoadParameters()
    # Initialize agent attention mask centered at (0.5, 0.5) (center of retina)
    agent_attention_mask = [0.5, 0.5]
    # Load existing offline controller or create new one if none exists
    off_control = LoadOrCreateOfflineController()
    
    # Training loop over epochs, resuming from last saved epoch
    FOR epoch FROM off_control.epoch TO params.epochs:
        # Clear/reset internal states of offline controller for new epoch
        ResetOfflineControllerStates(off_control)
        
        # Loop through all episodes within current epoch
        FOR episode FROM 0 TO params.episodes:
            # Determine world type: triangles (0) or other (1) based on epoch number
            world_type = red_triangle IF (epoch % 100 < params.triangles_percent) ELSE blue_square
            # Initialize environment with selected world type
            InitWorld(env, world_type)
            # Reset environment to initial state
            ResetEnv(env)
            
            # Loop through saccades (eye movements) per episode
            FOR saccade_idx FROM 0 TO params.saccade_num:
                # Initialize action (delta w.r.t. current retina center) as no movement
                action = [0, 0]
                
                # Loop through time steps within each saccade
                FOR time_step FROM 0 TO params.saccade_time:
                    # Execute action and get observation from environment
                    observation = Step(env, action)
                    
                    # At midpoint of saccade, generate new saccade target
                    IF time_step == midpoint:
                        # Generate saccade direction and competence from foveal input
                        saccade, competence = GenerateSaccade(off_control, observation["FOVEA"])
                        # Create Gaussian attention mask centered on saccade target
                        agent_attention_mask = GenerateGaussianMask(saccade, ComputeVariance(saccade))
                    ELSE:
                        # Use default centered Gaussian mask when not at midpoint
                        agent_attention_mask = GenerateGaussianMask([0.5, 0.5], ComputeVariance([0.5, 0.5]))
                    
                    # Extract visual features using Gabor filters on retinal input
                    saliency_map = ApplyGaborFilters(observation["RETINA"])
                    # Apply attention mask to saliency map
                    saliency_map = saliency_map * agent_attention_mask
                    # Sample most salient point from weighted saliency map
                    salient_point = SampleFromSaliencyMap(saliency_map)
                    # Convert salient point to eye movement action
                    action = ConvertToEyeMovement(salient_point)
                    # Store current state data for later learning
                    RecordState(off_control, episode, saccade_idx, time_step, observation, action, competence)
        
        # Filter recorded states to keep only salient/important ones
        # (for each sacade the state before saccade and the state after saccade)
        FilterSalientStates(off_control)
        # Update offline controller models with collected data
        UpdateOfflineController(off_control)

# Function to update the offline controller's internal models
FUNCTION UpdateOfflineController(off_control):
    # Get visual states from 2 time steps before current
    visual_conditions = GetVisualStates(off_control, -2)
    # Get visual states from 2 time steps after current
    visual_effects = GetVisualStates(off_control, +2)
    # Get attention states from 2 time steps after current
    attention_states = GetAttentionStates(off_control, +2)
    
    # Extract learned representations from attention and visual data
    representations = GetRepresentationsFromMaps(attention_states, visual_conditions, visual_effects)
    # Compute similarity/matching scores between representations.
    # match is the mean radial basis of the distance between the condition - effect representations from the action representation
    matches = ComputeMatches(representations)
    
    # Adjust learning hyperparameters based on current competence level (learning rates, neighborhoods)
    SetHyperparamsBasedOnCompetence(off_control)
    # Update self-organizing topological maps with new data
    UpdateTopologicalMaps(off_control, attention_states, visual_conditions, visual_effects)
    # Train predictor network on representations and their matches
    UpdatePredictor(off_control, representations, matches)

END PROGRAM
