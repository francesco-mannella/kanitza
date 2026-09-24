"""Save the visual-conditions prototypes of ./off_control_store as an image.

Usage: run inside a trained simulation folder (reads off_control_store and,
if present, loaded_params):
    python /path/to/src/test_weights.py

Normalizes each map unit's prototype to [0, 1] and tiles the units'
fovea-shaped prototypes (fovea_size x channels) on the map grid into
visual_weights.png.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from params import Parameters


params = Parameters()
try:
    params.load("loaded_params")
except FileNotFoundError:
    print("no local parameters")

store = torch.load("off_control_store", weights_only=False)
visual_weights = store["visual_conditions_map_state_dict"]["weights"]
visual_weights = visual_weights.detach().cpu().numpy()


def normalize(x):
    """Rescale x to [0, 1] (zeros if x is constant)."""
    span = np.ptp(x)
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


fovea_height, fovea_width = params.fovea_size
side = int(np.sqrt(params.maps_output_size))
channels = visual_weights.shape[0] // (fovea_height * fovea_width)

visual_weights = np.vstack([normalize(x) for x in visual_weights.T]).T
visual_weights = visual_weights.reshape(
    fovea_height, fovea_width, channels, side, side
)
visual_weights = visual_weights.transpose(3, 0, 4, 1, 2)
visual_weights = visual_weights.reshape(
    side * fovea_height, side * fovea_width, channels
)
plt.imsave("visual_weights.png", visual_weights.squeeze())
print("Saved visual_weights.png")
