"""Load the visual-conditions weights of ./off_control_store for inspection.

Usage: run inside a trained simulation folder:
    python /path/to/src/test_weights.py

Normalizes each map unit's prototype to [0, 1] and tiles the 10x10 units'
16x16x3 prototypes into a (160, 160, 3) image in `visual_weights`. Nothing
is displayed or saved (the imshow call is commented out); use it
interactively (e.g. python -i).
"""
import numpy as np
import torch


store = torch.load("off_control_store", weights_only=False)
visual_weights = store["visual_conditions_map_state_dict"]["weights"]
visual_weights = visual_weights.detach().cpu().numpy()


def normalize(x):
    return (x - x.min()) / np.ptp(x)


visual_weights = np.vstack([normalize(x) for x in visual_weights.T]).T
visual_weights = visual_weights.reshape(16, 16, 3, 10, 10)
visual_weights = visual_weights.transpose(3, 0,4 , 1, 2)
visual_weights = visual_weights.reshape(16 * 10, 16 * 10, -1)
#
# plt.imshow(visual_weights)
