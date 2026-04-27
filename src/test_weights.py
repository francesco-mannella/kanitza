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
