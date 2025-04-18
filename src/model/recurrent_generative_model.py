import pickle

import numpy as np
from params_recurrent_generative_model import ParamsFORCE


class RecurrentGenerativeModelUpdater:
    """
    A class to update the readout weights of a recurrent neural network using
    the FORCE learning algorithm.

    Attributes:
    -----------
    network : object
        A network object expected to have the attributes:
        - `reservoir_activity`: The activity of the reservoir at the current
          time step.
        - `readout_weights`: The weights from the reservoir to the readout
          layer.

    paramsFORCE : object
        Parameters object expected to have the attribute:
        - `num_recurrent_units`: Number of units in the reservoir.
        - `alpha`: Regularization parameter for the FORCE update.

    P : np.ndarray
        Inverse correlation matrix used in FORCE learning.

    """

    def __init__(self, network, paramsFORCE=None):

        if paramsFORCE is None:
            paramsFORCE = ParamsFORCE()

        self.network = network
        self.paramsFORCE = paramsFORCE
        self.P = (
            np.eye(self.paramsFORCE.num_recurrent_units)
            / self.paramsFORCE.alpha
        )

    def _readout_update(self, error):

        P_r = np.dot(self.P, self.network.reservoir_activity)
        r_P = np.dot(self.network.reservoir_activity.T, self.P)
        r_P_r = np.dot(self.network.reservoir_activity.T, P_r)
        coefficient = 1 / (1 + r_P_r)
        self.P = self.P - coefficient * np.outer(P_r, r_P)
        self.network.readout_weights = self.network.readout_weights - np.dot(
            np.dot(self.P, self.network.reservoir_activity), error.T
        )

    def __call__(self, error, update=True):

        if update is True:
            self._readout_update(error)


class RecurrentGenerativeModel:

    def __init__(self, paramsFORCE=None, feedback=True):

        if paramsFORCE is None:
            paramsFORCE = ParamsFORCE()
        self.paramsFORCE = paramsFORCE

        self.output_side = int(np.sqrt(self.paramsFORCE.num_output_units))

        self.tau = self.paramsFORCE.RNN_tau
        self.g = self.paramsFORCE.rec_gain
        self.phi = self.paramsFORCE.phi
        self.N_inputs = self.paramsFORCE.num_input_units
        self.N_rec = self.paramsFORCE.num_recurrent_units
        self.N_readouts = self.paramsFORCE.num_output_units
        self.dt = self.paramsFORCE.dt

        self.x = np.random.randn(self.N_rec, 1)
        self.reservoir_activity = np.tanh(self.x)
        self.readout_activity = np.zeros((self.N_readouts, 1))

        self.input_weights = np.random.uniform(
            self.paramsFORCE.uniform_dist,
            -self.paramsFORCE.uniform_dist,
            (
                self.N_rec,
                self.N_inputs,
            ),
        )

        self.rec_weights = np.random.normal(
            self.paramsFORCE.normal_dist_mean,
            self.paramsFORCE.normal_dist_sd,
            (self.N_rec, self.N_rec),
        )

        self.rec_weights[
            np.random.random_sample((self.N_rec, self.N_rec))
            > self.paramsFORCE.p_rec_connections
        ] = 0  # recurrent network has sparseness
        self.rec_weights *= (
            self.g
        )  # apply the gain to the recurrent connections

        self.readout_weights = np.random.uniform(
            self.paramsFORCE.uniform_dist * 0.1,
            -self.paramsFORCE.uniform_dist * 0.1,
            (self.N_rec, self.N_readouts),
        )

        if feedback is True:
            self.feedback_weights = self.phi * np.random.uniform(
                self.paramsFORCE.uniform_dist,
                -self.paramsFORCE.uniform_dist,
                (self.N_rec, self.N_readouts),
            )
        else:
            self.feedback_weights = np.zeros(self.N_rec, self.N_readouts)

    def reset(self):

        self.x = np.random.randn(self.N_rec, 1)
        self.reservoir_activity = np.tanh(self.x)
        self.readout_activity = np.zeros((self.N_readouts, 1))

    def update(self, inputs=None, mode="default"):

        if mode == "training":

            teacher_ratio = 1

            dx = (
                (
                    -self.x
                    + np.dot(self.rec_weights, self.reservoir_activity)
                    + teacher_ratio
                    * np.dot(
                        self.feedback_weights,
                        inputs
                        + np.random.uniform(
                            -self.paramsFORCE.teacher_noise,
                            self.paramsFORCE.teacher_noise,
                            (self.N_readouts, 1),
                        ),
                    )
                    + (1 - teacher_ratio)
                    * np.dot(self.feedback_weights, self.readout_activity)
                )
                * self.dt
                / self.tau
            )

            self.x = self.x + dx

            self.reservoir_activity = np.tanh(self.x)

            self.readout_activity = np.dot(
                self.readout_weights.T, self.reservoir_activity
            )

        elif mode == "input":

            dx = (
                (
                    -self.x
                    + np.dot(self.rec_weights, self.reservoir_activity)
                    + np.dot(self.feedback_weights, inputs)
                )
                * self.dt
                / self.tau
            )

            self.x = self.x + dx

            self.reservoir_activity = np.tanh(self.x)

            self.readout_activity = np.dot(
                self.readout_weights.T, self.reservoir_activity
            )

        else:

            dx = (
                (
                    -self.x
                    + np.dot(self.rec_weights, self.reservoir_activity)
                    + np.dot(self.feedback_weights, self.readout_activity)
                )
                * self.dt
                / self.tau
            )

            self.x = self.x + dx

            self.reservoir_activity = np.tanh(self.x)

            self.readout_activity = np.dot(
                self.readout_weights.T, self.reservoir_activity
            )

    def _target_shape(
        self,
        target_shape_lenght=None,
    ):
        """Generates the shape of target units activation"""

        if target_shape_lenght is None:
            target_shape_lenght = self.paramsFORCE.target_shape_lenght

        s = np.linspace(-400, 400, target_shape_lenght)
        temp = -((s) ** 2)
        return (temp - np.min(temp)) / np.max(temp - np.min(temp))

    def _compute_error(self, readouts, target_function):

        return readouts - target_function

    def _STM_to_RNN(self, goal):
        """

        Parameters
        ----------
        goal : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        linearized_scanpath = int(goal[0] * self.output_side + goal[1])
        target_sequence = np.zeros(
            (
                self.paramsFORCE.num_output_units,
                self.paramsFORCE.target_shape_lenght,
            )
        )
        target_sequence[int(linearized_scanpath)] = self._target_shape()

        return target_sequence

    def _RNN_to_STM(self, readouts_storage):
        """

        Parameters
        ----------
        goal : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        """
        Take the peak activation at the mean of the target shape lenght
        """
        peak_timestep = int(self.paramsFORCE.target_shape_lenght / 2)
        peak_activity = readouts_storage[:, peak_timestep]

        """
        Find the winning unit
        """
        winning_unit = np.where(peak_activity == np.max(peak_activity))[0][0]

        """
        Find the winning unit position in a 2D grid
        """
        pos_y = winning_unit // self.output_side
        pos_x = winning_unit % self.output_side
        winning_unit_pos = np.array([pos_y, pos_x])

        return winning_unit_pos

    def step(
        self,
        goal=None,
        timesteps=None,
        RNN_updater=None,
        mode="default",
    ):
        """
        Parameters
        ----------
        RNN : class object, recurrent neural network

        goal : tuple or array or list with x and y coordinates

        timesteps : time of network update

        RNN_updater : class object, RNN updater

        mode : the function can be run in three modalities.
              Mode default -> no input, no updater, it requires only how many timesteps the network has to run
              Mode input -> the network is given an input (goal)
              Mode training -> the network is given an input (goal) as target function. It requires the RNN updater too

        Returns
        -------
        None.

        """

        if timesteps is None:
            timesteps = self.paramsFORCE.target_shape_lenght

        if goal is None:
            mode = "default"
        elif RNN_updater is None:
            mode = "input"
        else:
            mode = "training"

        readouts_storage = np.zeros((self.N_readouts, timesteps))

        if mode == "default":
            for t in range(int(timesteps)):
                self.update(inputs=None, mode="default")
                readouts_storage[:, [t]] = self.readout_activity

        elif mode == "input":
            if goal is None:
                raise TypeError(
                    "Error! You tried to use the RNN in input mode, but the input was nan"
                )
            target_sequence = self._STM_to_RNN(goal)
            for t in range(len(target_sequence[1])):
                self.update(inputs=target_sequence[:, [t]], mode="input")
                readouts_storage[:, [t]] = self.readout_activity

        elif mode == "training":
            if goal is None:
                raise TypeError(
                    "Error! You tried to use the RNN in training mode, but the target function was nan"
                )
            if (
                isinstance(RNN_updater, RecurrentGenerativeModelUpdater)
                is False
            ):
                raise TypeError(
                    "Error! The RNN updater you are using is not a FORCE updater"
                )
            target_sequence = self._STM_to_RNN(goal)
            for t in range(len(target_sequence[1]) - 1):
                self.update(inputs=target_sequence[:, [t]], mode="training")
                error = self._compute_error(
                    self.readout_activity, target_sequence[:, [t + 1]]
                )
                self.updater(error)
                readouts_storage[:, [t]] = self.readout_activity

        predicted_goal = self._RNN_to_STM(readouts_storage)
        return predicted_goal

    def save(self, filename):
        """
        Saves the RecurrentGenerativeModel object to a file using pickle.

        Parameters:
        -----------
        filename : str
            The name of the file to save the object to.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def load(self, filename):
        """
        Loads a RecurrentGenerativeModel object from a file using pickle.

        Parameters:
        -----------
        filename : str
            The name of the file to load the object from.
        """
        with open(filename, "rb") as f:
            loaded_model = pickle.load(f)
            self.__dict__.update(loaded_model.__dict__)


if __name__ == "__main__":
    rnn = RecurrentGenerativeModel()
