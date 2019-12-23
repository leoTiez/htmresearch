# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import numpy as np
from htmresearch.algorithms.bayesian_column_pooler_base import BayesianColumnPoolerBase


class BayesianColumnPooler(BayesianColumnPoolerBase):
    """
    This class constitutes a temporary implementation for a cross-column pooler.
    The implementation goal of this class is to prove basic properties before
    creating a cleaner implementation.
    """

    def __init__(
            self,
            inputWidth,
            lateralInputWidths=(),
            cellCount=4096,
            sdrSize=40,
            maxSdrSize=None,
            minSdrSize=None,
            forgetting=0.1,
            # Proximal
            sampleSizeProximal=20,
            # Distal
            sampleSizeDistal=20,
            inertiaFactor=1.,
            # Bayesian
            noise=0.01,  # lambda
            learningRate=0.1,  # alpha
            activationThreshold=0.5,  # probability such that a cell becomes active
            useSupport=False,
            avoidWeightExplosion=True,
            resetProximalCounter=False,
            useProximalProbabilities=True,
            initMovingAverages=0.0,
            seed=42
    ):
        """
    Parameters:
    ----------------------------
    @param  inputWidth (int)
            The number of bits in the feedforward input

    @param  lateralInputWidths (list of ints)
            The number of bits in each lateral input

    @param  sdrSize (int)
            The number of active cells in an object SDR

    @param  onlineLearning (Bool)
            Whether or not the column pooler should learn in online mode.

    @param  maxSdrSize (int)
            The maximum SDR size for learning.  If the column pooler has more
            than this many cells active, it will refuse to learn.  This serves
            to stop the pooler from learning when it is uncertain of what object
            it is sensing.

    @param  minSdrSize (int)
            The minimum SDR size for learning.  If the column pooler has fewer
            than this many active cells, it will create a new representation
            and learn that instead.  This serves to create separate
            representations for different objects and sequences.

            If online learning is enabled, this parameter should be at least
            inertiaFactor*sdrSize.  Otherwise, two different objects may be
            incorrectly inferred to be the same, as SDRs may still be active
            enough to learn even after inertial decay.

    @param  synPermProximalInc (float)
            Permanence increment for proximal synapses

    @param  synPermProximalDec (float)
            Permanence decrement for proximal synapses

    @param  initialProximalPermanence (float)
            Initial permanence value for proximal synapses

    @param  sampleSizeProximal (int)
            Number of proximal synapses a cell should grow to each feedforward
            pattern, or -1 to connect to every active bit

    @param  minThresholdProximal (int)
            Number of active synapses required for a cell to have feedforward
            support

    @param  connectedPermanenceProximal (float)
            Permanence required for a proximal synapse to be connected

    @param  predictedInhibitionThreshold (int)
            How much predicted input must be present for inhibitory behavior
            to be triggered.  Only has effects if onlineLearning is true.

    @param  synPermDistalInc (float)
            Permanence increment for distal synapses

    @param  synPermDistalDec (float)
            Permanence decrement for distal synapses

    @param  sampleSizeDistal (int)
            Number of distal synapses a cell should grow to each lateral
            pattern, or -1 to connect to every active bit

    @param  initialDistalPermanence (float)
            Initial permanence value for distal synapses

    @param  activationThresholdDistal (int)
            Number of active synapses required to activate a distal segment

    @param  connectedPermanenceDistal (float)
            Permanence required for a distal synapse to be connected

    @param  inertiaFactor (float)
            The proportion of previously active cells that remain
            active in the next timestep due to inertia (in the absence of
            inhibition).  If onlineLearning is enabled, should be at most
            1 - learningTolerance, or representations may incorrectly become
            mixed.

    @param  seed (int)
            Random number generator seed
    """

        super(BayesianColumnPooler, self).__init__(
            inputWidth=inputWidth,
            lateralInputWidths=lateralInputWidths,
            cellCount=cellCount,
            sdrSize=sdrSize,
            maxSdrSize=maxSdrSize,
            minSdrSize=minSdrSize,
            sampleSizeProximal=sampleSizeProximal,
            sampleSizeDistal=sampleSizeDistal,
            inertiaFactor=inertiaFactor,
            noise=noise,
            forgetting=forgetting,
            activationThreshold=activationThreshold,
            useSupport=useSupport,
            avoidWeightExplosion=avoidWeightExplosion,
            resetProximalCounter=resetProximalCounter,
            useProximalProbabilities=useProximalProbabilities,
            seed=seed
        )
        self.learningRate = learningRate

        # Initialise weights to first segment randomly TODO check whether this is necessary. (commented out)
        # for d in self.distalWeights:
        #   d[:, :] = np.random.random(d[:, :].shape)
        # self.internalDistalWeights[:, :] = np.random.random(self.internalDistalWeights[:, :].shape)
        # self.proximalWeights[:, :] = np.random.random(self.proximalWeights[:, :].shape)

        if initMovingAverages == 0:
            initMovingAverages = self.noise**2

        self.distalMovingAverages = list(np.full((self.cellCount, n), initMovingAverages) for n in lateralInputWidths)
        self.internalDistalMovingAverages = np.full((self.cellCount, self.cellCount), initMovingAverages)
        self.proximalMovingAverages = np.full((self.cellCount, self.inputWidth), initMovingAverages)

        self.distalMovingAverageBias = list(np.full(self.cellCount, initMovingAverages) for _ in lateralInputWidths)
        self.internalDistalMovingAverageBias = np.full(self.cellCount, initMovingAverages)
        self.proximalMovingAverageBias = np.full(self.cellCount, initMovingAverages)

        self.distalMovingAverageInput = list(np.full(n, initMovingAverages) for n in lateralInputWidths)
        self.internalDistalMovingAverageInput = np.full(self.cellCount, initMovingAverages)
        self.proximalMovingAverageInput = np.full(self.inputWidth, initMovingAverages)

    def _beforeUpdate(self, connectionIndicator):
        pass

    def _updateConnectionData(self, connectionIndicator, **kwargs):
        if connectionIndicator == self.CONNECTION_ENUM["proximal"]:
            inputValues = kwargs["inputValues"]
            inputSize = self.inputWidth
            movingAverage = self.proximalMovingAverages
            movingAverageBias = self.proximalMovingAverageBias
            movingAverageInput = self.proximalMovingAverageInput

        elif connectionIndicator == self.CONNECTION_ENUM["internalDistal"]:
            inputValues = self.activeCells
            inputSize = self.cellCount
            movingAverage = self.internalDistalMovingAverages
            movingAverageBias = self.internalDistalMovingAverageBias
            movingAverageInput = self.internalDistalMovingAverageInput

        elif connectionIndicator == self.CONNECTION_ENUM["distal"]:
            inputValues = kwargs["inputValues"]
            index = kwargs["index"]
            inputSize = self.lateralInputWidths[index]
            movingAverage = self.distalMovingAverages[index]
            movingAverageBias = self.distalMovingAverageBias[index]
            movingAverageInput = self.distalMovingAverageInput[index]

        else:
            raise AssertionError("Connection indicator is set to an invalid value.\n"
                                 "Valid parameter values = proximal, internalDistal and distal")

        # Update only values for the active cells indices. Since we have a sparse representation it is likely that every
        # neuron is only active for one particular pattern or few ones. Hence, the more update steps are taken the less the
        # neuron becomes activated and thus the weights are more decreased. This leads to a forgetting mechanism that is
        # not desired in a sparse representation
        active_cells_indices = self.getActiveCellsIndices()
        # Updating moving average weights to input
        noisy_connection_matrix = np.outer((1 - self.noise ** 2) * self.activeCells, inputValues)
        # Consider only active segments
        noisy_connection_matrix[noisy_connection_matrix > 0] += self.noise ** 2
        noisy_connection_matrix = noisy_connection_matrix.reshape(self.cellCount, inputSize)
        movingAverage[active_cells_indices, :] += self.learningRate * (
                noisy_connection_matrix[active_cells_indices, :] - movingAverage[active_cells_indices, :]
        )

        # Updating moving average bias activity
        noisy_input_vector = (1 - self.noise) * self.activeCells
        # Consider only active segments
        noisy_input_vector[noisy_input_vector > 0] += self.noise
        movingAverageBias[active_cells_indices] += self.learningRate * (
                noisy_input_vector[active_cells_indices] - movingAverageBias[active_cells_indices]
        )

        # Updating moving average input activity
        input_mask = inputValues > 0
        noisy_input_vector = (1 - self.noise) * inputValues
        # Consider only active segments
        noisy_input_vector[noisy_input_vector > 0] += self.noise
        movingAverageInput[input_mask] += self.learningRate * (
                noisy_input_vector[input_mask] - movingAverageInput[input_mask]
        )

    def _afterUpdate(self, connectionIndicator):
        pass

    def _updateWeights(self, connectionIndicator, **kwargs):
        if connectionIndicator == self.CONNECTION_ENUM["proximal"]:
            movingAverages = self.proximalMovingAverages
            movingAverageBias = self.proximalMovingAverageBias
            movingAverageInput = self.proximalMovingAverageInput

        elif connectionIndicator == self.CONNECTION_ENUM["internalDistal"]:
            movingAverages = self.internalDistalMovingAverages
            movingAverageBias = self.internalDistalMovingAverageBias
            movingAverageInput = self.internalDistalMovingAverageInput

        elif connectionIndicator == self.CONNECTION_ENUM["distal"]:
            index = kwargs["index"]
            movingAverages = self.distalMovingAverages[index]
            movingAverageBias = self.distalMovingAverageBias[index]
            movingAverageInput = self.distalMovingAverageInput[index]

        else:
            raise AssertionError("Connection indicator is set to an invalid value.\n"
                                 "Valid parameter values = proximal, internalDistal and distal")

        weights = movingAverages / np.outer(
            movingAverageBias,
            movingAverageInput
        )

        return weights

    def _updateBias(self, connectionIndicator, **kwargs):
        if connectionIndicator == self.CONNECTION_ENUM["proximal"]:
            movingAverageBias = self.proximalMovingAverageBias

        elif connectionIndicator == self.CONNECTION_ENUM["internalDistal"]:
            movingAverageBias = self.internalDistalMovingAverageBias

        elif connectionIndicator == self.CONNECTION_ENUM["distal"]:
            index = kwargs["index"]
            movingAverageBias = self.distalMovingAverageBias[index]

        else:
          raise AssertionError("Connection indicator is set to an invalid value.\n"
                               "Valid parameter values = proximal, internalDistal and distal")

        bias = np.log(movingAverageBias)
        return bias