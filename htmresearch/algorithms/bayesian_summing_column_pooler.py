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


class BayesianSummingColumnPooler(BayesianColumnPoolerBase):
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
            # Proximal
            sampleSizeProximal=20,
            # Distal
            sampleSizeDistal=20,
            inertiaFactor=1.,
            # Bayesian
            noise=0.01,  # lambda
            activationThreshold=0.5,  # probability such that a cell becomes active
            forgetting=0.1,
            useSupport=False,
            avoidWeightExplosion=True,
            resetProximalCounter=False,
            useProximalProbabilities=True,
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

        super(BayesianSummingColumnPooler, self).__init__(
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
            activationThreshold=activationThreshold,
            forgetting=forgetting,
            useSupport=useSupport,
            avoidWeightExplosion=avoidWeightExplosion,
            resetProximalCounter=resetProximalCounter,
            useProximalProbabilities=useProximalProbabilities,
            seed=seed
        )

        self.distalConnectionCounts = list(np.zeros((self.cellCount, n)) for n in lateralInputWidths)
        self.internalDistalConnectionCount = np.zeros((self.cellCount, self.cellCount))
        self.proximalConnectionCount = np.zeros((self.cellCount, self.inputWidth))

        self.distalCellActivityCounts = list(np.zeros(self.cellCount) for n in lateralInputWidths)
        self.internalDistalCellActivityCount = np.zeros(self.cellCount)
        self.proximalCellActivityCount = np.zeros(self.cellCount)

        self.distalInputCounts = list(np.zeros(n) for n in lateralInputWidths)
        self.proximalInputCount = np.zeros(self.inputWidth)

        self.updateCounter = 0
        self.numberOfObjects = 0

    def _beforeUpdate(self, connectionIndicator):
        pass
        # if connectionIndicator == self.CONNECTION_ENUM["internalDistal"]:
        #     if self.avoidWeightExplosion:
        #         self.updateCounter = 0
        #     if self.resetProximalCounter:
        #         self._resetProximalCounter()

    def _updateConnectionData(self, connectionIndicator, **kwargs):
        if connectionIndicator == self.CONNECTION_ENUM["proximal"]:
            inputValues = kwargs["inputValues"]
            connectionCount = self.proximalConnectionCount
            cellActivityCount = self.proximalCellActivityCount
            inputCount = self.proximalInputCount

        elif connectionIndicator == self.CONNECTION_ENUM["internalDistal"]:
            inputValues = self.activeCells
            connectionCount = self.internalDistalConnectionCount
            cellActivityCount = self.internalDistalCellActivityCount
            inputCount = None

        elif connectionIndicator == self.CONNECTION_ENUM["distal"]:
            inputValues = kwargs["inputValues"]
            connectionCount = self.distalConnectionCounts
            cellActivityCount = self.distalCellActivityCounts
            inputCount = self.distalInputCounts

        else:
            raise AssertionError("Connection indicator is set to an invalid value.\n"
                                 "Valid parameter values = proximal, internalDistal and distal")

        # Updating connection count
        inputSize = connectionCount.shape[-1]
        connectionMatrix = np.outer(self.activeCells, inputValues)
        connectionMatrix = connectionMatrix.reshape(self.cellCount, inputSize)
        connectionCount += connectionMatrix

        # Updating cell activity count
        cellActivityCount += self.activeCells

        if inputCount is not None:
            # Updating input activity count
            inputCount += inputValues

    def _afterUpdate(self, connectionIndicator):
        if connectionIndicator == self.CONNECTION_ENUM["internalDistal"]:
            self.numberOfObjects += 1
        if connectionIndicator == self.CONNECTION_ENUM["distal"]:
            self.updateCounter += 1

    def _updateWeights(self, connectionIndicator, **kwargs):
        if connectionIndicator == self.CONNECTION_ENUM["proximal"]:
            connectionCount = self.proximalConnectionCount
            activationCount = self.proximalCellActivityCount
            inputCount = self.proximalInputCount
            updateCounter = self.updateCounter

        elif connectionIndicator == self.CONNECTION_ENUM["internalDistal"]:
            connectionCount = self.internalDistalConnectionCount
            activationCount = self.internalDistalCellActivityCount
            inputCount = self.internalDistalCellActivityCount
            updateCounter = self.numberOfObjects

        elif connectionIndicator == self.CONNECTION_ENUM["distal"]:
            index = kwargs["index"]
            connectionCount = self.distalConnectionCounts[index]
            activationCount = self.distalCellActivityCounts[index]
            inputCount = self.distalInputCounts[index]
            updateCounter = self.updateCounter[index]

        else:
            raise AssertionError("Connection indicator is set to an invalid value.\n"
                                 "Valid parameter values = proximal, internalDistal and distal")

        weights = (connectionCount * updateCounter) / np.outer(
            activationCount,
            inputCount
        ).reshape(connectionCount.shape)
        weights[weights == 0] = 1. / float(updateCounter)
        weights[np.isnan(weights)] = 1.

        return weights

    def _updateBias(self, connectionIndicator, **kwargs):
        if connectionIndicator == self.CONNECTION_ENUM["proximal"]:
            activationCount = self.proximalCellActivityCount
            updateCounter = self.updateCounter

        elif connectionIndicator == self.CONNECTION_ENUM["internalDistal"]:
            activationCount = self.internalDistalCellActivityCount
            updateCounter = self.numberOfObjects

        elif connectionIndicator == self.CONNECTION_ENUM["distal"]:
            index = kwargs["index"]
            activationCount = self.distalCellActivityCounts[index]
            updateCounter = self.updateCounter[index]

        else:
            raise AssertionError("Connection indicator is set to an invalid value.\n"
                                 "Valid parameter values = proximal, internalDistal and distal")

        bias = np.log(activationCount / float(updateCounter))
        bias[np.isneginf(bias)] = np.log(1 / float(updateCounter ** 2))
        return bias

    ###################################################################################################################
    # Newly added private functions
    ###################################################################################################################

    def _resetProximalCounter(self):
        self.proximalConnectionCount = np.zeros(self.proximalConnectionCount.shape)
        self.proximalCellActivityCount = np.zeros(self.proximalCellActivityCount.shape)
        self.proximalInputCount = np.zeros(self.proximalInputCount.shape)

