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
from nupic.bindings.math import Random

import os
VERBOSITY = os.getenv('NUMENTA_VERBOSITY', 0)

class BayesianColumnPoolerBase(object):

    CONNECTION_ENUM = {
        "proximal": 0,
        "internalDistal": 1,
        "distal": 2
    }

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
            maxSdrSize = None,
            minSdrSize = None,
            # Proximal
            sampleSizeProximal=20,
            # Distal
            sampleSizeDistal=20,
            inertiaFactor=1.,
            # Bayesian
            noise=0.01,  # lambda
            activationThreshold=0.5, # probability such that a cell becomes active
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

        assert maxSdrSize is None or maxSdrSize >= sdrSize
        assert minSdrSize is None or minSdrSize <= sdrSize

        self.inputWidth = inputWidth
        self.lateralInputWidths = lateralInputWidths
        self.cellCount = cellCount
        self.sdrSize = sdrSize

        self.maxSdrSize = sdrSize if maxSdrSize is None else maxSdrSize
        self.minSdrSize = sdrSize if minSdrSize is None else minSdrSize

        self.sampleSizeProximal = sampleSizeProximal
        self.sampleSizeDistal = sampleSizeDistal
        self.inertiaFactor = inertiaFactor

        self.prevActiveCells = np.zeros(self.cellCount, dtype="float64")
        self.activeCells = np.zeros(self.cellCount, dtype="float64")
        self.activePredictionCells = np.zeros(self.cellCount, dtype="float64")
        self._random = Random(seed)
        self.useInertia=True

        # Bayesian parameters (weights, bias, moving averages)
        # Each row represents one segment on a cell, so each cell potentially has

        # Weights 2D-Matrix - (1 segment per) cells x distalInput
        # Needs to be split up, because each segment only connects to the specified input
        self.distalWeights = list(np.zeros((self.cellCount, n)) for n in lateralInputWidths)
        self.internalDistalWeights = np.zeros((self.cellCount, self.cellCount))
        self.proximalWeights = np.zeros((self.cellCount, self.inputWidth))

        self.distalBias = list(np.zeros(self.cellCount) for n in lateralInputWidths)
        self.internalDistalBias = np.zeros(self.cellCount)
        self.proximalBias = np.zeros(self.cellCount)

        self.noise = noise
        self.activationThreshold = activationThreshold
        self.forgetting = forgetting

        self.useSupport = useSupport
        self.avoidWeightExplosion = avoidWeightExplosion
        self.resetProximalCounter = resetProximalCounter
        self.useProximalProbabilities = useProximalProbabilities

    def reset(self):
        """
        Reset internal states. When learning this signifies we are to learn a
        unique new object.
        """
        self.prevActiveCells = np.zeros(self.cellCount, dtype="float64")
        self.activeCells = np.zeros(self.cellCount, dtype="float64")
        self.activePredictionCells = np.zeros(self.cellCount, dtype="float64")

    def compute(
            self,
            feedforwardInput=(),
            lateralInputs=(),
            feedforwardGrowthCandidates=None,
            learn=True,
            predictedInput=None,
            onlyProximal=False
    ):
        """
        Runs one time step of the column pooler algorithm.

        @param  feedforwardInput (sequence)
                Sorted indices of active feedforward input bits

        @param  lateralInputs (list of sequences)
                For each lateral layer, a list of sorted indices of active lateral
                input bits

        @param  feedforwardGrowthCandidates (sequence or None)
                Sorted indices of feedforward input bits that active cells may grow
                new synapses to. If None, the entire feedforwardInput is used.

        @param  learn (bool)
                If True, we are learning a new object

        @param predictedInput (sequence)
               Sorted indices of predicted cells in the TM layer.
        """

        if feedforwardGrowthCandidates is None:
            feedforwardGrowthCandidates = feedforwardInput

        # inference step
        if not learn:
            self._computeInferenceMode(
                feedforwardInput,
                lateralInputs
            )

        # learning step
        else:
            self._computeLearningMode(
                feedforwardInput,
                lateralInputs,
                feedforwardGrowthCandidates
            )

    ###################################################################################################################
    # Private methods
    ###################################################################################################################

    # TODO: Here and in bayesian-apical-TM: Sampling the connections we learn on? (e.g. with growth candidates)
    # Currently all weights are changed
    def _computeLearningMode(self,
                             feedforwardInput,
                             lateralInputs,
                             feedforwardGrowthCandidates,
                             temporalLearningRate=None  # Makes it possible to update moving averages while not learning
                             # and to turn off updating moving averages
                             ):
        """
        Learning mode: we are learning a new object in an online fashion. If there
        is no prior activity, we randomly activate 'sdrSize' cells and create
        connections to incoming input. If there was prior activity, we maintain it.
        If we have a union, we simply do not learn at all.

        These cells will represent the object and learn distal connections to each
        other and to lateral cortical columns.

        Parameters:
        ----------------------------
        @param  feedforwardInput (sequence)
                Sorted indices of active feedforward input bits

        @param  lateralInputs (list of sequences)
                For each lateral layer, a list of sorted indices of active lateral
                input bits

        @param  feedforwardGrowthCandidates (sequence or None)
                Sorted indices of feedforward input bits that the active cells may
                grow new synapses to.  This is assumed to be the predicted active
                cells of the input layer.
        """
        # If there are not enough previously active cells, then we are no longer on
        # a familiar object.  Either our representation decayed due to the passage
        # of time (i.e. we moved somewhere else) or we were mistaken.  Either way,
        # create a new SDR and learn on it.
        # This case is the only way different object representations are created.
        # enforce the active cells in the output layer
        if self.numberOfActiveCells() < self.minSdrSize:
            # Randomly activate sdrSize bits from an array with cellCount bits
            self.activeCells = BayesianColumnPoolerBase._randomActivation(self.cellCount, self.sdrSize, self._random)

        self._beforeUpdate(connectionIndicator=BayesianColumnPoolerBase.CONNECTION_ENUM["internalDistal"])
        self._updateConnectionData(connectionIndicator=BayesianColumnPoolerBase.CONNECTION_ENUM["internalDistal"])
        self._afterUpdate(connectionIndicator=BayesianColumnPoolerBase.CONNECTION_ENUM["internalDistal"])
        # Internal distal learning
        # Pattern does not change while output neurons are kept constantly active, hence weights only needed to be
        # updated once
        self.internalDistalWeights, \
        self.internalDistalBias = self._learn(
            connectionIndicator=BayesianColumnPoolerBase.CONNECTION_ENUM["internalDistal"]
        )

        # Update Counts for proximal weights
        self._beforeUpdate(connectionIndicator=BayesianColumnPoolerBase.CONNECTION_ENUM["proximal"])
        self._updateConnectionData(
            connectionIndicator=BayesianColumnPoolerBase.CONNECTION_ENUM["proximal"],
            inputValues=feedforwardInput
        )
        self._afterUpdate(connectionIndicator=BayesianColumnPoolerBase.CONNECTION_ENUM["proximal"])

        # Update counts for lateral weights to other columns
        self._beforeUpdate(connectionIndicator=BayesianColumnPoolerBase.CONNECTION_ENUM["distal"])
        for i, lateralInput in enumerate(lateralInputs):
            self._updateConnectionData(
                connectionIndicator=BayesianColumnPoolerBase.CONNECTION_ENUM["distal"],
                inputValues=lateralInput,
                index=i
            )

        self._afterUpdate(connectionIndicator=BayesianColumnPoolerBase.CONNECTION_ENUM["distal"])

        # Update weights based on current frequency
        self.proximalWeights, \
        self.proximalBias = self._learn(connectionIndicator=BayesianColumnPoolerBase.CONNECTION_ENUM["proximal"])

        if VERBOSITY > 1:
            act = self._activation(self.proximalWeights, feedforwardInput, self.proximalBias, self.noise)
            print "Activation with current weights in learning" #, act[act > 0]

        # External distal learning
        for i, _ in enumerate(lateralInputs):
            self.distalWeights[i], \
            self.distalBias[i] = self._learn(
                connectionIndicator=BayesianColumnPoolerBase.CONNECTION_ENUM["distal"],
                index=i
            )


    def _computeInferenceMode(self, feedforwardInput, lateralInputs, onlyProximal=False):
        """
        Inference mode: if there is some feedforward activity, perform
        spatial pooling on it to recognize previously known objects, then use
        lateral activity to activate a subset of the cells with feedforward
        support. If there is no feedforward activity, use lateral activity to
        activate a subset of the previous active cells.

        Parameters:
        ----------------------------
        @param  feedforwardInput (sequence)
                Sorted indices of active feedforward input bits

        @param  lateralInputs (list of sequences)
                For each lateral layer, a list of sorted indices of active lateral
                input bits
        """

        self.prevActiveCells = self.activeCells.copy()
        # Calculate the feed forward activation
        # Support is only added if a prediction needs to be made
        feedForwardActivation = self._activation(self.proximalWeights, feedforwardInput, self.proximalBias, self.noise)

        # Forgetting process for values that doesn't get feed forward input
        self.activeCells -= self.forgetting*self.activeCells
        self.activeCells[self.activeCells < self.activationThreshold] = 0
        # Update cell values with new activation
        # Activation is either 1 (for every feed forward input) or it is set to the activation itself
        self.activeCells[
            feedForwardActivation >= self.activationThreshold
            ] = feedForwardActivation[feedForwardActivation >= self.activationThreshold] if self.useProximalProbabilities else 1

        activity = self.activeCells.copy()

        if self.useSupport:
            # first touch has no previous activation => exclude
            if np.any(self.prevActiveCells):
                activity *= self._activation(self.internalDistalWeights, self.prevActiveCells, self.internalDistalBias,
                                             self.noise)

            for i, lateralInput in enumerate(lateralInputs):
                activity *= self._activation(self.distalWeights[i], lateralInput, self.distalBias[i], self.noise)

        self.activePredictionCells = activity
        if VERBOSITY > 1:
            print "Column pooler inference input", feedforwardInput.nonzero()
            print "Column pooler activation output", self.activeCells[self.activeCells.nonzero()[0]]

    def _supportedActivation(self, activation, distalWeights):
        """
        Introduce concept of mutual entropy (similar to mutual information)
        activation_j = sum( - activation_i * E(log(w_ij)))
        :returns: mutual entropy
        """
        # Mask invalid values since it would otherwise lead to all values equal -inf
        weights = np.ma.masked_invalid(distalWeights)
        supportedActivation = weights.dot(activation)
        mask = np.ma.getmask(supportedActivation)
        supportedActivation[mask] = np.NINF
        return supportedActivation

    def _learn(self, connectionIndicator, **kwargs):
        weights = self._updateWeights(connectionIndicator=connectionIndicator, **kwargs)
        # set division by zero to zero since this represents unused segments
        weights = np.log(weights)

        bias = self._updateBias(connectionIndicator=connectionIndicator, **kwargs)

        return weights, bias

    ###################################################################################################################
    # Static methods
    ###################################################################################################################

    @staticmethod
    def _activation(weights, input, bias, noise, useBias=True, ignoreNinf=False):
        # To avoid explosion of activation value the input is normalised
        # It's made sure that all weight values are set. Hence we can make use of simple matrix multiplication
        normalization = float(input.sum())
        transformed_input = input / normalization if normalization > 0 else np.zeros(input.shape)
        activation = weights.dot(transformed_input) if not useBias else weights.dot(transformed_input) + bias
        return np.exp(activation)

    @staticmethod
    def _randomActivation(n, k, randomizer):
        # Activate k from n bits, randomly (using custom randomizer for reproducibility)
        activeCells = np.zeros(n, dtype="float64")
        indices = BayesianColumnPoolerBase._sampleRange(randomizer, 0, n, step=1, k=k)
        activeCells[indices] = 1
        return activeCells

    @staticmethod
    def _sampleRange(rng, start, end, step, k):
        """
        Equivalent to:

        random.sample(xrange(start, end, step), k)

        except it uses our random number generator.

        This wouldn't need to create the arange if it were implemented in C.
        """
        array = np.empty(k, dtype="uint32")
        rng.sample(np.arange(start, end, step, dtype="uint32"), array)
        return array

    ###################################################################################################################
    # Getter / setter methods
    ###################################################################################################################

    def numberOfInputs(self):
        """
        Returns the number of inputs into this layer
        """
        return self.inputWidth

    def numberOfCells(self):
        """
        Returns the number of cells in this layer.
        @return (int) Number of cells
        """
        return self.cellCount

    def numberOfActiveCells(self):
        return sum(self.activeCells > self.activationThreshold)

    def getActiveCellsIndices(self):
        # np.where returns tuple for all dimensions -> return array of fist dimension
        # return np.where(self.activeCells >= self.activationThreshold)[0]
        return np.nonzero(self.activeCells)[0]

    def getObjectPrediction(self):
        """
        :returns: Prediction of the most probable object
        """
        # If support is used, activity is no probabilities anymore
        # THus the threshold can be lower than 0
        activity = self.activeCells if not self.useSupport else self.activePredictionCells
        threshold = 0.01 if activity[activity > 0.01].shape[0] > 0 else 0.0 # if not self.useSupport else np.NINF

        # No probabilities anymore, thus do not filter for values greater than 0
        zippedActivation = filter(lambda x: x[1] > threshold, zip(range(len(activity)), activity))
        zippedActivation.sort(key=lambda t: t[1])
        zippedActivation.reverse()
        # if len(zippedActivation) >= self.sdrSize:
        #     indices, support = zip(*zippedActivation[:self.sdrSize])
        # else:
        #     indices, support = zip(*zippedActivation)
        indices, support = zip(*zippedActivation)

        return list(indices)

    def getActiveCellValues(self):
        return self.activeCells

    def getActiveCells(self):
        """
        Returns the indices of the active cells.
        @return (list) Indices of active cells.
        """
        return self.getActiveCellsIndices()

    def getUseInertia(self):
        """
        Get whether we actually use inertia  (i.e. a fraction of the
        previously active cells remain active at the next time step unless
        inhibited by cells with both feedforward and lateral support).
        @return (Bool) Whether inertia is used.
        """
        return self.useInertia

    def setUseInertia(self, useInertia):
        """
        Sets whether we actually use inertia (i.e. a fraction of the
        previously active cells remain active at the next time step unless
        inhibited by cells with both feedforward and lateral support).
        @param useInertia (Bool) Whether inertia is used.
        """
        self.useInertia = useInertia

    ###################################################################################################################
    # Abstract methods
    ###################################################################################################################

    def _beforeUpdate(self, connectionIndicator):
        pass

    def _updateConnectionData(self, connectionIndicator, **kwargs):
        pass

    def _afterUpdate(self, connectionIndicator):
        pass

    def _updateWeights(self, connectionIndicator, **kwargs):
        pass

    def _updateBias(self, connectionIndicator, **kwargs):
        pass

