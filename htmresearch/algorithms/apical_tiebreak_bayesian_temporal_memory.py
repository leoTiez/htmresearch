# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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

"""An implementation of TemporalMemory"""

import numpy as np

from htmresearch.algorithms.apical_tiebreak_bayesian_base import ApicalTiebreakBayesianTemporalMemoryBase


class BayesianApicalTiebreakPairMemory(ApicalTiebreakBayesianTemporalMemoryBase):
    """
    A generalized Temporal Memory with apical dendrites that add a "tiebreak".

    Basal connections are used to implement traditional Temporal Memory.

    The apical connections are used for further disambiguation. If multiple cells
    in a minicolumn have active basal segments, each of those cells is predicted,
    unless one of them also has an active apical segment, in which case only the
    cells with active basal and apical segments are predicted.

    In other words, the apical connections have no effect unless the basal input
    is a union of SDRs (e.g. from bursting minicolumns).

    This class is generalized in two ways:

    - This class does not specify when a 'timestep' begins and ends. It exposes
      two main methods: 'depolarizeCells' and 'activateCells', and callers or
      subclasses can introduce the notion of a timestep.
    - This class is unaware of whether its 'basalInput' or 'apicalInput' are from
      internal or external cells. They are just cell numbers. The caller knows
      what these cell numbers mean, but the TemporalMemory doesn't.
    """

    def __init__(
            self,
            columnCount=2048,
            basalInputSize=0,  # Must be non-equal zero
            apicalInputSize=0,  # Must be non-equal zero
            cellsPerColumn=32,
            initialPermanence=0.21,
            # Changed to float
            minThreshold=0.5,
            sampleSize=20,
            noise=0.01,  # lambda
            learningRate=0.1,  # alpha
            maxSegmentsPerCell=255,
            initMovingAverages=0.0,
            seed=42
    ):
        """
    @param columnCount (int)
    The number of minicolumns

    @param basalInputSize (sequence)
    The number of bits in the basal input

    @param apicalInputSize (int)
    The number of bits in the apical input

    @param cellsPerColumn (int)
    Number of cells per column

    @param reducedBasalThreshold (int)
    The activation threshold of basal (lateral) segments for cells that have
    active apical segments. If equal to activationThreshold (default),
    this parameter has no effect.

    @param initialPermanence (float)
    Initial permanence of a new synapse

    @param minThreshold (int)
    If the number of potential synapses active on a segment is at least this
    threshold, it is said to be "matching" and is eligible for learning.

    @param sampleSize (int)
    How much of the active SDR to sample with synapses.

    @param basalPredictedSegmentDecrement (float)
    Amount by which segments are punished for incorrect predictions.

    @param apicalPredictedSegmentDecrement (float)
    Amount by which segments are punished for incorrect predictions.

    @param maxSynapsesPerSegment
    The maximum number of synapses per segment.

    @param seed (int)
    Seed for the random number generator.
    """
        super(BayesianApicalTiebreakPairMemory, self).__init__(
            columnCount=columnCount,
            basalInputSize=basalInputSize,
            apicalInputSize=apicalInputSize,
            cellsPerColumn=cellsPerColumn,
            initialPermanence=initialPermanence,
            minThreshold=minThreshold,
            sampleSize=sampleSize,
            noise=noise,
            learningRate=learningRate,
            maxSegmentsPerCell=maxSegmentsPerCell,
            seed=seed
        )

        self.basalMovingAverages = np.full(
            (self.numBasalSegments, self.numberOfCells(), self.basalInputSize),
            initMovingAverages
        )
        self.apicalMovingAverages = np.full(
            (self.numApicalSegments, self.numberOfCells(), self.apicalInputSize),
            initMovingAverages
        )
        self.basalMovingAveragesBias = np.full((self.numBasalSegments, self.numberOfCells()), initMovingAverages)
        self.apicalMovingAveragesBias = np.full((self.numApicalSegments, self.numberOfCells()), initMovingAverages)
        self.basalMovingAverageInput = np.full(self.basalInputSize, initMovingAverages)
        self.apicalMovingAverageInput = np.full(self.apicalInputSize, initMovingAverages)


    def _addNewSegments(self, isBasal=True):
        input_size = self.basalInputSize if isBasal else self.apicalInputSize
        weight_matrix = self.basalWeights if isBasal else self.apicalWeights
        average_matrix = self.basalMovingAverages if isBasal else self.apicalMovingAverages
        bias_matrix = self.basalBias if isBasal else self.apicalBias
        bias_average = self.basalMovingAveragesBias if isBasal else self.apicalMovingAveragesBias
        active_segments = self.activeBasalSegments if isBasal else self.activeApicalSegments
        numSegments = self.numBasalSegments if isBasal else self.numApicalSegments

        if numSegments + 1 < self.maxSegmentsPerCell:
            weight_matrix = np.append(weight_matrix, np.zeros((1, self.numberOfCells(), input_size)), axis=0)
            average_matrix = np.append(average_matrix, np.zeros((1, self.numberOfCells(), input_size)), axis=0)
            bias_matrix = np.append(bias_matrix, np.zeros((1, self.numberOfCells())), axis=0)
            bias_average = np.append(bias_average, np.zeros((1, self.numberOfCells())), axis=0)
            active_segments = np.append(active_segments, np.zeros((1, self.numberOfCells())), axis=0)
            numSegments += 1

    def _updateConnectionData(self, isBasal=True):
        numSegments = self.numBasalSegments if isBasal else self.numApicalSegments
        inputValues = self.basalInput if isBasal else self.apicalInput
        movingAverageInput = self.basalMovingAverageInput if isBasal else self.apicalMovingAverageInput
        segments = self.activeBasalSegments if isBasal else self.activeApicalSegments
        inputSize = self.basalInputSize if isBasal else self.apicalInputSize
        movingAverage = self.basalMovingAverages if isBasal else self.apicalMovingAverages
        movingAverageBias = self.basalMovingAveragesBias if isBasal else self.apicalMovingAveragesBias

        # Updating moving average input activity
        noisy_input_vector = (1 - self.noise) * inputValues
        # Consider only active segments
        noisy_input_vector[noisy_input_vector > 0] += self.noise
        movingAverageInput += self.learningRate * (
                noisy_input_vector - movingAverageInput
        )

        # First update input values (includes learning rate)
        # Then use the probabilities of activity for movingAverage calculation
        # Instead of using 1 -> would lead to weight explosion, because we calculate weights based on MovingAverageInput
        inputProbabilities = inputValues
        inputProbabilities[inputValues.nonzero()] = movingAverageInput[inputValues.nonzero()]

        # Updating moving average weights to input
        noisy_connection_matrix = np.outer((1 - self.noise ** 2) * segments, inputProbabilities)
        # Consider only active segments
        noisy_connection_matrix[noisy_connection_matrix > 0] += self.noise ** 2
        noisy_connection_matrix = noisy_connection_matrix.reshape(numSegments, self.numberOfCells(), inputSize)
        movingAverage += self.learningRate * (
                noisy_connection_matrix - movingAverage
        )

        # Updating moving average bias of each segment
        noisy_activation_vector = (1 - self.noise) * segments
        # Consider only active segments
        noisy_activation_vector[noisy_activation_vector > 0] += self.noise
        movingAverageBias += self.learningRate * (
                noisy_activation_vector - movingAverageBias
        )

    def _afterUpdate(self):
        pass

    def _updateWeights(self, isBasal=True):
        movingAverages = self.basalMovingAverages if isBasal else self.apicalMovingAverages
        movingAveragesBias = self.basalMovingAveragesBias if isBasal else self.apicalMovingAveragesBias
        movingAveragesInput = self.basalMovingAverageInput if isBasal else self.apicalMovingAverageInput
        weights = movingAverages / np.outer(
            movingAveragesBias,
            movingAveragesInput
        ).reshape(movingAverages.shape)

        return weights

    def _updateBias(self, isBasal=True):
        movingAveragesBias = self.basalMovingAveragesBias if isBasal else self.apicalMovingAveragesBias
        bias = np.log(movingAveragesBias)

        return bias

