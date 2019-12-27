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


class SummingBayesianApicalTiebreakPairMemory(ApicalTiebreakBayesianTemporalMemoryBase):
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
            # Changed to float
            minThreshold=0.5,
            sampleSize=20,
            noise=0.01,  # lambda
            learningRate=0.1,  # alpha
            maxSegmentsPerCell=255,
            useApicalTiebreak=False,
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

        super(SummingBayesianApicalTiebreakPairMemory, self).__init__(
            columnCount=columnCount,
            basalInputSize=basalInputSize,
            apicalInputSize=apicalInputSize,
            cellsPerColumn=cellsPerColumn,
            minThreshold=minThreshold,
            sampleSize=sampleSize,
            noise=noise,
            learningRate=learningRate,
            maxSegmentsPerCell=maxSegmentsPerCell,
            useApicalTiebreak=useApicalTiebreak,
            seed=seed
        )

        self.basalConnectionCount = np.zeros((1, self.numberOfCells(), self.basalInputSize), dtype='int64')
        self.apicalConnectionCount = np.zeros((1, self.numberOfCells(), self.apicalInputSize), dtype='int64')
        self.basalSegmentActivationCount = np.zeros(self.numberOfCells(), dtype='int64')
        self.apicalSegmentActivationCount = np.zeros(self.numberOfCells(), dtype='int64')
        self.basalInputCount = np.zeros(self.basalInputSize, dtype='int64')
        self.apicalInputCount = np.zeros(self.apicalInputSize, dtype='int64')
        self.updateCounter = 0

    def _addNewSegments(self, isBasal=True):
        weights = self.basalWeights if isBasal else self.apicalWeights
        numSegments = self.numBasalSegments if isBasal else self.numApicalSegments
        connectionCount = self.basalConnectionCount if isBasal else self.apicalConnectionCount
        bias = self.basalBias if isBasal else self.apicalBias
        segmentActivationCount = self.basalSegmentActivationCount if isBasal else self.apicalSegmentActivationCount
        activeSegments = self.activeBasalSegments if isBasal else self.activeApicalSegments

        numberOfCells = weights.shape[1]
        inputSize = weights.shape[0]

        if numSegments + 1 < self.maxSegmentsPerCell:
            weights = np.append(weights, np.zeros((1, numberOfCells, inputSize)), axis=0)
            connectionCount = np.append(connectionCount, np.zeros((1, numberOfCells, inputSize)), axis=0)
            bias = np.append(bias, np.zeros((1, numberOfCells)), axis=0)
            segmentActivationCount = np.append(segmentActivationCount, np.zeros((1, numberOfCells)), axis=0)
            activeSegments = np.append(activeSegments, np.zeros((1, numberOfCells)), axis=0)
            numSegments += 1

        if isBasal:
            self._setBasalConnectionData(
                weight_matrix=weights,
                connection_count_matrix=connectionCount,
                bias_matrix=bias,
                activity_count_matrix=segmentActivationCount,
                active_segments=activeSegments,
                numSegments=numSegments
            )
        else:
            self._setApicalConnectionData(
                weight_matrix=weights,
                connection_count_matrix=connectionCount,
                bias_matrix=bias,
                activity_count_matrix=segmentActivationCount,
                active_segments=activeSegments,
                numSegments=numSegments
            )
    def _setBasalConnectionData(
            self,
            weight_matrix=None,
            connection_count_matrix=None,
            bias_matrix=None,
            activity_count_matrix=None,
            active_segments=None,
            numSegments=None,
            input_count_matrix=None
    ):
        if weight_matrix is not None:
            self.basalWeights = weight_matrix
        if connection_count_matrix is not None:
            self.basalConnectionCount = connection_count_matrix
        if bias_matrix is not None:
            self.basalBias = bias_matrix
        if activity_count_matrix is not None:
            self.basalSegmentActivationCount = activity_count_matrix
        if active_segments is not None:
            self.activeBasalSegments = active_segments
        if numSegments is not None:
            self.numBasalSegments = numSegments
        if input_count_matrix is not None:
            self.basalInputCount = input_count_matrix

    def _setApicalConnectionData(
            self,
            weight_matrix=None,
            connection_count_matrix=None,
            bias_matrix=None,
            activity_count_matrix=None,
            active_segments=None,
            numSegments=None,
            input_count_matrix=None
    ):
        if weight_matrix is not None:
            self.apicalWeights = weight_matrix
        if connection_count_matrix is not None:
            self.apicalConnectionCount = connection_count_matrix
        if bias_matrix is not None:
            self.apicalBias = bias_matrix
        if activity_count_matrix is not None:
            self.apicalSegmentActivationCount = activity_count_matrix
        if active_segments is not None:
            self.activeApicalSegments = active_segments
        if numSegments is not None:
            self.numApicalSegments = numSegments
        if input_count_matrix is not None:
            self.apicalInputCount = input_count_matrix


    def _updateConnectionData(self, isBasal=True):
        numSegments = self.numBasalSegments if isBasal else self.numApicalSegments
        segments = self.activeBasalSegments if isBasal else self.activeApicalSegments
        inputValues = self.basalInput if isBasal else self.apicalInput
        connectionCount = self.basalConnectionCount if isBasal else self.apicalConnectionCount
        segmentActivityCount = self.basalSegmentActivationCount if isBasal else self.apicalSegmentActivationCount
        inputCount = self.basalInputCount if isBasal else self.apicalInputCount

        # Updating moving average weights to input
        connection_matrix = np.outer(segments, inputValues)
        # Consider only active segments
        connection_matrix = connection_matrix.reshape(numSegments, self.numberOfCells(), connectionCount.shape[-1])
        connectionCount += connection_matrix.astype('int64')

        # Updating moving average bias of each segment
        segmentActivityCount += segments.reshape(-1).astype('int64')

        # Updating moving average input activity
        inputCount += inputValues.reshape(-1).astype('int64')

        if isBasal:
            self._setBasalConnectionData(
                connection_count_matrix=connectionCount,
                activity_count_matrix=segmentActivityCount,
                input_count_matrix=inputCount
            )
        else:
            self._setApicalConnectionData(
                connection_count_matrix=connectionCount,
                activity_count_matrix=segmentActivityCount,
                input_count_matrix=inputCount
            )

    def _afterUpdate(self):
        self.updateCounter += 1

    def _updateWeights(self, isBasal=True):
        connectionCount = self.basalConnectionCount if isBasal else self.apicalConnectionCount
        activationCount = self.basalSegmentActivationCount if isBasal else self.apicalSegmentActivationCount
        inputCount = self.basalInputCount if isBasal else self.apicalInputCount

        weights = (connectionCount.astype('float64') * self.updateCounter) / np.outer(
            activationCount,
            inputCount
        ).reshape(connectionCount.shape).astype('float64')
        weights[weights == 0] = 1. / float(self.updateCounter)
        weights[np.isnan(weights)] = 1.
        return weights

    def _updateBias(self, isBasal=True):
        activationCount = self.basalSegmentActivationCount if isBasal else self.apicalSegmentActivationCount
        bias = np.log(activationCount / float(self.updateCounter))

        return bias

