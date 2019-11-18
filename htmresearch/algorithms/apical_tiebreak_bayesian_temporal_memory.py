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

from htmresearch.support import numpy_helpers as np2
from nupic.bindings.math import Random, SparseMatrixConnections


class ApicalTiebreakBayesianTemporalMemory(object):
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

  def __init__(self,
               columnCount=2048,
               basalInputSize=0,  # Must be non-equal zero
               apicalInputSize=0,  # Must be non-equal zero
               cellsPerColumn=32,
               initialPermanence=0.21,
               # Changed to float
               minThreshold=0.5,
               sampleSize=20,
               noise=0.01,  # lambda
               learning_rate=0.1,  # alpha
               maxSegmentsPerCell=255,
               seed=42):
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

    self.columnCount = columnCount
    self.cellsPerColumn = cellsPerColumn
    self.initialPermanence = initialPermanence
    self.minThreshold = minThreshold
    self.maxSegmentsPerCell = maxSegmentsPerCell
    self.basalInputSize = basalInputSize
    self.apicalInputSize = apicalInputSize

    # We use a continuous weight matrix
    # Three dimensional to have weight from every input to every segment with mapping from segment to cell
    self.basalWeights = np.zeros(
      (self.maxSegmentsPerCell, self.columnCount*self.cellsPerColumn, self.basalInputSize))
    # Initialise weights to first segment randomly
    # TODO check whether this is necessary. Setting it zero should conceptually work. What is the outcome?
    self.basalWeights[0, :, :] = np.random.random(self.basalWeights[0, :, :].shape)
    self.apicalWeights = np.zeros(
      (self.maxSegmentsPerCell, self.columnCount*self.cellsPerColumn, self.apicalInputSize))
    # Initialise weights to first segment randomly
    self.apicalWeights[0, :, :] = np.random.random(self.apicalWeights[0, :, :].shape)
    self.basalBias = np.zeros((self.maxSegmentsPerCell, self.columnCount*self.cellsPerColumn))
    self.apicalBias = np.zeros((self.maxSegmentsPerCell, self.columnCount*self.cellsPerColumn))

    self.basalMovingAverages = np.zeros(
      (self.maxSegmentsPerCell, self.columnCount*self.cellsPerColumn, self.basalInputSize))
    self.apicalMovingAverages = np.zeros(
      (self.maxSegmentsPerCell, self.columnCount*self.cellsPerColumn, self.apicalInputSize))
    self.basalMovingAveragesBias = np.zeros((self.maxSegmentsPerCell, self.columnCount * self.cellsPerColumn))
    self.apicalMovingAveragesBias = np.zeros((self.maxSegmentsPerCell, self.columnCount * self.cellsPerColumn))
    self.basalMovingAverageInput = np.zeros(self.basalInputSize)
    self.apicalMovingAverageInput = np.zeros(self.apicalInputSize)

    self.noise = noise
    self.learningRate = learning_rate

    self.rng = Random(seed)
    # Changed already to float64
    self.predictedCells = np.zeros(self.numberOfCells(), dtype="float64")
    self.activeBasalSegments = np.zeros((self.maxSegmentsPerCell, self.numberOfCells()), dtype="float64")
    self.activeApicalSegments = np.zeros((self.maxSegmentsPerCell, self.numberOfCells()), dtype="float64")
    self.apicalInput = np.zeros(self.apicalInputSize, dtype="float64")
    self.basalInput = np.zeros(self.basalInputSize, dtype="float64")
    self.activeCells = np.zeros(self.numberOfCells(), dtype="float64")
    # TODO check where they are needed outside the module and eventually remove and change getter + setter
    # Still needs to be changed. However, not clear whether they are necessary
    self.winnerCells = np.empty(0, dtype="uint32")
    self.predictedActiveCells = np.empty(0, dtype="uint32")
    self.matchingBasalSegments = np.empty(0, dtype="uint32")
    self.matchingApicalSegments = np.empty(0, dtype="uint32")
    self.basalPotentialOverlaps = np.empty(0, dtype="int32")
    self.apicalPotentialOverlaps = np.empty(0, dtype="int32")

    # TODO what do these parameters do? Necessary?
    self.useApicalTiebreak=True
    self.useApicalModulationBasalThreshold=True

  def reset(self):
    """
    Clear all cell and segment activity.
    """
    # Changed already to float64
    self.predictedCells = np.zeros(self.numberOfCells(), dtype="float64")
    self.activeBasalSegments = np.zeros((self.maxSegmentsPerCell, self.numberOfCells()), dtype="float64")
    self.activeApicalSegments = np.zeros((self.maxSegmentsPerCell, self.numberOfCells()), dtype="float64")
    self.apicalInput = np.zeros(self.apicalInputSize, dtype="float64")
    self.basalInput = np.zeros(self.basalInputSize, dtype="float64")
    self.activeCells = np.zeros(self.numberOfCells(), dtype="float64")
    # Still needs to be changed. However, not clear whether they are necessary
    self.winnerCells = np.empty(0, dtype="uint32")
    self.predictedActiveCells = np.empty(0, dtype="uint32")
    self.matchingBasalSegments = np.empty(0, dtype="uint32")
    self.matchingApicalSegments = np.empty(0, dtype="uint32")
    self.basalPotentialOverlaps = np.empty(0, dtype="int32")
    self.apicalPotentialOverlaps = np.empty(0, dtype="int32")

  def depolarizeCells(self, basalInput, apicalInput, learn=None):
    """
    Calculate predictions.

    Depolarization means in this case to calculate the respective probability
    of cells detecting a particular pattern / participating in firing.

    @param basalInput (numpy array)
    List of active input bits for the basal dendrite segments

    @param apicalInput (numpy array)
    List of active input bits for the apical dendrite segments

    @param learn (bool)
    Whether learning is enabled. Some TM implementations may depolarize cells
    differently or do segment activity bookkeeping when learning is enabled.
    """
    activation_basal = self._calculateSegmentActivity(self.basalWeights, basalInput, self.basalBias, use_bias=True)
    activation_apical = self._calculateSegmentActivity(self.apicalWeights, apicalInput, self.apicalBias, use_bias=False)

    activation_basal = np.exp(activation_basal)
    activation_apical = np.exp(activation_apical)
    self.predictedCells = self._calculatePredictedValues(activation_basal, activation_apical)
    self.activeBasalSegments = activation_basal
    self.activeApicalSegments = activation_apical
    # Save basal and apical input values for learning
    self.basalInput = basalInput
    self.apicalInput = apicalInput

  def activateCells(self,
                    activeColumns,
                    basalReinforceCandidates=None, # Do not delete since most likely needed for interface communication
                    apicalReinforceCandidates=None,
                    basalGrowthCandidates=None,
                    apicalGrowthCandidates=None,
                    learn=True,
                    temporalLearningRate=None # Makes it possible to update moving averages while not learning
                                              # and to turn off updating moving averages
                    ):
    """
    Activate cells in the specified columns, using the result of the previous
    'depolarizeCells' as predictions. Then learn.

    @param activeColumns (numpy array)
    List of active columns

    @param basalReinforceCandidates (numpy array)
    List of bits that the active cells may reinforce basal synapses to.

    @param apicalReinforceCandidates (numpy array)
    List of bits that the active cells may reinforce apical synapses to.

    @param basalGrowthCandidates (numpy array)
    List of bits that the active cells may grow new basal synapses to.

    @param apicalGrowthCandidates (numpy array)
    List of bits that the active cells may grow new apical synapses to

    @param learn (bool)
    Whether to grow / reinforce / punish synapses
    """
    # List of active columns is expected to be an array with indices
    all_columns = np.arange(0, self.numberOfColumns())
    # Get all inactive columns and set their respective predicted cells to zero
    inactive_columns = np.setdiff1d(all_columns, activeColumns)
    active_cells_before_burst = self._reshapeCellsToColumnBased(self.predictedCells.copy())
    active_cells_before_burst[:, inactive_columns] = 0
    # find bursting columns
    bursting_columns = activeColumns[np.where(
      np.abs(active_cells_before_burst[:, activeColumns].sum(axis=0)
             ) < self.minThreshold)]  # Use this because of numerical issues. Sum over all cells should be zero
                                      # If there is one active cell, the sum over all cells >= thresold
                                      # TODO consider change if normalisation is changed

    # Calculate basal segment activity after bursting
    if bursting_columns.shape[0] > 0:
      self.activeBasalSegments = self._setMaxSegmentsAfterBursting(bursting_columns, self.activeBasalSegments)
      self.activeApicalSegments = self._setMaxSegmentsAfterBursting(bursting_columns, self.activeApicalSegments)
      # Sets bursting cells to 1 due to normalisation in the column
      self.predictedCells = self._calculatePredictedValues(self.activeBasalSegments, self.activeApicalSegments)

    # Reset active cells values
    self.activeCells = self._reshapeCellsToColumnBased(self.predictedCells.copy())
    self.activeCells[:, inactive_columns] = 0
    self.activeCells = self._reshapeCellsFromColumnBased(self.activeCells)

    # All non-active segments should be set to zero
    # All segments above the threshold could have activated the cell and hence should be included
    # in the learning process
    self.activeBasalSegments = self._setNonActiveSegments(self.activeBasalSegments, inactive_columns)
    self.activeApicalSegments = self._setNonActiveSegments(self.activeApicalSegments, inactive_columns)

    # Update moving averages
    # TODO updates moving averages of segments when they have sufficient activiation even if they have not been previously used -> Required?
    self.basalMovingAverages, self.basalMovingAveragesBias, self.basalMovingAverageInput = self._updateMovingAverage(
      self.activeBasalSegments,
      self.basalMovingAverages,
      self.basalMovingAveragesBias,
      self.basalMovingAverageInput,
      self.basalInput,
      self.basalInputSize,
      temporalLearningRate
    )
    self.apicalMovingAverages, self.apicalMovingAveragesBias, self.apicalMovingAverageInput = self._updateMovingAverage(
      self.activeApicalSegments,
      self.apicalMovingAverages,
      self.apicalMovingAveragesBias,
      self.apicalMovingAverageInput,
      self.apicalInput,
      self.apicalInputSize,
      temporalLearningRate
    )

    # Learn
    if learn:
      self.basalWeights, self.basalBias = self._learn(
        self.basalMovingAverages,
        self.basalMovingAveragesBias,
        self.basalMovingAverageInput
      )
      self.apicalWeights, self.apicalBias = self._learn(
        self.apicalMovingAverages,
        self.apicalMovingAveragesBias,
        self.apicalMovingAverageInput
      )

  def _calculatePredictedValues(self, activation_basal, activation_apical):
    predicted_cells = self._calculatePredictedCells(activation_basal, activation_apical)
    normalisation = self._reshapeCellsToColumnBased(predicted_cells).sum(axis=0)
    predicted_cells = self._reshapeCellsFromColumnBased(
      self._reshapeCellsToColumnBased(predicted_cells)
      / normalisation.reshape((1, normalisation.shape[0]))
    )
    predicted_cells[np.isnan(predicted_cells)] = 0
    # TODO when apply threshold -> experiments
    predicted_cells[predicted_cells < self.minThreshold] = 0.0
    return predicted_cells

  def _setMaxSegmentsAfterBursting(self, burstingColumns, segments):
    # Calculate bursting columns
    # This makes sure that only segments are learnt that are active
    # Reshaping active segments for easy access per column
    segments_column_based = self._reshapeSegmentsToColumnBased(segments)
    # TODO check update of values
    max_cells_ind = self._getMaxCellIndexPerColumn(segments_column_based[:, :, burstingColumns], self.cellsPerColumn)
    segments_to_update = segments_column_based[:, max_cells_ind, burstingColumns].argmin(axis=0)
    segments_column_based[segments_to_update, max_cells_ind, burstingColumns] = self.minThreshold
    return self._reshapeSegmetsFromColumnBased(segments_column_based)

  @staticmethod
  def _getMaxCellIndexPerColumn(segmentMatrix, cellsPerColumn):
    return np.row_stack(segmentMatrix).argmax(axis=0) % cellsPerColumn

  def _setNonActiveSegments(self, segments, inactiveColumns):
    segments[segments < self.minThreshold] = 0.0
    segments = self._reshapeSegmentsToColumnBased(segments)
    segments[:, :, inactiveColumns] = 0.0
    return self._reshapeSegmetsFromColumnBased(segments)

  def _reshapeSegmentsToColumnBased(self, segments, numOfColumns=None):
    if numOfColumns is None:
      numOfColumns = self.numberOfColumns()

    return segments.reshape(
      self.maxSegmentsPerCell,
      self.cellsPerColumn,
      numOfColumns
    )

  def _reshapeSegmetsFromColumnBased(self, segments, numOfColumns=None):
    if numOfColumns is None:
      numOfColumns = self.numberOfColumns()

    return segments.reshape(
      self.maxSegmentsPerCell,
      self.cellsPerColumn*numOfColumns
    )

  def _reshapeCellsToColumnBased(self, cells, numOfColumns=None):
    if numOfColumns is None:
      numOfColumns = self.numberOfColumns()

    return cells.reshape(
      self.cellsPerColumn,
      numOfColumns
    )

  def _reshapeCellsFromColumnBased(self, cells):
    return cells.reshape(-1)

  def _updateMovingAverage(
          self,
          segments,
          movingAverage,
          movingAverageBias,
          movingAverageInput,
          inputValues,
          inputSize,
          learningRate
  ):
    if learningRate is None:
      learningRate = self.learningRate

    # Updating moving average weights to input
    noisy_connection_matrix = np.outer((1 - self.noise**2) * segments, inputValues)
    # Consider only active segments
    noisy_connection_matrix[noisy_connection_matrix > 0] += self.noise**2
    noisy_connection_matrix = noisy_connection_matrix.reshape(self.maxSegmentsPerCell, self.numberOfCells(), inputSize)
    movingAverage += learningRate * (
            noisy_connection_matrix - movingAverage
    )

    # Updating moving average bias of each segment
    noisy_activation_vector = (1 - self.noise) * segments
    # Consider only active segments
    noisy_activation_vector[noisy_activation_vector > 0] += self.noise
    movingAverageBias += learningRate * (
            noisy_activation_vector - movingAverageBias
    )
    # Updating moving average input activity
    noisy_input_vector = (1 - self.noise) * inputValues
    # Consider only active segments
    noisy_input_vector[noisy_input_vector > 0] += self.noise
    movingAverageInput += learningRate * (
            noisy_input_vector - movingAverageInput
    )
    return movingAverage, movingAverageBias, movingAverageInput

  @staticmethod
  def _calculateSegmentActivity(weights, activeInput, bias, use_bias=True):
    # Runtime warnings for negative infinity can be ignored here
    activation = np.log(np.multiply(weights, activeInput)).sum(axis=2)
    return activation if not use_bias else activation + bias

  def _calculatePredictedCells(self, activeBasalSegments, activeApicalSegments):
    """
    Calculate the predicted cells, given the set of active segments.

    An active basal segment is enough to predict a cell.
    An active apical segment is *not* enough to predict a cell.

    When a cell has both types of segments active, other cells in its minicolumn
    must also have both types of segments to be considered predictive.

    @param activeBasalSegments (numpy array) two dimensional (segments x cells)
    @param activeApicalSegments (numpy array) two dimentsional (segments x cells)

    @return (numpy array)
    """
    max_cells = (activeBasalSegments + activeApicalSegments).max(axis=0)

    return max_cells

  @staticmethod
  def _learn(movingAverages, movingAveragesBias, movingAveragesInput):
    weights = movingAverages / np.outer(
      movingAveragesBias,
      movingAveragesInput
    ).reshape(movingAverages.shape)
    # set division by zero to zero since this represents unused segments
    weights[np.isnan(weights)] = 0

    # Unused segments are set to -inf. That is desired since we take the exp function for the activation
    # exp(-inf) = 0 what is the desired outcome
    bias = np.log(movingAveragesBias)
    return weights, bias

  @classmethod
  def _getCellsWithFewestSegments(cls, connections, rng, columns,
                                  cellsPerColumn):
    """
    For each column, get the cell that has the fewest total basal segments.
    Break ties randomly.

    @param connections (SparseMatrixConnections)
    @param rng (Random)
    @param columns (numpy array) Columns to check

    @return (numpy array)
    One cell for each of the provided columns
    """
    candidateCells = np2.getAllCellsInColumns(columns, cellsPerColumn)

    # Arrange the segment counts into one row per minicolumn.
    segmentCounts = np.reshape(connections.getSegmentCounts(candidateCells),
                               newshape=(len(columns),
                                         cellsPerColumn))

    # Filter to just the cells that are tied for fewest in their minicolumn.
    minSegmentCounts = np.amin(segmentCounts, axis=1, keepdims=True)
    candidateCells = candidateCells[np.flatnonzero(segmentCounts ==
                                                   minSegmentCounts)]

    # Filter to one cell per column, choosing randomly from the minimums.
    # To do the random choice, add a random offset to each index in-place, using
    # casting to floor the result.
    (_,
     onePerColumnFilter,
     numCandidatesInColumns) = np.unique(candidateCells / cellsPerColumn,
                                         return_index=True, return_counts=True)

    offsetPercents = np.empty(len(columns), dtype="float32")
    rng.initializeReal32Array(offsetPercents)

    np.add(onePerColumnFilter,
           offsetPercents*numCandidatesInColumns,
           out=onePerColumnFilter,
           casting="unsafe")

    return candidateCells[onePerColumnFilter]


  def getActiveCells(self):
    """
    @return (numpy array)
    Active cells
    """
    return self.activeCells


  def getPredictedActiveCells(self):
    """
    @return (numpy array)
    Active cells that were correctly predicted
    """
    return self.predictedActiveCells


  def getWinnerCells(self):
    """
    @return (numpy array)
    Cells that were selected for learning
    """
    return self.winnerCells


  def getActiveBasalSegments(self):
    """
    @return (numpy array)
    Active basal segments for this timestep
    """
    return self.activeBasalSegments


  def getActiveApicalSegments(self):
    """
    @return (numpy array)
    Matching basal segments for this timestep
    """
    return self.activeApicalSegments


  def numberOfColumns(self):
    """ Returns the number of columns in this layer.

    @return (int) Number of columns
    """
    return self.columnCount


  def numberOfCells(self):
    """
    Returns the number of cells in this layer.

    @return (int) Number of cells
    """
    return self.numberOfColumns() * self.cellsPerColumn


  def getCellsPerColumn(self):
    """
    Returns the number of cells per column.

    @return (int) The number of cells per column.
    """
    return self.cellsPerColumn

  def getMinThreshold(self):
    """
    Returns the min threshold.
    @return (int) The min threshold.
    """
    return self.minThreshold


  def setMinThreshold(self, minThreshold):
    """
    Sets the min threshold.
    @param minThreshold (int) min threshold.
    """
    self.minThreshold = minThreshold


  def getSampleSize(self):
    """
    Gets the sampleSize.
    @return (int)
    """
    return self.sampleSize


  def setSampleSize(self, sampleSize):
    """
    Sets the sampleSize.
    @param sampleSize (int)
    """
    self.sampleSize = sampleSize

  def getUseApicalTieBreak(self):
    """
    Get whether we actually use apical tie-break.
    @return (Bool) Whether apical tie-break is used.
    """
    return self.useApicalTiebreak


  def setUseApicalTiebreak(self, useApicalTiebreak):
    """
    Sets whether we actually use apical tie-break.
    @param useApicalTiebreak (Bool) Whether apical tie-break is used.
    """
    self.useApicalTiebreak = useApicalTiebreak


  def getUseApicalModulationBasalThreshold(self):
    """
    Get whether we actually use apical modulation of basal threshold.
    @return (Bool) Whether apical modulation is used.
    """
    return self.useApicalModulationBasalThreshold


  def setUseApicalModulationBasalThreshold(self, useApicalModulationBasalThreshold):
    """
    Sets whether we actually use apical modulation of basal threshold.
    @param useApicalModulationBasalThreshold (Bool) Whether apical modulation is used.
    """
    self.useApicalModulationBasalThreshold = useApicalModulationBasalThreshold


# TODO adapt class for the compute method which is used as a common interface
# class ApicalTiebreakPairMemory(ApicalTiebreakTemporalMemory):
#   """
#   Pair memory with apical tiebreak.
#   """
#
#   def compute(self,
#               activeColumns,
#               basalInput,
#               apicalInput=(),
#               basalGrowthCandidates=None,
#               apicalGrowthCandidates=None,
#               learn=True):
#     """
#     Perform one timestep. Use the basal and apical input to form a set of
#     predictions, then activate the specified columns, then learn.
#
#     @param activeColumns (numpy array)
#     List of active columns
#
#     @param basalInput (numpy array)
#     List of active input bits for the basal dendrite segments
#
#     @param apicalInput (numpy array)
#     List of active input bits for the apical dendrite segments
#
#     @param basalGrowthCandidates (numpy array or None)
#     List of bits that the active cells may grow new basal synapses to.
#     If None, the basalInput is assumed to be growth candidates.
#
#     @param apicalGrowthCandidates (numpy array or None)
#     List of bits that the active cells may grow new apical synapses to
#     If None, the apicalInput is assumed to be growth candidates.
#
#     @param learn (bool)
#     Whether to grow / reinforce / punish synapses
#     """
#     activeColumns = np.asarray(activeColumns)
#     basalInput = np.asarray(basalInput)
#     apicalInput = np.asarray(apicalInput)
#
#     if basalGrowthCandidates is None:
#       basalGrowthCandidates = basalInput
#     basalGrowthCandidates = np.asarray(basalGrowthCandidates)
#
#     if apicalGrowthCandidates is None:
#       apicalGrowthCandidates = apicalInput
#     apicalGrowthCandidates = np.asarray(apicalGrowthCandidates)
#
#     self.depolarizeCells(basalInput, apicalInput, learn)
#     self.activateCells(activeColumns, basalInput, apicalInput,
#                        basalGrowthCandidates, apicalGrowthCandidates, learn)
#
#
#   def getPredictedCells(self):
#     """
#     @return (numpy array)
#     Cells that were predicted for this timestep
#     """
#     return self.predictedCells
#
#
#   def getBasalPredictedCells(self):
#     """
#     @return (numpy array)
#     Cells with active basal segments
#     """
#     return np.unique(
#       self.basalConnections.mapSegmentsToCells(
#         self.activeBasalSegments))
#
#
#   def getApicalPredictedCells(self):
#     """
#     @return (numpy array)
#     Cells with active apical segments
#     """
#     return np.unique(
#       self.apicalConnections.mapSegmentsToCells(
#         self.activeApicalSegments))
#
#
#
#
# class ApicalTiebreakSequenceMemory(ApicalTiebreakTemporalMemory):
#   """
#   Sequence memory with apical tiebreak.
#   """
#
#   def __init__(self,
#                columnCount=2048,
#                apicalInputSize=0,
#                cellsPerColumn=32,
#                activationThreshold=13,
#                reducedBasalThreshold=13,
#                initialPermanence=0.21,
#                connectedPermanence=0.50,
#                minThreshold=10,
#                sampleSize=20,
#                permanenceIncrement=0.1,
#                permanenceDecrement=0.1,
#                basalPredictedSegmentDecrement=0.0,
#                apicalPredictedSegmentDecrement=0.0,
#                maxSynapsesPerSegment=-1,
#                seed=42):
#     params = {
#       "columnCount": columnCount,
#       "basalInputSize": columnCount * cellsPerColumn,
#       "apicalInputSize": apicalInputSize,
#       "cellsPerColumn": cellsPerColumn,
#       "activationThreshold": activationThreshold,
#       "reducedBasalThreshold": reducedBasalThreshold,
#       "initialPermanence": initialPermanence,
#       "connectedPermanence": connectedPermanence,
#       "minThreshold": minThreshold,
#       "sampleSize": sampleSize,
#       "permanenceIncrement": permanenceIncrement,
#       "permanenceDecrement": permanenceDecrement,
#       "basalPredictedSegmentDecrement": basalPredictedSegmentDecrement,
#       "apicalPredictedSegmentDecrement": apicalPredictedSegmentDecrement,
#       "maxSynapsesPerSegment": maxSynapsesPerSegment,
#       "seed": seed,
#     }
#
#     super(ApicalTiebreakSequenceMemory, self).__init__(**params)
#
#     self.prevApicalInput = np.empty(0, dtype="uint32")
#     self.prevApicalGrowthCandidates = np.empty(0, dtype="uint32")
#     self.prevPredictedCells = np.empty(0, dtype="uint32")
#
#
#   def reset(self):
#     """
#     Clear all cell and segment activity.
#     """
#     super(ApicalTiebreakSequenceMemory, self).reset()
#
#     self.prevApicalInput = np.empty(0, dtype="uint32")
#     self.prevApicalGrowthCandidates = np.empty(0, dtype="uint32")
#     self.prevPredictedCells = np.empty(0, dtype="uint32")
#
#
#   def compute(self,
#               activeColumns,
#               apicalInput=(),
#               apicalGrowthCandidates=None,
#               learn=True):
#     """
#     Perform one timestep. Activate the specified columns, using the predictions
#     from the previous timestep, then learn. Then form a new set of predictions
#     using the new active cells and the apicalInput.
#
#     @param activeColumns (numpy array)
#     List of active columns
#
#     @param apicalInput (numpy array)
#     List of active input bits for the apical dendrite segments
#
#     @param apicalGrowthCandidates (numpy array or None)
#     List of bits that the active cells may grow new apical synapses to
#     If None, the apicalInput is assumed to be growth candidates.
#
#     @param learn (bool)
#     Whether to grow / reinforce / punish synapses
#     """
#     activeColumns = np.asarray(activeColumns)
#     apicalInput = np.asarray(apicalInput)
#
#     if apicalGrowthCandidates is None:
#       apicalGrowthCandidates = apicalInput
#     apicalGrowthCandidates = np.asarray(apicalGrowthCandidates)
#
#     self.prevPredictedCells = self.predictedCells
#
#     self.activateCells(activeColumns, self.activeCells, self.prevApicalInput,
#                        self.winnerCells, self.prevApicalGrowthCandidates, learn)
#     self.depolarizeCells(self.activeCells, apicalInput, learn)
#
#     self.prevApicalInput = apicalInput.copy()
#     self.prevApicalGrowthCandidates = apicalGrowthCandidates.copy()
#
#
#   def getPredictedCells(self):
#     """
#     @return (numpy array)
#     The prediction from the previous timestep
#     """
#     return self.prevPredictedCells
#
#
#   def getNextPredictedCells(self):
#     """
#     @return (numpy array)
#     The prediction for the next timestep
#     """
#     return self.predictedCells
#
#
#   def getNextBasalPredictedCells(self):
#     """
#     @return (numpy array)
#     Cells with active basal segments
#     """
#     return np.unique(
#       self.basalConnections.mapSegmentsToCells(
#         self.activeBasalSegments))
#
#
#   def getNextApicalPredictedCells(self):
#     """
#     @return (numpy array)
#     Cells with active apical segments
#     """
#     return np.unique(
#       self.apicalConnections.mapSegmentsToCells(
#         self.activeApicalSegments))
