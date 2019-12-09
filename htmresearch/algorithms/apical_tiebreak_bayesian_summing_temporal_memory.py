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
from nupic.bindings.math import Random


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
               learningRate=0.1,  # alpha
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
    self.noise = noise
    self.learningRate = learningRate

    self.rng = Random(seed)

    self.numBasalSegments = 1
    self.numApicalSegments = 1
    # We use a continuous weight matrix
    # Three dimensional to have weight from every input to every segment with mapping from segment to cell
    self.basalWeights = np.zeros(
      (self.numBasalSegments, self.columnCount*self.cellsPerColumn, self.basalInputSize))

    self.apicalWeights = np.zeros(
      (self.numApicalSegments, self.columnCount*self.cellsPerColumn, self.apicalInputSize))
    self.basalBias = np.zeros((self.numBasalSegments, self.columnCount*self.cellsPerColumn))
    self.apicalBias = np.zeros((self.numApicalSegments, self.columnCount*self.cellsPerColumn))

    # Set to zero to use randomly initialized first weight value
    self.basalSegmentCount = np.zeros((self.columnCount, self.cellsPerColumn))
    self.apicalSegmentCount = np.zeros((self.columnCount, self.cellsPerColumn))

    self.basalConnectionCount = np.zeros((1, self.numberOfCells(), self.basalInputSize))
    self.apicalConnectionCount = np.zeros((1, self.numberOfCells(), self.apicalInputSize))
    self.basalSegmentActivationCount = np.zeros(self.numberOfCells())
    self.apicalSegmentActivationCount = np.zeros(self.numberOfCells())
    self.basalInputCount = np.zeros(self.basalInputSize)
    self.apicalInputCount = np.zeros(self.apicalInputSize)
    self.updateCounter = 0

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

  def depolarizeCells(self, basalInput, apicalInput):
    """
    Calculate predictions.

    Depolarization means in this case to calculate the respective probability
    of cells detecting a particular pattern / participating in firing.

    @param basalInput (numpy array)
    List of active input bits for the basal dendrite segments

    @param apicalInput (numpy array)
    List of active input bits for the apical dendrite segments
    """
    activation_basal = self._calculateSegmentActivity(self.basalWeights, basalInput, self.basalBias, self.noise, use_bias=True)
    activation_apical = self._calculateSegmentActivity(self.apicalWeights, apicalInput, self.apicalBias, self.noise, use_bias=False)

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
    predicted_cells_before_burst = self._reshapeCellsToColumnBased(self.predictedCells.copy())
    predicted_cells_before_burst[:, inactive_columns] = 0
    # find bursting columns
    bursting_columns = activeColumns[np.where(
      np.abs(predicted_cells_before_burst[:, activeColumns].sum(axis=0)
             ) < self.minThreshold)]  # Use this because of numerical issues. Sum over all cells should be zero
                                      # If there is one active cell, the sum over all cells >= thresold
                                      # TODO consider change if normalisation is changed

    # Calculate basal segment activity after bursting
    if bursting_columns.shape[0] > 0:
      if np.any(self.numBasalSegments == np.min(self.basalSegmentCount[activeColumns, :], axis=1)):
         self._addNewSegments(
           self.basalWeights,
           self.basalConnectionCount,
           self.basalBias,
           self.basalSegmentActivationCount,
           self.activeBasalSegments,
           self.numBasalSegments,
           self.maxSegmentsPerCell
         )
      if np.any(self.numApicalSegments == np.min(self.apicalSegmentCount[activeColumns, :], axis=1)):
        self._addNewSegments(
          self.apicalWeights,
          self.apicalConnectionCount,
          self.apicalBias,
          self.apicalSegmentActivationCount,
          self.activeApicalSegments,
          self.numApicalSegments,
          self.maxSegmentsPerCell
        )

      self.activeBasalSegments, self.basalSegmentCount = self._setMaxSegmentsAfterBursting(
        bursting_columns,
        self.activeBasalSegments,
        self.basalSegmentCount
      )
      self.activeApicalSegments, self.apicalSegmentCount = self._setMaxSegmentsAfterBursting(
        bursting_columns,
        self.activeApicalSegments,
        self.apicalSegmentCount,
        isApical=True
      )
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
    self.activeApicalSegments = self._setNonActiveSegments(self.activeApicalSegments, inactive_columns, isApical=True)

    # Learn
    if learn:
      # Update moving averages
      # TODO updates moving averages of segments when they have sufficient activiation even if they have not been previously used -> Required?
      # moved to the if statement to prevent update while learning TODO implement temporal learning rate as parameter such that it can be passed from outside

      self._updateCount(
        self.activeBasalSegments,
        self.basalConnectionCount,
        self.basalSegmentActivationCount,
        self.basalInputCount,
        self.basalInput,
        isApical=False
      )
      self._updateCount(
        self.activeApicalSegments,
        self.apicalConnectionCount,
        self.apicalSegmentActivationCount,
        self.apicalInputCount,
        self.apicalInput,
        isApical=True
      )
      self.updateCounter += 1

      self.basalWeights, self.basalBias = self._learn(
        self.basalConnectionCount,
        self.basalSegmentActivationCount,
        self.basalInputCount,
        self.updateCounter
      )
      self.apicalWeights, self.apicalBias = self._learn(
        self.apicalConnectionCount,
        self.apicalSegmentActivationCount,
        self.apicalInputCount,
        self.updateCounter
      )

  def _calculatePredictedValues(self, activation_basal, activation_apical):
    # TODO: We can not interpret it as probability anymore when adding apical+basal activation
    predicted_cells = self._calculatePredictedCells(activation_basal, activation_apical)
    # TODO: Update status report to normalize before activation
    predicted_cells[predicted_cells < self.minThreshold] = 0.0
    normalisation = self._reshapeCellsToColumnBased(predicted_cells).sum(axis=0)
    predicted_cells = self._reshapeCellsFromColumnBased(
      self._reshapeCellsToColumnBased(predicted_cells)
      / normalisation.reshape((1, normalisation.shape[0]))
    )
    predicted_cells[np.isnan(predicted_cells)] = 0
    # TODO when apply threshold -> experiments
    return predicted_cells

  @staticmethod
  def _addNewSegments(
          weights,
          connectionCount,
          bias,
          segmentActivationCount,
          activeSegments,
          numSegments,
          maxSegments
  ):
    numberOfCells = weights.shape[1]
    inputSize = weights.shape[0]
    if numSegments + 1 < maxSegments:
      weights = np.append(weights, np.zeros((1, numberOfCells, inputSize)), axis=0)
      connectionCount = np.append(connectionCount, np.zeros((1, numberOfCells, inputSize)), axis=0)
      bias = np.append(bias, np.zeros((1, numberOfCells)), axis=0)
      segmentActivationCount = np.append(segmentActivationCount, np.zeros((1, numberOfCells)), axis=0)
      activeSegments = np.append(activeSegments, np.zeros((1, numberOfCells)), axis=0)
      numSegments += 1

  def _setMaxSegmentsAfterBursting(self, burstingColumns, segments, segmentCount, isApical=False):
    # Calculate bursting columns
    # This makes sure that only segments are learnt that are active
    # Reshaping active segments for easy access per column
    segments_column_based = self._reshapeSegmentsToColumnBased(segments)
    # TODO check update of values
    min_segment_cells_ind = self._getCellWithMinSegments(segmentCount[burstingColumns, :])
    segments_column_based[
      segmentCount[burstingColumns, min_segment_cells_ind].astype('int32'),
      min_segment_cells_ind,
      burstingColumns
    ] = 1 # self.minThreshold
    update_ind = segmentCount[
      burstingColumns,
      min_segment_cells_ind
    ][segmentCount[burstingColumns, min_segment_cells_ind] < self.maxSegmentsPerCell].astype('int32')
    segmentCount[burstingColumns, min_segment_cells_ind][update_ind] += 1
    return self._reshapeSegmetsFromColumnBased(segments_column_based, isApical=isApical), segmentCount

  @staticmethod
  def _getMaxCellIndexPerColumn(segmentMatrix, cellsPerColumn):
    return np.row_stack(segmentMatrix).argmax(axis=0) % cellsPerColumn

  @staticmethod
  def _getCellWithMinSegments(segmentCount):
    return np.argmin(segmentCount, axis=1)

  def _setNonActiveSegments(self, segments, inactiveColumns, isApical=False):
    # Set all segments to zero, whose cells did not get activated
    segments[:, self.predictedCells < self.minThreshold] = 0.0
    segments = self._reshapeSegmentsToColumnBased(segments, isApical=isApical)
    segments[:, :, inactiveColumns] = 0.0
    return self._reshapeSegmetsFromColumnBased(segments, isApical=isApical)

  def _reshapeSegmentsToColumnBased(self, segments, numOfColumns=None, isApical=False):
    if numOfColumns is None:
      numOfColumns = self.numberOfColumns()

    numSegments = self.numApicalSegments if isApical else self.numBasalSegments
    return segments.reshape(
      numSegments,
      self.cellsPerColumn,
      numOfColumns
    )

  def _reshapeSegmetsFromColumnBased(self, segments, numOfColumns=None, isApical=False):
    if numOfColumns is None:
      numOfColumns = self.numberOfColumns()

    numSegments = self.numApicalSegments if isApical else self.numBasalSegments
    return segments.reshape(
      numSegments,
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

  def _updateCount(
          self,
          segments,
          connectionCount,
          segmentActivityCount,
          inputCount,
          inputValues,
          isApical=False
  ):
    numSegments = self.numApicalSegments if isApical else self.numBasalSegments
    # Updating moving average weights to input
    connection_matrix = np.outer(segments, inputValues)
    # Consider only active segments
    connection_matrix = connection_matrix.reshape(numSegments, self.numberOfCells(), connectionCount.shape[-1])
    connectionCount += connection_matrix

    # Updating moving average bias of each segment
    segmentActivityCount += segments.reshape(-1)

    # Updating moving average input activity
    inputCount += inputValues.reshape(-1)

  @staticmethod
  def _calculateSegmentActivity(weights, activeInput, bias, noise, use_bias=True):
    # Runtime warnings for negative infinity can be ignored here
    activeMask = activeInput > 0
    # Only sum over active input -> otherwise large negative sum due to sparse activity and 0 inputs with noise
    activation = np.log(np.multiply(weights[:, :, activeMask], activeInput[activeMask]) + noise)
    # Special case if active mask has no active inputs (e.g initialisation)
    # then activation becomes 0 and hence the exp of it 1
    activation = activation.sum(axis=2) if np.any(activeMask) else activation.sum(axis=2) + np.NINF
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
  def _learn(connectionCount, activationCount, inputCount, updateCounter):
    weights = connectionCount / np.outer(
      activationCount,
      inputCount
    ).reshape(connectionCount.shape)
    # set division by zero to zero since this represents unused segments
    weights[np.isnan(weights)] = 0

    # Unused segments are set to -inf. That is desired since we take the exp function for the activation
    # exp(-inf) = 0 what is the desired outcome
    bias = np.log(activationCount / float(updateCounter))
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

  def getActiveCellsValues(self):
    return self.activeCells

  def getActiveCellsIndices(self):
    return np.where(self.activeCells.reshape(-1) >= self.minThreshold)

  def getActiveCells(self):
    """
    @return (numpy array)
    Active cells
    """
    return self.getActiveCellsIndices()

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


# TODO adapt class for the compute method which is used as a common interface
class SummingBayesianApicalTiebreakPairMemory(ApicalTiebreakBayesianTemporalMemory):
  """
  Pair memory with apical tiebreak.
  """

  def compute(self,
              activeColumns,
              basalInput,
              apicalInput=(),
              learn=True):
    """
    Perform one timestep. Use the basal and apical input to form a set of
    predictions, then activate the specified columns, then learn.

    @param activeColumns (numpy array)
    List of active columns

    @param basalInput (numpy array)
    List of active input bits for the basal dendrite segments

    @param apicalInput (numpy array)
    List of active input bits for the apical dendrite segments

    @param basalGrowthCandidates (numpy array or None)
    List of bits that the active cells may grow new basal synapses to.
    If None, the basalInput is assumed to be growth candidates.

    @param apicalGrowthCandidates (numpy array or None)
    List of bits that the active cells may grow new apical synapses to
    If None, the apicalInput is assumed to be growth candidates.

    @param learn (bool)
    Whether to grow / reinforce / punish synapses
    """
    activeColumns = np.asarray(activeColumns)
    basalInput = np.asarray(basalInput)
    apicalInput = np.asarray(apicalInput)

    # Special case if indices of active values are passed
    if basalInput.shape[0] < self.basalInputSize:
      if basalInput.dtype == np.int64:
        basalInput_temp = np.zeros(self.basalInputSize)
        basalInput_temp[basalInput] = 1.0
        basalInput = basalInput_temp

    self.depolarizeCells(basalInput, apicalInput)
    self.activateCells(activeColumns, learn=learn, temporalLearningRate=None)


  def getPredictedCellsValues(self):
    return self.predictedCells

  def getPredictedCellsIndices(self):
    return np.where(self.predictedCells.reshape(-1) >= self.minThreshold)

  def getPredictedCells(self):
    """
    @return (numpy array)
    Cells that were predicted for this timestep
    """
    return self.getPredictedCellsIndices()
    # return self.predictedCells


  def getBasalPredictedCellValues(self):
    """
    @return (numpy array)
    Cells with active basal segments
    """
    return self.predictedCells

  def getBasalPredictedCellIndices(self):
    return self.getPredictedCellsIndices()

  def getApicalPredictedCellValues(self):
    """
    @return (numpy array)
    Cells with active apical segments
    """
    return self.predictedCells


  def getApicalPredictedCellIndices(self):
    return self.getPredictedCellsIndices()