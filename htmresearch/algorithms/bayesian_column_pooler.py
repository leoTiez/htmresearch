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



class ColumnPooler(object):
  """
  This class constitutes a temporary implementation for a cross-column pooler.
  The implementation goal of this class is to prove basic properties before
  creating a cleaner implementation.
  """

  def __init__(self,
               inputWidth,
               lateralInputWidths=(),
               cellCount=4096,
               sdrSize=40,
               # onlineLearning = False,
               maxSdrSize = None,
               minSdrSize = None,

               # Proximal
               # synPermProximalInc=0.1,
               # synPermProximalDec=0.001,
               # initialProximalPermanence=0.6,
               sampleSizeProximal=20,
               # minThresholdProximal= 10,
               # connectedPermanenceProximal=0.50,
               # predictedInhibitionThreshold=20,

               # Distal
               # synPermDistalInc=0.1,
               # synPermDistalDec=0.001,
               # initialDistalPermanence=0.6,
               sampleSizeDistal=20,
               # activationThresholdDistal=13,
               # connectedPermanenceDistal=0.50,
               inertiaFactor=1.,

               # Bayesian
               noise=0.01,  # lambda
               learningRate=0.1,  # alpha
               activationThreshold=0.5, # probability such that a cell becomes active

               seed=42):
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
    if maxSdrSize is None:
      self.maxSdrSize = sdrSize
    else:
      self.maxSdrSize = maxSdrSize
    if minSdrSize is None:
      self.minSdrSize = sdrSize
    else:
      self.minSdrSize = minSdrSize
    self.sampleSizeProximal = sampleSizeProximal
    self.sampleSizeDistal = sampleSizeDistal
    self.inertiaFactor = inertiaFactor

    self.activeCells = np.zeros(self.cellCount, dtype="float64")
    self._random = Random(seed)
    self.useInertia=True

    # Bayesian parameters (weights, bias, moving averages)
    # Each row represents one segment on a cell, so each cell potentially has
    # 1 proximal segment and 1+len(lateralInputWidths) distal segments.

    # Weights 2D-Matrix - (1 segment per) cells x distalInput +1 (include bias)
    # Needs to be split up, because each segment only connects to the specified input
    self.distalWeights = list(np.zeros((self.cellCount, n+1)) for n in lateralInputWidths)
    self.internalDistalWeights = np.zeros((self.cellCount, self.cellCount+1))
    self.proximalWeights = np.zeros((self.cellCount, self.inputWidth+1))

    # Initialise weights to first segment randomly TODO check whether this is necessary. (commented out)
    # for d in self.distalWeights:
    #   d[:, :] = np.random.random(d[:, :].shape)
    # self.internalDistalWeights[:, :] = np.random.random(self.internalDistalWeights[:, :].shape)
    # self.proximalWeights[:, :] = np.random.random(self.proximalWeights[:, :].shape)

    self.distalMovingAverages = list(np.zeros((self.cellCount, n)) for n in lateralInputWidths)
    self.internalDistalMovingAverages = np.zeros((self.cellCount, self.cellCount))
    self.proximalMovingAverages = np.zeros((self.cellCount, self.inputWidth))

    # TODO: Separate bias per cell context needed?
    self.distalMovingAverageBias = list(np.zeros(self.cellCount) for n in lateralInputWidths)
    self.internalDistalMovingAverageBias = np.zeros(self.cellCount)
    self.proximalMovingAverageBias = np.zeros(self.cellCount)

    self.distalMovingAverageInput = list(np.zeros(n) for n in lateralInputWidths)
    self.internalDistalMovingAverageInput = np.zeros(self.cellCount)
    self.proximalMovingAverageInput = np.zeros(self.inputWidth)

    self.noise = noise
    self.learningRate = learningRate
    self.activationThreshold = activationThreshold


  def compute(self, feedforwardInput=(), lateralInputs=(),
              feedforwardGrowthCandidates=None, learn=True,
              predictedInput = None,):
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
      self._computeInferenceMode(feedforwardInput, lateralInputs)

    # learning step
    else:
      self._computeLearningMode(feedforwardInput, lateralInputs,
                                feedforwardGrowthCandidates)


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
    prevActiveCells = self.activeCells

    # If there are not enough previously active cells, then we are no longer on
    # a familiar object.  Either our representation decayed due to the passage
    # of time (i.e. we moved somewhere else) or we were mistaken.  Either way,
    # create a new SDR and learn on it.
    # This case is the only way different object representations are created.
    # enforce the active cells in the output layer
    if self.numberOfActiveCells() < self.minSdrSize:
      # Randomly activate sdrSize bits from an array with cellCount bits
      self.activeCells = _randomActivation(self.cellCount, self.sdrSize)

    # Update moving averages -> could be extended to be updated in inference mode too
    self.proximalMovingAverages, \
    self.proximalMovingAverageBias,\
    self.proximalMovingAverageInput = self._updateMovingAverage(
      self.activeCells,
      self.proximalMovingAverages,
      self.proximalMovingAverageBias,
      self.proximalMovingAverageInput,
      feedforwardInput,
      self.inputWidth,
      temporalLearningRate,
    )

    for i, lateralInput in enumerate(lateralInputs):
      self.distalMovingAverages[i], \
      self.distalMovingAverageBias[i],\
      self.distalMovingAverageInput[i] = self._updateMovingAverage(
        self.activeCells,
        self.distalMovingAverages[i],
        self.distalMovingAverageBias[i],
        self.distalMovingAverageInput[i],
        lateralInput,
        self.lateralInputWidths[i],
        temporalLearningRate,
      )

    self.internalDistalMovingAverages, \
    self.internalDistalMovingAverageBias, \
    self.internalDistalMovingAverageInput = self._updateMovingAverage(
      self.activeCells,
      self.internalDistalMovingAverages,
      self.internalDistalMovingAverageBias,
      self.internalDistalMovingAverageInput,
      self.activeCells,
      self.cellCount,
      temporalLearningRate,
    )



    # Learning
    # If we have a union of cells active, don't learn.  This primarily affects
    # online learning.
    if self.numberOfActiveCells() > self.maxSdrSize:
      return

    # Finally, now that we have decided which cells we should be learning on, do
    # the actual learning.
    if len(feedforwardInput) > 0:
      self.proximalWeights = self._learn(self.proximalMovingAverages, self.proximalMovingAverageBias, self.proximalMovingAverageInput)

      # External distal learning
      for i, lateralInput in enumerate(lateralInputs):
        self.distalWeights[i] = self._learn(self.distalMovingAverages[i], self.distalMovingAverageBias[i], self.distalMovingAverageInput[i])

      # Internal distal learning
      self.internalDistalWeights = self._learn(self.internalDistalMovingAverages, self.internalDistalMovingAverageBias, self.internalDistalMovingAverageInput)


  def _computeInferenceMode(self, feedforwardInput, lateralInputs):
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

    prevActiveCells = self.activeCells

    # # Calculate the feedforward supported cells
    # input = np.append(feedforwardInput, 1) # include bias term
    # feedForwardActivation = np.multiply(self.proximalWeights, input).sum(axis=1)
    #
    # # Calculate the number of lateral support on each cell
    # internalDistalActivation = np.multiply(self.internalDistalWeights, prevActiveCells).sum(axis=1)
    # lateralSupport = internalDistalActivation
    # for i, lateralInput in enumerate(lateralInputs):
    #   distalActivation = np.multiply(self.distalWeights[i], lateralInput).sum(axis=1)
    #   lateralSupport *= distalActivation  # TODO: SUM, MULTIPLY OR AVG?
    #
    #
    # support = feedForwardActivation * lateralSupport

    # Calculate the feedforward supported cells
    overlaps = self.proximalPermanences.rightVecSumAtNZGteThresholdSparse(
      feedforwardInput, self.connectedPermanenceProximal)
    feedforwardSupportedCells = np.where(
      overlaps >= self.minThresholdProximal)[0]

    # Calculate the number of active segments on each cell
    numActiveSegmentsByCell = np.zeros(self.cellCount, dtype="int")
    overlaps = self.internalDistalPermanences.rightVecSumAtNZGteThresholdSparse(
      prevActiveCells, self.connectedPermanenceDistal)
    numActiveSegmentsByCell[overlaps >= self.activationThresholdDistal] += 1
    for i, lateralInput in enumerate(lateralInputs):
      overlaps = self.distalPermanences[i].rightVecSumAtNZGteThresholdSparse(
        lateralInput, self.connectedPermanenceDistal)
      numActiveSegmentsByCell[overlaps >= self.activationThresholdDistal] += 1

    chosenCells = []

    # First, activate the FF-supported cells that have the highest number of
    # lateral active segments (as long as it's not 0)
    if len(feedforwardSupportedCells) == 0:
      pass
    else:
      numActiveSegsForFFSuppCells = numActiveSegmentsByCell[
        feedforwardSupportedCells]

      # This loop will select the FF-supported AND laterally-active cells, in
      # order of descending lateral activation, until we exceed the sdrSize
      # quorum - but will exclude cells with 0 lateral active segments.
      ttop = np.max(numActiveSegsForFFSuppCells)
      while ttop > 0 and len(chosenCells) < self.sdrSize:
        chosenCells = np.union1d(chosenCells,
                    feedforwardSupportedCells[numActiveSegsForFFSuppCells >= ttop])
        ttop -= 1

    # If we haven't filled the sdrSize quorum, add in inertial cells.
    if len(chosenCells) < self.sdrSize:
      if self.useInertia:
        prevCells = np.setdiff1d(prevActiveCells, chosenCells)
        inertialCap = int(len(prevCells) * self.inertiaFactor)
        if inertialCap > 0:
          numActiveSegsForPrevCells = numActiveSegmentsByCell[prevCells]
          # We sort the previously-active cells by number of active lateral
          # segments (this really helps).  We then activate them in order of
          # descending lateral activation.
          sortIndices = np.argsort(numActiveSegsForPrevCells)[::-1]
          prevCells = prevCells[sortIndices]
          numActiveSegsForPrevCells = numActiveSegsForPrevCells[sortIndices]

          # We use inertiaFactor to limit the number of previously-active cells
          # which can become active, forcing decay even if we are below quota.
          prevCells = prevCells[:inertialCap]
          numActiveSegsForPrevCells = numActiveSegsForPrevCells[:inertialCap]

          # Activate groups of previously active cells by order of their lateral
          # support until we either meet quota or run out of cells.
          ttop = np.max(numActiveSegsForPrevCells)
          while ttop >= 0 and len(chosenCells) < self.sdrSize:
            chosenCells = np.union1d(chosenCells,
                        prevCells[numActiveSegsForPrevCells >= ttop])
            ttop -= 1

    # If we haven't filled the sdrSize quorum, add cells that have feedforward
    # support and no lateral support.
    discrepancy = self.sdrSize - len(chosenCells)
    if discrepancy > 0:
      remFFcells = np.setdiff1d(feedforwardSupportedCells, chosenCells)

      # Inhibit cells proportionally to the number of cells that have already
      # been chosen. If ~0 have been chosen activate ~all of the feedforward
      # supported cells. If ~sdrSize have been chosen, activate very few of
      # the feedforward supported cells.

      # Use the discrepancy:sdrSize ratio to determine the number of cells to
      # activate.
      n = (len(remFFcells) * discrepancy) // self.sdrSize
      # Activate at least 'discrepancy' cells.
      n = max(n, discrepancy)
      # If there aren't 'n' available, activate all of the available cells.
      n = min(n, len(remFFcells))

      if len(remFFcells) > n:
        selected = _sample(self._random, remFFcells, n)
        chosenCells = np.append(chosenCells, selected)
      else:
        chosenCells = np.append(chosenCells, remFFcells)

    chosenCells.sort()
    self.activeCells = np.asarray(chosenCells, dtype="uint32")


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
    return np.where(self.activeCells >= self.activationThreshold)[0]

  def getActiveCells(self):
    """
    Returns the indices of the active cells.
    @return (list) Indices of active cells.
    """
    return self.getActiveCellsIndices()

  def numberOfConnectedProximalSynapses(self, cells=None):
    """
    Returns the number of proximal connected synapses on these cells.

    Parameters:
    ----------------------------
    @param  cells (iterable)
            Indices of the cells. If None return count for all cells.
    """
    if cells is None:
      cells = xrange(self.numberOfCells())

    return _countWhereGreaterEqualInRows(self.proximalPermanences, cells,
                                         self. connectedPermanenceProximal)


  def numberOfProximalSynapses(self, cells=None):
    """
    Returns the number of proximal synapses with permanence>0 on these cells.

    Parameters:
    ----------------------------
    @param  cells (iterable)
            Indices of the cells. If None return count for all cells.
    """
    if cells is None:
      cells = xrange(self.numberOfCells())

    n = 0
    for cell in cells:
      n += self.proximalPermanences.nNonZerosOnRow(cell)
    return n


  def numberOfDistalSegments(self, cells=None):
    """
    Returns the total number of distal segments for these cells.

    A segment "exists" if its row in the matrix has any permanence values > 0.

    Parameters:
    ----------------------------
    @param  cells (iterable)
            Indices of the cells
    """
    if cells is None:
      cells = xrange(self.numberOfCells())

    n = 0

    for cell in cells:
      if self.internalDistalPermanences.nNonZerosOnRow(cell) > 0:
        n += 1

      for permanences in self.distalPermanences:
        if permanences.nNonZerosOnRow(cell) > 0:
          n += 1

    return n


  def numberOfConnectedDistalSynapses(self, cells=None):
    """
    Returns the number of connected distal synapses on these cells.

    Parameters:
    ----------------------------
    @param  cells (iterable)
            Indices of the cells. If None return count for all cells.
    """
    if cells is None:
      cells = xrange(self.numberOfCells())

    n = _countWhereGreaterEqualInRows(self.internalDistalPermanences, cells,
                                      self.connectedPermanenceDistal)

    for permanences in self.distalPermanences:
      n += _countWhereGreaterEqualInRows(permanences, cells,
                                         self.connectedPermanenceDistal)

    return n


  def numberOfDistalSynapses(self, cells=None):
    """
    Returns the total number of distal synapses for these cells.

    Parameters:
    ----------------------------
    @param  cells (iterable)
            Indices of the cells
    """
    if cells is None:
      cells = xrange(self.numberOfCells())
    n = 0
    for cell in cells:
      n += self.internalDistalPermanences.nNonZerosOnRow(cell)

      for permanences in self.distalPermanences:
        n += permanences.nNonZerosOnRow(cell)
    return n


  def reset(self):
    """
    Reset internal states. When learning this signifies we are to learn a
    unique new object.
    """
    self.activeCells = np.zeros(self.cellCount, dtype="float64")

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


  def _updateMovingAverage(
          self,
          cells,
          movingAverage,
          movingAverageBias,
          movingAverageInput,
          inputValues,
          inputSize,
          learningRate,
  ):
    if learningRate is None:
      learningRate = self.learningRate

    # Updating moving average weights to input
    noisy_connection_matrix = np.outer((1 - self.noise**2) * cells, inputValues)
    # Consider only active segments
    noisy_connection_matrix[noisy_connection_matrix > 0] += self.noise**2
    noisy_connection_matrix = noisy_connection_matrix.reshape(self.cellCount, inputSize)
    movingAverage += learningRate * (
            noisy_connection_matrix - movingAverage
    )

    # Updating moving average bias activity
    noisy_input_vector = (1 - self.noise) * cells
    # Consider only active segments
    noisy_input_vector[noisy_input_vector > 0] += self.noise
    movingAverageBias += learningRate * (
            noisy_input_vector - movingAverageBias
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
  def _learn(movingAverages, movingAverageBias, movingAveragesInput):
    weights = movingAverages / np.outer(
      movingAverageBias,
      movingAveragesInput
    )
    # set division by zero to zero since this represents unused segments
    weights[np.isnan(weights)] = 0

    # Unused segments are set to -inf. That is desired since we take the exp function for the activation
    # exp(-inf) = 0 what is the desired outcome
    bias = np.log(movingAverageBias).reshape(1, movingAverageBias.shape[0])
    weights = np.concatenate((weights, bias.T), axis=1)

    return weights

#
# Functionality that could be added to the C code or bindings
#

def _randomActivation(n, k):
  # Activate k from n bits, randomly (by permutation)
  activeCells = np.append(np.ones(k), np.zeros(n-k))
  np.random.shuffle(activeCells)
  return activeCells



def _sample(rng, arr, k):
  """
  Equivalent to:

  random.sample(arr, k)

  except it uses our random number generator.
  """
  selected = np.empty(k, dtype="uint32")
  rng.sample(np.asarray(arr, dtype="uint32"),
             selected)
  return selected


def _countWhereGreaterEqualInRows(sparseMatrix, rows, threshold):
  """
  Like countWhereGreaterOrEqual, but for an arbitrary selection of rows, and
  without any column filtering.
  """
  return sum(sparseMatrix.countWhereGreaterOrEqual(row, row+1,
                                                   0, sparseMatrix.nCols(),
                                                   threshold)
             for row in rows)
