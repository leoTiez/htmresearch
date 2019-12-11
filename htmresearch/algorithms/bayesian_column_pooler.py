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



class BayesianColumnPooler(object):
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
               maxSdrSize = None,
               minSdrSize = None,

               # Proximal
               sampleSizeProximal=20,

               # Distal
               sampleSizeDistal=20,
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

    # Weights 2D-Matrix - (1 segment per) cells x distalInput
    # Needs to be split up, because each segment only connects to the specified input
    self.distalWeights = list(np.zeros((self.cellCount, n)) for n in lateralInputWidths)
    self.internalDistalWeights = np.zeros((self.cellCount, self.cellCount))
    self.proximalWeights = np.zeros((self.cellCount, self.inputWidth))

    self.distalBias = list(np.zeros(self.cellCount) for n in lateralInputWidths)
    self.internalDistalBias = np.zeros(self.cellCount)
    self.proximalBias = np.zeros(self.cellCount)

    # Initialise weights to first segment randomly TODO check whether this is necessary. (commented out)
    # for d in self.distalWeights:
    #   d[:, :] = np.random.random(d[:, :].shape)
    # self.internalDistalWeights[:, :] = np.random.random(self.internalDistalWeights[:, :].shape)
    # self.proximalWeights[:, :] = np.random.random(self.proximalWeights[:, :].shape)

    self.distalMovingAverages = list(np.zeros((self.cellCount, n)) for n in lateralInputWidths)
    self.internalDistalMovingAverages = np.zeros((self.cellCount, self.cellCount))
    self.proximalMovingAverages = np.zeros((self.cellCount, self.inputWidth))

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
    prevActiveCells = self.activeCells

    # If there are not enough previously active cells, then we are no longer on
    # a familiar object.  Either our representation decayed due to the passage
    # of time (i.e. we moved somewhere else) or we were mistaken.  Either way,
    # create a new SDR and learn on it.
    # This case is the only way different object representations are created.
    # enforce the active cells in the output layer
    if self.numberOfActiveCells() < self.minSdrSize:
      # Randomly activate sdrSize bits from an array with cellCount bits
      self.activeCells = _randomActivation(self.cellCount, self.sdrSize, self._random)

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
      prevActiveCells,
      self.cellCount,
      temporalLearningRate,
    )

    # Learning
    # If we have a union of cells active, don't learn.  This primarily affects
    # online learning.
    # if self.numberOfActiveCells() > self.maxSdrSize:
    #  return

    # Finally, now that we have decided which cells we should be learning on, do
    # the actual learning.
    if len(feedforwardInput) > 0: # Length of feed-forward input should always be equal to self.inputWidth
                                  # Thus this is a tautology
      self.proximalWeights, \
      self.proximalBias = self._learn(self.proximalMovingAverages, self.proximalMovingAverageBias, self.proximalMovingAverageInput)

      # External distal learning
      for i, lateralInput in enumerate(lateralInputs):
        self.distalWeights[i], \
        self.distalBias[i] = self._learn(self.distalMovingAverages[i], self.distalMovingAverageBias[i], self.distalMovingAverageInput[i])

      # Internal distal learning
      self.internalDistalWeights, \
      self.internalDistalBias = self._learn(
        self.internalDistalMovingAverages,
        self.internalDistalMovingAverageBias,
        self.internalDistalMovingAverageBias  # Having recurrent connections, thus weights between each other are learnt
      )


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
    prevActiveCellIndices = np.where(prevActiveCells >= self.activationThreshold)[0]

    # Calculate the feedforward supported cells
    feedForwardActivation = self._activation(self.proximalWeights, feedforwardInput, self.proximalBias, self.noise)
    feedforwardSupportedCells = np.where(feedForwardActivation >= self.activationThreshold)[0]

    # Calculate the number of active distal segments (internal and lateral) on each cell
    numActiveSegmentsByCell = np.zeros(self.cellCount, dtype="int")

    # Internal Distal
    internalDistalActivation = self._activation(self.internalDistalWeights, prevActiveCells, self.internalDistalBias, self.noise)
    numActiveSegmentsByCell[internalDistalActivation >= self.activationThreshold] += 1

    # Lateral connections to other cortical columns
    for i, lateralInput in enumerate(lateralInputs):
      distalActivation = self._activation(self.distalWeights[i], lateralInput, self.distalBias[i], self.noise)
      numActiveSegmentsByCell[distalActivation >= self.activationThreshold] += 1

    chosenCells = np.array([], dtype="int")

    # First, activate the FF-supported cells that have the highest number of
    # lateral active segments (as long as it's not 0)
    if len(feedforwardSupportedCells) == 0:
      pass
    else:
      numActiveSegsForFFSuppCells = numActiveSegmentsByCell[feedforwardSupportedCells]

      # This loop will select the FF-supported AND laterally-active cells, in
      # order of descending lateral activation, until we exceed the sdrSize
      # quorum - but will exclude cells with 0 lateral active segments.
      ttop = np.max(numActiveSegsForFFSuppCells)
      while ttop > 0 and len(chosenCells) < self.sdrSize:
        supported = feedforwardSupportedCells[numActiveSegsForFFSuppCells >= ttop]
        chosenCells = np.union1d(chosenCells, supported)
        ttop -= 1

    # If we haven't filled the sdrSize quorum, add in inertial cells.
    if len(chosenCells) < self.sdrSize:
      if self.useInertia:
        prevCells = np.setdiff1d(prevActiveCellIndices, chosenCells)
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
            chosenCells = np.union1d(chosenCells, prevCells[numActiveSegsForPrevCells >= ttop])
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
    self.activeCells = np.zeros(self.cellCount, dtype="float64")
    # TODO: Inference mode only sets them to 1, but we could calculate a value from the proximal/distal support
    self.activeCells[chosenCells] = 1 # feedForwardActivation[chosenCells] + distal/lateral mult/sum?

    print "Column pooler inference input/distalSupport/output"
    print feedforwardInput.nonzero()
    print numActiveSegmentsByCell[numActiveSegmentsByCell > 0]
    print self.activeCells.nonzero()

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

  def getActiveCellValues(self):
    return self.activeCells

  def getActiveCells(self):
    """
    Returns the indices of the active cells.
    @return (list) Indices of active cells.
    """
    return self.getActiveCellsIndices()

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

    # Update only values for the active cells indices. Since we have a sparse representation it is likely that every
    # neuron is only active for one particular pattern or few ones. Hence, the more update steps are taken the less the
    # neuron becomes activated and thus the weights are more decreased. This leads to a forgetting mechanism that is
    # not desired in a sparse representation
    active_cells_indices = self.getActiveCellsIndices()
    # Updating moving average weights to input
    noisy_connection_matrix = np.outer((1 - self.noise**2) * cells, inputValues)
    # Consider only active segments
    noisy_connection_matrix[noisy_connection_matrix > 0] += self.noise**2
    noisy_connection_matrix = noisy_connection_matrix.reshape(self.cellCount, inputSize)
    movingAverage[active_cells_indices, :] += learningRate * (
            noisy_connection_matrix[active_cells_indices, :] - movingAverage[active_cells_indices, :]
    )

    # Updating moving average bias activity
    noisy_input_vector = (1 - self.noise) * cells
    # Consider only active segments
    noisy_input_vector[noisy_input_vector > 0] += self.noise
    movingAverageBias[active_cells_indices] += learningRate * (
            noisy_input_vector[active_cells_indices] - movingAverageBias[active_cells_indices]
    )

    # Updating moving average input activity
    input_mask = inputValues > 0
    noisy_input_vector = (1 - self.noise) * inputValues
    # Consider only active segments
    noisy_input_vector[noisy_input_vector > 0] += self.noise
    movingAverageInput[input_mask] += learningRate * (
            noisy_input_vector[input_mask] - movingAverageInput[input_mask]
    )
    return movingAverage, movingAverageBias, movingAverageInput

  @staticmethod
  def _activation(weights, input, bias, noise, use_bias=True):
    # Runtime warnings for negative infinity can be ignored here
    activeMask = input > 0
    # Only sum over active input -> otherwise large negative sum due to sparse activity and 0 inputs with noise
    activation = np.log(np.multiply(weights[:, activeMask], input[activeMask]) + noise)
    # Special case if active mask has no active inputs (e.g initialisation)
    # then activation becomes 0 and hence the exp of it 1
    activation = activation.sum(axis=1) if np.any(activeMask) else activation.sum(axis=1) + np.NINF
    activation = activation if not use_bias else activation + bias
    return np.exp(activation)

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
    bias = np.log(movingAverageBias)

    return weights, bias

#
# Functionality that could be added to the C code or bindings
#

def _randomActivation(n, k, randomizer):
  # Activate k from n bits, randomly (using custom randomizer for reproducibility)
  activeCells = np.zeros(n, dtype="float64")
  indices = _sampleRange(randomizer, 0, n, step=1, k=k)
  activeCells[indices] = 1
  return activeCells

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
