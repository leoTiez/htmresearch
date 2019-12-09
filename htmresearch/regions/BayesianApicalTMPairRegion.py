# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
Region for Temporal Memory with various Bayesian apical implementations.
"""

import numpy as np

from nupic.bindings.regions.PyRegion import PyRegion



class BayesianApicalTMPairRegion(PyRegion):
  """
  Implements pair memory with the TM for the HTM network API. The temporal
  memory uses basal and apical dendrites.
  """

  @classmethod
  def getSpec(cls):
    """
    Return the Spec for ApicalTMPairRegion
    """

    spec = {
      "description": BayesianApicalTMPairRegion.__doc__,
      "singleNodeOnly": True,
      "inputs": {
        "activeColumns": {
          "description": ("An array of 0's and 1's representing the active "
                          "minicolumns, i.e. the input to the TemporalMemory"),
          "dataType": "Real32",
          "count": 0,
          "required": True,
          "regionLevel": True,
          "isDefaultInput": True,
          "requireSplitterMap": False
        },
        "resetIn": {
          "description": ("A boolean flag that indicates whether"
                          " or not the input vector received in this compute cycle"
                          " represents the first presentation in a"
                          " new temporal sequence."),
          "dataType": "Real32",
          "count": 1,
          "required": False,
          "regionLevel": True,
          "isDefaultInput": False,
          "requireSplitterMap": False
        },
        "basalInput": {
          "description": "An array of 0's and 1's representing basal input",
          "dataType": "Real32",
          "count": 0,
          "required": False,
          "regionLevel": True,
          "isDefaultInput": False,
          "requireSplitterMap": False
        },
        "basalGrowthCandidates": {
          "description": ("An array of 0's and 1's representing basal input " +
                          "that can be learned on new synapses on basal " +
                          "segments. If this input is a length-0 array, the " +
                          "whole basalInput is used."),
          "dataType": "Real32",
          "count": 0,
          "required": False,
          "regionLevel": True,
          "isDefaultInput": False,
          "requireSplitterMap": False
        },
        "apicalInput": {
          "description": "An array of 0's and 1's representing top down input."
          " The input will be provided to apical dendrites.",
          "dataType": "Real32",
          "count": 0,
          "required": False,
          "regionLevel": True,
          "isDefaultInput": False,
          "requireSplitterMap": False
        },
        "apicalGrowthCandidates": {
          "description": ("An array of 0's and 1's representing apical input " +
                          "that can be learned on new synapses on apical " +
                          "segments. If this input is a length-0 array, the " +
                          "whole apicalInput is used."),
          "dataType": "Real32",
          "count": 0,
          "required": False,
          "regionLevel": True,
          "isDefaultInput": False,
          "requireSplitterMap": False},
      },
      "outputs": {
        "predictedCells": {
          "description": ("A binary output containing a 1 for every "
                          "cell that was predicted for this timestep."),
          "dataType": "Real32",
          "count": 0,
          "regionLevel": True,
          "isDefaultOutput": False
        },

        "predictedActiveCells": {
          "description": ("A binary output containing a 1 for every "
                          "cell that transitioned from predicted to active."),
          "dataType": "Real32",
          "count": 0,
          "regionLevel": True,
          "isDefaultOutput": False
        },

        "activeCells": {
          "description": ("A binary output containing a 1 for every "
                          "cell that is currently active."),
          "dataType": "Real32",
          "count": 0,
          "regionLevel": True,
          "isDefaultOutput": True
        },

        "winnerCells": {
          "description": ("A binary output containing a 1 for every "
                          "'winner' cell in the TM."),
          "dataType": "Real32",
          "count": 0,
          "regionLevel": True,
          "isDefaultOutput": False
        },
      },

      "parameters": {

        # Input sizes (the network API doesn't provide these during initialize)
        "columnCount": {
          "description": ("The size of the 'activeColumns' input "
                          "(i.e. the number of columns)"),
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1,
          "constraints": ""
        },

        "basalInputWidth": {
          "description": "The size of the 'basalInput' input",
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1,
          "constraints": ""
        },

        "apicalInputWidth": {
          "description": "The size of the 'apicalInput' input",
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1,
          "constraints": ""
        },

        "learn": {
          "description": "True if the TM should learn.",
          "accessMode": "ReadWrite",
          "dataType": "Bool",
          "count": 1,
          "defaultValue": "true"
        },
        "cellsPerColumn": {
          "description": "Number of cells per column",
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1,
          "constraints": ""
        },

        "initialPermanence": {
          "description": "Initial permanence of a new synapse.",
          "accessMode": "Read",
          "dataType": "Real32",
          "count": 1,
          "constraints": ""
        },
        "connectedPermanence": {
          "description": ("If the permanence value for a synapse is greater "
                          "than this value, it is said to be connected."),
          "accessMode": "Read",
          "dataType": "Real32",
          "count": 1,
          "constraints": ""
        },
        "minThreshold": {
          "description": ("Minimal excitation of a segment required to be considered"
                          "to be active"),
          "accessMode": "Read",
          "dataType": "Real32",
          "count": 1,
          "constraints": ""
        },
        "sampleSize": {
          "description": ("The desired number of active synapses for an "
                          "active cell"),
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1
        },
        "maxSynapsesPerSegment": {
          "description": "The maximum number of synapses per segment",
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1
        },
        "maxSegmentsPerCell": {
          "description": "The maximum number of segments per cell",
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1
        },
        "noise": {
          "description": ("Noise added to the Bayesian learning procedure to avoid"
                          "taking the logarithm of 0"),
          "accessMode": "Read",
          "dataType": "Real32",
          "count": 1
        },
        "learningRate": {
          "description": "Learning rate of the Bayesian learning rule",
          "accessMode": "Read",
          "dataType": "Real32",
          "count": 1
        },
        "seed": {
          "description": "Seed for the random number generator.",
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1
        },
        "implementation": {
          "description": "Apical implementation",
          "accessMode": "Read",
          "dataType": "Byte",
          "count": 0,
          "constraints": "enum: BayesianApicalTiebreak",
          "defaultValue": "BayesianApicalTiebreak"
        },
      },
    }

    return spec


  def __init__(self,

               # Input sizes
               columnCount,
               basalInputWidth,
               apicalInputWidth=0,

               # TM params
               cellsPerColumn=32,
               initialPermanence=0.21,
               minThreshold=0.5,
               sampleSize=20,
               maxSegmentsPerCell=255,
               maxSynapsesPerSegment=255,
               seed=42,
               noise=0.01,  # lambda
               learningRate=0.1,  # alpha
               # Region params
               implementation="BayesianApicalTiebreak",
               learn=True,
               **kwargs):

    # Input sizes (the network API doesn't provide these during initialize)
    self.columnCount = columnCount
    self.basalInputWidth = basalInputWidth
    self.apicalInputWidth = apicalInputWidth

    # TM params
    self.cellsPerColumn = cellsPerColumn
    self.initialPermanence = initialPermanence
    self.minThreshold = minThreshold
    self.sampleSize = sampleSize
    self.maxSegmentsPerCell = maxSegmentsPerCell
    self.maxSynapsesPerSegment = maxSynapsesPerSegment
    self.seed = seed
    self.noise = noise
    self.learningRate = learningRate

    # Region params
    self.implementation = implementation
    self.learn = learn

    PyRegion.__init__(self, **kwargs)

    # TM instance
    self._tm = None


  def initialize(self):
    """
    Initialize the self._tm if not already initialized.
    """

    if self._tm is None:
      params = {
        "columnCount": self.columnCount,
        "basalInputSize": self.basalInputWidth,
        "apicalInputSize": self.apicalInputWidth,
        "cellsPerColumn": self.cellsPerColumn,
        "initialPermanence": self.initialPermanence,
        "minThreshold": self.minThreshold,
        "sampleSize": self.sampleSize,
        "maxSegmentsPerCell": self.maxSegmentsPerCell,
        "seed": self.seed,
        "noise": self.noise,
        "learningRate": self.learningRate,
      }

      if self.implementation == "BayesianApicalTiebreak":
        import htmresearch.algorithms.apical_tiebreak_bayesian_temporal_memory as btm
        cls = btm.BayesianApicalTiebreakPairMemory

      else:
        raise ValueError("Unrecognized implementation %s" % self.implementation)

      self._tm = cls(**params)

  def compute(self, inputs, outputs):
    """
    Run one iteration of TM's compute.
    """

    # If there's a reset, don't call compute. In some implementations, an empty
    # input might cause unwanted effects.
    if "resetIn" in inputs:
      assert len(inputs["resetIn"]) == 1
      if inputs["resetIn"][0] != 0:
        # send empty output
        self._tm.reset()
        outputs["activeCells"][:] = 0
        outputs["predictedActiveCells"][:] = 0
        outputs["winnerCells"][:] = 0
        return

    activeColumns = inputs["activeColumns"].nonzero()[0]

    if "basalInput" in inputs:
      basalInput = inputs["basalInput"].nonzero()[0]
    else:
      basalInput = np.empty(0, dtype="uint32")

    if "apicalInput" in inputs:
      apicalInput = np.asarray(inputs["apicalInput"], dtype="float64")
    else:
      apicalInput = np.empty(0, dtype="uint32")

    self._tm.compute(activeColumns, basalInput, apicalInput, self.learn)

    activeCellsIndices = self._tm.getActiveCellsIndices()
    predictedCellsIndices = self._tm.getPredictedCellsIndices()
    activeCellsValues = self._tm.getActiveCellsValues()
    predictedCellsValues = self._tm.getPredictedCellsValues()
    predictedActivatedIndices = np.intersect1d(activeCellsIndices, predictedCellsIndices)
    outputs["activeCells"][:] = activeCellsValues
    outputs["predictedCells"][:] = predictedCellsValues
    outputs["predictedCells"][predictedCellsIndices] = predictedCellsValues[predictedCellsIndices]
    outputs["predictedActiveCells"][:] = 0
    outputs["predictedActiveCells"][predictedActivatedIndices] = activeCellsValues[predictedActivatedIndices]
    # (outputs["activeCells"] * outputs["predictedCells"])
    # Treat winner and active cells the same way
    outputs["winnerCells"][:] = 0
    outputs["winnerCells"][activeCellsIndices] = activeCellsValues[activeCellsIndices]


  def getParameter(self, parameterName, index=-1):
    """
      Get the value of a NodeSpec parameter. Most parameters are handled
      automatically by PyRegion's parameter get mechanism. The ones that need
      special treatment are explicitly handled here.
    """
    return PyRegion.getParameter(self, parameterName, index)


  def setParameter(self, parameterName, index, parameterValue):
    """
    Set the value of a Spec parameter. Most parameters are handled
    automatically by PyRegion's parameter set mechanism. The ones that need
    special treatment are explicitly handled here.
    """
    if hasattr(self, parameterName):
      setattr(self, parameterName, parameterValue)
    else:
      raise Exception("Unknown parameter: " + parameterName)


  def getOutputElementCount(self, name):
    """
    Return the number of elements for the given output.
    """
    if name in ["activeCells", "predictedCells", "predictedActiveCells",
                "winnerCells"]:
      return self.cellsPerColumn * self.columnCount
    else:
      raise Exception("Invalid output name specified: %s" % name)
