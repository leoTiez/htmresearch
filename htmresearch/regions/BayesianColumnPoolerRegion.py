# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

import copy
import numpy as np
import inspect

from nupic.bindings.regions.PyRegion import PyRegion


def getConstructorArguments():
  """
  Return constructor argument associated with BayesianColumnPooler.
  @return defaults (list)   a list of args and default values for each argument
  """
  argspec = inspect.getargspec(BayesianColumnPooler.__init__)
  return argspec.args[1:], argspec.defaults


class BayesianColumnPoolerRegion(PyRegion):
  """
  The BayesianColumnPoolerRegion implements an L2 layer within a single cortical column / cortical
  module.

  The layer supports feed forward (proximal) and lateral inputs.
  """

  @classmethod
  def getSpec(cls):
    """
    Return the Spec for BayesianColumnPoolerRegion.

    The parameters collection is constructed based on the parameters specified
    by the various components (tmSpec and otherSpec)
    """
    spec = dict(
      description=BayesianColumnPoolerRegion.__doc__,
      singleNodeOnly=True,
      inputs=dict(
        feedforwardInput=dict(
          description="The primary feed-forward input to the layer, this is a"
                      " binary array containing 0's and 1's",
          dataType="Real32",
          count=0,
          required=True,
          regionLevel=True,
          isDefaultInput=True,
          requireSplitterMap=False),

        feedforwardGrowthCandidates=dict(
          description=("An array of 0's and 1's representing feedforward input " +
                       "that can be learned on new proximal synapses. If this " +
                       "input isn't provided, the whole feedforwardInput is "
                       "used."),
          dataType="Real32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        predictedInput=dict(
          description=("An array of 0s and 1s representing input cells that " +
                       "are predicted to become active in the next time step. " +
                       "If this input is not provided, some features related " +
                       "to online learning may not function properly."),
          dataType="Real32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        lateralInput=dict(
          description="Lateral binary input into this column, presumably from"
                      " other neighboring columns.",
          dataType="Real32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        resetIn=dict(
          description="A boolean flag that indicates whether"
                      " or not the input vector received in this compute cycle"
                      " represents the first presentation in a"
                      " new temporal sequence.",
          dataType='Real32',
          count=1,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

      ),
      outputs=dict(
        feedForwardOutput=dict(
          description="The default output of BayesianColumnPoolerRegion. By default this"
                      " outputs the active cells. You can change this "
                      " dynamically using the defaultOutputType parameter.",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=True),

        activeCells=dict(
          description="A binary output containing a 1 for every"
                      " cell that is currently active.",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=False),

      ),
      parameters=dict(
        learningMode=dict(
          description="Whether the node is learning (default True).",
          accessMode="ReadWrite",
          dataType="Bool",
          count=1,
          defaultValue="true"),
        onlineLearning=dict(
          description="Whether to use onlineLearning or not (default False).",
          accessMode="ReadWrite",
          dataType="Bool",
          count=1,
          defaultValue="false"),
        learningTolerance=dict(
          description="How much variation in SDR size to accept when learning. "
                      "Only has an effect if online learning is enabled. "
                      "Should be at most 1 - inertiaFactor.",
          accessMode="ReadWrite",
          dataType="Real32",
          count=1,
          defaultValue="false"),
        cellCount=dict(
          description="Number of cells in this layer",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        inputWidth=dict(
          description='Number of inputs to the layer.',
          accessMode='Read',
          dataType='UInt32',
          count=1,
          constraints=''),
        numOtherCorticalColumns=dict(
          description="The number of lateral inputs that this L2 will receive. "
                      "This region assumes that every lateral input is of size "
                      "'cellCount'.",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        sdrSize=dict(
          description="The number of active cells invoked per object",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        maxSdrSize=dict(
          description="The largest number of active cells in an SDR tolerated "
                      "during learning. Stops learning when unions are active.",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        minSdrSize=dict(
          description="The smallest number of active cells in an SDR tolerated "
                      "during learning.  Stops learning when possibly on a "
                      "different object or sequence",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),

        #
        # Proximal
        #
        synPermProximalInc=dict(
          description="Amount by which permanences of proximal synapses are "
                      "incremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        synPermProximalDec=dict(
          description="Amount by which permanences of proximal synapses are "
                      "decremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        initialProximalPermanence=dict(
          description="Initial permanence of a new proximal synapse.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        sampleSizeProximal=dict(
          description="The desired number of active synapses for an active cell",
          accessMode="Read",
          dataType="Int32",
          count=1),
        minThresholdProximal=dict(
          description="If the number of synapses active on a proximal segment "
                      "is at least this threshold, it is considered as a "
                      "candidate active cell",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        connectedPermanenceProximal=dict(
          description="If the permanence value for a synapse is greater "
                      "than this value, it is said to be connected.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        predictedInhibitionThreshold=dict(
          description="How many predicted cells are required to cause "
                      "inhibition in the pooler.  Only has an effect if online "
                      "learning is enabled.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),

        #
        # Distal
        #
        synPermDistalInc=dict(
          description="Amount by which permanences of synapses are "
                      "incremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        synPermDistalDec=dict(
          description="Amount by which permanences of synapses are "
                      "decremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        initialDistalPermanence=dict(
          description="Initial permanence of a new synapse.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        sampleSizeDistal=dict(
          description="The desired number of active synapses for an active "
                      "segment.",
          accessMode="Read",
          dataType="Int32",
          count=1),
        activationThresholdDistal=dict(
          description="If the number of synapses active on a distal segment is "
                      "at least this threshold, the segment is considered "
                      "active",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        connectedPermanenceDistal=dict(
          description="If the permanence value for a synapse is greater "
                      "than this value, it is said to be connected.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        inertiaFactor=dict(
          description="Controls the proportion of previously active cells that "
                      "remain active through inertia in the next timestep (in  "
                      "the absence of inhibition).",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),

        #
        # Bayesian
        #
        noise=dict(
          description="Noise added to the Bayesian learning procedure to avoid"
                      "taking the logarithm of 0",
          accessMode="Read",
          dataType="Real32",
          count=1),
        learningRate=dict(
          description="Learning rate of the Bayesian learning rule",
          accessMode="Read",
          dataType="Real32",
          count=1),
        activationThreshold=dict(
          description="Activation threshold for the output of the region. (For proximal & distal activation)",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),

        forgetting=dict(
          description="Forgetting for non activated values",
          accessMode="Read",
          dataType="Real32",
          defaultValue=0.1,
          count=1
        ),

        initMovingAverages=dict(
          description="Init values for moving averages",
          accessMode="Read",
          dataType="Real32",
          defaultValue=0.0,
          count=1
        ),
        useSupport=dict(
          description="Flag whether or not to use support",
          accessMode="Read",
          dataType="Bool",
          defaultValue=False,
          count=1
        ),
        avoidWeightExplosion=dict(
          description="Flag whether or not to suppress weight explosion",
          accessMode="Read",
          dataType="Bool",
          defaultValue=True,
          count=1
        ),
        resetProximalCounter=dict(
          description="Flag whether or not to reset proximal counter when learning new object",
          accessMode="Read",
          dataType="Bool",
          defaultValue=False,
          count=1
        ),
        useProximalProbabilities=dict(
          description="Flag whether or not to use proximal ff activities as activation value",
          accessMode="Read",
          dataType="Bool",
          defaultValue=True,
          count=1
        ),
        implementation = dict(
          description="Bayesian implementation",
          accessMode="Read",
          dataType="Byte",
          count= 0,
          constraints="enum: Bayesian, SummingBayesian",
          defaultValue="Bayesian"
        ),

        seed=dict(
          description="Seed for the random number generator.",
          accessMode="Read",
          dataType="UInt32",
          count=1),
        defaultOutputType=dict(
          description="Controls what type of cell output is placed into"
                      " the default output 'feedForwardOutput'",
          accessMode="ReadWrite",
          dataType="Byte",
          count=0,
          constraints="enum: active,predicted,predictedActiveCells",
          defaultValue="active"),
      ),
      commands=dict(
        reset=dict(description="Explicitly reset TM states now."),
      )
    )

    return spec


  def __init__(self,
               cellCount=4096,
               inputWidth=16384,
               numOtherCorticalColumns=0,
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
               activationThreshold=0.5,  # probability such that a cell becomes active
               forgetting=0.1,
               initMovingAverages=0.0,
               useSupport=False,
               avoidWeightExplosion=True,
               resetProximalCounter=False,
               useProximalProbabilities=True,
               implementation="Bayesian",
               seed=42,
               defaultOutputType = "active",
               **kwargs):

    # Used to derive Column Pooler params
    self.numOtherCorticalColumns = numOtherCorticalColumns

    # Column Pooler params
    self.inputWidth = inputWidth
    self.cellCount = cellCount
    self.sdrSize = sdrSize
    self.maxSdrSize = maxSdrSize
    self.minSdrSize = minSdrSize
    self.sampleSizeProximal = sampleSizeProximal
    self.sampleSizeDistal = sampleSizeDistal
    self.inertiaFactor = inertiaFactor
    self.seed = seed

    self.activationThreshold = activationThreshold
    self.learningRate = learningRate
    self.noise = noise

    self.implementation = implementation

    self.forgetting = forgetting
    self.initMovingAverages = initMovingAverages
    self.useSupport = useSupport
    self.avoidWeightExplosion = avoidWeightExplosion
    self.resetProximalCounter = resetProximalCounter
    self.useProximalProbabilities = useProximalProbabilities
    # Region params
    self.learningMode = True
    self.defaultOutputType = defaultOutputType

    self._pooler = None

    PyRegion.__init__(self, **kwargs)


  def initialize(self):
    """
    Initialize the internal objects.
    """
    if self._pooler is None:
      params = {
        "inputWidth": self.inputWidth,
        "lateralInputWidths": [self.cellCount] * self.numOtherCorticalColumns,
        "cellCount": self.cellCount,
        "sdrSize": self.sdrSize,
        "maxSdrSize": self.maxSdrSize,
        "minSdrSize": self.minSdrSize,
        "sampleSizeProximal": self.sampleSizeProximal,
        "sampleSizeDistal": self.sampleSizeDistal,
        "inertiaFactor": self.inertiaFactor,
        "noise": self.noise,
        "activationThreshold": self.activationThreshold,
        "seed": self.seed,
      }

      if self.implementation == "Bayesian":
        from htmresearch.algorithms.bayesian_column_pooler import BayesianColumnPooler as cls
        params["learningRate"] = self.learningRate
        params["initMovingAverages"] = self.initMovingAverages

      elif self.implementation == "SummingBayesian":
        from htmresearch.algorithms.bayesian_summing_column_pooler import BayesianSummingColumnPooler as cls

      else:
        raise ValueError("Unrecognized implementation %s" % self.implementation)

      params["forgetting"] = self.forgetting
      params["useSupport"] = self.useSupport
      params["avoidWeightExplosion"] = self.avoidWeightExplosion
      params["resetProximalCounter"] = self.resetProximalCounter
      params["useProximalProbabilities"] = self.useProximalProbabilities

      self._pooler = cls(**params)

  def compute(self, inputs, outputs):
    """
    Run one iteration of compute.

    Note that if the reset signal is True (1) we assume this iteration
    represents the *end* of a sequence. The output will contain the
    representation to this point and any history will then be reset. The output
    at the next compute will start fresh, presumably with bursting columns.
    """
    # Handle reset first (should be sent with an empty signal)
    if "resetIn" in inputs:
      assert len(inputs["resetIn"]) == 1
      if inputs["resetIn"][0] != 0:
        # send empty output
        self.reset()
        outputs["feedForwardOutput"][:] = 0
        outputs["activeCells"][:] = 0
        return

    feedforwardInput = np.asarray(inputs["feedforwardInput"], dtype="float64")

    if "feedforwardGrowthCandidates" in inputs:
      feedforwardGrowthCandidates = np.asarray(inputs["feedforwardGrowthCandidates"], dtype="float64")
    else:
      feedforwardGrowthCandidates = feedforwardInput

    if "lateralInput" in inputs:
      lateralInputs = tuple(np.asarray(singleInput, dtype="float64")
                            for singleInput
                            in np.split(inputs["lateralInput"], self.numOtherCorticalColumns))
    else:
      lateralInputs = ()

    if "predictedInput" in inputs:
      predictedInput = np.asarray(inputs["predictedInput"], dtype="float64")
    else:
      predictedInput = None

    # Send the inputs into the Column Pooler.
    self._pooler.compute(feedforwardInput, lateralInputs,
                         feedforwardGrowthCandidates, learn=self.learningMode,
                         predictedInput=predictedInput)

    # Extract the active / predicted cells and put them into value arrays.
    outputs["activeCells"][:] = self._pooler.getActiveCellValues()

    # Send appropriate output to feedForwardOutput.
    if self.defaultOutputType == "active":
      outputs["feedForwardOutput"][:] = outputs["activeCells"]
    else:
      raise Exception("Unknown outputType: " + self.defaultOutputType)


  def reset(self):
    """ Reset the state of the layer"""
    if self._pooler is not None:
      self._pooler.reset()


  def getParameter(self, parameterName, index=-1):
    """
    Get the value of a NodeSpec parameter. Most parameters are handled
    automatically by PyRegion's parameter get mechanism. The ones that need
    special treatment are explicitly handled here.
    """
    return PyRegion.getParameter(self, parameterName, index)


  def setParameter(self, parameterName, index, parameterValue):
    """
    Set the value of a Spec parameter.
    """
    if hasattr(self, parameterName):
      setattr(self, parameterName, parameterValue)
    else:
      raise Exception("Unknown parameter: " + parameterName)


  def getOutputElementCount(self, name):
    """
    Return the number of elements for the given output.
    """
    if name in ["feedForwardOutput", "activeCells"]:
      return self.cellCount
    else:
      raise Exception("Invalid output name specified: " + name)
