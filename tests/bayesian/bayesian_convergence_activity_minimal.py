#!/usr/bin/python2.7
import random
import matplotlib
matplotlib.use("Agg")
from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment
from htmresearch.frameworks.layers.object_machine_factory import (createObjectMachine)

"""
  Minimal version of the convergence_activity experiment from the htmpapers/frontiers repository.
  Reproducing the experiment from the "How the neocortex learns the structure of the world" (Columns-paper).
  Will be used to test and compare the results with the bayesian implementation of apical-TM and column pooler.
"""

def runExperiment():
  """
  We will run two experiments side by side, with either single column
  or 3 columns
  """
  numColumns = 3
  numFeatures = 3
  numPoints = 10
  numLocations = 10
  numObjects = 10
  numRptsPerSensation = 2

  objectMachine = createObjectMachine(
    machineType="simple",
    numInputBits=20,
    sensorInputSize=1024,
    externalInputSize=1024,
    numCorticalColumns=3,
    seed=40,
  )
  objectMachine.createRandomObjects(numObjects, numPoints=numPoints,
                                    numLocations=numLocations,
                                    numFeatures=numFeatures)

  objects = objectMachine.provideObjectsToLearn()

  # single-out the inputs to the column #1
  objectsSingleColumn = {}
  for i in range(numObjects):
    featureLocations = []
    for j in range(numLocations):
      featureLocations.append({0: objects[i][j][0]})
    objectsSingleColumn[i] = featureLocations

  # we will run two experiments side by side, with either single column
  # or 3 columns
  exp3 = L4L2Experiment(
    'three_column',
    numCorticalColumns=3,
    seed=1
  )

  exp1 = L4L2Experiment(
    'single_column',
    numCorticalColumns=1,
    seed=1
  )

  print "train single column "
  exp1.learnObjects(objectsSingleColumn)
  print "train multi-column "
  exp3.learnObjects(objects)

  # test on the first object
  objectId = 0
  obj = objectMachine[objectId]

  # Create sequence of sensations for this object for all columns
  # We need to set the seed to get specific convergence points for the red
  # rectangle in the graph.
  objectSensations = {}
  random.seed(12)
  for c in range(numColumns):
    objectCopy = [pair for pair in obj]
    random.shuffle(objectCopy)
    # stay multiple steps on each sensation
    sensations = []
    for pair in objectCopy:
      for _ in xrange(numRptsPerSensation):
        sensations.append(pair)
    objectSensations[c] = sensations

  sensationStepsSingleColumn = []
  sensationStepsMultiColumn = []
  for step in xrange(len(objectSensations[0])):
    pairs = [
      objectSensations[col][step] for col in xrange(numColumns)
    ]
    sdrs = objectMachine._getSDRPairs(pairs)
    sensationStepsMultiColumn.append(sdrs)
    sensationStepsSingleColumn.append({0: sdrs[0]})

  print "inference: multi-columns "
  exp3.sendReset()
  l2ActiveCellsMultiColumn = []
  L2ActiveCellNVsTimeMultiColumn = []
  for sensation in sensationStepsMultiColumn:
    exp3.infer([sensation], objectName=objectId, reset=False)
    l2ActiveCellsMultiColumn.append(exp3.getL2Representations())
    activeCellNum = 0
    for c in range(numColumns):
      activeCellNum += len(exp3.getL2Representations()[c])
    L2ActiveCellNVsTimeMultiColumn.append(activeCellNum / numColumns)

  print "inference: single column "
  exp1.sendReset()
  l2ActiveCellsSingleColumn = []
  L2ActiveCellNVsTimeSingleColumn = []
  for sensation in sensationStepsSingleColumn:
    exp1.infer([sensation], objectName=objectId, reset=False)
    l2ActiveCellsSingleColumn.append(exp1.getL2Representations())
    L2ActiveCellNVsTimeSingleColumn.append(len(exp1.getL2Representations()[0]))

if __name__ == "__main__":
  runExperiment()