#!/usr/bin/python2.7
import random
import matplotlib
matplotlib.use("Agg")
from htmresearch.frameworks.layers.object_machine_factory import (createObjectMachine)
from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment


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
  numColumns = 1 # 3
  numFeatures = 3
  numPoints = 10
  numLocations = 10
  numObjects = 2 # 10
  numRptsPerSensation = 2

  objectMachine = createObjectMachine(
    machineType="simple",
    numInputBits=20,
    sensorInputSize=1024,
    externalInputSize=1024,
    numCorticalColumns=numColumns,
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

  maxNumSegemnts = 2
  # we will run two experiments side by side, with either single column
  # or 3 columns
  # exp3 = L4L2Experiment(
  #   'three_column',
  #   implementation='BayesianApicalTiebreak',
  #   L4RegionType="py.BayesianApicalTMPairRegion",
  #   numCorticalColumns=3,
  #   maxSegmentsPerCell=5,
  #   seed=1
  # )

  exp1 = L4L2Experiment(
    'single_column',
    implementation='BayesianApicalTiebreak',
    L4RegionType="py.BayesianApicalTMPairRegion",
    numCorticalColumns=1,
    maxSegmentsPerCell=maxNumSegemnts,
    seed=1
  )

  print "train single column "
  exp1.learnObjects(objectsSingleColumn)
  # print "train multi-column "
  # exp3.learnObjects(objects)

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

  # print "inference: multi-columns "
  # exp3.sendReset()
  # l2ActiveCellsMultiColumn = []
  # L2ActiveCellNVsTimeMultiColumn = []
  # for sensation in sensationStepsMultiColumn:
  #   exp3.infer([sensation], objectName=objectId, reset=False)
  #   l2ActiveCellsMultiColumn.append(exp3.getL2Representations())
  #   activeCellNum = 0
  #   for c in range(numColumns):
  #     activeCellNum += len(exp3.getL2Representations()[c])
  #   L2ActiveCellNVsTimeMultiColumn.append(activeCellNum / numColumns)

  print "inference: single column "
  exp1.sendReset()
  l2ActiveCellsSingleColumn = []
  L2ActiveCellNVsTimeSingleColumn = []
  for sensation in sensationStepsSingleColumn:
    exp1.infer([sensation], objectName=objectId, reset=False)
    rep = exp1.getL2Representations()
    l2ActiveCellsSingleColumn.append(rep)
    print "\n\nRepresentation", rep
    print "Length Representation", len(rep[0])
    L2ActiveCellNVsTimeSingleColumn.append(len(rep[0]))

  # Used to figure out where to put the red rectangle!
  sdrSize = exp1.config["L2Params"]["sdrSize"]
  singleColumnHighlight = next(
    (idx for idx, value in enumerate(l2ActiveCellsSingleColumn)
     if len(value[0]) == sdrSize), None)

  print "SDR size", sdrSize
  print singleColumnHighlight

if __name__ == "__main__":
  runExperiment()
