#!/usr/bin/python2.7
import random
import matplotlib
matplotlib.use("Agg")
import warnings
import json
import os

from htmresearch.frameworks.layers.object_machine_factory import (createObjectMachine)
from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment


def runExperiments():
    # default parameters
    seed = 40
    suppressWarnings = True
    maxNumSegments = 2
    sdrSize = 40
    columns_count = 2048
    cells_per_column = 8

    # parameters to iterate over
    num_columns_p = [1, 3]
    implementation_p = ["SummingBayesian", "Bayesian"]
    learning_rate_l2_p = [0.1, 0.2]
    learning_rate_l4_p = [0.1, 0.2]
    activation_threshold_p = [0.3, 0.5, 0.7]
    min_threshold_p = [0.1, 0.3, 0.5]
    forgetting_p = [0.01, 0.1, 0.15]
    use_support_p = [True, False]
    use_proximal_probabilities_p = [True, False]
    use_apical_tiebreak_p = [True, False]

    for num_columns in num_columns_p:
        for implementation in implementation_p:
            for learning_rate_l2 in learning_rate_l2_p:
                for learning_rate_l4 in learning_rate_l4_p:
                    for activation_threshold in activation_threshold_p:
                        for min_threshold in min_threshold_p:
                            for forgetting in forgetting_p:
                                for use_support in use_support_p:
                                    for use_proximal_probabilities in use_proximal_probabilities_p:
                                        for use_apical_tiebreak in use_apical_tiebreak_p:
                                            L2Overrides = {
                                                "noise": 1e-8,
                                                "cellCount": 512,  # new: 256 # original: 4096
                                                "inputWidth": cells_per_column * columns_count,  # new: 8192 # original: 16384 (?) = cells per column * column count
                                                "sdrSize": sdrSize,
                                                "useProximalProbabilities": use_proximal_probabilities,
                                                "avoidWeightExplosion": True,
                                                "useSupport": use_support,
                                                "seed": seed,
                                                "learningMode": True,
                                                "learningRate": learning_rate_l2,  # alpha
                                                "activationThreshold": activation_threshold,  # used for cell activation & competition through distal segment activity
                                                "forgetting": forgetting,
                                                "resetProximalCounter": False,
                                            }

                                            L4Overrides = {
                                                "noise": 1e-8,
                                                "cellsPerColumn": cells_per_column,  # new: 4 # original 32
                                                "columnCount": columns_count,  # new: 2048 # original: 2048
                                                "minThreshold": min_threshold,
                                                "learn": True,
                                                "sampleSize": 40,
                                                "useApicalTiebreak": use_apical_tiebreak,
                                                "learningRate": learning_rate_l4,
                                                "maxSegmentsPerCell": maxNumSegments,
                                                "implementation": "Bayesian",
                                                "seed": seed
                                            }

                                            title = 'Experiment with config \n\tL2Overrides=%s \n\tL4Overrides=%s \n\tImplementation=%s \n\tnumCC=%s' % (L2Overrides, L4Overrides, implementation, num_columns)
                                            id = hash(title)
                                            print "\n\n#################################\n", title
                                            exp = L4L2Experiment(
                                                title,
                                                implementation=implementation,
                                                L2RegionType="py.BayesianColumnPoolerRegion",
                                                L4RegionType="py.BayesianApicalTMPairRegion",
                                                numCorticalColumns=num_columns,
                                                maxSegmentsPerCell=maxNumSegments,
                                                seed=1,
                                                L2Overrides=L2Overrides,
                                                L4Overrides=L4Overrides,
                                            )

                                            with warnings.catch_warnings():
                                                if (suppressWarnings):
                                                    warnings.simplefilter("ignore")
                                                results = runExperiment(exp, num_columns)

                                                filename = 'results/%s_%s_%s.json' % (results["convergedSteps"], results["bestOverlapRatio"], id)
                                                if not os.path.exists('results'):
                                                    os.makedirs('results')
                                                with open(filename, 'w') as outfile:
                                                    json.dump(results, outfile, indent=4)

                                            print "\n#################################"

"""
  Minimal version of the convergence_activity experiment from the htmpapers/frontiers repository.
  Reproducing the experiment from the "How the neocortex learns the structure of the world" (Columns-paper).
  Will be used to test and compare the results with the bayesian implementation of apical-TM and column pooler.
"""

def runExperiment(exp, numColumns):
  """
  We will run two experiments side by side, with either single column
  or 3 columns
  """
  numFeatures = 3
  numPoints = 10
  numLocations = 10
  numObjects = 10 # 2
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
  objectsFeatures = {}
  for i in range(numObjects):
    featureLocations = []
    for j in range(numLocations):
      featureLocations.append({0: objects[i][j][0]})
    objectsFeatures[i] = featureLocations

  print "train "
  exp.learnObjects(objectsFeatures)
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

  sensationSteps = []
  for step in xrange(len(objectSensations[0])):
    pairs = [
      objectSensations[col][step] for col in xrange(numColumns)
    ]
    sdrs = objectMachine._getSDRPairs(pairs)
    sensationSteps.append(sdrs)

  print "inference "
  exp.sendReset()

  # 1.) Inference: Get active predictions for each sensation/location input
  l2ActiveCells = []
  for sensation in sensationSteps:
    exp.infer([sensation], objectName=objectId, reset=False)
    rep = exp.getL2Prediction()
    l2ActiveCells.append(rep)
    activeCellNum = 0
    for c in range(numColumns):
        activeCellNum += len(rep[c])

  # 2.) Save active cells of first object for convergence test (and possible drawing)
  # Used to figure out where to put the red rectangle!
  sdrSize = exp.config["L2Params"]["sdrSize"]
  singleColumnHighlight = next(
    (idx for idx, value in enumerate(l2ActiveCells)
     if len(value[0]) == sdrSize), None)
  firstObjectRepresentation = exp.objectL2Representations[0][0]
  converged = next(
    (idx for idx, value in enumerate(l2ActiveCells)
     if (value[0] == firstObjectRepresentation)), None)

  print "Exactly SDR-Size activity (%s) after %s steps" % (sdrSize, singleColumnHighlight)
  print "Converged to first object representation after %s steps" % converged

  # 3.) Iterate over all active predictions and print the overlap to all objects
  print "Overlaps of each l2-representation (after new sensation) to each object"
  overlaps = {}
  bestOverlapRatio = 0
  for idx in range(0, len(l2ActiveCells)):
      print "overlap of l2-representation after %s sensation (overlap/active-neurons)" % (idx+1)
      overlaps[idx] = {}
      for i in range(0, len(exp.objectL2Representations)):
          object = exp.objectL2Representations[i][0]
          l2Representation = l2ActiveCells[idx][0]
          overlap = len(l2Representation.intersection(object))
          overlaps[idx][i] = "%s/%s" % (overlap, len(l2Representation))
          if (overlap*1.0/len(l2Representation) > bestOverlapRatio):
              bestOverlapRatio = overlap*1.0/len(l2Representation)
          print "\tTo object %s is %s/%s" % (i, overlap, len(l2Representation))

  result = {
      "convergedSteps": converged,
      "overlaps": overlaps,
      "bestOverlapRatio": bestOverlapRatio,
      "params": exp.name,
  }

  return result

if __name__ == "__main__":
  runExperiments()
