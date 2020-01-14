#!/usr/bin/env python
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

"""
This file plots activity of single vs multiple columns as they converge.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import random
import numpy as np

import plotly
import plotly.graph_objs as go
from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment
from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)

import argparse
import time

def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_cmd():
    parser = argparse.ArgumentParser(description='L4L2 Experiment')
    parser.add_argument('--implementation', type=str)
    parser.add_argument('--learningRate', type=float)
    parser.add_argument('--forgetting', type=float)
    parser.add_argument('--outputCount', type=int)
    parser.add_argument('--cellCount', type=int)
    parser.add_argument('--sdrSize', type=int)
    parser.add_argument('--outputActivation', type=float)
    parser.add_argument('--useSupport', type=str2bool)
    parser.add_argument('--useApicalTiebreak', type=str2bool)

    parsed_args = parser.parse_args()

    return parsed_args


# Plotly requires a valid user to be able to save High Res images
plotlyUser = os.environ.get('PLOTLY_USERNAME')
plotlyAPIKey = os.environ.get('PLOTLY_API_KEY')
if plotlyAPIKey is not None:
  plotly.plotly.sign_in(plotlyUser, plotlyAPIKey)



def plotActivity(l2ActiveCellsMultiColumn, highlightTouch):
  maxTouches = 15
  numTouches = min(maxTouches, len(l2ActiveCellsMultiColumn))
  numColumns = len(l2ActiveCellsMultiColumn[0])
  fig = plotly.tools.make_subplots(
    rows=1, cols=numColumns, shared_yaxes=True,
    subplot_titles=('Column 1', 'Column 2', 'Column 3')[0:numColumns]
  )

  data = go.Scatter(x=[], y=[])

  shapes = []
  for t, sdrs in enumerate(l2ActiveCellsMultiColumn):
    if t <= numTouches:
      for c, activeCells in enumerate(sdrs):
        # print t, c, len(activeCells)
        for cell in activeCells:
          shapes.append(
            {
              'type': 'rect',
              'xref': 'x' + str((c + 1)),
              'yref': 'y1',
              'x0': t,
              'x1': t + 0.6,
              'y0': cell,
              'y1': cell + 1,
              'line': {
                # 'color': 'rgba(128, 0, 128, 1)',
                'width': 2,
              },
              # 'fillcolor': 'rgba(128, 0, 128, 0.7)',
            },
          )
        if t == highlightTouch:
          # Add red rectangle
          shapes.append(
            {
              'type': 'rect',
              'xref': 'x' + str((c + 1)),
              'x0': t,
              'x1': t + 0.6,
              'y0': -95,
              'y1': 4100,
              'line': {
                'color': 'rgba(255, 0, 0, 0.5)',
                'width': 3,
              },
            },
          )

  # Legend for x-axis and appropriate title
  fig['layout']['annotations'].append({
    'font': {'size': 20},
    'xanchor': 'center',
    'yanchor': 'bottom',
    'text': 'Number of touches',
    'xref': 'paper',
    'yref': 'paper',
    'x': 0.5,
    'y': -0.15,
    'showarrow': False,
  })
  fig['layout']['annotations'].append({
    'font': {'size': 24},
    'xanchor': 'center',
    'yanchor': 'bottom',
    'text': ['', '<b>One cortical column</b>', '',
             '<b>Three cortical columns</b>'][numColumns],
    'xref': 'paper',
    'yref': 'paper',
    'x': 0.5,
    'y': 1.1,
    'showarrow': False,
  })
  layout = {
    'height': 600,
    'font': {'size': 18},
    'yaxis': {
      'title': "Neuron #",
      'range': [-100, 4201],
      'showgrid': False,
    },
    'shapes': shapes,
  }

  if numColumns == 1:
    layout.update(width=320)
  else:
    layout.update(width=700)

  for c in range(numColumns):
    fig.append_trace(data, 1, c + 1)
    fig['layout']['xaxis' + str(c + 1)].update({
      'title': "",
      'range': [0, numTouches],
      'showgrid': False,
      'showticklabels': True,
    }),

  fig['layout'].update(layout)

  # Save plots as HTM and/or PDF
  basename = 'plots/activity_c' + str(numColumns)
  plotly.offline.plot(fig, filename=basename + '.html', auto_open=True)

  # Can't save image files in offline mode
  if plotlyAPIKey is not None:
    plotly.plotly.image.save_as(fig, filename=basename + '.pdf', scale=4)



def plotL2ObjectRepresentations(exp1):
  shapes = []
  numObjects = len(exp1.objectL2Representations)
  for obj in range(numObjects):
    activeCells = exp1.objectL2Representations[obj][0]
    for cell in activeCells:
      shapes.append(
        {
          'type': 'rect',
          'x0': obj,
          'x1': obj + 0.75,
          'y0': cell,
          'y1': cell + 2,
          'line': {
            # 'color': 'rgba(128, 0, 128, 1)',
            'width': 2,
          },
          # 'fillcolor': 'rgba(128, 0, 128, 0.7)',
        },
      )

  # Add red rectangle
  shapes.append(
    {
      'type': 'rect',
      'x0': 0,
      'x1': 0.9,
      'y0': -95,
      'y1': 4100,
      'line': {
        'color': 'rgba(255, 0, 0, 0.5)',
        'width': 3,
      },
    },
  )

  data = [go.Scatter(x=[], y=[])]
  layout = {
    'width': 320,
    'height': 600,
    'font': {'size': 20},
    'xaxis': {
      'title': "Object #",
      'range': [0, 10],
      'showgrid': False,
      'showticklabels': True,
    },
    'yaxis': {
      'title': "Neuron #",
      'range': [-100, 4201],
      'showgrid': False,
    },
    'shapes': shapes,
    'annotations': [{
      'xanchor': 'middle',
      'yanchor': 'bottom',
      'text': 'Target object',
      'x': 1,
      'y': 4100,
      'ax': 10,
      'ay': -25,
      'arrowcolor': 'rgba(255, 0, 0, 1)',
    },
      {
        'font': {'size': 24},
        'xanchor': 'center',
        'yanchor': 'bottom',
        'text': '<b>Object representations</b>',
        'xref': 'paper',
        'yref': 'paper',
        'x': 0.5,
        'y': 1.1,
        'showarrow': False,
      }
    ]
  }
  fig = {
    'data': data,
    'layout': layout,
  }
  plotPath = plotly.offline.plot(fig, filename='plots/shapes-rectangle.html',
                                 auto_open=True)
  print "url=", plotPath

  # Can't save image files in offline mode
  if plotlyAPIKey is not None:
    plotly.plotly.image.save_as(fig,
                                filename='plots/target_object_representations.pdf',
                                scale=4)

def plotAverageActivity(activities_over_time, converged_list, legend_names, name="tiebreak_support_test"):
  plt.clf()
  for activity_over_time, converged, legend_name in zip(activities_over_time, converged_list, legend_names):
    average_cell_activity = np.asarray(np.mean(activity_over_time, axis=1))
    plt.plot(range(1, average_cell_activity.shape[0] + 1), average_cell_activity, label=legend_name)
    if converged is not None:
      plt.plot(converged, average_cell_activity[converged], 'o')
  plt.xlabel('Sensation')
  plt.ylabel('Activity')
  plt.legend()
  plt.savefig('%s.png' % name)

def runExperiment(arguments):
  """
  We will run two experiments side by side, with either single column
  or 3 columns
  """
  numColumns = 3
  numFeatures = 3 # new: 3 # original: 3
  numPoints = 5 # new: 5 # original: 10
  numLocations = 5 # new: 5 # original: 10
  numObjects = 5 # new: 2 # original: 10
  numRptsPerSensation = 1

  objectMachine = createObjectMachine(
    machineType="simple",
    numInputBits=20,
    sensorInputSize=1024,
    externalInputSize=1024,
    numCorticalColumns=1,
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

  cellsPerColumn = arguments.cellCount
  outputCells = arguments.outputCount
  # params
  maxNumSegments = 16
  L2Overrides = {
    "noise": 1e-10,
    "cellCount": outputCells, # new: 256 # original: 4096
    "inputWidth": 2048 * cellsPerColumn, # new: 8192 # original: 16384 (?)
    "activationThreshold": arguments.outputActivation,
    "sdrSize": arguments.sdrSize,
    "forgetting": arguments.forgetting,
    "initMovingAverages": 1/float(outputCells),
    "useSupport": arguments.useSupport,
    "useProximalProbabilities": True,
    "avoidWeightExplosion": False
  }

  L4Overrides = {
    "noise": 1e-10,
    "cellsPerColumn": cellsPerColumn, # new: 4 # original 32
    "columnCount": 2048, # new: 2048 # original: 2048
    "initMovingAverages": 1/float(2048 * cellsPerColumn),
    "minThreshold": 1/float(cellsPerColumn),
    "useApicalTiebreak": arguments.useApicalTiebreak
  }

  if arguments.implementation is None or "Bayesian" in arguments.implementation:
    if "Summing" not in arguments.implementation:
      L2Overrides["learningRate"] = arguments.learningRate
      L4Overrides["learningRate"] = arguments.learningRate

    exp1 = L4L2Experiment(
      'single_column',
      implementation=arguments.implementation,
      L2RegionType="py.BayesianColumnPoolerRegion",
      L4RegionType="py.BayesianApicalTMPairRegion",
      L2Overrides=L2Overrides,
      L4Overrides=L4Overrides,
      numCorticalColumns=1,
      maxSegmentsPerCell=maxNumSegments,
      numLearningPoints=7,
      seed=1
    )
  else:
    exp1 = L4L2Experiment(
      'single_column',
      numCorticalColumns=1,
      maxSegmentsPerCell=maxNumSegments,
      numLearningPoints=3,
      seed=1
    )

  print "train single column "
  exp1.learnObjects(objectsSingleColumn)

  # test on the first object
  objectId = 2
  obj = objectMachine[objectId]

  # Create sequence of sensations for this object for all columns
  # We need to set the seed to get specific convergence points for the red
  # rectangle in the graph.
  objectSensations = {}
  random.seed(12)
  for c in range(numColumns):
    objectCopy = [pair for pair in obj]
    # random.shuffle(objectCopy)
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

  print  "inference: single column "
  exp1.sendReset()
  l2ActiveCellsSingleColumn = []
  L2ActiveCellNVsTimeSingleColumn = []
  l2ActiveValues = []
  l2ActiveValuesRepresentation = []
  for sensation in sensationStepsSingleColumn:
    exp1.infer([sensation], objectName=objectId, reset=False)
    l2ActiveCellsSingleColumn.append(exp1.getL2Prediction())
    cellActivity = exp1.getActiveCellValues()[0]
    l2ActiveValuesRepresentation.append(cellActivity[list(exp1.objectL2Representations[objectId][0])])
    l2ActiveValues.append(cellActivity)
    L2ActiveCellNVsTimeSingleColumn.append(len(exp1.getL2Prediction()[0]))

  # Used to figure out where to put the red rectangle!
  sdrSize = exp1.config["L2Params"]["sdrSize"]
  singleColumnHighlight = next(
    (idx for idx, value in enumerate(l2ActiveCellsSingleColumn)
     if len(value[0]) == sdrSize), None)
  firstObjectRepresentation = exp1.objectL2Representations[objectId][0]
  converged = next(
    (idx for idx, value in enumerate(l2ActiveCellsSingleColumn)
     if (value[0] == firstObjectRepresentation)), None)

  print  "Converged to first object representation after %s steps" % converged
  print  "Exactly SDR-Size activity (%s) after %s steps" % (sdrSize, singleColumnHighlight)
  print  "Overlaps of each l2-representation (after new sensation) to each object"
  for idx in range(0, len(l2ActiveCellsSingleColumn)):
    print
    "overlap of l2-representation %s" % idx
    for i in range(0, len(exp1.objectL2Representations)):
      object = exp1.objectL2Representations[i][0]
      l2Representation = l2ActiveCellsSingleColumn[idx][0]
      overlap = len(l2Representation.intersection(object))
      print "\tTo object %s is %s/%s" % (i, overlap, len(l2Representation))

  return l2ActiveValuesRepresentation, l2ActiveValues, converged
  #
  # print "First Object representation", np.sort(list(firstObjectRepresentation))
  #
  # print "\n\nL2 Output over steps"
  # for idx in range(0, len(l2ActiveCellsSingleColumn)):
  #   rep = np.sort(list(l2ActiveCellsSingleColumn[idx][0]))
  #   print len(rep), rep
  # print "\n\nObject representations L2"
  # for idx in range(0, len(exp1.objectL2Representations)):
  #   obj = np.sort(list(exp1.objectL2Representations[idx][0]))
  #   print len(obj), obj

  # plotActivity(l2ActiveCellsSingleColumn, singleColumnHighlight)
  # plotL2ObjectRepresentations(exp1)

  # Multi column experiment
  # exp3 = L4L2Experiment(
  #   'three_column',
  #   numCorticalColumns=3,
  #   seed=1
  # )
  #
  # print "train multi-column "
  # exp3.learnObjects(objects)
  #
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
  #
  # sdrSize = exp3.config["L2Params"]["sdrSize"]
  # multiColumnHighlight = next(
  #   (idx for idx, value in enumerate(l2ActiveCellsMultiColumn)
  #    if len(value[0]) == sdrSize), None)
  #
  # plotActivity(l2ActiveCellsMultiColumn, multiColumnHighlight)



if __name__ == "__main__":
  initial_args = argparse.Namespace(
    implementation="Bayesian",
    learningRate=0.01,
    forgetting=0.1,
    outputCount=128,
    sdrSize=5,
    outputActivation=0.3,
    cellCount=8,
    useApicalTiebreak=True,
    useSupport=True
  )

  print "Summing Implementations"
  parsed_args = initial_args
  # implementations = ["Bayesian", "SummingBayesian"]
  activation_list = []
  converged_list = []
  time_diff = 0
  counter = 0

  start = time.time()
  for _ in range(5):
    parsed_args.implementation = "SummingBayesian"
    result = runExperiment(parsed_args)
    # activation_list.append(result[0])
    # converged_list.append(result[2])
    counter += 1

  end = time.time()
  time_diff += end - start

  # legend_names = ["Incremental", "Summing"]
  # plotAverageActivity(activation_list, converged_list, legend_names, name="incremental_summing")
  #
  # print  "Apical"
  # parsed_args = initial_args
  # flags = [True, False]
  # activation_list = []
  # converged_list = []
  #
  # start = time.time()
  # for element in flags:
  #   parsed_args.useApicalTiebreak = element
  #   result = runExperiment(parsed_args)
  #   activation_list.append(result[0])
  #   converged_list.append(result[2])
  #   counter += 1
  # end = time.time()
  # time_diff += end - start
  #
  # legend_names = ["Use apical", "No apical"]
  # plotAverageActivity(activation_list, converged_list, legend_names, name="apical")
  #
  # print  "Average activity 5"
  # parsed_args = initial_args
  # activation_list = []
  # converged_list = []
  #
  # start = time.time()
  # result = runExperiment(parsed_args)
  # activation_list.append(result[0])
  # activation_list.append(result[1])
  # converged_list.append(result[2])
  # converged_list.append(result[2])
  # counter += 1
  # end = time.time()
  # time_diff += end - start
  #
  # legend_names = ["Object representation", "All cells"]
  # plotAverageActivity(activation_list, converged_list, legend_names, name="average_activity_5")
  #
  # print "Average activity 20"
  # parsed_args = initial_args
  # parsed_args.sdrSize = 20
  # activation_list = []
  # converged_list = []
  #
  # start = time.time()
  # result = runExperiment(parsed_args)
  # activation_list.append(result[0])
  # activation_list.append(result[1])
  # converged_list.append(result[2])
  # converged_list.append(result[2])
  # counter += 1
  # end = time.time()
  # time_diff += end - start
  #
  # legend_names = ["Object representation", "All cells"]
  # plotAverageActivity(activation_list, converged_list, legend_names, name="average_activity_20")

  # print  "Learning rate"
  # parsed_args = initial_args
  # learningRates = [0.01, 0.1, 0.5]
  # activation_list = []
  # converged_list = []

  # start = time.time()
  # for element in learningRates:
  #   parsed_args.learningRate = element
  #   result = runExperiment(parsed_args)
  #   activation_list.append(result[0])
  #   converged_list.append(result[2])
  #   counter += 1
  # end = time.time()
  # time_diff += end - start
  #
  # legend_names = ["$\alpha$=0.01", "$\alpha$=0.1", "$\alpha$=0.5"]
  # plotAverageActivity(activation_list, converged_list, legend_names, name="learning_rate")

  print  time_diff / float(counter)
