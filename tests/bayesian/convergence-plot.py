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

import os
import random

import plotly
import plotly.graph_objs as go
from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment
from htmresearch.frameworks.layers.object_machine_factory import (
    createObjectMachine
)

Y_SIZE = 128

# Plotly requires a valid user to be able to save High Res images
plotlyUser = os.environ.get('PLOTLY_USERNAME')
plotlyAPIKey = os.environ.get('PLOTLY_API_KEY')
if plotlyAPIKey is not None:
    plotly.plotly.sign_in(plotlyUser, plotlyAPIKey)


def plotActivity(l2ActiveCellsMultiColumn, highlightTouches):
    maxTouches = 15
    numTouches = min(maxTouches, len(l2ActiveCellsMultiColumn))
    numColumns = len(l2ActiveCellsMultiColumn[0])
    fig = plotly.tools.make_subplots(
        rows=1, cols=numColumns, shared_yaxes=True,
        subplot_titles=('Numenta (Single Column)', 'Bayesian Incremental', 'Bayesian Summing')[0:numColumns]
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
                # highlight touch for the column (or approach)
                highlightTouch = highlightTouches[c]
                if t == highlightTouch:
                    # Add red rectangle
                    shapes.append(
                        {
                            'type': 'rect',
                            'xref': 'x' + str((c + 1)),
                            'x0': t,
                            'x1': t + 0.6,
                            'y0': 0,
                            'y1': Y_SIZE,
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
                 '<b>Convergence with different configurations</b>'][numColumns],
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
            'title': "Neuron # (Output Layer)",
            'range': [0, Y_SIZE],
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
            'y0': 0,
            'y1': Y_SIZE,
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
            'range': [0, Y_SIZE],
            'showgrid': False,
        },
        'shapes': shapes,
        'annotations': [{
            'xanchor': 'middle',
            'yanchor': 'bottom',
            'text': 'Target object',
            'x': 1,
            'y': Y_SIZE,
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


def runExperiment():
    """
  We will run two experiments side by side, with either single column
  or 3 columns
  """
    numColumns = 1
    numFeatures = 3  # new: 3 # original: 3
    numPoints = 5  # new: 5 # original: 10
    numLocations = 5  # new: 5 # original: 10
    numObjects = 5  # new: 2 # original: 10
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

    cellsPerColumn = 8
    outputCells = 128
    # params
    maxNumSegments = 16
    L2Overrides = {
        "learningRate": 0.01,
        "noise": 1e-10,
        "cellCount": outputCells,  # new: 256 # original: 4096
        "inputWidth": 2048 * cellsPerColumn,  # new: 8192 # original: 16384 (?)
        "activationThreshold": 0.3,
        "sdrSize": 5,
        "forgetting": 0.1,
        "initMovingAverages": 1 / float(outputCells),
        "useSupport": True,
        "useProximalProbabilities": True,
        "avoidWeightExplosion": False,
    }

    L4Overrides = {
        "learningRate": 0.01,
        "noise": 1e-10,
        "cellsPerColumn": cellsPerColumn,  # new: 4 # original 32
        "columnCount": 2048,  # new: 2048 # original: 2048
        "initMovingAverages": 1 / float(2048 * cellsPerColumn),
        "minThreshold": 1 / float(cellsPerColumn),
        "useApicalTiebreak": True
    }

    # EXP 1 - Summing Bayesian

    exp1 = L4L2Experiment(
        'single_column_summing_bayesian',
        implementation='SummingBayesian',
        L2RegionType="py.BayesianColumnPoolerRegion",
        L4RegionType="py.BayesianApicalTMPairRegion",
        L2Overrides=L2Overrides,
        L4Overrides=L4Overrides,
        numCorticalColumns=1,
        maxSegmentsPerCell=maxNumSegments,
        numLearningPoints=7,
        seed=1
    )

    # EXP 2 - Incremental Bayesian

    exp2 = L4L2Experiment(
        'single_column_bayesian',
        implementation='Bayesian',
        L2RegionType="py.BayesianColumnPoolerRegion",
        L4RegionType="py.BayesianApicalTMPairRegion",
        L2Overrides=L2Overrides,
        L4Overrides=L4Overrides,
        numCorticalColumns=1,
        maxSegmentsPerCell=maxNumSegments,
        numLearningPoints=7,
        seed=1
    )

    # EXP 3 - Numenta

    exp3 = L4L2Experiment(
        'single_column',
        numCorticalColumns=1,
        L2Overrides={
            'cellCount': outputCells,
            'sdrSize': 5,
        },
        numLearningPoints=7,
        seed=1
    )

    print "train three approaches (summing, incremental, numenta)"
    exp1.learnObjects(objectsSingleColumn)
    exp2.learnObjects(objectsSingleColumn)
    exp3.learnObjects(objectsSingleColumn)

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

    print "inference: single column "
    exp1.sendReset()
    exp2.sendReset()
    exp3.sendReset()
    l2MultipleTypes = []
    for sensation in sensationStepsSingleColumn:
        exp1.infer([sensation], objectName=objectId, reset=False)
        exp2.infer([sensation], objectName=objectId, reset=False)
        exp3.infer([sensation], objectName=objectId, reset=False)

        rep1 = exp3.getL2Representations()[0]
        rep2 = exp1.getL2Prediction()[0]
        rep3 = exp2.getL2Prediction()[0]
        l2MultipleTypes.append([rep1, rep2, rep3])

    # Used to figure out where to put the red rectangle!
    firstObjectRepresentation1 = exp1.objectL2Representations[objectId][0]
    firstObjectRepresentation2 = exp2.objectL2Representations[objectId][0]
    firstObjectRepresentation3 = exp3.objectL2Representations[objectId][0]
    firstObjectRepresentations = [firstObjectRepresentation1,
                                  firstObjectRepresentation2,
                                  firstObjectRepresentation3]

    print("OBJECT REPRESENTATIONS")
    print(firstObjectRepresentations)

    converged1 = next(
        (idx for idx, value in enumerate(l2MultipleTypes)
         if (value[0] == firstObjectRepresentations[0])), None)
    converged2 = next(
        (idx for idx, value in enumerate(l2MultipleTypes)
         if (value[1] == firstObjectRepresentations[1])), None)
    converged3 = next(
        (idx for idx, value in enumerate(l2MultipleTypes)
         if (value[2] == firstObjectRepresentations[2])), None)
    converged = [converged1, converged2, converged3]

    plotActivity(l2MultipleTypes, converged)
    plotL2ObjectRepresentations(exp1)
    plotL2ObjectRepresentations(exp2)
    plotL2ObjectRepresentations(exp3)
    print("Finished")

if __name__ == "__main__":
    runExperiment()
