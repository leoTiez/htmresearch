#!/usr/bin/python2.7
import unittest
import numpy as np

# Import apical tiebreak bayesian tm
import os
import sys
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), '../../htmresearch/algorithms')))
import bayesian_column_pooler as bcp


class BayesianCPTest(unittest.TestCase):
    def setUp(self):
        self.inputWidth = 4
        self.lateralInputWidths = ()
        self.cellCount = 4
        self.sdrSize = 2
        self.maxSdrSize = None
        self.minSdrSize = None
        # Proximal
        self.sampleSizeProximal = 20 # not used
        # Distal
        self.sampleSizeDistal = 20
        self.inertiaFactor = 1.
        # Bayesian
        self.noise = 0.01  # lambda
        self.learningRate = 0.1  # alpha
        self.activationThreshold = 0.5  # probability such that a cell becomes active

        # Experiment params
        self.distalInputValue = 0.5
        self.proximalInputValue = 0.2

        # Reduce network size
        self.bcp = bcp.ColumnPooler(
            inputWidth=self.inputWidth,
            lateralInputWidths=self.lateralInputWidths,
            cellCount=self.cellCount,
            sdrSize=self.sdrSize,
            noise=self.noise,
            learningRate=self.learningRate,
            activationThreshold=self.activationThreshold,
        )

    def test_compute_proximal_lateral(self):
        lateralInputWidth = 6
        lateralInputWidths = [lateralInputWidth]
        lateralInputValue = 0.2

        cp = bcp.ColumnPooler(
            inputWidth=self.inputWidth,
            lateralInputWidths=lateralInputWidths,
            cellCount=self.cellCount,
            sdrSize=self.sdrSize,
            noise=self.noise,
            learningRate=self.learningRate,
            activationThreshold=self.activationThreshold,
        )

        proximalInput = np.full(self.inputWidth, self.proximalInputValue)
        lateralInput = np.full(lateralInputWidth, lateralInputValue)
        lateralInputValues = [lateralInput]

        cp.compute(
            feedforwardInput=proximalInput,
            lateralInputs=lateralInputValues,
            learn=True,
        )

        activeCellIndicies = cp.getActiveCells()
        self.assertTrue(len(activeCellIndicies) == self.sdrSize,
                        "New object-representation was not corretly initialized with sdrSize active bits.")

    def test_compute_proximal_only(self):
        proximalInput = np.full(self.inputWidth, self.proximalInputValue)

        self.bcp.compute(
            feedforwardInput=proximalInput,
            learn=True,
        )

        activeCellIndicies = self.bcp.getActiveCells()
        self.assertTrue(len(activeCellIndicies) == self.sdrSize,
                        "New object-representation was not corretly initialized with sdrSize active bits.")

    def test_moving_average(self):
        # Object representation example (randomly sampled)
        activeCells = np.array([0, 1, 0, 1])

        # Initialized Moving Averages at very first step
        internalDistalMovingAverages = np.zeros((self.cellCount, self.inputWidth))
        internalDistalMovingAverageBias = np.zeros(self.cellCount)
        internalDistalMovingAverageInput = np.zeros(self.cellCount)
        input = activeCells # E.g. distal-internal => same cells as input

        updatedMovingAverage, \
        updatedMovingAverageBias,\
        updatedMovingAverageInput = self.bcp._updateMovingAverage(
            activeCells,
            internalDistalMovingAverages,
            internalDistalMovingAverageBias,
            internalDistalMovingAverageInput,
            input,
            self.cellCount,
            None
        )

        movingAveragesFromActiveCells = updatedMovingAverage[activeCells > 0][:, input > 0]
        self.assertTrue(
            np.all(
                movingAveragesFromActiveCells > 0.0
            ), 'Not all moving averages from active cells to input have been updated'
        )

        movingAveragesFromActiveInputs = updatedMovingAverageInput[input > 0]
        self.assertTrue(
            np.all(
                movingAveragesFromActiveInputs > 0.0
            ), 'Not all moving averages of the input have been updated'
        )

        movingAverageBiasFromActiveCells = internalDistalMovingAverageBias[activeCells > 0]
        self.assertTrue(
            np.all(
                movingAverageBiasFromActiveCells > 0.0
            ), 'Not all moving average bias of the cells have been updated'
        )

    def test_learn(self):
        # Moving averages after one iteration
        internalDistalMovingAverages = np.array([
            [0, 0, 0, 0],
            [0, 0.1, 0, 0.1],
            [0, 0, 0, 0],
            [0, 0.1, 0, 0.1],
        ])

        internalDistalMovingAverageBias = np.array([0,  0.1,  0,  0.1])
        internalDistalMovingAverageInput = np.array([0,  0.1,  0,  0.1])

        # Initialized Moving Averages at very first step

        weights = self.bcp._learn(
            internalDistalMovingAverages,
            internalDistalMovingAverageBias,
            internalDistalMovingAverageInput,
        )

        weightsFromActiveCellsAndInput = weights[:, :-1][internalDistalMovingAverages > 0]
        self.assertTrue(
            np.all(
                weightsFromActiveCellsAndInput > 0.0
            ), 'Not all weights and bias values from active cells to input have been updated. (greater 0)'
        )

        weightsFromInactiveCellsAndInput = weights[:, :-1][internalDistalMovingAverages <= 0]
        self.assertTrue(
            np.all(
                weightsFromInactiveCellsAndInput <= 0.0
            ), 'Not all weights and bias values from active cells to input are zero.'
        )

        biasFromActiveCellsAndInput = weights[:, -1:][internalDistalMovingAverageBias > 0]
        self.assertTrue(
            np.all(
                np.exp(biasFromActiveCellsAndInput) > 0.0
            ), 'Not all bias values from active cells have been updated. (greater 0)'
        )

        biasFromInactiveCellsAndInput = weights[:, -1:][internalDistalMovingAverageBias <= 0]
        self.assertTrue(
            np.all(
                np.exp(biasFromInactiveCellsAndInput) <= 0.0
            ), 'Not all bias values from inactive cells are zero. (exp(-inf))'
        )

if __name__ == '__main__':
    unittest.main()
