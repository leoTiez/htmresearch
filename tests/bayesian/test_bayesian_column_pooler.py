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
        self.bcp = bcp.BayesianColumnPooler(
            inputWidth=self.inputWidth,
            lateralInputWidths=self.lateralInputWidths,
            cellCount=self.cellCount,
            sdrSize=self.sdrSize,
            noise=self.noise,
            learningRate=self.learningRate,
            activationThreshold=self.activationThreshold,
        )

    # runs multiple learning iterations, but always with the same pattern # TODO: Make meaningful by changing input
    def test_compute_proximal_lateral_multiple_iterations(self):
        learning_times = 5
        inference_times = 1
        cellCount = 20
        sdrSize = 4
        lateralInputWidth = 10
        inputWidth = 10
        lateralInputWidths = [lateralInputWidth, lateralInputWidth]
        lateralInputValue = 0.2

        cp = bcp.BayesianColumnPooler(
            inputWidth=inputWidth,
            lateralInputWidths=lateralInputWidths,
            cellCount=cellCount,
            sdrSize=sdrSize,
            noise=self.noise,
            learningRate=self.learningRate,
            activationThreshold=self.activationThreshold,
        )

        proximalInput = np.full(inputWidth, self.proximalInputValue)
        lateralInput = np.full(lateralInputWidth, lateralInputValue)
        lateralInputValues = [lateralInput, lateralInput]

        # Multiple step learn
        for i in range(0, learning_times):
            cp.compute(
                feedforwardInput=proximalInput,
                lateralInputs=lateralInputValues,
                learn=True,
            )
        activationLearn = cp.getActiveCells()

        # Multiple step inference
        for i in range(0, inference_times):
            cp.compute(
                feedforwardInput=proximalInput,
                lateralInputs=lateralInputValues,
            )
            activationInference = cp.getActiveCells()

            print("Two activations", activationLearn, activationInference)

            self.assertTrue(len(activationLearn) == sdrSize,
                            "New object-representation was not correctly initialized with sdrSize active bits.")

            self.assertTrue(len(activationInference) == sdrSize,
                            "New activation was not correctly initialized with sdrSize active bits during inference.")

            self.assertTrue(np.array_equal(np.sort(activationLearn), np.sort(activationInference)),
                            "After one step the activation should be similar to the learned activation. (same input)")


    def test_compute_inference_proximal_lateral(self):
        cellCount = 20
        sdrSize = 4
        lateralInputWidth = 10
        inputWidth = 10
        lateralInputWidths = [lateralInputWidth, lateralInputWidth]
        lateralInputValue = 0.2

        cp = bcp.BayesianColumnPooler(
            inputWidth=inputWidth,
            lateralInputWidths=lateralInputWidths,
            cellCount=cellCount,
            sdrSize=sdrSize,
            noise=self.noise,
            learningRate=self.learningRate,
            activationThreshold=self.activationThreshold,
        )

        proximalInput = np.full(inputWidth, self.proximalInputValue)
        lateralInput = np.full(lateralInputWidth, lateralInputValue)
        lateralInputValues = [lateralInput, lateralInput]

        # One step learn
        cp.compute(
            feedforwardInput=proximalInput,
            lateralInputs=lateralInputValues,
            learn=True,
        )
        activationLearn = cp.getActiveCells()

        # One step inference
        cp.compute(
            feedforwardInput=proximalInput,
            lateralInputs=lateralInputValues,
        )
        activationInference = cp.getActiveCells()

        print("Two activations", activationLearn, activationInference)

        self.assertTrue(len(activationLearn) == sdrSize,
                        "New object-representation was not correctly initialized with sdrSize active bits.")

        self.assertTrue(len(activationInference) == sdrSize,
                        "New activation was not correctly initialized with sdrSize active bits during inference.")

        self.assertTrue(np.array_equal(np.sort(activationLearn), np.sort(activationInference)),
                        "After one step the activation should be similar to the learned activation. (same input)")


    def test_compute_inference_proximal_only(self):
        proximalInput = np.full(self.inputWidth, self.proximalInputValue)

        # One step learn
        self.bcp.compute(feedforwardInput=proximalInput, learn=True)
        activationLearn = self.bcp.getActiveCells()
        # One step inference
        self.bcp._computeInferenceMode(feedforwardInput=proximalInput, lateralInputs=[])
        activationInference = self.bcp.getActiveCells()

        self.assertTrue(len(activationLearn) == self.sdrSize,
                        "New object-representation was not correctly initialized with sdrSize active bits.")

        self.assertTrue(len(activationInference) == self.sdrSize,
                        "New activation was not correctly initialized with sdrSize active bits during inference.")

        self.assertTrue(np.array_equal(np.sort(activationLearn), np.sort(activationInference)),
                        "After one step the activation should be similar to the learned activation. (same input)")


    def test_compute_learn_proximal_lateral(self):
        lateralInputWidth = 6
        lateralInputWidths = [lateralInputWidth]
        lateralInputValue = 0.2

        cp = bcp.BayesianColumnPooler(
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

        activeCellIndices = cp.getActiveCells()
        self.assertTrue(len(activeCellIndices) == self.sdrSize,
                        "New object-representation was not correctly initialized with sdrSize active bits.")

    def test_compute_learn_proximal_only(self):
        proximalInput = np.full(self.inputWidth, self.proximalInputValue)

        self.bcp.compute(
            feedforwardInput=proximalInput,
            learn=True,
        )

        activeCellIndices = self.bcp.getActiveCells()
        self.assertTrue(len(activeCellIndices) == self.sdrSize,
                        "New object-representation was not correctly initialized with sdrSize active bits.")

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
