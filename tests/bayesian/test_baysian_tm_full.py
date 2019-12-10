import unittest
import numpy as np
import pprint as pp

from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment
from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)

class BayesianTMTest(unittest.TestCase):
    def setUp(self):
        # params
        maxNumSegments = 2
        L2Overrides = {
            "learningRate": 0.1,
            "noise": 1e-6,
            "cellCount": 256,  # new: 256 # original: 4096
            "inputWidth": 8192,  # new: 8192 # original: 16384 (?)
        }

        L4Overrides = {
            "learningRate": 0.1,
            "noise": 1e-6,
            "cellsPerColumn": 4,  # new: 4 # original 32
            "columnCount": 2048,  # new: 2048 # original: 2048
            "minThreshold": 0.35,
        }

        self.exp1 = L4L2Experiment(
            'single_column',
            implementation='BayesianApicalTiebreak',
            L2RegionType="py.BayesianColumnPoolerRegion",
            L4RegionType="py.BayesianApicalTMPairRegion",
            L2Overrides=L2Overrides,
            L4Overrides=L4Overrides,
            numCorticalColumns=1,
            maxSegmentsPerCell=maxNumSegments,
            numLearningPoints=3,  # number repetitions for learning
            seed=1
        )

        numFeatures = 3  # new: 3 # original: 3
        numPoints = 5  # new: 5 # original: 10
        numLocations = 5  # new: 5 # original: 10
        numObjects = 5  # new: 2 # original: 10
        numRptsPerSensation = 2

        self.objectMachine = createObjectMachine(
            machineType="simple",
            numInputBits=20,
            sensorInputSize=1024,
            externalInputSize=1024,
            numCorticalColumns=3,
            seed=40,
        )
        self.objectMachine.createRandomObjects(numObjects, numPoints=numPoints,
                                          numLocations=numLocations,
                                          numFeatures=numFeatures)

    def test_full_toy_sensations(self):
        # 4 sensations in a sequence fixed
        objects = {
            "simple": [
                # location, feature for CC0
                {0: (set([1, 2, 3]), set([1, 2, 3]))},
                {0: (set([4, 5, 6]), set([4, 5, 6]))},
                {0: (set([7, 8, 9]), set([7, 8, 9]))},
                {0: (set([10, 11, 12]), set([10, 11, 12]))},
            ]
        }
        # learn the sensations
        print "train single column: 4-sequence"
        self.exp1.learnObjects(objects)

        sensationStepsSingleColumn = objects["simple"]

        print "inference: single column"
        self.exp1.sendReset()
        l2ActiveCellsSingleColumn = []
        l4PredictedCellsSingleColumn = []
        l4ActiveCellsSingleColumn = []
        l4PredictedActiveCellsSingleColumn = []

        for sensation in sensationStepsSingleColumn:
            self.exp1.infer([sensation], objectName="simple", reset=False)

            active = self.exp1.getL4Representations()[0]
            predicted = self.exp1.getL4PredictedCells()[0]
            predictedActive = self.exp1.getL4PredictedActiveCells()[0]

            l2ActiveCellsSingleColumn.append(self.exp1.getL2Representations())
            l4PredictedCellsSingleColumn.append(predicted)
            l4ActiveCellsSingleColumn.append(active)
            l4PredictedActiveCellsSingleColumn.append(predictedActive)

            print
            print("SENSATION", sensation)
            print("L4 active", active)

            print("L4 predicted", predicted)

            print("L4 predicted active", predictedActive)

            self.assertTrue(np.array_equal(active, predictedActive), 'Not all active cells were predicted')

        active = self.exp1.getL4Representations()[0]
        predicted = self.exp1.getL4PredictedCells()[0]
        self.assertTrue(np.array_equal(active, predicted),
                        '%s additional predictions were made, did not converge' % (len(predicted) - len(active)))

    def test_full_one_object_sensations(self):
        # Only first object with sensations
        objects = { 0:self.objectMachine.provideObjectsToLearn()[0] }

        print("OBJECT", objects)
        # learn the sensations
        print "train single column: 4-sequence"
        self.exp1.learnObjects(objects)

        # test only on first object
        sensationStepsSingleColumn = objects[0]

        print "inference: single column"
        self.exp1.sendReset()
        l2ActiveCellsSingleColumn = []
        l4PredictedCellsSingleColumn = []
        l4ActiveCellsSingleColumn = []
        l4PredictedActiveCellsSingleColumn = []

        for sensation in sensationStepsSingleColumn:
            self.exp1.infer([sensation], objectName=0, reset=False)

            active = self.exp1.getL4Representations()[0]
            predicted = self.exp1.getL4PredictedCells()[0]
            predictedActive = self.exp1.getL4PredictedActiveCells()[0]

            l2ActiveCellsSingleColumn.append(self.exp1.getL2Representations())
            l4PredictedCellsSingleColumn.append(predicted)
            l4ActiveCellsSingleColumn.append(active)
            l4PredictedActiveCellsSingleColumn.append(predictedActive)

            print
            print("SENSATION", sensation)
            print("L4 active", active)

            print("L4 predicted", predicted)

            print("L4 predicted active", predictedActive)

            self.assertTrue(np.array_equal(active, predictedActive), 'Not all active cells were predicted')

        active = self.exp1.getL4Representations()[0]
        predicted = self.exp1.getL4PredictedCells()[0]
        self.assertTrue(np.array_equal(active, predicted),
                        '%s additional predictions were made, did not converge' % (len(predicted) - len(active)))

    def test_full_multi_object_sensations(self):
        # Only first object with sensations
        objects = self.objectMachine.provideObjectsToLearn()

        print("OBJECT", objects)
        # learn the sensations
        print "train single column: 4-sequence"
        self.exp1.learnObjects(objects)

        # test only on first object
        sensationStepsSingleColumn = objects[0]

        print "inference: single column"
        self.exp1.sendReset()
        l2ActiveCellsSingleColumn = []
        l4PredictedCellsSingleColumn = []
        l4ActiveCellsSingleColumn = []
        l4PredictedActiveCellsSingleColumn = []

        for sensation in sensationStepsSingleColumn:
            self.exp1.infer([sensation], objectName=0, reset=False)

            active = self.exp1.getL4Representations()[0]
            predicted = self.exp1.getL4PredictedCells()[0]
            predictedActive = self.exp1.getL4PredictedActiveCells()[0]

            l2ActiveCellsSingleColumn.append(self.exp1.getL2Representations())
            l4PredictedCellsSingleColumn.append(predicted)
            l4ActiveCellsSingleColumn.append(active)
            l4PredictedActiveCellsSingleColumn.append(predictedActive)

            print
            print("SENSATION", sensation)
            print("L4 active", active)

            print("L4 predicted", predicted)

            print("L4 predicted active", predictedActive)

            self.assertTrue(np.array_equal(active, predictedActive), 'Not all active cells were predicted')

        active = self.exp1.getL4Representations()[0]
        predicted = self.exp1.getL4PredictedCells()[0]
        self.assertTrue(np.array_equal(active, predicted),
                        '%s additional predictions were made, did not converge' % (len(predicted) - len(active)))

if __name__ == '__main__':
    unittest.main()
