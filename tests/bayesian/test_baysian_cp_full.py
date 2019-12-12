import unittest
import numpy as np
import pprint as pp

from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment
from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)

class BayesianCPTest(unittest.TestCase):
    def setUp(self):
        # params
        maxNumSegments = 2
        L2Overrides = {
            "learningRate": 0.1,
            "noise": 1e-8,
            "cellCount": 256,  # new: 256 # original: 4096
            "inputWidth": 8192,  # new: 8192 # original: 16384 (?)
            "sdrSize": 5,
            "activationThreshold": 0.01,
        }

        L4Overrides = {
            "learningRate": 0.1,
            "noise": 1e-8,
            "cellsPerColumn": 4,  # new: 4 # original 32
            "columnCount": 2048,  # new: 2048 # original: 2048
            "minThreshold": 0.35,
        }

        self.exp1 = L4L2Experiment(
            'single_column',
            implementation='SummingBayesian',
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

    def test_one_toy_l2(self):
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

        simpleObjectRepresentation = self.exp1.objectL2Representations["simple"][0]
        for sensation in sensationStepsSingleColumn:
            self.exp1.infer([sensation], objectName="simple", reset=False)

            active = self.exp1.getL2Representations()[0]
            l2ActiveCellsSingleColumn.append(active)

            print
            print("SENSATION", sensation)
            print("Target object", simpleObjectRepresentation)
            print("L2 active", active)

            self.assertTrue(len(simpleObjectRepresentation.difference(active)) == 0,
                            'Not all object cells were activated missing %s' % simpleObjectRepresentation.difference(
                                    active))

        active = self.exp1.getL2Representations()[0]
        objectPrediction = self.exp1.getL2Prediction()[0]
        self.assertTrue(np.array_equal(objectPrediction, simpleObjectRepresentation),
                        '%s additional active cells were made, did not converge' % (len(objectPrediction) - len(simpleObjectRepresentation)))
        # With a single object even the activation should be a sufficient indicator for activation
        self.assertTrue(np.array_equal(active, simpleObjectRepresentation),
                        '%s additional active cells were made, did not converge' % (
                                    len(active) - len(simpleObjectRepresentation)))

    def test_two_unique_toys_l2(self):
        # 4 sensations in a sequence fixed
        objects = {
            "simple": [
                # location, feature for CC0
                {0: (set([1, 2, 3]), set([1, 2, 3]))},
                {0: (set([4, 5, 6]), set([4, 5, 6]))},
                {0: (set([7, 8, 9]), set([7, 8, 9]))},
                {0: (set([10, 11, 12]), set([10, 11, 12]))},
            ],
            "simple2": [
                # location, feature for CC0
                {0: (set([13, 14, 15]), set([13, 14, 15]))},
                {0: (set([16, 17, 18]), set([16, 17, 18]))},
                {0: (set([19, 20, 21]), set([19, 20, 21]))},
                {0: (set([22, 23, 24]), set([22, 23, 24]))},
            ]
        }
        # learn the sensations
        print "train single column: 4-sequence"
        self.exp1.learnObjects(objects)
        self.exp1.sendReset()

        simpleObjectRepresentation = self.exp1.objectL2Representations["simple"][0]
        simpleObjectRepresentation2 = self.exp1.objectL2Representations["simple2"][0]

        for o in ["simple", "simple2"]:

            sensationStepsSingleColumn = objects[o]
            objectRepresentation = self.exp1.objectL2Representations[o][0]

            print "inference test: single column - object %s" % o
            print("Object %s representation" % o, objectRepresentation)

            for i in range(len(sensationStepsSingleColumn)):
                sensation = sensationStepsSingleColumn[i]
                self.exp1.infer([sensation], objectName=o, reset=False)

                active = self.exp1.getL2Representations()[0]
                objectPrediction = self.exp1.getL2Prediction()[0]
                overlap = simpleObjectRepresentation.intersection(active)
                overlap2 = simpleObjectRepresentation2.intersection(active)

                print
                print("Overlap %s %s/%s %s" % ("simple", len(overlap), len(simpleObjectRepresentation), overlap))
                print("Overlap %s %s/%s %s" % ("simple2", len(overlap2), len(simpleObjectRepresentation2), overlap2))
                print("L2 active", active)

                self.assertTrue(len(objectRepresentation.difference(active)) == 0,
                                'Not all object cells were activated missing %s' % objectRepresentation.difference(
                                    active))

                # Check if last sensation converged
                if (i == len(sensationStepsSingleColumn) - 1):
                    print "Final object prediction: %s" % objectPrediction
                    self.assertTrue(np.array_equal(objectPrediction, objectRepresentation),
                                    '%s additional active cells were made, did not converge' % (
                                                len(objectPrediction) - len(objectRepresentation)))

            # reset after object to avoid temporal sequence
            self.exp1.sendReset()


    def test_two_toys_common_sensations_l2(self):
        # 4 sensations in a sequence fixed
        objects = {
            "simple": [
                # location, feature for CC0
                {0: (set([1, 2, 3]), set([1, 2, 3]))},
                {0: (set([4, 5, 6]), set([4, 5, 6]))},
                {0: (set([7, 8, 9]), set([7, 8, 9]))},
                {0: (set([10, 11, 12]), set([10, 11, 12]))},
            ],
            "simple2": [
                # location, feature for CC0
                {0: (set([1, 2, 3]), set([1, 2, 3]))}, # same sensation -> both should be predicted
                {0: (set([16, 17, 18]), set([16, 17, 18]))},
                {0: (set([19, 20, 21]), set([19, 20, 21]))},
                {0: (set([22, 23, 24]), set([22, 23, 24]))},
            ]
        }
        # learn the sensations
        print "train single column: 4-sequence"
        self.exp1.learnObjects(objects)
        self.exp1.sendReset()

        simpleObjectRepresentation = self.exp1.objectL2Representations["simple"][0]
        simpleObjectRepresentation2 = self.exp1.objectL2Representations["simple2"][0]

        for o in ["simple", "simple2"]:

            sensationStepsSingleColumn = objects[o]
            objectRepresentation = self.exp1.objectL2Representations[o][0]

            print "inference test: single column - object %s" % o
            print("Object %s representation" % o, objectRepresentation)

            for i in range(len(sensationStepsSingleColumn)):
                sensation = sensationStepsSingleColumn[i]
                self.exp1.infer([sensation], objectName=o, reset=False)

                active = self.exp1.getL2Representations()[0]
                objectPrediction = self.exp1.getL2Prediction()[0]
                overlap = simpleObjectRepresentation.intersection(active)
                overlap2 = simpleObjectRepresentation2.intersection(active)

                print
                print("Overlap %s %s/%s %s" % ("simple", len(overlap), len(simpleObjectRepresentation), overlap))
                print("Overlap %s %s/%s %s" % ("simple2", len(overlap2), len(simpleObjectRepresentation2), overlap2))
                print("L2 active", active)


                self.assertTrue(len(objectRepresentation.difference(active)) == 0,
                               'Not all object cells were activated missing %s' % objectRepresentation.difference(active))

                # First sensation should predict both objects possible
                if (i == 0):
                    self.assertTrue(len(simpleObjectRepresentation.difference(active)) == 0,
                                    'First (overlapping) sensation did not predict both objects')
                    self.assertTrue(len(simpleObjectRepresentation2.difference(active)) == 0,
                                    'First (overlapping) sensation did not predict both objects')

                # Check if last sensation converged
                if (i == len(sensationStepsSingleColumn)-1):
                    print "Final object prediction: %s" % objectPrediction
                    self.assertTrue(np.array_equal(objectPrediction, objectRepresentation),
                                    '%s additional active cells were made, did not converge' % (len(objectPrediction) - len(objectRepresentation)))

            # reset after object to avoid temporal sequence
            self.exp1.sendReset()

if __name__ == '__main__':
    unittest.main()
