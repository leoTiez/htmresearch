#!/usr/bin/python2.7
import unittest
import numpy as np

# Import apical tiebreak bayesian tm
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), '../algorithms')))
import apical_tiebreak_bayesian_temporal_memory as btm


class BayesianTMTest(unittest.TestCase):
    def setUp(self):
        self.column_count = 5
        self.basal_input_size = 4
        self.apical_input_size = 6
        self.cells_per_column = 2
        # Reduce network size
        self.btm = btm.ApicalTiebreakBayesianTemporalMemory(
            columnCount=self.column_count,
            basalInputSize=self.basal_input_size,
            apicalInputSize=self.apical_input_size,
            cellsPerColumn=self.cells_per_column
        )

    def test_reset(self):
        self.btm.reset()
        self.assertEqual(self.btm.predictedCells.shape, (self.btm.numberOfCells(),),
                         'Dimensionality mismatch of predicted cells')
        self.assertTrue(np.all(self.btm.predictedCells == 0), 'Predicted cells after reset non-zero')

        self.assertEqual(self.btm.activeBasalSegments.shape, (self.btm.maxSegmentsPerCell, self.btm.numberOfCells()),
                         'Dimensionality mismatch of active basal segments')
        self.assertTrue(np.all(self.btm.activeBasalSegments == 0), 'Active basal segments after reset non-zero')

        self.assertEqual(self.btm.activeApicalSegments.shape, (self.btm.maxSegmentsPerCell, self.btm.numberOfCells()),
                         'Dimensionality mismatch of active apical segments')
        self.assertTrue(np.all(self.btm.activeApicalSegments == 0), 'Active apical segments after reset non-zero')

        self.assertEqual(self.btm.apicalInput.shape, (self.btm.apicalInputSize, ),
                         'Dimensionality mismatch of apical input')
        self.assertTrue(np.all(self.btm.apicalInput == 0), 'Apical input after reset non-zero')

        self.assertEqual(self.btm.basalInput.shape, (self.btm.basalInputSize,),
                         'Dimensionality mismatch of basal input')
        self.assertTrue(np.all(self.btm.basalInput == 0), 'Basal input after reset non-zero')

        self.assertEqual(self.btm.activeCells.shape, (self.btm.numberOfCells(),),
                         'Dimensionality mismatch of active cells')
        self.assertTrue(np.all(self.btm.basalInput == 0), 'Active cells after reset non-zero')

    def test_depolarize_cells(self):
        basal_input = np.full(self.basal_input_size, 0.5)
        apical_input = np.full(self.apical_input_size, 0.2)

        self.btm.depolarizeCells(basal_input, apical_input)


if __name__ == '__main__':
    unittest.main()

