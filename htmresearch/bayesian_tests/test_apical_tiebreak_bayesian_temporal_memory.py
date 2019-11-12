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
        self.max_segments_per_cell = 255
        self.column_count = 5
        self.basal_input_size = 4
        self.apical_input_size = 6
        self.cells_per_column = 2
        self.min_threshold = 0.3
        # Reduce network size
        self.btm = btm.ApicalTiebreakBayesianTemporalMemory(
            columnCount=self.column_count,
            basalInputSize=self.basal_input_size,
            apicalInputSize=self.apical_input_size,
            cellsPerColumn=self.cells_per_column,
            maxSegmentsPerCell=self.max_segments_per_cell,
            minThreshold = self.min_threshold
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

    def test_activate_cells(self):
        active_columns = np.asarray([1, 3, 4])
        all_columns = np.arange(0, self.column_count)
        inactive_columns = np.setdiff1d(all_columns, active_columns)
        active_cells = self.btm.predictedCells.copy().reshape(
            self.cells_per_column,
            self.column_count
        )
        active_cells[:, inactive_columns] = 0.0

        self.btm.activateCells(active_columns)

        self.assertTrue(
            np.all(
                active_cells[:, inactive_columns] == self.btm.activeCells.reshape(
                    self.cells_per_column,
                    self.column_count
                )[:, inactive_columns]
            ), 'Inactive cells should be set to zero')

        self.assertTrue(
            np.all(
                self.btm.activeCells.reshape(
                    self.cells_per_column,
                    self.column_count
                )[:, active_columns] >= self.min_threshold
            ), 'Active cells should be set to a value equal or above min threshold')

        active_cells = self.btm.activeCells.reshape(
            self.cells_per_column,
            self.column_count
        )[:, active_columns].argmax(axis=1)

        self._movingAverageWeightsTests(self.btm.basalMovingAverages, active_columns, active_cells)

    def _movingAverageWeightsTests(self, moving_average, active_columns, active_cells):
        self.assertTrue(
            np.all(
                moving_average.reshape((
                    self.max_segments_per_cell,
                    self.cells_per_column,
                    self.column_count,
                    self.basal_input_size
                ))[0, :, active_columns, :][active_cells, :, :] > 0.0
            )
        )

        self.assertTrue(
            np.all(
                moving_average.reshape((
                    self.max_segments_per_cell,
                    self.cells_per_column,
                    self.column_count,
                    self.basal_input_size
                ))[1:, :, :, :] == 0.0
            )
        )

if __name__ == '__main__':
    unittest.main()

