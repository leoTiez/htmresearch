#!/usr/bin/python2.7
import unittest
import numpy as np

# Import apical tiebreak bayesian tm
import htmresearch.algorithms.apical_tiebreak_bayesian_temporal_memory as btm


class BayesianTMTest(unittest.TestCase):
    def setUp(self):
        self.max_segments_per_cell = 255
        self.column_count = 5
        self.basal_input_size = 4
        self.apical_input_size = 6
        self.cells_per_column = 2
        self.min_threshold = 0.3
        self.basal_input_value = 0.5
        self.apical_input_value = 0.2
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
        basal_input = np.full(self.basal_input_size, self.basal_input_value)
        apical_input = np.full(self.apical_input_size, self.apical_input_value)

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

        self.btm.basalInput = np.full(self.basal_input_size, self.basal_input_value)
        self.btm.apicalInput = np.full(self.apical_input_size, self.apical_input_value)
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

        # Moving average weight
        self._movingAverageWeightsTests(
            self.btm.basalMovingAverages,
            active_columns,
            active_cells,
            self.basal_input_size
        )
        self._movingAverageWeightsTests(
            self.btm.apicalMovingAverages,
            active_columns,
            active_cells,
            self.apical_input_size
        )

        # Moving average bias
        self._movingAverageBiasTests(
            self.btm.basalMovingAveragesBias,
            active_columns,
            active_cells
        )
        self._movingAverageBiasTests(
            self.btm.apicalMovingAveragesBias,
            active_columns,
            active_cells
        )

        # Moving average input
        self._movingAverageInputTests(self.btm.basalMovingAverageInput, self.basal_input_value)
        self._movingAverageInputTests(self.btm.apicalMovingAverageInput, self.apical_input_value)

        # Weights
        self._movingAverageWeightsTests(self.btm.basalWeights, active_columns, active_cells, self.basal_input_size)
        self._movingAverageWeightsTests(self.btm.apicalWeights, active_columns, active_cells, self.apical_input_size)

        # Bias
        self.assertTrue(np.all(self.btm.basalBias == np.log(self.btm.basalMovingAveragesBias)),
                        'Basal bias does not match the logarithm of the moving average of the bias')
        self.assertTrue(np.all(self.btm.apicalBias == np.log(self.btm.apicalMovingAveragesBias)),
                        'Apical bias does not match the logarithm of the moving average of the bias')

    def _movingAverageWeightsTests(self, moving_average, active_columns, active_cells, input_size):
        self.assertTrue(
            np.all(
                moving_average.reshape((
                    self.max_segments_per_cell,
                    self.cells_per_column,
                    self.column_count,
                    input_size
                ))[0, :, active_columns, :][active_cells, :, :] > 0.0
            ), 'Not all weights to active segments have been updated'
        )

        self.assertTrue(
            np.all(
                moving_average.reshape((
                    self.max_segments_per_cell,
                    self.cells_per_column,
                    self.column_count,
                    input_size
                ))[1:, :, :, :] == 0.0
            ), 'Weights to inactive segments have been updated'
        )

    def _movingAverageBiasTests(self, moving_average_bias, active_columns, active_cells):
        self.assertTrue(
            np.all(
                moving_average_bias.reshape((
                    self.max_segments_per_cell,
                    self.cells_per_column,
                    self.column_count
                ))[0, :, active_columns][active_cells, :] > 0.0
            ), 'Not all biases to active segments have been updated'
        )

        self.assertTrue(
            np.all(
                moving_average_bias.reshape((
                    self.max_segments_per_cell,
                    self.cells_per_column,
                    self.column_count
                ))[1:, :, :] == 0.0
            ), 'Biases to inactive segments have been updated'
        )

    def _movingAverageInputTests(self, moving_average_input, input_value):
        moving_average_value = self.btm.learningRate * ((1 - self.btm.noise) * input_value + self.btm.noise)
        self.assertTrue(
            np.all(
                moving_average_input == moving_average_value
            ), 'Not all input moving averages have same value while having same input'
        )


if __name__ == '__main__':
    unittest.main()
