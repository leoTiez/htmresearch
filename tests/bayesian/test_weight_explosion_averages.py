import unittest
import numpy as np

# Test all cases of co-activity/activity and ensure that the weights do not explode
# Co-activity should never be higher than the Neuron-activities multiplied
class WeightExplosionTest(unittest.TestCase):
    def setUp(self):
        self.numSegments = 1
        self.cellCount = 1
        self.inputSize = 1
        self.learningRate = 0.1
        self.noise = 0.00001

    # test 4 cases:
    # no input & no bias
    # no input
    # no bias
    # input & bias (+ co-activity)

    def test_no_input_and_bias_activity(self):
        inputValues = np.array([0.0])
        segments = np.array([[0.0]])

        movingAverage = np.full((self.numSegments, self.cellCount, self.inputSize), self.noise**2) # 1.0 / (self.cellCount * self.inputSize))
        movingAverageBias = np.full((self.numSegments, self.cellCount), self.noise) # 1.0 /self. cellCount)
        movingAverageInput = np.full((self.inputSize), self.noise) # 1.0 / self.inputSize)

        for i in range(1000):

            # Updating moving average input activity
            noisy_input_vector = (1 - self.noise) * inputValues
            # Consider only active segments
            noisy_input_vector += self.noise
            movingAverageInput += self.learningRate * (
                    noisy_input_vector - movingAverageInput
            )

            # First update input values (includes learning rate)
            # Then use the probabilities of activity for movingAverage calculation
            # Instead of using 1 -> would lead to weight explosion, because we calculate weights based on MovingAverageInput
            inputProbabilities = inputValues
            inputProbabilities[inputValues.nonzero()] = movingAverageInput[inputValues.nonzero()]

            # Updating moving average weights to input
            noisy_connection_matrix = np.outer((1 - self.noise ** 2) * segments, inputProbabilities)
            # Consider only active segments
            noisy_connection_matrix += self.noise ** 2
            noisy_connection_matrix = noisy_connection_matrix.reshape(self.numSegments, self.cellCount, self.inputSize)
            movingAverage += self.learningRate * (
                    noisy_connection_matrix - movingAverage
            )

            # Updating moving average bias of each segment
            noisy_activation_vector = (1 - self.noise) * segments
            # Consider only active segments
            noisy_activation_vector += self.noise
            movingAverageBias += self.learningRate * (
                    noisy_activation_vector - movingAverageBias
            )

        # AFTER TEST
        weights = movingAverage / np.outer(
            movingAverageBias,
            movingAverageInput
        ).reshape(movingAverage.shape)
        # set division by zero to zero since this represents unused segments
        weights[np.isnan(weights)] = 0

        self.assertTrue(weights.max() < 1.1, "Weights exploded")


if __name__ == '__main__':
    unittest.main()
