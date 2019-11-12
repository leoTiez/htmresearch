#!/usr/bin/python2.7
import unittest

# Import apical tiebreak bayesian tm
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), '../algorithms')))
import apical_tiebreak_bayesian_temporal_memory as btm


class BayesianTMTest(unittest.TestCase):
    def test_initialisation(self):
        pass

