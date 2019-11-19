#!/usr/bin/env bash
pip install .
echo "Updated environment. Run experiments"
echo "####################################"
python2.7 tests/bayesian/bayesian_convergence_activity_minimal.py