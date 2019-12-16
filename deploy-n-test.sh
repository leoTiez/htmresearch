#!/usr/bin/env bash
pip install .
echo "Updated environment. Run experiments"
echo "####################################"
python2.7 tests/bayesian/experiment_suite.py