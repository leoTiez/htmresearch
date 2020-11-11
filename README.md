# Incorporating A Continuous Bayesian Learning Rule Into A Discrete Hierarchical Temporal Model

This repository contains the Python code that was used to extend Numenta's HTM approach
to a continuous stochastical network that embeds a Hebbian-like Bayesian learning rule
to adapt to changing input statistics. The project was conducted within the research
class DD2465 at KTH, and the report can be found [here](Numenta_BCPNN_Report.pdf).

The goal is to compare the approaches regarding their respective convergence behavior.
We hypothesised that the network will achieve similar results, while increasing
interpretability and reducing the number of model parameters that are needed to be tuned.

Unfortunately, we could verify our claim. The algorithm comes with the drawback of
large computational costs for both, memory and time consumption. Moreover, we could not
obtain a similar performance as the original model (measured in number of sensation-location
pairs needed to infer correctly an object).

Nevertheless, we are convinced that the work opens a new way to investigate thoroughly 
the dynamical behaviour of the network. Moreover, the output activity of the neurons of
the object representation can be seen as a measure of confidence. Thus, we believe
that with a larger time frame it is possible to tune the performance of the Bayesian
version to be similar to the original approach, while increasing interpretability.

The original README that was published by Numenta can be found [here](NumentaREADME.md)

## Installation
To install the project, change to the project directory and run 
```bash
pip install .
```  

## Execution
The first file for a large parameter search can be executed via
```bash
python experiments.py
```

The other file that creates the plots is run through the command
```bash
python tests/bayesian/bayesian_conergence_activity_plot.py
```

## Contribution
Please feel free to contribute the work of [Numenta](https://numenta.com/) and to our own
adaptions. Clone, fork and try. For hints and recommendations, don't hesitate to contact
us via email: llze@kth.se and heyder@kth.se. 