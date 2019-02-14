# kaic-analysis

This repository accompanies the manuscript Marsland, Cui and Horowitz "The Thermodynamic Uncertainty Relation in Biochemical Oscillations" (2018, https://arxiv.org/abs/1901.00548). It includes three main items:

- **KMC_KaiC_rev2** contains the C++ code for the thermodynamically consistent computational model of the KaiABC oscillator, as described in Paijmans, et al. (2017, https://doi.org/10.1371/journal.pcbi.1005415), with some small adjustments as described in the Readme file contained in that folder. The Readme also has instructions for installing and running the code (requires scrolling down past the changelog).
- **kaic_analysis** contains a Python package with functions for simulating and analyzing the toy oscillator model developed in Marsland et al. (2018), and for loading and analyzing the data produced by the KaiABC model. To install the package, navigate to this folder in a Linux terminal and type `pip install -e .`. The `-e` makes sure any edits are updated in the appropriate place in your Python library.
- **data** contains the simulation data used in Marsland et al. (2018), along with a Jupyter Notebook for loading and analyzing the data, and making all the plots in the paper.
