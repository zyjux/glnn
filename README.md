# GLNN

Repository for work on Group Lipschitz Neural Networks (GLNNs) by Emily J. King, Dustin Mixon, Michael Perlmutter, and Lander Ver Hoef.
Prior versions of this work used the "Group Lipschitz Pooling" or "GLP" name and acronym, but to be more accurate (and avoid an acronym now commonly used outside of mathematics) we are now using GLNN.
However, the acronym GLP is used often within this package and should be viewed as synonymous with GLNN wherever it occurs.

# Structure of this repository

This repository is organized into three main folders.

* The [`verifying_continuity`](./verifying_continuity) folder contains code, scripts, and experimental results relating to verifying that the $\Psi$ and $\Xi$ operators are well-behaved.
* Second, the [`synthetic_data`](./synthetic_data) folder contains everything necessary to show that the GLNN performs better than CNNs on smaller datasets, including synthetic data generation scripts, model definition and training code, and experimental scripts and results.
* Finally, the [`glp_ri`](./glp_ri) folder contains the relevant code for training the GLNN and CNN networks on tropical cyclone (TC) rapid intensification (RI) data, as well as the trained model files and results.
More data on those data can be found in [this paper](https://journals.ametsoc.org/view/journals/wefo/40/7/WAF-D-24-0166.1.xml).

[`environment.yml`](./environment.yml) contains a listing of all the packages needed for this repo.
Installation was generally performed via pip, and the majority of the work was performed using PyTorch version 2.11.0 with CUDA version 13.3.
