# Tropical Cyclone Rapid Intensification Prediction

This folder contains the details necessary to run the experiments from the paper on real-world tropical cyclone (TC) rapid intensification (RI) data.
More details on the model, data, and preprocessing can be found in [this paper](https://journals.ametsoc.org/view/journals/wefo/40/7/WAF-D-24-0166.1.xml).
Given valid TC data as described in the Lagerquist et al. paper, [`generate_labels.py`](./generate_labels.py) contains all the code necessary to create ground truth labels for the task of predicting RI.
The [`data_utils.py`](./data_utils.py) and [`network_def.py`](./network_def.py) files include all the necessary imports (including custom dataset classes), while [`train_model.py`](./train_model.py) is the script that actually trains a model, given a config file.
[`apply_model.py`](./apply_model.py) again requires a model config file and applies the given model to a test or validation dataset and saves the outputs as a netcdf file.
Finally, [`evaluate_model.py`](./evaluate_model.py) computes a variety of statistics on the provided netcdf files, which are assumed to be the outputs of various runs of `apply_model.py`.

The [experiments](./experiments) folder contains subfolders with the config files, bash scripts, log files, and experimental results for the experiments performed.

The [`plot_generator_output.py`](./plot_generator_output.py) script is present to allow for the inspection of the custom dataset output to ensure that preprocessing and augmentation are being performed correctly.
