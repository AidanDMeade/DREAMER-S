# DREAMER-S: **D**eep Lea**R**ning-**E**nabled **A**ttention-based **M**ultiple-instance approaches with **E**xplainable **R**epresentations for **S**patial Biology

This repository contains the codes necessary to run the hyperparameter optimsition of a ResNet-inspired Linear MIL model (refer to the figure below). The pre-processing and explainable AI step are not included in this repository.

![Workflow](src/fig_workflow.png)

## ⚙️ Installation and use

### A. Test run training:

1. We use a training loop which is deployed within the BiospecML python module; install this module according to the instructions available at: `https://github.com/rafsanlab/BioSpecML`
2. Install other dependencies and modules according to the paper (i.e Pytorch etc.)
3. Clone this repo and download the chemical image data into a folder named `data/` in this repository.
4. Run `python run.py`

### B. For custom training using your own data:

If you have your own data, model, dataset class, training loop or even of you wish to choose which parameter to optimise, follow these steps after following the instructions in A (step 1-3) above.

1.  Place your .mat files in a folder named `data/` in this repository.
2.  Generate two dataframes (train and validation) containing a filename column and label column for your data.
3.  Find the mean and std from the training dataset and save as tensor `.pt` file.
4.  Save all the dataframes and tensor `.pt` files in `metadata/` folder.
5.  Edit `run.py` file for your custom parameter and set paths to the prepared files. Also set the label and filename column.
6.  Run `python run.py`.
7.  Training results will be in the `results/` folder (will be generated later).

Suggested edits:

- Set custom labels in `labels_dict` for custom dataset.
- Number of epochs (i.e 50, 100).
- Additonal hyperparameter (i.e weight decay, learning rate etc.)
- Your own custom model architecture.
- Your own `training_loop` with custom metrics i.e precision/recall

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17179431.svg)](https://doi.org/10.5281/zenodo.17179431)
