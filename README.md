# TDAF undergrad project

Topological data analysis features for molecular machine learning for undergrad project students. I've given you two example notebooks that between them cover the how to do both regression and classification as well as how to load in the standard deepchem datasets and how to load in the topological datasets. You will need to train models on one of the topological datasets I've given you and the standard deepchem version of that dataset.

I've included the topological data that you will need, it's take a while to create these features for the whole dataset so I have done that for you.

There are several notebooks to give you examples on how to write the code. You probably want to go through in order.

## How to install

Download the code, it's best to download the zip file and unpack it to somewhere sensible.

Install Anaconda 3, available from the internet or the uni software repository

To create a fresh conda environment with the packages you need and start jupyter.

First open anaconda prompt (Anaconda3) and navigate to the folder where you saved the files. Then run the code below. The first line sets up a new environment for you. Then you activate it to open that environment, then run jupyter notebook from there.

for windows:
```
conda env create --file .\requirements_win.yml
conda activate tdaf-tf2p7h2
jupyter notebook
```

This will install the GPU accelerated version of tensorflow. 

If you don't have a compatible GPU, then edit the appropriate `.yml` file in a text editor and change `tensorflow-gpu` to `tensorflow`.

## Notebooks and instructions

These are the example notebooks and suggested order.

1. Persistence homology for molecules: this contains demos of how the persistence homology stuff works and allows you to produce persistence diagrams for simple molecules. You can put in your own SMILES strings or load in a pdbfile. I would recommend also reading the giotta examples, especially the second one (see below)

2. Undergrad Project Tox21 Starter: This demonstrates code to train a classifier on standard featurisations in deepchem. Switching dataset and from classification to regression should be straight-forward.

3. Undergrad TDAF example: This demonstrates how to read in the topological datasets and how to build and train regression models to run on that dataset. You can switch dataset and switch from regression to classification.

I've also given you Do_QM7.py, Do_QM8.py and Do_BBBP.py: these build the topological datasets and take a while to run. To use, open an anaconda terminal and set up the environment, then type:
`python 3 Do_BBBP.py` and hit return, this should build the dataset for you, if you need it. You shouldn't need to run these.

Giotta-tda tutorials

[1] The features we are using are built using the code here:
https://giotto-ai.github.io/gtda-docs/latest/notebooks/vietoris_rips_quickstart.html

[2] And the method followed is here:
https://giotto-ai.github.io/gtda-docs/latest/notebooks/classifying_shapes.html