# swav_transcriptomics
This repertory presents a python package and execution notebooks to show the benefits of a self-supervised pretraining (using SwAV procedure) on transcriptomics data (TCGA RNA-seq especially).

The SwAV procedure paper, by Mathilde Caron et al. : https://arxiv.org/abs/2006.09882

To download TCGA RNA-seq data : https://portal.gdc.cancer.gov/
## cancerclassification package
This package contains all the modules needed to run the full pipeline : preparing the data, training SwAV, running experiments.
- data.py : functions around the preprocessing of data and the preparation of the datasets
- early_stopping.py : simple class that implements early stopping of the training
- nn.py : functions around the training of a neural network on transcriptomics data for a classification task
- survival.py : functions around the training of a neural network on transcriptomics data for a survival analysis task
- swav.py : functions around the pretraining of a neural network on transcriptomics data with the SwAV procedure
## notebooks
Both files "demo_swav_microarray.ipynb" and "demo_swav_tcga.ipynb" present an example of code execution to train, finetune and evaluate the benefits of SWAV procedure on transcriptomics datasets.
