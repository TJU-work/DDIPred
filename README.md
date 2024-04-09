## Requirements
- Python 3.6.13
- tensorflow-gpu 1.14.0
- numpy 1.16.4
- keras 2.3.0
- pandas 1.1.5
- scikit-learn 0.22.2
- dgl 0.9.0
- dgllife 0.3.0
- torch 1.9.0
- rdkit 2020.09.1.0 

## Installation
We suggest using the conda environment to install dependencies.
To install the redkit tool correctly,
please use the conda command:
```
    conda install -c rdkit rdkit
```
Other dependencies can be installed by the command:
```
    pip install -r requirements.txt
```

## File description

#### MSKG-DDI_Multi
This folder contains the model for multi-classification problem and dataset.

#### MSKG-DDI_Binary
This folder contains the model for binary-classification problem and dataset.

## Usage
To run the code, call the following command:
```
    python run.py
```

To prepare drug's smiles, first set your file to "data/smile",
name it "smilies.csv" and call the following command:
```
    python smiles.py
```

## Dataset
All data of datasets for __MSKG-DDI_Multi__ and __MSKG-DDI_Binary__  are stored in the "raw_data" folder
and are named by the names of datasets.
To run a custom dataset, please create the following files:
1. "approved_example.txt" is list of all drug interactions
2. "entity2id.txt" list of all entities
3. "train2id.txt" KG of drugs
4. "smiles.npy" pretrained smiles of drugs

