# 1. Installation
Go to the doa directory where setup.py is located and type,

```
pip install .
```

We provide a complete example in the tutorials/examples.ipynb


# 2. Obtaining results
#### 2.1 Run models to get prediction errors for molecules

```
python scripts/run_model.py --config config.yaml --run-id 1
```

The main configurations are sotred in scripts/config/main.yaml. 

```
data-path: Path to the csv file containing smiles and molecular descriptors of the molecules.
n-runs: Number of models to train to get model predictions and associated errors.
```

The results will be saved to the folder specified by the save-path
