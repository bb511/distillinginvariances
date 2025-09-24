# ModelNet40 Experiments
---

The `configs` folder holds all the configuration files that reproduce every experiment contained in the [Understanding Knowledge Distillation by Transferring Symmetries]().

## Dependencies

Try to set up a conda environment by using the environment file in this directory.
```
conda env create --name [your_env_name] --file=conda_env.yml
```

If this does not work, install the following packages manually, with python 3.11 using
conda or pip:

```
- numpy
- wget
- h5py
- pytar
- pytorch (with the cuda version that works for you, see https://pytorch.org/get-started/locally/)
- deepspeed (only available through pip)
- torchinfo
- matplotlib
```

## Running trainings

To run a training, use, for example
```
python train.py --config configs/config_deepsinv_16cosnt.yml --device cuda:0
```

This will train an invariant deepsets model on the physics data downsampled to 16
constituents and initialisation seed = 42.
The hyperparameters of the model and of the training procedure are found in the
respective config file.

## Running distillations

Running distillation is similar to the trainings, but make sure the distillation
config file points to an already trained teacher! For example:
```
python distill.py --config configs/config_deepsinv_to_mlp_16const.yml --device cuda:0
```

which relies on the existence of a deespinv model at the location
`trained_models/deepsinv_16const/seed42`.
For different distillation temperatures, change the temperature entry in the config
file for the respective experiment.

## Validating the models

After training or distilling a model, you can obtain the metrics of interest for our
experiments by running, for example
```
python test.py --models_dir trained_models/deepsinv_16const --device cuda:0
```

