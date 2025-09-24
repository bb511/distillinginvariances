# MNIST experiments

This is the code to run the experiments to assess distillation for shift invariance.

## Installation & running

install requirements
```bash
cd "path_to_repo"
pip install -r requirements.txt
```

NOTE: installing deepspeed on a Windows environment will be cumbersome. Without deepspeed, you will still be able to replicate all shift invariance results, apart from running ```calc_flops.py```.

* **To exactly replicate the shift invariance experiments:** run the script ```experiments.py``` (without any arguments). This script will produce a ```solutions_processed.xlsx``` excel file with all metrics and sensitivities (max value over all random seeds - min value over all random seeds) for each metric.
* In ```experiments.py``` the partial results of the script are saved in case of script failure (or exit through ctrl+C) in the file ```solutions_backup.xlsx```. To use those partial results and bring them in the format of ```solutions_processed.xlsx``` you can take a look at the notebook ```processing_results.ipynb```.
* Run ```calc_flops.py``` to calculate the flops for each model. 






