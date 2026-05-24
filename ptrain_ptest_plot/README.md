# p_train / p_test MNIST replacement-noise experiment

This folder contains the code and raw results for the MNIST replacement-noise robustness experiment.
MNIST train/test files are not included; the scripts use torchvision MNIST and will download/load MNIST under `./data`.

## Experiment

- Model: MLP, ReLU, 3 hidden layers, width 128
- Optimizer: Adam
- Epochs: 20
- Training data: full MNIST training set with replacement noise applied to every sample at `p_train`
- Test data: fixed noisy MNIST test sets generated for each `p_test`
- Grid: `p_train = np.linspace(0, 1, 21)`, `p_test = np.linspace(0, 1, 81)`
- Repeats: 20 independent trainings per `p_train`

## Contents

- `scripts/train_noisy_mnist_ptrain_ptest.py`: training/evaluation script
- `scripts/launch_dual_gpu_ptrain_ptest.sh`: dual-GPU launcher used for the run
- `scripts/plot_noisy_mnist_ptrain_ptest.py`: plots test accuracy vs `p_test`, colored by `p_train`
- `scripts/plot_auc_ptrain_ptest.py`: computes and plots AUC over `p_test` vs `p_train`
- `data/ptrain_ptest_full_merged.csv`: merged raw test loss/accuracy results
- `data/ptrain_ptest_full_auc_summary.csv`: AUC mean/stderr by `p_train`
- `figures/`: generated PNG/PDF plots

## Recreate plots from raw CSV

From this folder:

```bash
conda run -n pytorch python scripts/plot_noisy_mnist_ptrain_ptest.py \
  --csv data/ptrain_ptest_full_merged.csv \
  --output-png figures/ptrain_ptest_full_accuracy.png \
  --output-pdf figures/ptrain_ptest_full_accuracy.pdf

conda run -n pytorch python scripts/plot_auc_ptrain_ptest.py \
  --csv data/ptrain_ptest_full_merged.csv \
  --summary-csv data/ptrain_ptest_full_auc_summary.csv \
  --output-png figures/ptrain_ptest_full_auc_vs_ptrain.png \
  --output-pdf figures/ptrain_ptest_full_auc_vs_ptrain.pdf
```

## Best AUC result

The AUC analysis maximized at approximately:

```text
p_train = 0.65
mean AUC = 0.777689
stderr = 0.000543
num_runs = 20
```
