# Noisy neural networks

Research code for studying how neural networks learn from corrupted inputs, and
how their predictions at high noise relate to simple class-centroid models.
The main experiments train image classifiers across noise levels, compare
network logits and accuracy with centroid predictions, and measure how
signal-to-noise ratio (SNR) changes with training-set size and input dimension.
Most experiments use MNIST, Fashion-MNIST, and KMNIST; extensions cover CIFAR,
tabular data, and ECG data.

This is a collection of experiment scripts, rather than an installable Python
package. Configuration lives in command-line arguments, environment variables,
or a script's `USER INPUTS` / manual-configuration block, depending on the runner.

## Environment and data

Run the commands below from the repository root in an existing Python 3.10+
environment. The core experiments require `torch`, `torchvision`, `numpy`,
`scipy`, `pandas`, `matplotlib`, `scikit-learn`, `Pillow`, and `tqdm`.
Notebooks additionally need Jupyter; the optional NTK code in `modules/ntk.py`
needs JAX and `neural_tangents`. There is no pinned dependency manifest.

Check your interpreter before launching a run:

```bash
python -c 'import torch, torchvision, numpy, scipy, pandas, matplotlib, sklearn, PIL, tqdm; print("Imports OK; CUDA:", torch.cuda.is_available())'
```

Use an environment with compatible PyTorch and torchvision builds. CPU runs
are supported, but the nested-ensemble trainers require `ALLOW_CPU = True`
when neither CUDA nor MPS is available. For headless plotting, set
`MPLBACKEND=Agg`.

The torchvision-based runners download datasets when needed. The nested-ensemble
trainers instead read existing IDX files. To prepare a shared MNIST-family cache:

```bash
python - <<'PY'
from torchvision import datasets
root = "noisy_mnist_asad/data/raw"
for dataset in (datasets.MNIST, datasets.FashionMNIST, datasets.KMNIST):
    for train in (True, False):
        dataset(root=root, train=train, download=True)
PY
```

This creates `MNIST/raw/`, `FashionMNIST/raw/`, and `KMNIST/raw/` below the
specified root. Dataset caches and generated results may be absent in a fresh
checkout.

## 1. Quick start: separate training and test noise

`ptrain_ptest_plot/scripts/train_noisy_mnist_ptrain_ptest.py` trains a ReLU MLP
on MNIST with replacement noise at `p_train`, then evaluates it on fixed noisy
test sets at each `p_test`. Training corruption is sampled on access; test
corruption is fixed so models can be compared on the same inputs.

This small CPU run checks the training and plotting workflow without launching
the full experiment:

```bash
PYTHONPATH="$PWD/noisy_mnist_asad${PYTHONPATH:+:$PYTHONPATH}" \
python ptrain_ptest_plot/scripts/train_noisy_mnist_ptrain_ptest.py \
  --p-train-values 0,0.5 --p-test-values 0,0.5,1 \
  --num-runs 1 --epochs 1 \
  --max-train-samples 1000 --max-test-samples 500 \
  --results-path ptrain_ptest_plot/data/smoke.csv

MPLBACKEND=Agg python ptrain_ptest_plot/scripts/plot_noisy_mnist_ptrain_ptest.py \
  --csv ptrain_ptest_plot/data/smoke.csv \
  --output-png ptrain_ptest_plot/figures/smoke.png
```

`PYTHONPATH` makes the shared `src.nnet_models` module available. This runner
stores MNIST in `./data` relative to the working directory, separately from the
shared cache above. The smoke run is a functionality check, not a scientific result.

For the full default grid (21 training-noise levels, 81 test-noise levels,
20 repeats, and 20 epochs), run:

```bash
PYTHONPATH="$PWD/noisy_mnist_asad${PYTHONPATH:+:$PYTHONPATH}" \
python ptrain_ptest_plot/scripts/train_noisy_mnist_ptrain_ptest.py \
  --use-cuda --results-path ptrain_ptest_plot/data/ptrain_ptest_run.csv
```

Omit `--use-cuda` for CPU. The CSV contains test loss and accuracy per training
run and test-noise level. See [the experiment README](ptrain_ptest_plot/README.md)
for plots from existing results and area-under-the-curve analysis.

## 2. Three-dataset corruption curves

The main runner is
`noisy_mnist_asad/corruption_curve_replacement/run_figure_corruption_curve_three_datasets.py`.
It trains MLPs on fixed corrupted training examples and evaluates on clean test
images. Defaults are MNIST/KMNIST/Fashion-MNIST, three hidden layers of width 128,
ReLU, Adam, 11 noise levels, and 10 repeats. A calibration run determines the
epoch budget, with a default maximum of 200 epochs.

Before running:

- The default balanced-subset protocol needs `mnist_balanced_indices.json`,
  `kmnist_balanced_indices.json`, and `fmnist_balanced_indices.json` in
  `noisy_mnist_asad/data/external/balanced_indices/`. Each contains
  `train_indices` and `test_indices`. These are separate artifacts; downloading
  the images does not create them. To explore without those files, set
  `use_balanced_subsets = False` in `main()`; this changes the sampling protocol.
- Override the CPU affinity list: the checked-in default is cores 70–90.
  On Linux, `python -c 'import os; print(sorted(os.sched_getaffinity(0)))'`
  shows the cores available to the current process.
- Architecture, datasets, and `use_cuda` are manual settings in `main()`.
  CUDA requires a single worker, selected by a single CPU in the affinity list.

For example, after choosing a valid CPU ID (replace `0` below if necessary):

```bash
NOISY_MNIST_THREE_DATASETS_CPU_CORES=0 \
NOISY_MNIST_THREE_DATASETS_DATA_WORKERS=0 \
NOISY_MNIST_THREE_DATASETS_P_VALUES=0,0.5,1 \
NOISY_MNIST_THREE_DATASETS_REPEATS=1 \
NOISY_MNIST_THREE_DATASETS_EPOCHS=1 \
NOISY_MNIST_THREE_DATASETS_EPOCH_STEP=1 \
NOISY_MNIST_THREE_DATASETS_MAX_TRAIN_SAMPLES=1000 \
NOISY_MNIST_THREE_DATASETS_SUFFIX=smoke \
MPLBACKEND=Agg \
python noisy_mnist_asad/corruption_curve_replacement/run_figure_corruption_curve_three_datasets.py
```

For a full sweep, remove the overrides for noise levels, repeats, epochs,
epoch step, sample limit, and suffix; retain CPU settings appropriate for your
machine. Each selected CPU creates one worker.

Related runners reuse this pipeline and environment-variable prefix:

| Noise model | Runner under `noisy_mnist_asad/` | Meaning of `p` |
| --- | --- | --- |
| Uniform replacement | `corruption_curve_replacement/run_figure_corruption_curve_three_datasets.py` | Probability of replacing a pixel with uniform noise; 0 is clean |
| Clipped-Gaussian replacement | `corruption_curve_clipped_gaussian/run_figure_corruption_curve_three_datasets_clipped_gaussian.py` | Probability of replacing a pixel with clipped Gaussian noise; 0 is clean |
| Gaussian mixture | `corruption_curve_additive_gaussian/run_figure_corruption_curve_three_datasets_additive_gaussian.py` | **Signal** weight in `clip(p*x + (1-p)*noise, 0, 1)`; 1 is clean |

Each experiment writes per-run and summary CSVs, per-epoch training losses,
ensemble accuracies, and runtime logs under its own `data/` folder,
with PDF plots and embedded metadata under `figures/`. These runners change the working directory to
`noisy_mnist_asad/`; use absolute paths for custom output roots.

## 3. Nested ensembles, centroid predictions, and SNR

`only_code_06.26_justin/` contains four related experiment families:

| Directory | Purpose |
| --- | --- |
| `logit_comparison/` | Fit centroid-based predictions to initialization-averaged network logits |
| `test_accuracy_comparison/` | Compare fitted noisy-centroid and empirical network test accuracy across noise levels |
| `snr/` | Vary training size and image resolution to study accuracy and SNR scaling |
| `activation_comparison/` | Compare activation functions using saved test logits |

Each contains `train_noisy_mlp_ensemble.py`, `sweep.py`, and a local `modules/`
directory. The trainer creates **K independent noisy training datasets**, then
trains **M independently initialized networks on each dataset**, for K × M
models. Evaluation uses clean test inputs. Images are centered and normalized
to unit RMS. Here Gaussian mixing uses `(1-p)*x + p*noise`, so larger `p` means
more noise, unlike the three-dataset Gaussian-mixture wrapper above.

### Train and inspect one ensemble

Edit the `USER INPUTS` block in
`only_code_06.26_justin/logit_comparison/train_noisy_mlp_ensemble.py`:

- Set `DATA_ROOT` to the **absolute path** of the shared IDX cache prepared above.
- Choose `DATASET`, `NOISE_TYPE`, `NOISE_PROBABILITY`, `TRAIN_SIZE`, `IMG_HW`,
  `ACTIVATION`, `DEPTH`, `WIDTH`, and training settings. In this trainer `DEPTH`
  counts linear layers, including the output layer; the corruption-curve runner
  counts hidden layers.
- For a small check, use `NUM_NOISE_DATASETS = 2`,
  `NUM_MODELS_PER_NOISE_DATASET = 2`, `NUM_MODELS = 4`, `TRAIN_SIZE = 1000`,
  `WIDTH = 32`, and `EPOCHS = 1`. Set `ALLOW_CPU = True` for CPU operation.
- Keep `SAVE_TEST_OUTPUTS = True` for downstream analyses, and choose a distinct
  `OUTPUT_ROOT` for each campaign.

```bash
python only_code_06.26_justin/logit_comparison/train_noisy_mlp_ensemble.py
```

Relative data/output roots resolve beside the trainer. The printed output
directory contains `metadata.json` (configuration, seeds, and selected indices),
`training_summary.json`, `model*.pth` checkpoints, and `test_outputs.csv.gz`
(one row per model and test example).

Use that directory explicitly; several analysis scripts otherwise default to
paths on the original author's machine:

```bash
ENSEMBLE_DIR=/absolute/path/to/the/printed/mlp_ensemble__directory
python only_code_06.26_justin/logit_comparison/compare_centroid_logits_subset.py \
  --ensemble-dir "$ENSEMBLE_DIR" --max-test-examples 1000
MPLBACKEND=Agg python only_code_06.26_justin/logit_comparison/plot_centroid_scatter.py \
  --input-csv "$ENSEMBLE_DIR/centroid_comparison_test1000.csv.gz"
```

### Run sweeps and downstream analyses

Edit the selected family's `sweep.py` configuration independently of the
single-ensemble trainer. Set `DRY_RUN = True` first to inspect the Cartesian
product and output locations, then set it to `False` to train:

```bash
python only_code_06.26_justin/logit_comparison/sweep.py
```

Check K, M, and grid sizes before launching: checked-in settings can request
thousands of networks and large compressed logit files. `SKIP_COMPLETED` skips
finished ensembles; partial ensembles are not automatically resumed.

For test-accuracy comparison, set `ROOT_DIR` in `logit_comparison/extract_full_fit.py`
to the calibration ensemble root and run it to create `centroid_fit_summary.csv`.
Then run `test_accuracy_comparison/eval_noisy_centroid_and_network_accuracy.py`
with explicit `--ensemble-root`, `--fit-csv`, and `--output-dir`; use its `--help`
to match dataset, noise type, architecture, and fit probability to your runs.
For activation comparison, run
`activation_comparison/evaluate_activation_comparison_test_accuracy.py`
with `--ensemble-root` pointing at that family's completed sweep.

For SNR analysis of existing nested ensembles:

```bash
MPLBACKEND=Agg python only_plotting_09.11.26/snr/analyze_nested_snr_scaling.py \
  --root /absolute/path/to/trained_mlp_nested_ensembles \
  --num-test-points 500 --output true_class
```

This averages logits over initializations, then computes squared mean divided
by variance across noisy datasets. It requires at least two noisy datasets per
ensemble and writes tables and plots beneath the ensemble root's `snr_scaling/`
directory by default. Vary `TRAIN_SIZES` and `IMG_HWS` in the SNR sweep to study
training-size and dimension scaling.

## Other experiments and repository layout

- [CIFAR CNN experiments](noisy_mnist_asad/noisy_CNNs/README.md) and
  [vision transformers](noisy_mnist_asad/Noisy%20ViT/README.md).
- [High-noise effective-model validation](noisy_mnist_asad/high_noise_effective_model_validation/README.md),
  [logit collapse](noisy_mnist_asad/high_noise_logit_collapse/README.md), and
  [normalization ablations](noisy_mnist_asad/train_test_normalization_ablation/README.md).
- `noisy_mnist_asad/higgs_gaussian_noise/` and `ptbxl_gaussian_noise/` contain
  tabular/ECG sweep entry points; inspect their `--help` for dataset and cache options.
- `noisy_mnist_asad/src/` holds shared models, corruption functions, and data
  helpers; `noisy_mnist_asad/tests/` contains their tests.
- `only_plotting_*/`, `plotting/`, `SNR_N_scaling_plotting/`, and
  `SNR_d_scaling_plotting/` contain figure and aggregation scripts. Check input
  paths before reusing them with a new result set.
- `manuscript/` contains manuscript sources and
  [revision/reproduction workflows](manuscript/revision_review/README.md).

Keep the configuration, seeds, preprocessing, and data-selection metadata with
each result. Noise conventions, depth conventions, and normalization differ
between experiment families, so matching only `p` and width is insufficient to
reproduce a run. Use new output paths or suffixes for exploratory runs: some
runners overwrite files with matching names.
