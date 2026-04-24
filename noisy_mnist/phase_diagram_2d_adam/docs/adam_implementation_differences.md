# Adam Phase-Diagram Implementation Differences

This note explains why the two Adam cache families under [`noisy_mnist/phase_diagram_2d_adam`](/home/abhuiyan/nnet_error_project/noisy_mnist/phase_diagram_2d_adam) are not directly comparable as "the same experiment with more seeds."

Relevant code paths:
- Corrected Adam runner: [run_experiments_2d_phase_diagram_adam.py](/home/abhuiyan/nnet_error_project/noisy_mnist/phase_diagram_2d/run_experiments_2d_phase_diagram_adam.py)
- Legacy Adam training semantics reference: [nnet_models.py](/home/abhuiyan/nnet_error_project/noisy_mnist/src/nnet_models.py)
- Legacy Adam experiment driver reference: [run_experiments_noisy_training_data.py](/home/abhuiyan/nnet_error_project/noisy_mnist/scripts/run_experiments_noisy_training_data.py)
- SGD phase-diagram runner that the earlier Adam variant was derived from: [run_experiments_2d_phase_diagram.py](/home/abhuiyan/nnet_error_project/noisy_mnist/phase_diagram_2d/run_experiments_2d_phase_diagram.py)

## Caches Being Compared

### Corrected Adam caches
- [`phase2d_adam_mnist_balanced_perclass_noreplace_v3_r25_e20_p0.9000-0.9950_b16-54120_tpc500-5412_w512_lr0.001_wd0_loss-cross_entropy_logtr100_logte500`](/home/abhuiyan/nnet_error_project/noisy_mnist/phase_diagram_2d_adam/phase2d_adam_mnist_balanced_perclass_noreplace_v3_r25_e20_p0.9000-0.9950_b16-54120_tpc500-5412_w512_lr0.001_wd0_loss-cross_entropy_logtr100_logte500)
- [`phase2d_adam_mnist_balanced_perclass_noreplace_v3_r5_e20_p0.9000-0.9950_b16-54120_tpc500-5412_w512_lr0.001_wd0_loss-cross_entropy_logtr100_logte500`](/home/abhuiyan/nnet_error_project/noisy_mnist/phase_diagram_2d_adam/phase2d_adam_mnist_balanced_perclass_noreplace_v3_r5_e20_p0.9000-0.9950_b16-54120_tpc500-5412_w512_lr0.001_wd0_loss-cross_entropy_logtr100_logte500)

### Earlier Adam cache
- [`phase2d_adam_mnist_balanced_perclass_noreplace_v3_r5_e20_p0.9000-0.9950_b16-54120_tpc500-5412_w512_lr0.01_logtr100_logte500`](/home/abhuiyan/nnet_error_project/noisy_mnist/phase_diagram_2d_adam/phase2d_adam_mnist_balanced_perclass_noreplace_v3_r5_e20_p0.9000-0.9950_b16-54120_tpc500-5412_w512_lr0.01_logtr100_logte500)

## Short Version

The earlier Adam cache was effectively a patched SGD phase-diagram runner with `SGD` swapped to `Adam`. The corrected Adam runner was later rewritten to match the legacy Adam pipeline used elsewhere in the repo.

Because of that, the older Adam cache differs from the corrected Adam caches in:
- learning rate
- loss/optimizer metadata
- corruption and seeding path
- model/loss helper path
- step-count and epoch semantics

So the earlier Adam cache should not be treated as a clean baseline for comparison against either the corrected Adam caches or the SGD cache.

## Shared Experiment Structure

All of these phase-diagram caches still share the same top-level grid:
- balanced no-replacement MNIST
- `p` values from `0.9000` to `0.9950`
- batch sizes from `16` to `54120`
- train samples per class `500, 1000, 2000, 4000, 5412`
- width `512`
- reference epochs `20`

The differences are in the training implementation.

## Main Implementation Differences

### 1. Adam hyperparameters

The corrected Adam runner matches the legacy Adam defaults from [run_experiments_noisy_training_data.py](/home/abhuiyan/nnet_error_project/noisy_mnist/scripts/run_experiments_noisy_training_data.py):
- `learning_rate = 1e-3`
- `weight_decay = 0.0`
- `loss_type = "cross_entropy"`

The earlier Adam cache used:
- `learning_rate = 0.01`
- no recorded `weight_decay`
- no recorded `loss_type`

This alone is a major change in experiment definition.

### 2. Shared legacy helpers vs custom phase-diagram code

The corrected Adam runner imports and uses the shared helpers from [nnet_models.py](/home/abhuiyan/nnet_error_project/noisy_mnist/src/nnet_models.py):
- `build_model(...)`
- `compute_loss(...)`
- `CorruptedDataset(...)`
- `set_seed(...)`

The earlier Adam implementation followed the older custom phase-diagram path inherited from [run_experiments_2d_phase_diagram.py](/home/abhuiyan/nnet_error_project/noisy_mnist/phase_diagram_2d/run_experiments_2d_phase_diagram.py), which used:
- a local `SaltAndPepperDataset`
- a local hand-written MLP
- its own loss/model plumbing
- separate derived shuffle/noise seeds

### 3. RNG and corruption semantics

The corrected Adam runner uses one legacy-style run seed via `set_seed(spec.seed)` and the shared corruption path through `CorruptedDataset`.

The earlier Adam implementation used the same RNG structure as the SGD runner:
- separate `noise_seed`
- separate `shuffle_seed`
- local corruption dataset implementation

This means the stochastic path was not the same.

### 4. Training-loop semantics

The corrected Adam runner is epoch-based:
- `steps_per_epoch = ceil(N / effective_batch_size)`
- `num_steps = reference_epochs * steps_per_epoch`
- explicit `for epoch in range(reference_epochs)` training loop

The earlier Adam implementation used the same sample-budget loop as the SGD runner:
- compute `total_samples_seen = N * reference_epochs`
- compute `num_steps = ceil(total_samples_seen / effective_batch_size)`
- train with `while samples_seen < total_samples_seen`

These are not equivalent when `N` is not divisible by `batch_size`.

Example:
- with `N = 10000`, `B = 1024`, `reference_epochs = 20`
- corrected Adam: `steps_per_epoch = ceil(10000 / 1024) = 10`, so `num_steps = 200`
- earlier Adam: `num_steps = ceil(200000 / 1024) = 196`

So the old and corrected Adam runs do not even take the same number of optimizer steps at the same nominal setting.

### 5. Final train loss semantics

The corrected Adam runner records final train loss as the mean loss over the last epoch, matching the legacy Adam training style more closely.

The earlier Adam implementation inherited the phase-diagram runner's SGD-style bookkeeping, which was not aligned to the legacy Adam path.

### 6. Cache-key and metadata discipline

The corrected Adam runner encodes the important Adam details in the cache key:
- `lr0.001`
- `wd0`
- `loss-cross_entropy`

It also records these in config and row metadata:
- `optimizer = "adam"`
- `learning_rate = 0.001`
- `weight_decay = 0.0`
- `loss_type = "cross_entropy"`

The earlier Adam cache key only included `lr0.01`, and the run/summary metadata did not clearly record the full optimizer/loss definition.

## Practical Interpretation

The earlier Adam cache is best thought of as:
- an intermediate, incorrect Adam phase-diagram implementation
- not legacy-Adam-equivalent
- not directly comparable to the corrected Adam caches except as a historical artifact

The corrected Adam caches are the ones aligned to the legacy Adam code path in this repo.

## Comparison Table

| Feature | Earlier Adam cache | Corrected Adam caches |
| --- | --- | --- |
| Runner path | Adam-swapped phase-diagram code | Rewritten Adam runner |
| Default LR | `0.01` | `0.001` |
| Weight decay recorded | No | Yes, `0.0` |
| Loss type recorded | No | Yes, `cross_entropy` |
| Model builder | Local phase-diagram MLP | Shared `build_model(...)` |
| Loss computation | Local phase-diagram path | Shared `compute_loss(...)` |
| Corruption dataset | Local custom dataset | Shared `CorruptedDataset(...)` |
| RNG style | Separate derived seeds | Shared `set_seed(...)` |
| Step accounting | `ceil(total_samples_seen / B)` style | `reference_epochs * ceil(N / B)` |
| Metadata quality | Limited | Explicit optimizer/loss metadata |

## Recommended Usage

If you want Adam results that are consistent with the legacy Adam training setup in this repo, use the corrected Adam caches:
- the `lr0.001_wd0_loss-cross_entropy` cache family

If you need to compare Adam to SGD in a controlled way, compare:
- corrected Adam caches
- current SGD phase-diagram caches

But note that even that comparison still changes both:
- optimizer type
- learning-rate default

So it is not a "same hyperparameters, optimizer only" study unless you explicitly force the same LR in both runners.
