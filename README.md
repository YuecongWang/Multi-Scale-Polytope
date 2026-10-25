# Multi-scale Polytope Poisoning

A clean-label poisoning research project built on **Bullseye Polytope (BP)** and **Convex Polytope (CP)**. MSP uses features from several network stages to construct poisons, with pooled concatenation, flattened concatenation, and equally weighted layerwise objectives.

The repository provides a shared CIFAR-10 workflow for **linear transfer** and **full-parameter victim fine-tuning**. Pretraining and victim training stay matched across methods so that experiments measure the effect of the attack representation and loss.

[Chinese guide](README_zh.md) · [Script catalog](docs/SCRIPT_CATALOG.md) · [Parameters and protocol](docs/PARAMETERS_AND_PROTOCOL.md) · [Migration and validation](docs/MIGRATION.md) · [Model feature interfaces](models/README.md)

## Methods and training scenarios

Attack representation and victim training are independent choices:

| Setting | Values | Meaning |
| --- | --- | --- |
| `attack.representation` | `features` / `blocks` | Use the selected feature-mode objective, or the original BP/CP block objective |
| `victim.training` | `linear` / `end2end` | Update only the classifier head, or all pretrained victim parameters |
| `victim.model_mode` | `eval` / `train` | Control BatchNorm/dropout behavior during victim optimization; linear training requires `eval` |

**MSP supports both victim training scenarios with `attack.representation="features"`.** For example, `MSP-Flat` remains the method name when `victim.training` changes from `linear` to `end2end`. Full-parameter optimization in `eval` mode still computes gradients and updates the backbone; it keeps BatchNorm running statistics fixed.

| Method | Objective | Attack features |
| --- | --- | --- |
| CP-Original | Simplex-constrained convex-hull reconstruction | Original penultimate features |
| BP-Original | Normalized poison-centroid residual | Original penultimate features |
| MSP-Pool | BP centroid objective | Global-average-pooled features from three stages, concatenated |
| MSP-Flat | BP centroid objective | Flattened features from three stages, concatenated |
| MSP-Flat-Layerwise | Mean of per-layer normalized residuals | The same three stages, flattened separately |
| BP-3Layer-Pool-Control | Mean of per-layer normalized residuals | Three separately pooled stages |
| BP-E2E-Original / CP-E2E-Original | Original BP/CP block objective | Original `block=True` interface; compatibility method names |

The first six methods use `features` and can be evaluated with either linear or full-parameter victims. MSP does not require a block interface. All seven model families can serve as victims: ResNet, SENet, DPN, ResNeXt, DenseNet, GoogLeNet, and MobileNetV2.

The original `blocks` attack requires **proxy** models with a block interface: ResNet, ResNeXt, DPN, or SENet. Its diagnostic victim uses final features, so the victim does not need that interface. Use a separate configuration for this historical block baseline.

Here, end-to-end means fine-tuning all parameters of a **pretrained** victim. Poison construction still freezes proxies and optimizes pixels. The workflow does not differentiate through an unrolled victim-training procedure, and end-to-end does not mean training a randomly initialized victim from scratch.

## Repository layout

~~~text
configs/                         Linear and full-parameter smoke/benchmark configurations
prepare_data.py                  Build a documented split from official CIFAR-10 training data
workflow.py                      Plan, pretrain, craft, and evaluate
train_cifar10_models.py          Clean pretraining, resume, and checkpoint-only evaluation
craft_poisons_transfer.py        Staged CP/BP/MSP experiments
trainer.py                      Shared attack objectives and victim training
models/                         Classifiers and multistage feature interfaces
summarize_results.py             ASR, conditional ASR, and clean-accuracy summaries
tests/                          Planning, data contracts, statistics, and tensor regressions
verify_layerwise.py             Feature/gradient checks for seven real architectures
launch/  analysis/              Historical experiment wrappers and result analysis
Multi-Target-Mode/              Separate historical external-target experiments
runs/                           New experiment outputs; ignored by Git
~~~

## 1. Install the environment

Run the commands from the repository root. Install a matching torch/torchvision pair for your CPU or CUDA environment using the [PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies. `requirements.txt` specifies dependency ranges, rather than a fully pinned environment.

~~~powershell
conda create -n msp python=3.11 -y
conda activate msp
# Run the appropriate torch/torchvision installation command here.
python -m pip install -r requirements.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
~~~

Planning and dependency-free contract tests can run without PyTorch:

~~~powershell
python workflow.py --config configs/smoke.json --stage plan
python workflow.py --config configs/smoke_end2end.json --stage plan
python -m unittest discover -s tests -v
~~~

Real tensor tests explicitly skip when torch is unavailable. Use CUDA for substantial experiments; CPU is useful for small execution checks. Windows pretraining defaults to `workers=0`.

## 2. Prepare the data

If you already have the upstream split, place it at `datasets/CIFAR10_TRAIN_Split.pth` and retain its provenance. Matching a filename does not establish that two splits contain the same images.

Otherwise, create a new deterministic research split:

~~~powershell
python prepare_data.py --download --seed 2026
~~~

This writes `datasets/CIFAR10_TRAIN_Split.pth` and a companion JSON manifest containing the source indices, seed, and split SHA-256. Existing outputs are not overwritten.

| Data | Default size | Purpose |
| --- | --- | --- |
| `clean_train` | 4,800/class; 48,000 total | Clean proxy/victim pretraining |
| `others` | 200/class; 2,000 total | Victim fine-tuning, poison bases, and held-out attack targets |
| Official CIFAR-10 test set | 1,000/class; 10,000 total | Clean test accuracy |

This is a new split, not a claim to reproduce the authors' original split. Train compatible checkpoints on it when starting a new study.

The default victim set uses class positions 0–49 in `others`: 500 images total. Five poisons **replace their bases in place**, keeping the total at 500. Target index 0 selects class position 50; target index 49 selects position 99. These targets stay outside the victim-training window. Changes to `num_per_class` or `subset_group` must also keep target and base selection consistent.

## 3. Prepare clean checkpoints

Both victim scenarios share clean pretrained checkpoints. If suitable checkpoints already exist, verify the model names, filenames, and provenance, then skip pretraining.

The smoke configurations need a ResNet18 proxy and a ResNet50 victim:

~~~powershell
python workflow.py --config configs/smoke.json --stage pretrain
~~~

This is still **300 epochs of actual pretraining per model**. Smoke configurations shorten attack optimization and victim fine-tuning, not clean pretraining. Do not run this stage again merely to switch the victim training scenario.

Individual training, continuation, and checkpoint evaluation:

~~~powershell
python train_cifar10_models.py --net ResNet18 --epochs 300 --device cuda --output-dir model-chks
python train_cifar10_models.py --net ResNet18 --epochs 300 --device cuda --resume model-chks/ckpt-ResNet18-4800-latest.t7
python train_cifar10_models.py --net ResNet18 --device cuda --evaluate-only --resume model-chks/ckpt-ResNet18-4800.t7
python clean_test_evaluation.py --nets ResNet18 ResNet50 --device cuda --output runs/pretrained-clean-evaluation.json
~~~

The best checkpoint defaults to `model-chks/ckpt-ResNet18-4800.t7`. The `-latest` file also saves optimizer, scheduler, and RNG state. `--epochs` is the total epoch budget; `--resume` continues at the next epoch. Keep the model and training configuration compatible with the checkpoint.

## 4. Run one target through each scenario

The two smoke configurations use one target, one proxy, one victim, five methods, two poison iterations, and two victim-training epochs. They check execution and output consistency; their ASR is not a benchmark result.

### Linear transfer

~~~powershell
python verify_layerwise.py
python workflow.py --config configs/smoke.json --stage plan
python workflow.py --config configs/smoke.json --stage craft
python workflow.py --config configs/smoke.json --stage evaluate
python summarize_results.py runs/smoke-seed2026 --config configs/smoke.json
~~~

The effective settings are `features + linear + eval`. The backbone is frozen, its final features are cached, and only the classifier head is trained with Adam at learning rate 0.1.

### MSP with full-parameter victim fine-tuning

Use the same prepared data and clean checkpoints:

~~~powershell
python workflow.py --config configs/smoke_end2end.json --stage plan
python workflow.py --config configs/smoke_end2end.json --stage craft
python workflow.py --config configs/smoke_end2end.json --stage evaluate
python summarize_results.py runs/smoke-end2end-seed2026 --config configs/smoke_end2end.json
~~~

The effective settings are `features + end2end + eval`. The attack retains its selected MSP representation. Victim fine-tuning updates the backbone and classifier with Adam at learning rate 1e-4 and weight decay 0. The smoke budget is two epochs; the formal configuration uses 60.

In either scenario, `craft` only optimizes poison pixels. `evaluate` loads existing poisons, reloads the same victim checkpoint independently for clean and poisoned conditions, and uses paired training seeds and sample positions.

`workflow.py --stage all` runs **pretrain → craft → evaluate**, including the expensive pretraining stage. Data preparation and result summarization are separate commands. With existing checkpoints, use the staged commands above.

## 5. Run controlled benchmarks

Use [benchmark.json](configs/benchmark.json) for linear transfer or [benchmark_end2end.json](configs/benchmark_end2end.json) for full-parameter victims.

| Parameter | Shared attack settings | Linear victim | Full-parameter victim |
| --- | --- | --- | --- |
| Methods / targets | Six methods; targets 0–49 | Same | Same |
| Proxy ensemble | Six architectures × three checkpoint templates = 18 instances | Same | Same |
| Proxy dropout / repeats | 0.2, 0.25, 0.3 / `net_repeat=1` | Same | Same |
| Poison budget | Five images; 800 iterations; Adam lr=0.04; pixel-space L∞ ε=0.1 | Same | Same |
| Attack representation | `features` | Same | Same |
| Victims | DenseNet121 and ResNet18 | Same | Same |
| Victim updates / module mode | — | `linear / eval` | `end2end / eval` |
| Victim optimizer / lr / weight decay | — | Adam / 0.1 / 0 | Adam / 1e-4 / 0 |
| Victim epochs / batch / decay | — | 60 / 64 / epochs 30,45 | 60 / 64 / epochs 30,45 |
| Seed | 2026 | Same | Same |

The end-to-end presets explicitly specify their learning rate. When writing a custom CLI command, also pass the intended learning rate; selecting the training scenario alone is not a substitute for planning its optimizer settings.

Start with a copy of a formal configuration using `"targets": [0]` and a new pilot `output_root`. Complete one target at the formal 800-iteration/60-epoch budget before expanding.

Linear benchmark:

~~~powershell
python workflow.py --config configs/benchmark.json --stage plan
# Only if the required 18 proxy and two victim checkpoints are missing:
python workflow.py --config configs/benchmark.json --stage pretrain
python workflow.py --config configs/benchmark.json --stage craft
python workflow.py --config configs/benchmark.json --stage evaluate
python summarize_results.py runs/benchmark-seed2026 --config configs/benchmark.json
~~~

Full-parameter benchmark, reusing the same clean checkpoints:

~~~powershell
python workflow.py --config configs/benchmark_end2end.json --stage plan
python workflow.py --config configs/benchmark_end2end.json --stage craft
python workflow.py --config configs/benchmark_end2end.json --stage evaluate
python summarize_results.py runs/benchmark-end2end-seed2026 --config configs/benchmark_end2end.json
~~~

If starting directly with the full-parameter configuration and no checkpoints, run its `--stage pretrain` first. Each benchmark plans 300 poison-generation jobs and 1,200 victim-training runs: 300 jobs × two victims × clean/poisoned conditions.

Keep separate output roots for the two victim scenarios, and for changes to seed, epsilon, weights, iterations, or training settings. A recorded protocol mismatch prevents reuse of an incompatible run directory. Completed poison generation can be reused; explicitly running evaluation retrains the victims.

For repeatability, copy the configuration for several seeds, use a separate output root for each, and report per-seed results before aggregating. Keep method budgets matched within a scenario.

### Direct CLI and legacy compatibility

For one MSP-Flat full-parameter experiment with the default checkpoint filenames:

~~~powershell
python craft_poisons_transfer.py --stage all --device cuda --mode mean --feature-mode flat --attack-representation features --victim-training end2end --victim-model-mode eval --substitute-nets ResNet18 --subs-chk-name ckpt-%s-4800.t7 --subs-dp 0 --target-net ResNet50 --target-index 0 --poison-num 5 --poison-ites 800 --poison-epsilon 0.1 --retrain-opt adam --retrain-lr 1e-4 --retrain-wd 0 --retrain-epochs 60 --chk-path runs/direct-msp-flat-end2end
~~~

For linear transfer, select `--victim-training linear --victim-model-mode eval` and the planned classifier-head learning rate, such as 0.1.

The old `--end2end true` shorthand remains supported: with `feature-mode=original` it selects `blocks + end2end`; with an MSP feature mode it selects `features + end2end`. Prefer the three explicit options in new experiments. Conflicting legacy and explicit victim-training settings are rejected.

To study the original block attack, use `attack.representation="blocks"` with `BP-E2E-Original`/`CP-E2E-Original`, block-capable proxies, and an explicitly specified victim scenario. DenseNet, GoogLeNet, or MobileNetV2 may still be victims.

## 6. Outputs and metrics

~~~text
runs/benchmark-end2end-seed2026/
  MSP-Flat/
    flat/mean/800/0/
      protocol.json
      log.txt
      craft.json
      poison_00000.pth
      poison_00050.pth
      ...
      poison_00799.pth
      evaluation-DenseNet121-seed2026.json
      evaluation-ResNet18-seed2026.json
      evaluation.json
  summary.json
  summary.csv
~~~

Poison filenames use zero-based iteration indices: 800 updates end at `poison_00799.pth`. Snapshots are saved every 50 iterations and at the final iteration.

The run protocol uses schema version 2; configuration files still use version 1. The protocol records the method, attack representation, victim training and module mode, parameters, and data/checkpoint hashes. Generation records include time and peak GPU allocation. Evaluation records include base indices, measured perturbations, paired predictions, probabilities, test accuracy, trainable-parameter counts, and a sampled backbone parameter change. The backbone probe is a diagnostic, not a complete parameter-by-parameter audit. Use a new output root when migrating an older protocol.

| Metric | Meaning |
| --- | --- |
| ASR | Fraction of all targets predicted as the poison class by the poisoned victim |
| Conditional ASR | ASR among targets correctly classified after matched clean victim fine-tuning |
| Eligible targets | Conditional-ASR denominator; zero gives null/N/A |
| Clean accuracy drop | Clean-condition test accuracy minus poisoned-condition accuracy, in percentage points |
| Seconds / peak allocation | Recorded resource cost; GPU-memory fields are null for CPU runs |

The summarizer checks configured coverage and rejects duplicate identities, mixed training/data protocols, or inconsistent base selection. Incomplete coverage exits nonzero and is listed in `summary.json`; use `--allow-incomplete` explicitly to inspect a run in progress. Summarize linear and full-parameter runs separately.

## 7. Tests, historical results, and scope

~~~powershell
python -m unittest discover -s tests -v
python verify_layerwise.py
~~~

Contract tests exercise production selection/loading logic through small mocks. Tensor regressions require torch and report skips explicitly when it is absent. The real-architecture verifier covers seven architectures without data or checkpoints. Tests and historical CI checks do not establish the ASR of the current pipeline; validation details are recorded in [MIGRATION.md](docs/MIGRATION.md).

Earlier GitHub CI passed the contract/tensor checks and seven-architecture verification for the revision it tested. Historical end-to-end result folders are retained as evidence of earlier experiments; they do not establish that the new pipeline used the same data, checkpoints, hyperparameters, or execution path. New claims require running the documented configuration and retaining its provenance.

The runnable unified pipeline currently targets CIFAR-10 single-target experiments. Historical CIFAR-100 outputs are not a claim of strict CIFAR-100 reproduction by this pipeline. Availability experiments and external-car multi-target experiments use different protocols; see the [complete script catalog](docs/SCRIPT_CATALOG.md). Availability accuracy degradation is not targeted ASR, and directory names do not prove proxy/victim pretraining overlap.

The [archived GitHub introduction and visualizations](docs/LEGACY_GITHUB_README.md) retain the original remote figure links. The [upstream README](docs/UPSTREAM_README.md) preserves the original experiment description.

## Acknowledgments and citation

The implementation builds on [Bullseye Polytope](https://arxiv.org/abs/2005.00191), which in turn builds on [Convex Polytope](https://github.com/zhuchen03/ConvexPolytopePosioning/). Model implementations originate from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar). Please retain the upstream authors' credit and cite their work when using it.

MSP is this repository's multistage attack-feature extension; the documentation does not claim official paper reproduction or a measured performance improvement.

~~~bibtex
@article{aghakhani2020bullseye,
  title={Bullseye Polytope: A Scalable Clean-Label Poisoning Attack with Improved Transferability},
  author={Aghakhani, Hojjat and Meng, Dongyu and Wang, Yu-Xiang and Kruegel, Christopher and Vigna, Giovanni},
  journal={arXiv preprint arXiv:2005.00191},
  year={2020}
}
~~~
