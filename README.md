# An Efficient 3D Gaussian Splatting for Underwater 3D Reconstruction

**Muhammad Irham Dinan · BINUS University**

Research implementation for efficient novel-view synthesis in static underwater scenes. The code adapts SeaSplat to reduce the number of stored Gaussians and the storage required for selected attributes. The study evaluates how these changes affect stored model size and agreement with held-out underwater images.

## Method

| Component | Implementation | Role |
| --- | --- | --- |
| **M1: Dense correspondence initialization** | [`source/roma_init.py`](implementation/source/roma_init.py) | Triangulates RoMa matches to initialize a dense point cloud. M1 disables Gaussian cloning and splitting during training; pruning remains active. |
| **M2: Importance-based population simplification** | [`source/simplify.py`](implementation/source/simplify.py) | Samples and prunes Gaussians by importance, then fits the water parameters with the retained Gaussians fixed after each selection event. |
| **M3: Attribute vector quantization** | [`source/quantize.py`](implementation/source/quantize.py) and [`source/storage.py`](implementation/source/storage.py) | Encodes DC appearance, log-scale, and rotation quaternions with separate codebooks. Positions and opacity logits remain continuous. |

The [experiment configuration](implementation/configs/cells.json) defines A0 as the internal SeaSplat control, A1–A3 as the individual mechanisms, A4–A6 as the pairwise combinations, and A7 as the full combination. A0D is a separate gradient-routing diagnostic. SS is a separately trained upstream SeaSplat reference, evaluated outside the factorial design.

## Installation

Use Linux with an NVIDIA GPU, a CUDA toolkit and host compiler compatible with your PyTorch build, and matching CUDA-enabled PyTorch and torchvision. The campaign used an NVIDIA A100. Install COLMAP if you need to undistort the original inputs.

Run from the repository root:

```bash
bash implementation/tools/setup_colab.sh
cd implementation
python -m tools.verify_rasterizer
```

The setup script builds the bundled rasterizer and nearest-neighbor extensions against the active PyTorch environment. The default architecture targets the A100. If you use another GPU, set `TORCH_CUDA_ARCH_LIST` for that device and record the hardware change. For M1, setup installs RoMa. Set `ROMA_INSTALL_SPEC` to a commit-qualified Git URL if you need to hold its version fixed. RoMa and LPIPS may download weights at first use.

## Data preparation

Obtain the images and COLMAP reconstructions from [SeaThru-NeRF](https://sea-thru-nerf.github.io/). The experimental scene directory keys are `Curasao`, `IUI3-RedSea`, `JapaneseGradens-RedSea`, and `Panama`. Keep the literal keys when running the configured campaign. Each prepared scene needs undistorted images and the corresponding `sparse/0` camera model.

Example for Curaçao, run from `implementation/`. Set the three paths to locations on your machine before running:

```bash
export RAW_ROOT=/absolute/path/to/SeaThru-NeRF
export DATA_ROOT=/absolute/path/to/prepared-scenes
export RUN_ROOT=/absolute/path/to/new-run-directory

python -m source.undistort \
  --source "$RAW_ROOT/Curasao" --output "$DATA_ROOT/Curasao"

mkdir -p "$RUN_ROOT/dense"
python -m source.roma_init \
  --source_path "$DATA_ROOT/Curasao" --images images \
  --preset dense --seed 0 --output "$RUN_ROOT/dense/Curasao.ply"
```

Repeat preparation for the other scenes when running the full design. Dense clouds are required for A1, A4, A5, and A7. If your raw image directory is named differently, pass its name to the undistortion command with `--images`.

## Training and evaluation

Example single-scene runs, from `implementation/`:

```bash
python train.py --cell A0 --seed 0 --images images \
  -s "$DATA_ROOT/Curasao" --model_path "$RUN_ROOT/runs/A0/Curasao/s0"

python train.py --cell A7 --seed 0 --images images \
  -s "$DATA_ROOT/Curasao" --model_path "$RUN_ROOT/runs/A7/Curasao/s0" \
  --pcd_path "$RUN_ROOT/dense/Curasao.ply" --n_bud 200000
```

The September configuration uses 30,000 nominal iterations, M2 selection at 15,000 and 20,000, a first-event budget of 200,000 primitives, a second-event cumulative-importance threshold of 0.99, and 200 medium-only continuation passes after each event. M3 uses 4,096 requested codebook entries per group and activates after iteration 22,000. Record the resolved arguments; explicit CLI values override configuration defaults.

For the full campaign, prepare all four scenes and a separate [upstream SeaSplat checkout](https://github.com/dxyang/seasplat) with its extensions built. The configuration declares `ddc6259` as the upstream reference. Record the actual commit and resolved environment before running. Use a fresh output directory:

```bash
export SS_REPO=/absolute/path/to/upstream-seasplat
python -m tools.run_ledger init --output_root "$RUN_ROOT"
python -m tools.run_queue --output_root "$RUN_ROOT" \
  --data_root "$DATA_ROOT" --ref_repo "$SS_REPO" --dry_run
python -m tools.run_queue --output_root "$RUN_ROOT" \
  --data_root "$DATA_ROOT" --ref_repo "$SS_REPO"

mkdir -p "$RUN_ROOT/analysis"
python -m tools.collect_results --output_root "$RUN_ROOT" --split Test
python -m tools.analyse --output_root "$RUN_ROOT" --split Test \
  --weighting unweighted --json "$RUN_ROOT/analysis/analysis_unweighted.json"
```

The September design records three repeats per configuration and scene: 96 internal factorial runs, 12 SS reference runs, and 12 A0D runs. Evaluation uses pooled PSNR, SSIM, and LPIPS on held-out underwater views. Report final Gaussian count and stored model size separately. The Gaussian color-pass profiler does not measure end-to-end underwater rendering, and saved size does not measure memory after loading. The supplied code and configuration support new runs; matching archived measurements requires the original inputs, recorded environment, and run manifests.

## CPU checks

Run from `implementation/`. These checks use synthetic fixtures and do not need the dataset or CUDA:

```bash
python -m tools.verify_analysis
python -m tools.verify_collect_results
python -m tools.verify_ledger
python -m tools.verify_config_layer
python -m tools.verify_measure_reference
python -m tools.verify_undistort
```

The real-scene undistortion check is optional. Set `E3DGSUW_DATASET="$RAW_ROOT"` to include it.

## Citation and attribution

If you use this implementation, cite the thesis and identify the repository URL and commit in your artifact record.

```bibtex
@mastersthesis{dinan2026efficient,
  author = {Muhammad Irham Dinan},
  title  = {An Efficient 3D Gaussian Splatting for Underwater 3D Reconstruction},
  school = {BINUS University},
  year   = {2026}
}
```

The implementation builds on [SeaSplat](https://doi.org/10.1109/ICRA55743.2025.11128502) and [3D Gaussian Splatting](https://doi.org/10.1145/3592433). The adaptations draw on [EDGS](https://arxiv.org/abs/2504.13204v1), [RoMa](https://doi.org/10.1109/CVPR52733.2024.01871), [Mini-Splatting](https://doi.org/10.1007/978-3-031-72980-5_10), and [CompGS](https://doi.org/10.1007/978-3-031-73411-3_19). The dataset is described by [Levy et al.](https://doi.org/10.1109/CVPR52729.2023.00014). See the [inherited implementation license](implementation/LICENSE.md), the [rasterizer notice](implementation/third_party/diff-gaussian-rasterization_ms/LICENSE.md), the [nearest-neighbor extension notice](implementation/third_party/simple-knn/LICENSE.md), and the [GLM notice](implementation/third_party/diff-gaussian-rasterization_ms/third_party/glm/copying.txt).

This source distribution was prepared from research repository commit `2a99449d4e2be339680afd054a14c56f12930ef1`. It excludes manuscripts, presentation materials, datasets, weights, and experiment outputs.
