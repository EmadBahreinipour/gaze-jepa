# GazeJEPA: Saliency-Driven Gaze Simulation

**Course:** Natural Interaction, Università degli Studi di Milano, 2025–2026
**Supervisor:** Prof. Giuseppe Boccignone
**Authors:** Arash Khosropour, Emad Bahreinipour

## Overview

The starting point is [SaccadeJEPA](https://github.com/LumenPallidium/jepa), a self-supervised vision model that claims to simulate saccadic eye movements. The assignment was to reverse-engineer it and build something that genuinely simulates gaze using real eye-tracking data.

What we found: SaccadeJEPA picks where to look with a uniform random translation (`saccade.py:66`). There is no content-based attention — the saccade is augmentation, not output. From there the project splits into two tracks:

- **Arash** — saliency prediction: implement and evaluate four saliency sources against human fixations.
- **Emad** — gaze dynamics: replace the random cropper inside upstream `SaccadeJepa` with a saliency × inhibition-of-return sequential cropper and retrain on FIND.

Both tracks share a single saliency package, the FIND data loader, and the 50/6/6 train/val/test split (`seed=42`).

## Dataset

The FIND dataset (Liu & Xu, 2016) contains eye-tracking from 39 observers on 65 multi-face social videos at 1280×720. Fixations live in `.mat` files under `Our_database/fix_data/`. The dataset is not shipped with the repo; set `FIND_DATA_ROOT` to your local copy:

```
FIND_DATA_ROOT/Our_database/raw_videos/001.mp4
FIND_DATA_ROOT/Our_database/fix_data/001.mat
```

## Arash — saliency prediction

Four sources evaluated on the test split (310 frames, 39 observers):

| Saliency source               | AUC       | NSS       | CC        |
| ----------------------------- | --------- | --------- | --------- |
| Random (SaccadeJEPA baseline) | 0.499     | −0.004    | 0.000     |
| Center bias                   | 0.774     | 1.056     | 0.460     |
| Spectral residual (Hou-Zhang) | 0.687     | 0.391     | 0.132     |
| ResNetSaliency (ours)         | **0.891** | **2.549** | **0.713** |

ResNetSaliency uses an ImageNet-pretrained ResNet-18 with a lightweight 1×1 head trained on FIND with KL divergence. Its high score is partly because FIND is face-dominated and pretrained features already locate faces well.

## Emad — less obscure reuse of SaccadeJEPA

Upstream `SaccadeCropper` (random affine) is swapped for `GazeCropper`, which draws sequential fixations from a saliency map under an inhibition-of-return mask. The encoder, EMA target, predictor, and Huber + cycle + VICReg loss are reused unmodified. A 5-epoch CPU demo drops validation prediction MSE from 0.2996 (untrained) to 0.0462 — a ~84.6% reduction.

Any `SaliencySource` plugs into the factory:

```python
from gazejepa.saliency import ResNetSaliency
from gazejepa.jepa_reuse import make_gaze_jepa

source = ResNetSaliency.load("checkpoints/saliency/resnet_saliency_best.pt")
model  = make_gaze_jepa(saliency_source=source, full_input_size=(200, 200))
```

## Setup

```bash
conda env create -f environment.yml
conda activate gaze-jepa
pip install -e .
```

## Running

**Arash — train and evaluate saliency:**

```bash
python scripts/train_saliency.py --config configs/saliency_train.yml --data-root data/find_dataset
python scripts/evaluate_saliency.py --split test \
    --resnet-checkpoint checkpoints/saliency/resnet_saliency_best.pt
```

Notebook: `notebooks/arash_saliency_comparison.ipynb`

**Emad — train and evaluate the patched JEPA:**

```bash
python scripts/train_jepa_reuse.py \
    --n-epochs 5 --batch-size 4 --frames-per-video 4 \
    --variance-weight 1.0 --covariance-weight 0.1 --vicreg-gamma 1.0 \
    --output-dir outputs/jepa_reuse --device cpu

python scripts/evaluate_jepa_reuse.py --splits val,test
```

Notebook: `notebooks/saccade_to_gazejepa_steps.ipynb`

All scripts read FIND from `--data-root` (or `FIND_DATA_ROOT`). Full citations and discussion live in `report_arash/` and `report_emad/`.
