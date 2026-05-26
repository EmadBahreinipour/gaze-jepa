# GazeJEPA: Saliency-Driven Gaze Simulation

**Course:** Natural Interaction, Università degli Studi di Milano, 2025–2026
**Supervisors:** Prof. Giuseppe Boccignone
**Authors:** Arash Khosropour, Emad Bahreinipour

## Overview

The starting point is [SaccadeJEPA](https://github.com/LumenPallidium/jepa), a self-supervised vision model that claims to simulate saccadic eye movements. The assignment was to reverse-engineer it and build something that genuinely simulates gaze using real eye-tracking data.

What we found: SaccadeJEPA picks where to look with a uniform random translation (`saccade.py:66`). There is no content-based attention — the saccade is augmentation, not output. From there the project splits into two tracks:

- **Arash** — saliency prediction: implement and evaluate five saliency sources against human fixations.
- **Emad** — gaze dynamics: replace the random cropper inside upstream `SaccadeJepa` with a saliency × inhibition-of-return sequential cropper and retrain on FIND.

Both tracks share a single saliency package, the FIND data loader, and the 50/6/6 train/val/test split (`seed=42`).

## Dataset

The FIND dataset contains eye-tracking from 39 observers on 65 multi-face social videos at 1280×720. Fixations live in `.mat` files under `Our_database/fix_data_NEW/`. The dataset is not shipped with the repo; set `FIND_DATA_ROOT` to your local copy:

```
FIND_DATA_ROOT/Our_database/raw_videos/001.mp4
FIND_DATA_ROOT/Our_database/fix_data_NEW/001.mat
```

## Arash — saliency prediction

Five sources evaluated on the test split (310 frames, 39 observers):

| Saliency source               | AUC       | NSS       | CC        |
| ----------------------------- | --------- | --------- | --------- |
| Random (SaccadeJEPA baseline) | 0.502     | 0.007     | 0.001     |
| Center bias                   | 0.772     | 1.045     | 0.458     |
| Spectral residual (Hou-Zhang) | 0.688     | 0.394     | 0.136     |
| ResNetSaliency (ours)         | 0.881     | 2.525     | 0.723     |
| I-JEPA Saliency (ours)        | **0.892** | **2.768** | **0.772** |

Both learned models use a frozen pretrained backbone with a lightweight 3×3 conv head trained on FIND with KL divergence. The high scores are partly because FIND is face-dominated and pretrained features already locate faces well. I-JEPA's self-supervised transformer features give it a consistent edge over ResNet-18 across all three metrics but not that noticable.

## Emad — less obscure reuse of SaccadeJEPA

Upstream `SaccadeCropper` (random affine) is swapped for `GazeCropper`, which draws sequential fixations from a saliency map under an inhibition-of-return mask. The encoder, EMA target, predictor, and Huber + cycle + VICReg loss are reused unmodified. A 5-epoch CPU demo drops validation prediction MSE from 0.2996 (untrained) to 0.0462 — a ~84.6% reduction.

Any `SaliencySource` plugs into the factory:

```python
from gazejepa.saliency import ResNetSaliency
from gazejepa.jepa_reuse import make_gaze_jepa

source = ResNetSaliency.load("gazejepa/checkpoints/resnet_saliency_best.pt")
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
python scripts/train_saliency.py --model resnet --data-root data/find_dataset
python scripts/train_saliency.py --model ijepa  --data-root data/find_dataset

python scripts/evaluate_saliency.py --split test \
    --resnet-checkpoint gazejepa/checkpoints/resnet_saliency_best.pt \
    --ijepa-checkpoint gazejepa/checkpoints/ijepa_saliency_best.pt
```

Notebook: `notebooks/arash_notebook.ipynb`

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
