"""
Extend notebooks/saccade_jepa_assessment.ipynb with the actual
"less obscure reuse" demonstration.

The original notebook (cells 0-9) does the reverse engineering and
sketches what the integration would look like in markdown. This script
appends cells that *implement* the integration: it builds a GazeJepa via
the gazejepa.jepa_reuse machinery and runs it on a real FIND frame,
producing the numerics that answer the prof's "is reuse feasible"
question.

Run idempotently — it strips any cells with id starting with
``reuse-`` before appending, so re-running replaces the demonstration
section without duplicating it.
"""
from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "saccade_jepa_assessment.ipynb"

REUSE_PREFIX = "reuse-"


def md(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": REUSE_PREFIX + cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "code",
        "id": REUSE_PREFIX + cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


NEW_CELLS = [
    md(
        "section-header",
        """## Less obscure reuse: wiring SaccadeJEPA into an explicit gaze loop

The cells above reverse-engineer SaccadeJEPA and sketch (in markdown) what an explicit gaze-dynamics version would look like. Below we *implement* that sketch with the smallest possible change to SaccadeJEPA itself.

**The change in one line**: replace `SaccadeJepa.saccade_cropper` (random affine per image) with `GazeCropper` (saliency × IOR-driven sequential fixations). Everything else — ConvNeXt-Tiny encoder, EMA target, MLP predictor, NeRF affine embedder, JEPA + cycle-consistency loss — is reused unmodified.

This is what the prof's brief asks for: *"reused in a less obscure way for an explicit simulation of gaze dynamics."*
""",
    ),
    code(
        "imports",
        """import torch
import numpy as np
import matplotlib.pyplot as plt

from gazejepa.jepa_reuse import GazeCropper, make_gaze_jepa
from gazejepa.saliency import IttiKochSaliency, RandomSaliency, CenterBiasSaliency
from gazejepa.data import resolve_data_root, load_frame, get_split

torch.manual_seed(42)
np.random.seed(42)
""",
    ),
    md(
        "build",
        """### Build the patched model

`make_gaze_jepa(...)` instantiates upstream `SaccadeJepa` (via the LumenPallidium clone) and substitutes the cropper. The returned object is a real `SaccadeJepa` instance — its forward signature, output tuple, and parameter list are unchanged.
""",
    ),
    code(
        "build-code",
        """FULL_INPUT = (200, 200)   # SaccadeJepa default — image fed to the model
MODEL_INPUT = (128, 128)  # SaccadeJepa default — size of each fixation crop
T = 5                     # number of fixations per image (yields T-1 transitions)

# Build the patched model with classical Itti-Koch saliency.
saliency = IttiKochSaliency()
model = make_gaze_jepa(
    saliency_source=saliency,
    full_input_size=FULL_INPUT,
    model_input_size=MODEL_INPUT,
    n_fixations=T,
    ior_sigma=20.0,
    ior_decay=0.7,
    sampling_mode="stochastic",
)
model.eval()

print(f"Model class:                {type(model).__name__}")
print(f"Cropper class (after swap): {type(model.saccade_cropper).__name__}")
print(f"affine_embed_dim (cropper): {model.saccade_cropper.affine_embed_dim}")
print(f"affine_embedder accepts:    {model.affine_embedder[1].in_features}")
print(f"\\nTotal trainable params:   {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
""",
    ),
    md(
        "find-frame",
        """### Forward pass on a real FIND frame

Load a frame from the FIND test split, resize to the model's expected `full_input_size`, and run the patched forward pass. SaccadeJepa returns `(target, context_copy, target_pred, cycle_loss)` exactly as before — we just get T-1 transitions per image instead of one random pair.
""",
    ),
    code(
        "find-frame-code",
        """import os
from pathlib import Path

# Locate FIND data: env var first, then the project's data/find_dataset/
# inferred from the notebook's location (so the notebook works without
# requiring the env var to be set).
if "FIND_DATA_ROOT" in os.environ:
    data_root = resolve_data_root()
else:
    nb_dir = Path.cwd()
    candidate = nb_dir / "data" / "find_dataset"
    if not candidate.is_dir():
        candidate = nb_dir.parent / "data" / "find_dataset"
    data_root = resolve_data_root(candidate)
print(f"FIND data root: {data_root}")

split = get_split(data_root, seed=42)
test_video = split["test"][0]

# Load and resize to the model's full_input_size.
frame = load_frame(data_root, test_video, frame_idx=100)  # (720, 1280, 3) uint8 RGB
frame_small = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # (3, 720, 1280)
frame_small = torch.nn.functional.interpolate(
    frame_small.unsqueeze(0), size=FULL_INPUT, mode="bilinear", align_corners=False
).squeeze(0)
print(f"FIND frame: video {test_video}, frame 100, resized to {tuple(frame_small.shape)}")

# Run the model. .forward returns 4-tuple; we want the prediction error.
with torch.no_grad():
    target, context_copy, target_pred, cycle_loss = model(frame_small.unsqueeze(0))

n_transitions = T - 1
pred_mse = torch.nn.functional.mse_loss(target_pred, target).item()
print(f"\\nForward outputs:")
print(f"  target         : {tuple(target.shape)}    (encoded fix_(t+1) representations)")
print(f"  context_copy   : {tuple(context_copy.shape)}")
print(f"  target_pred    : {tuple(target_pred.shape)}    (predictor's guess at target)")
print(f"  cycle_loss     : {cycle_loss.item():.4f}")
print(f"\\n  prediction MSE : {pred_mse:.4f}    (untrained — reflects the predictor's bias)")
print(f"  transitions    : {n_transitions}    ({T} fixations - 1)")
""",
    ),
    md(
        "viz",
        """### Visualizing the reused gaze loop in action

To make what just happened legible, we re-run the cropper alone (it's deterministic given the seed) to extract the actual fixation locations, then plot:
- The input frame with the scanpath overlaid;
- The saliency map driving the sampler;
- The (view_t, view_{t+1}) pairs the predictor sees for each transition.
""",
    ),
    code(
        "viz-code",
        """from gazejepa.gaze_loop import GazeLoop

# Re-run the GazeLoop independently to recover the scanpath. Same seed,
# same saliency, same IOR -> same fixations as the cropper produced
# (the cropper internally reseeds on every forward, so we reseed here
# to keep visuals matched to the model run above).
torch.manual_seed(42)
np.random.seed(42)
loop = GazeLoop(
    saliency_source=saliency,
    n_fixations=T,
    image_size=FULL_INPUT,
    ior_sigma=20.0,
    ior_decay=0.7,
    sampling_mode="stochastic",
)
loop_result = loop(frame_small)
scanpath = loop_result["scanpath"].numpy()  # (T, 2)
sal_map = loop_result["saliency_map"].numpy()  # (H, W)

fig, axes = plt.subplots(2, T, figsize=(3 * T, 6))

# Top row: frame + scanpath, then saliency, then context_copy norms.
axes[0, 0].imshow(frame_small.permute(1, 2, 0).numpy())
axes[0, 0].plot(scanpath[:, 0], scanpath[:, 1], "r-o", linewidth=2, markersize=8)
for i, (x, y) in enumerate(scanpath, start=1):
    axes[0, 0].annotate(str(i), (x, y), color="white", fontsize=10,
                         xytext=(4, 4), textcoords="offset points")
axes[0, 0].set_title(f"FIND vid {test_video} frame 100\\n+ saliency-driven scanpath")
axes[0, 0].axis("off")

axes[0, 1].imshow(sal_map, cmap="hot")
axes[0, 1].set_title("Itti-Koch saliency\\n(drives the sampler)")
axes[0, 1].axis("off")

# Per-transition prediction MSE.
per_trans_mse = ((target_pred - target) ** 2).mean(dim=1).numpy()
axes[0, 2].bar(range(1, n_transitions + 1), per_trans_mse, color="steelblue")
axes[0, 2].set_title("Predictor MSE per transition\\n(untrained)")
axes[0, 2].set_xlabel("transition fix_t -> fix_{t+1}")
axes[0, 2].set_ylabel("MSE")
for i in range(3, T):
    axes[0, i].axis("off")

# Bottom row: the (view_t, view_{t+1}) pairs the predictor was trained on.
# We re-extract them from the cropper to make visible exactly what the
# model saw.
torch.manual_seed(42)
np.random.seed(42)
v1, v2, _ = model.saccade_cropper(frame_small.unsqueeze(0))
for t in range(min(T - 1, T)):
    if t < T - 1:
        # Show view_1 (fix_t) and view_2 (fix_{t+1}) side by side, but
        # since we have n_transitions cells, alternate them. Actually
        # simpler: show view_2 (the target) for each transition.
        axes[1, t].imshow(v2[t].permute(1, 2, 0).numpy())
        axes[1, t].set_title(f"target crop\\n(fix_{t + 2})")
        axes[1, t].axis("off")
axes[1, T - 1].axis("off")

plt.tight_layout()
plt.savefig("../outputs/jepa_reuse_demo.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved ../outputs/jepa_reuse_demo.png")
""",
    ),
    md(
        "ablation",
        """### Sanity check: does saliency-driven sampling change the predictor's task?

The strongest version of the brief's question is: **does this reuse actually do something gaze-relevant, or is it just window dressing?**

A concrete falsifiable test: run the same untrained model with two different saliency sources — random and Itti-Koch — on the same frame. If both produce the same prediction MSE, then the saliency mechanism isn't really shaping what the predictor sees. If Itti-Koch produces meaningfully different (typically *lower-variance*, more concentrated near content) crops than random, the predictor's MSE distribution should differ.

Note: this is an *untrained* check. The prediction MSE here reflects the convex hull of crop variability, not learned predictive accuracy. A real training run on FIND would amplify this difference.
""",
    ),
    code(
        "ablation-code",
        """# Run the same model with two different saliency sources, on the same
# 5 FIND test frames, T=5 fixations each. Report mean prediction MSE.
N_FRAMES = 5
results = {}

# data_root is reused from the previous cell.

for src_name, src in [("random", RandomSaliency()),
                      ("itti_koch", IttiKochSaliency())]:
    # Rebuild the model with this source. The encoder/predictor weights are
    # re-randomised for each (no shared state), so this is a same-conditions
    # comparison of how the cropper choice affects the prediction task.
    torch.manual_seed(42)
    np.random.seed(42)
    m = make_gaze_jepa(
        saliency_source=src,
        full_input_size=FULL_INPUT,
        model_input_size=MODEL_INPUT,
        n_fixations=T,
    )
    m.eval()

    mses = []
    for vi, vid in enumerate(split["test"][:N_FRAMES]):
        try:
            f = load_frame(data_root, vid, frame_idx=100)
        except Exception as e:
            print(f"  skip {vid}: {e}")
            continue
        f_small = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
        f_small = torch.nn.functional.interpolate(
            f_small.unsqueeze(0), size=FULL_INPUT,
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        with torch.no_grad():
            tgt, _, tgt_pred, _ = m(f_small.unsqueeze(0))
        mses.append(torch.nn.functional.mse_loss(tgt_pred, tgt).item())
    results[src_name] = mses

print(f"Untrained predictor MSE on {N_FRAMES} FIND test frames, T={T} fixations each:\\n")
print(f"  {'source':<12} | {'mean MSE':>10} | {'std':>8} | per-frame MSE")
print(f"  {'-'*12} | {'-'*10} | {'-'*8} | {'-'*40}")
for name, mses in results.items():
    if mses:
        m_mean = float(np.mean(mses))
        m_std = float(np.std(mses))
        per = ", ".join(f"{x:.3f}" for x in mses)
        print(f"  {name:<12} | {m_mean:>10.4f} | {m_std:>8.4f} | {per}")
""",
    ),
    md(
        "trained-header",
        """## After training: does the predictor learn anything?

The cells above run an *untrained* `make_gaze_jepa` on FIND, which is enough to prove the architectural substitution works but says nothing about whether the JEPA training signal produces useful representations on FIND.

[`scripts/train_jepa_reuse.py`](../scripts/train_jepa_reuse.py) wraps the upstream Huber + cycle + VICReg loss (from [`jepa/train.py:saccade_loss`](https://github.com/LumenPallidium/jepa/blob/main/jepa/train.py)) around the patched model and trains it on FIND. The cells below load the resulting checkpoint and report the answer.

Run the training with:

```bash
FIND_DATA_ROOT=$(pwd)/data/find_dataset \\
    python scripts/train_jepa_reuse.py \\
        --n-epochs 3 --batch-size 4 --frames-per-video 4 \\
        --variance-weight 1.0 --covariance-weight 0.1 --vicreg-gamma 1.0 \\
        --output-dir outputs/jepa_reuse --device cpu
```

If the checkpoint isn't there yet, the cells below print a message and skip — re-run them after training completes.
""",
    ),
    code(
        "trained-load",
        """from pathlib import Path

CKPT_DIR = Path("../outputs/jepa_reuse/checkpoints")
METRICS_DIR = Path("../outputs/jepa_reuse/metrics")
final_ckpt = CKPT_DIR / "final.pt"
have_ckpt = final_ckpt.exists()

if have_ckpt:
    print(f"Found trained checkpoint: {final_ckpt}")
    # Build a fresh model with the same architecture and load the trained weights.
    trained_model = make_gaze_jepa(
        saliency_source=IttiKochSaliency(),
        full_input_size=FULL_INPUT,
        model_input_size=MODEL_INPUT,
        n_fixations=T,
        ior_sigma=20.0,
        ior_decay=0.7,
        sampling_mode="stochastic",
    )
    state = torch.load(final_ckpt, map_location="cpu", weights_only=False)
    trained_model.load_state_dict(state["model_state_dict"])
    trained_model.eval()
    print(f"Loaded {sum(p.numel() for p in trained_model.parameters() if p.requires_grad):,} params")
else:
    print(f"No checkpoint at {final_ckpt} yet — run scripts/train_jepa_reuse.py first.")
    trained_model = None
""",
    ),
    code(
        "trained-mse",
        """if trained_model is not None:
    torch.manual_seed(42)
    np.random.seed(42)
    with torch.no_grad():
        tgt_t, _, tgt_pred_t, cyc_t = trained_model(frame_small.unsqueeze(0))
    trained_mse = torch.nn.functional.mse_loss(tgt_pred_t, tgt_t).item()

    print(f"Same FIND frame, T={T}, T-1={T-1} transitions:\\n")
    print(f"  Untrained predictor MSE: {pred_mse:.4f}    (random init)")
    print(f"  Trained   predictor MSE: {trained_mse:.4f}    (after training)")
    print(f"  Reduction:               {(pred_mse - trained_mse) / pred_mse * 100:+.1f}%")
    print(f"\\n  Trained cycle loss:      {cyc_t.item():.4f}")

    # Show the loss curve from the training run.
    loss_curve = METRICS_DIR / "loss_curve.png"
    if loss_curve.exists():
        from IPython.display import Image, display
        print(f"\\nLoss curve from {loss_curve}:")
        display(Image(str(loss_curve)))
else:
    print("Skipping (no checkpoint).")
""",
    ),
    md(
        "trained-scanpaths",
        """### Generated scanpaths from the trained model

Even when the predictor's MSE moves only modestly, the *cropper* — `GazeCropper`, which sits unchanged in the trained model — keeps producing saliency × IOR-driven sequences. The figure below extracts the actual scanpaths the trained model is sampling on FIND test frames, so we can compare them qualitatively against the untrained run.
""",
    ),
    code(
        "trained-viz-code",
        """if trained_model is not None:
    # Use the trained model's GazeCropper directly to extract scanpaths
    # on a handful of FIND frames. Same fixations the trained model saw
    # at inference time.
    N_FRAMES = 4
    fig, axes = plt.subplots(1, N_FRAMES, figsize=(4 * N_FRAMES, 4))

    for ax, vid in zip(axes, split["test"][:N_FRAMES]):
        try:
            f = load_frame(data_root, vid, frame_idx=100)
        except Exception as e:
            ax.set_title(f"vid {vid}: {e}")
            ax.axis("off")
            continue
        f_small = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
        f_small = torch.nn.functional.interpolate(
            f_small.unsqueeze(0), size=FULL_INPUT,
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        # Re-run the GazeLoop standalone to recover the scanpath. Same
        # saliency × IOR machinery the trained cropper uses internally.
        torch.manual_seed(42)
        np.random.seed(42)
        rl = GazeLoop(
            saliency_source=IttiKochSaliency(),
            n_fixations=T,
            image_size=FULL_INPUT,
            ior_sigma=20.0,
            ior_decay=0.7,
            sampling_mode="stochastic",
        )(f_small)
        sp = rl["scanpath"].numpy()

        ax.imshow(f_small.permute(1, 2, 0).numpy())
        ax.plot(sp[:, 0], sp[:, 1], "r-o", linewidth=2, markersize=8)
        for i, (x, y) in enumerate(sp, start=1):
            ax.annotate(str(i), (x, y), color="white", fontsize=10,
                         xytext=(4, 4), textcoords="offset points")
        ax.set_title(f"vid {vid} frame 100")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("../outputs/jepa_reuse_scanpaths_trained.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved ../outputs/jepa_reuse_scanpaths_trained.png")
else:
    print("Skipping (no checkpoint).")
""",
    ),
    md(
        "verdict",
        """## What this answers about the brief

The prof's question, verbatim: *"to understand what has actually been implemented and whether it can be reused in a less obscure way for an explicit simulation of gaze dynamics (i.e., saccades followed by fixations)."*

**Reverse engineering** (cells 0–4 above): SaccadeJEPA is a self-supervised representation-learning scheme using one random spatial perturbation per image as the pretext task. Confirmed by running its `SaccadeCropper` and `SaccadeJepa.forward` directly. Not a gaze simulator.

**"Less obscure reuse"** (this section): yes, it can be reused, with one architectural substitution. Replace `SaccadeJepa.saccade_cropper` with [`GazeCropper`](../gazejepa/jepa_reuse/gaze_cropper.py) — a saliency × IOR-driven sequential cropper whose `affine_embed_dim` matches `SaccadeCropper`'s by construction, so the prebuilt `affine_embedder` accepts our embeddings without surgery. The forward pass, encoder, EMA target, predictor, and JEPA+cycle losses are reused unmodified.

**Training** (cells above): the patched model is trained on FIND with the upstream Huber + cycle + VICReg loss, via [`scripts/train_jepa_reuse.py`](../scripts/train_jepa_reuse.py). The before/after MSE comparison and the loss curve are the evidence that the JEPA training signal does (or doesn't) produce useful next-fixation representations on FIND.

The architectural change is one line:

```python
sj = SaccadeJepa(...)              # unchanged upstream model
sj.saccade_cropper = GazeCropper(...)  # the entire substitution
```

That is the literal "less obscure reuse" the brief asks for, with the training run closing the "if feasible" hedge.
""",
    ),
]


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text())

    # Strip any previous reuse-* cells so re-runs don't duplicate.
    nb["cells"] = [
        c for c in nb["cells"]
        if not (c.get("id", "").startswith(REUSE_PREFIX))
    ]

    nb["cells"].extend(NEW_CELLS)

    NOTEBOOK.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Wrote {NOTEBOOK} ({len(nb['cells'])} cells, "
          f"{len(NEW_CELLS)} reuse cells appended).")


if __name__ == "__main__":
    main()
