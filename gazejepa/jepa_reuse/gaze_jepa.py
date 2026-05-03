"""Build a SaccadeJepa with its cropper swapped for GazeCropper."""

from __future__ import annotations

from typing import Any

from gazejepa.jepa_reuse.saccade_jepa import SaccadeJepa
from gazejepa.jepa_reuse.gaze_cropper import GazeCropper


def make_gaze_jepa(
    saliency_source: Any,
    *,
    full_input_size: tuple[int, int] = (200, 200),
    model_input_size: tuple[int, int] = (128, 128),
    n_fixations: int = 5,
    ior_sigma: float = 20.0,
    ior_decay: float = 0.7,
    sampling_mode: str = "stochastic",
    saccade_jepa_kwargs: dict | None = None,
):
    """Instantiate ``SaccadeJepa`` and swap its cropper for :class:`GazeCropper`.

    ``saliency_source`` must conform to :class:`gazejepa.saliency.SaliencySource`.
    ``saccade_jepa_kwargs`` forwards extras (e.g. ``in_channels``,
    ``predictor_depth``) through to ``SaccadeJepa``.
    """
    base_kwargs = dict(
        full_input_size=full_input_size,
        model_input_size=model_input_size,
    )
    if saccade_jepa_kwargs:
        base_kwargs.update(saccade_jepa_kwargs)

    sj = SaccadeJepa(**base_kwargs)

    # Mirror the upstream NeRF dims so affine_embedder accepts our embeddings.
    upstream_cropper = sj.saccade_cropper
    gaze_cropper = GazeCropper(
        saliency_source=saliency_source,
        input_h=full_input_size[0],
        input_w=full_input_size[1],
        target_h=model_input_size[0],
        target_w=model_input_size[1],
        n_fixations=n_fixations,
        ior_sigma=ior_sigma,
        ior_decay=ior_decay,
        sampling_mode=sampling_mode,
        translation_L=upstream_cropper.translation_L,
        rotation_L=upstream_cropper.rotation_L,
    )
    assert gaze_cropper.affine_embed_dim == upstream_cropper.affine_embed_dim, (
        f"GazeCropper affine_embed_dim ({gaze_cropper.affine_embed_dim}) "
        f"must equal upstream SaccadeCropper "
        f"({upstream_cropper.affine_embed_dim}); the prebuilt "
        f"affine_embedder Linear layer would otherwise reject our embeddings."
    )

    sj.saccade_cropper = gaze_cropper

    return sj


if __name__ == "__main__":
    import torch

    H, W = 200, 200
    crop_h, crop_w = 128, 128
    T = 4
    B = 1

    class _BlobSaliency:
        name = "_blob"

        def __call__(self, images: torch.Tensor) -> torch.Tensor:
            b, _, h, w = images.shape
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing="ij",
            )
            blob = torch.exp(
                -((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (2 * 30.0**2)
            )
            blob = blob / blob.sum()
            return blob.unsqueeze(0).expand(b, -1, -1).contiguous()

    torch.manual_seed(0)
    model = make_gaze_jepa(
        saliency_source=_BlobSaliency(),
        full_input_size=(H, W),
        model_input_size=(crop_h, crop_w),
        n_fixations=T,
    )
    model.eval()

    x = torch.rand(B, 3, H, W)
    with torch.no_grad():
        outs = model(x)

    assert len(outs) == 4, f"Expected 4 outputs, got {len(outs)}"
    target, context_copy, target_pred, cycle_loss = outs

    n_transitions = B * (T - 1)
    expected_class_dim = model.class_dim
    assert target.shape == (n_transitions, expected_class_dim), (
        f"target shape {tuple(target.shape)} != "
        f"({n_transitions}, {expected_class_dim})"
    )
    assert target_pred.shape == target.shape, (
        f"target_pred {tuple(target_pred.shape)} != target {tuple(target.shape)}"
    )

    print(
        f"GazeJepa OK — forward returned 4 outputs, "
        f"target {tuple(target.shape)}, "
        f"target_pred {tuple(target_pred.shape)}, "
        f"cycle_loss={float(cycle_loss):.4f}"
    )
    pred_err = torch.nn.functional.mse_loss(target_pred, target).item()
    print(f"  Untrained prediction MSE on this synthetic input: {pred_err:.4f}")
