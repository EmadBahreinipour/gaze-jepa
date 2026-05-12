"""Evaluation: saliency metrics, scanpath metrics, gaze-loop driver."""

from gazejepa.evaluation.evaluate import (
    collect_generated_amplitudes,
    collect_human_amplitudes,
    collect_paired_human_amplitudes,
    collect_video_saccade_amplitudes,
    evaluate_gaze_loop,
)
from gazejepa.evaluation.saliency_comparison import (
    evaluate_source,
    run_full_comparison,
)

__all__ = [
    "collect_generated_amplitudes",
    "collect_human_amplitudes",
    "collect_paired_human_amplitudes",
    "collect_video_saccade_amplitudes",
    "evaluate_gaze_loop",
    "evaluate_source",
    "run_full_comparison",
]
