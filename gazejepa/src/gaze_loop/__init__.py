# Emad's territory — gaze controller, scanpath generation, ablation study.
# Interface contract: any SaliencySource from src/saliency/ can be plugged in.
# The only requirement is: predict(images: Tensor[B,3,H,W]) -> Tensor[B,H,W]
