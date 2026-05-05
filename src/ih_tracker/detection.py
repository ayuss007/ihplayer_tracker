from typing import List

import numpy as np

from .types import Detection


class RFDETRDetector:
    """RF-DETR adapter.

    Replace `predict` internals with your model loading/inference code.
    """

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def predict(self, frame: np.ndarray) -> List[Detection]:
        _ = frame
        # TODO: model inference. Placeholder output keeps architecture runnable.
        return []
