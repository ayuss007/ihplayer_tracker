from collections import defaultdict
from typing import Dict, Iterable, Tuple


class JerseyRecognizer:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def predict(self, _crop) -> Tuple[str, float]:
        # TODO: PARSeq inference
        return "", 0.0


class ConfidenceWeightedVoting:
    def __init__(self):
        self.memory: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def update(self, track_id: int, text: str, confidence: float) -> str:
        if text:
            self.memory[track_id][text] += max(confidence, 0.0)
        if not self.memory[track_id]:
            return ""
        return max(self.memory[track_id].items(), key=lambda x: x[1])[0]

    def bulk_update(self, updates: Iterable[Tuple[int, str, float]]):
        return {track_id: self.update(track_id, text, conf) for track_id, text, conf in updates}
