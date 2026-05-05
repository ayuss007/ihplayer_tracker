from dataclasses import dataclass
from typing import List

from .types import Detection, TrackState


@dataclass
class _SimpleTrack:
    track_id: int
    bbox: tuple
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    class_name: str = "player"


class MultiObjectTracker:
    """Baseline tracker skeleton (Kalman hooks can be extended here)."""

    def __init__(self, max_age: int = 40):
        self.max_age = max_age
        self.next_id = 1
        self.tracks: List[_SimpleTrack] = []

    def update(self, detections: List[Detection]) -> List[TrackState]:
        # Placeholder association: assign new track for each detection.
        self.tracks = []
        for det in detections:
            self.tracks.append(
                _SimpleTrack(
                    track_id=self.next_id,
                    bbox=det.bbox,
                    class_name=det.class_name,
                )
            )
            self.next_id += 1

        return [
            TrackState(
                track_id=t.track_id,
                bbox=t.bbox,
                age=t.age,
                hits=t.hits,
                time_since_update=t.time_since_update,
                class_name=t.class_name,
            )
            for t in self.tracks
        ]
