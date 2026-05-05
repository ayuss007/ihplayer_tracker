from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]


@dataclass
class Detection:
    bbox: BBox
    score: float
    class_name: str
    embedding: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    jersey_text: Optional[str] = None
    jersey_confidence: float = 0.0


@dataclass
class TrackState:
    track_id: int
    bbox: BBox
    age: int
    hits: int
    time_since_update: int
    class_name: str
    jersey_votes: Dict[str, float] = field(default_factory=dict)


@dataclass
class FrameResult:
    frame_index: int
    tracks: List[TrackState]
    homography: Optional[np.ndarray] = None
