from typing import Iterator

import cv2

from .camera_motion import CameraMotionCompensator
from .config import RuntimeConfig
from .detection import RFDETRDetector
from .recognition import ConfidenceWeightedVoting, JerseyRecognizer
from .tracking import MultiObjectTracker
from .types import FrameResult


class HockeyTrackingPipeline:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.detector = RFDETRDetector(config.get("models", "rf_detr_checkpoint", default=""))
        self.recognizer = JerseyRecognizer(config.get("models", "parseq_checkpoint", default=""))
        self.voter = ConfidenceWeightedVoting()
        self.tracker = MultiObjectTracker(max_age=config.get("tracker", "max_age", default=40))
        self.cam_comp = CameraMotionCompensator(
            orb_features=config.get("camera_motion", "orb_features", default=1500),
            ecc_iterations=config.get("camera_motion", "ecc_iterations", default=50),
            ecc_eps=config.get("camera_motion", "ecc_eps", default=1e-5),
        )

    def run(self, video_path: str) -> Iterator[FrameResult]:
        cap = cv2.VideoCapture(video_path)
        prev_gray = None
        frame_idx = 0

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h = self.cam_comp.estimate(prev_gray, gray) if prev_gray is not None else None

            detections = self.detector.predict(frame)
            tracks = self.tracker.update(detections)

            for t in tracks:
                voted = self.voter.update(t.track_id, "", 0.0)
                if voted:
                    t.jersey_votes[voted] = 1.0

            yield FrameResult(frame_index=frame_idx, tracks=tracks, homography=h)
            prev_gray = gray
            frame_idx += 1

        cap.release()
