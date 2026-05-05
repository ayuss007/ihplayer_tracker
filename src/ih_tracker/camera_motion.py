from typing import Optional

import cv2
import numpy as np


class CameraMotionCompensator:
    def __init__(self, orb_features: int = 1500, ecc_iterations: int = 50, ecc_eps: float = 1e-5):
        self.orb = cv2.ORB_create(nfeatures=orb_features)
        self.ecc_iterations = ecc_iterations
        self.ecc_eps = ecc_eps

    def estimate(self, prev_gray: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
        kp1, des1 = self.orb.detectAndCompute(prev_gray, None)
        kp2, des2 = self.orb.detectAndCompute(gray, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        if len(matches) < 8:
            return None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        h, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
        return h
