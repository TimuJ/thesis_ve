"""Synthetic artefact: progressive Gaussian blur within detected face bboxes.

Stresses LR-VCC sub-metric I (Identity slow-fast). Other sub-metrics should
show minimal effect because the blur is confined to face regions (typically
<5% of frame area for our base videos).

Mechanism:
    sigma = severity * 10.0
    kernel = odd integer ~= 6*sigma + 1 (clamped >= 3)
    For each detected frontal face bbox in the frame (Haar cascade), apply
    GaussianBlur with that sigma to a 10% padded crop.

If no face is detected in a given frame, the frame is passed through
unchanged. This is correct -- the artefact only fires where identity is
actually present.
"""
import cv2
import numpy as np


_FACE_DETECTOR = None


def _get_face_detector():
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        _FACE_DETECTOR = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if _FACE_DETECTOR.empty():
            raise RuntimeError("Failed to load Haar face cascade")
    return _FACE_DETECTOR


def apply_identity_degradation(frame_bgr: np.ndarray, idx: int, severity: float) -> np.ndarray:
    """Blur detected faces in the frame by sigma=severity*10.0.

    Parameters
    ----------
    frame_bgr : np.ndarray
        (H, W, 3) BGR uint8 array (OpenCV format).
    idx : int
        Zero-based frame index (unused, kept for API parity with other artefacts).
    severity : float
        Blur magnitude in [0, 1]. severity=0 returns the frame unchanged.
        severity=0.40 -> sigma=4.0, a clearly visible face blur.

    Returns
    -------
    np.ndarray
        (H, W, 3) BGR uint8 array of the same shape as input.
    """
    if severity <= 0.0:
        return frame_bgr.copy()

    sigma = float(severity) * 10.0
    k = max(3, int(round(sigma * 6)) | 1)  # odd

    detector = _get_face_detector()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
    )
    if len(faces) == 0:
        return frame_bgr.copy()

    out = frame_bgr.copy()
    h_img, w_img = frame_bgr.shape[:2]
    for (x, y, w, h) in faces:
        pad = int(0.1 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w_img, x + w + pad), min(h_img, y + h + pad)
        roi = out[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        blurred = cv2.GaussianBlur(roi, (k, k), sigmaX=sigma, sigmaY=sigma)
        out[y0:y1, x0:x1] = blurred
    return out
