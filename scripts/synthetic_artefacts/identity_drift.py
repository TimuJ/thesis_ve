"""Synthetic artefact: progressive face-region morph toward a reference identity.

Stresses LR-VCC sub-metric I (Identity slow-fast). Unlike `identity_degradation`,
which applies a static blur to the face region, this artefact tests the
canonical long-term consistency failure mode of long-video super-resolution:
the subject's identity slowly drifts across the duration of the video.

Mechanism:
    For each frame i of T:
        blend = severity * (i / (T - 1))      # 0 at frame 0, severity at frame T-1
    For each detected frontal face bbox in the frame:
        resize the reference face image to fit the bbox,
        blend: (1 - blend) * face_region + blend * reference_face,
        paste back into the frame.

    Frames with no detected face pass through unchanged.

severity = 0 leaves the video unchanged.
severity = 1.0 means a full morph to the reference identity by the final frame.

By design the morph is confined to the detected face bounding box (no padding),
so the surrounding background and hair remain unchanged. This keeps the
artefact targeted at sub-metric I (Identity) rather than spilling onto the
other sub-metrics.
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


def apply_identity_drift(frame_bgr: np.ndarray,
                          idx: int,
                          n_frames: int,
                          reference_face_bgr: np.ndarray,
                          severity: float) -> np.ndarray:
    """Morph detected faces in the frame toward `reference_face_bgr`.

    The blend factor ramps linearly from 0 at frame 0 to `severity` at frame
    ``n_frames - 1``. Faces are detected with a Haar cascade. If no face is
    detected on a given frame the frame is returned unchanged.

    Parameters
    ----------
    frame_bgr : np.ndarray
        (H, W, 3) BGR uint8 array (OpenCV format).
    idx : int
        Zero-based frame index.
    n_frames : int
        Total number of frames in the video.
    reference_face_bgr : np.ndarray
        (H_ref, W_ref, 3) BGR uint8 image of the reference face to drift toward.
        Will be resized to each detected bbox at composition time.
    severity : float
        In [0, 1]. severity = 0 returns the frame unchanged.
        severity = 1.0 means full morph (blend factor = 1) at the final frame.

    Returns
    -------
    np.ndarray
        (H, W, 3) BGR uint8 array of the same shape as input.
    """
    if severity <= 0.0 or n_frames <= 1:
        return frame_bgr.copy()

    # Linear ramp: blend factor goes from 0 at frame 0 to severity at frame n_frames - 1.
    blend = float(severity) * float(idx) / float(n_frames - 1)
    blend = max(0.0, min(1.0, blend))
    if blend <= 0.0:
        return frame_bgr.copy()

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
        # No padding: confine the morph to the detected face bbox so we don't
        # blend the reference into background pixels and contaminate sub-metric D
        # (color histogram) or sub-metric T (temporal warping error).
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w_img, x + w), min(h_img, y + h)
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            continue
        roi = out[y0:y1, x0:x1]
        ref_resized = cv2.resize(
            reference_face_bgr, (bw, bh),
            interpolation=cv2.INTER_AREA,
        )
        # Linear alpha blend.
        blended = cv2.addWeighted(roi, 1.0 - blend, ref_resized, blend, 0.0)
        out[y0:y1, x0:x1] = blended
    return out
