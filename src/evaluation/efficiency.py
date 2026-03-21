"""
Efficiency metrics: FPS and VRAM usage tracking.
These require GPU — will gracefully skip if CUDA unavailable.
"""
import time
from contextlib import contextmanager

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


class FPSTracker:
    """Track frames per second during inference."""

    def __init__(self):
        self.frame_count = 0
        self.total_time = 0.0
        self._start = None

    def start(self):
        if HAS_CUDA:
            torch.cuda.synchronize()
        self._start = time.perf_counter()

    def tick(self, num_frames: int = 1):
        if HAS_CUDA:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start
        self.total_time += elapsed
        self.frame_count += num_frames
        self._start = time.perf_counter()

    @property
    def fps(self) -> float:
        if self.total_time == 0:
            return 0.0
        return self.frame_count / self.total_time

    def reset(self):
        self.frame_count = 0
        self.total_time = 0.0
        self._start = None


class VRAMTracker:
    """Track peak VRAM usage during inference."""

    def __init__(self):
        self.peak_mb = 0.0

    def reset(self):
        if HAS_CUDA:
            torch.cuda.reset_peak_memory_stats()
        self.peak_mb = 0.0

    def update(self):
        if HAS_CUDA:
            self.peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    @property
    def peak_gb(self) -> float:
        return self.peak_mb / 1024


@contextmanager
def track_efficiency():
    """Context manager that tracks FPS and VRAM for a block of inference code.

    Usage:
        with track_efficiency() as tracker:
            for frame in frames:
                tracker['fps'].start()
                result = model(frame)
                tracker['fps'].tick()
        print(tracker['fps'].fps, tracker['vram'].peak_gb)
    """
    fps = FPSTracker()
    vram = VRAMTracker()
    vram.reset()
    tracker = {"fps": fps, "vram": vram}
    try:
        yield tracker
    finally:
        vram.update()
