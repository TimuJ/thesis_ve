"""Per-clip frame-index alignment guard (build-first invariant).

The scoring freezes indices on GT and re-indexes into LQ and every SR output, so
a wrong index map silently compares the wrong frames. This guard asserts that the
decoded frame count agrees with every per-frame annotation and the manifest, and
that the stream is constant-rate. A clip that fails is quarantined, never scored.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlignmentResult:
    ok: bool
    reasons: list


def check_alignment(counts: dict, r_frame_rate: str, avg_frame_rate: str) -> AlignmentResult:
    """counts maps a source name (e.g. decoded, groundtruth, out_of_view,
    full_occlusion, manifest) to a frame count; all must be equal, and the stream
    must be constant-rate (r_frame_rate == avg_frame_rate)."""
    reasons: list[str] = []
    ref = counts.get("groundtruth", next(iter(counts.values()), None))
    for name, value in counts.items():
        if value != ref:
            reasons.append(f"{name} count {value} != groundtruth {ref}")
    if r_frame_rate != avg_frame_rate:
        reasons.append(
            f"variable frame rate: r_frame_rate {r_frame_rate} != avg_frame_rate {avg_frame_rate}"
        )
    return AlignmentResult(ok=(len(reasons) == 0), reasons=reasons)
