"""The three-axis consistency-judge prompt (Texture / Geometry / Semantic, 1-5).

The judge sees three images -- the clean REFERENCE frame, the GROUND-TRUTH frame,
and the RECONSTRUCTION crop -- and rates how faithfully the reconstruction brings
the object back on each axis. The output format is pinned so the harness can find
each axis's answer position and read the '1'..'5' logits for probability-weighted
scoring (see scorer.py). Rubric anchors mirror the LR-VCC consistency report so
the judge, DINO, and the human ranking are talking about the same thing.

Image assembly into a model's chat format (Qwen3-VL) lives in judge.py; this
module is the pure text + rubric so it can be unit-tested and version-controlled
on its own.
"""
from __future__ import annotations

AXES = ("Texture", "Geometry", "Semantic")

RUBRIC = {
    "Texture": (
        "fine surface detail -- does the object's pattern come back at the right "
        "scale and sharpness? 1 = markings lost, collaged, or drifted to a generic "
        "over-fine mesh; 3 = pattern partially recovered but soft, busy, or slightly "
        "wrong-scale; 5 = pattern restored at the reference scale with clean markings."
    ),
    "Geometry": (
        "shape, proportion, and pose -- is the structure coherent and consistent "
        "with the reference? 1 = distorted proportions or incoherent outline; "
        "3 = roughly right with visible errors; 5 = proportions, pose, and body "
        "outline coherent and consistent with the reference."
    ),
    "Semantic": (
        "identity -- is it unmistakably the same object with the same high-level "
        "appearance? 1 = ambiguous or a different-looking object; 3 = same kind of "
        "object but altered appearance; 5 = unmistakably the same instance."
    ),
}

SYSTEM = (
    "You are a meticulous visual-consistency judge. You compare a reconstructed "
    "video frame of an object that disappeared and reappeared against a clean "
    "reference and the ground truth, and rate how faithfully the reconstruction "
    "preserves the object. Judge only fidelity to the reference object; ignore "
    "background. Reason briefly before each rating, be decisive, and use the full "
    "1-5 range -- a faithful reconstruction earns a 5 and a poor one a 1; do not "
    "default every axis to 3."
)


def build_instruction() -> str:
    """The text instruction shown after the three images. Lays out each axis with
    its 1-5 rubric anchors and pins the output to one single digit per axis, in a
    fixed order, so the answer positions are known for logit extraction."""
    lines = [
        "The three images are, in order: (1) REFERENCE (clean, before the gap), "
        "(2) GROUND TRUTH, (3) RECONSTRUCTION to be judged.",
        "",
        "Rate the RECONSTRUCTION's fidelity to the object on three axes, each a "
        "single integer from 1 to 5 (5 = most faithful):",
        "",
    ]
    for axis in AXES:
        lines.append(f"- {axis}: {RUBRIC[axis]}")
    lines += [
        "",
        "First reason briefly -- two or three sentences comparing the RECONSTRUCTION "
        "to the reference object, noting where its texture, shape, and identity are "
        "faithful or have drifted. Commit to real differences rather than hedging to "
        "the middle. Then, on three separate lines with nothing after them, give the "
        "ratings as single integers 1-5:",
        "Texture: <digit>",
        "Geometry: <digit>",
        "Semantic: <digit>",
    ]
    return "\n".join(lines)
