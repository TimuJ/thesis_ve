"""Tests for the 3-axis judge prompt builder."""
from scripts.lr_vcc.mllm_judge.prompt import AXES, RUBRIC, build_instruction


def test_axes_are_texture_geometry_semantic_in_order():
    assert AXES == ("Texture", "Geometry", "Semantic")


def test_rubric_defines_every_axis_with_1_and_5_anchors():
    for axis in AXES:
        text = RUBRIC[axis]
        assert "1 =" in text and "5 =" in text


def test_instruction_names_each_axis_and_pins_the_output_format():
    instr = build_instruction()
    for axis in AXES:
        assert axis in instr
    # the answer must be machine-parseable in a fixed order for logit extraction
    assert instr.index("Texture") < instr.index("Geometry") < instr.index("Semantic")
    assert "1" in instr and "5" in instr


def test_instruction_requests_single_digit_per_axis():
    instr = build_instruction().lower()
    assert "single" in instr or "one digit" in instr or "1-5" in instr or "1 to 5" in instr


def test_instruction_asks_for_reasoning_before_the_ratings():
    # Chain-of-thought: the model must deliberate before committing a score,
    # otherwise it hedges every axis to 3. The parseable rating lines stay.
    instr = build_instruction().lower()
    assert "reason" in instr
    assert "texture: <digit>" in instr
    assert "geometry: <digit>" in instr
    assert "semantic: <digit>" in instr
