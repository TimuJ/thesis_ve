"""Tests for the judge harness's pure seams (the model forward pass is GPU-only
and validated on the server, not here). Importing judge must NOT require torch --
these tests passing on a box without a working torch is the proof that the heavy
imports are lazy."""
from scripts.lr_vcc.mllm_judge.judge import (
    AxisScores,
    _input_device,
    answer_prefixes,
    build_messages,
)
from scripts.lr_vcc.mllm_judge.prompt import SYSTEM, build_instruction


def test_module_imports_without_torch():
    import scripts.lr_vcc.mllm_judge.judge  # noqa: F401  (must not import torch at top)


def test_build_messages_orders_ref_gt_recon_then_instruction():
    msgs = build_messages("ref.png", "gt.png", "recon.png")
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]["content"]
    images = [c for c in user if c["type"] == "image"]
    assert [c["image"] for c in images] == ["ref.png", "gt.png", "recon.png"]
    texts = [c for c in user if c["type"] == "text"]
    assert len(texts) == 1 and texts[0]["text"] == build_instruction()
    assert msgs[0]["content"][0]["text"] == SYSTEM


def test_axis_scores_as_dict_uses_axis_names():
    a = AxisScores(2.0, 3.5, 5.0)
    assert a.as_dict() == {"Texture": 2.0, "Geometry": 3.5, "Semantic": 5.0}


def test_answer_prefixes_truncate_just_before_each_rating_digit():
    gen = ("The markings are soft but present and the shape holds.\n"
           "Texture: 3\nGeometry: 4\nSemantic: 5")
    pre = answer_prefixes(gen)
    # each prefix ends right after "Axis: " so the next token is that axis's digit
    assert pre["Texture"].endswith("Texture: ")
    assert pre["Geometry"].endswith("Geometry: ")
    assert pre["Semantic"].endswith("Semantic: ")
    # a later axis's prefix carries the reasoning and the earlier ratings (conditioning)
    assert pre["Geometry"].startswith("The markings are soft")
    assert "Texture: 3" in pre["Geometry"]
    assert "Geometry: 4" in pre["Semantic"]


def test_answer_prefixes_ignore_axis_word_not_followed_by_a_rating_digit():
    # The model may mention an axis in prose; only "Axis: <1-5 digit>" is the rating.
    gen = "Texture: the surface is busy.\nTexture: 2\nGeometry: 3\nSemantic: 4"
    pre = answer_prefixes(gen)
    assert pre["Texture"].endswith("Texture: ")
    assert "the surface is busy" in pre["Texture"]   # skipped the prose mention
    assert pre["Texture"].count("Texture:") == 2      # kept up to the real rating


def test_answer_prefixes_none_when_axis_rating_missing():
    pre = answer_prefixes("Texture: 3\nGeometry: 4")   # Semantic rating absent
    assert pre["Semantic"] is None
    assert pre["Texture"].endswith("Texture: ")


def test_input_device_concrete_returns_that_device_without_probing_the_model():
    # A single-GPU device is used verbatim; the model is never probed for its
    # embedding device (so a concrete device needs no loaded, sharded model).
    probed = []
    dev = _input_device("cuda:0", lambda: probed.append(True) or "cuda:1")
    assert dev == "cuda:0"
    assert probed == []


def test_input_device_auto_uses_the_input_embedding_device():
    # device_map="auto" shards weights across GPUs; inputs must land on the input
    # embedding's device (cuda:0 here), whence accelerate forwards across shards.
    assert _input_device("auto", lambda: "cuda:0") == "cuda:0"
