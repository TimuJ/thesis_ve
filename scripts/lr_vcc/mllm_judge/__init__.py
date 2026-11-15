"""Reproducible open-MLLM consistency judge (Qwen3-VL) for the LR-VCC pipeline.

Rates a reconstructed reappearance crop against a reference frame and the GT on
three axes (Texture / Geometry / Semantic), each 1-5, via probability-weighted
scoring over the model's '1'..'5' token logits. See scorer.py for the scoring
core; the model/tokenizer wiring lives in judge.py (GPU-only).
"""
