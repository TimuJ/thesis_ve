"""Qwen3-VL three-axis consistency judge -- GPU inference harness.

SCAFFOLD STATUS. The structure, the deterministic recipe, and the probability-
weighted scoring are in place. Three model-specific details MUST be validated
against the live Qwen3-VL tokenizer/model before the numbers are trusted -- each
is marked `# VALIDATE:` below:
  1. the chat-template image-content format (`{"type":"image", ...}`) and the
     qwen_vl_utils.process_vision_info call,
  2. the token ids for the digits '1'..'5' at the answer position (leading-space
     variants, single-token-ness),
  3. that `apply_chat_template(..., add_generation_prompt=True)` + our appended
     answer prefix land the next-token position exactly on the digit.

torch / transformers / qwen_vl_utils are imported LAZILY (inside functions), so
this module -- and the pure scorer/prompt tests -- import fine on a box without a
working torch (e.g. the M1 dev machine).

Scoring protocol (deterministic, teacher-forced): for each axis in order we build
the prompt up to "<Axis>: ", run one forward pass, read the logits of the five
digit tokens, and take the probability-weighted expected value (scorer.py). We
then append the argmax digit so the next axis is conditioned on the model's own
answer. Greedy, batch=1, seeds fixed, sdpa attention -- no sampling.

Usage (on the server, in the `mllmjudge` env):
    from scripts.lr_vcc.mllm_judge.judge import Qwen3VLJudge
    judge = Qwen3VLJudge("~/models/Qwen3-VL-8B-Instruct", device="cuda:0")
    scores = judge.score(ref_img, gt_img, recon_img)   # PIL.Image or path
    print(scores.as_dict())
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .prompt import AXES, SYSTEM, build_instruction
from .scorer import weighted_score

DEFAULT_MODEL = os.environ.get(
    "MLLM_JUDGE_MODEL", os.path.expanduser("~/models/Qwen3-VL-8B-Instruct")
)
SEED = 42


@dataclass
class AxisScores:
    """The three per-axis 1-5 scores (probability-weighted expected values)."""

    texture: float
    geometry: float
    semantic: float

    def as_dict(self) -> dict[str, float]:
        return {"Texture": self.texture, "Geometry": self.geometry, "Semantic": self.semantic}


def build_messages(ref, gt, recon) -> list[dict]:
    """The chat messages: system prompt, then the three images (REFERENCE, GROUND
    TRUTH, RECONSTRUCTION, in that order) followed by the instruction. ``ref``/
    ``gt``/``recon`` may be PIL images or file paths. Pure -- no model needed."""
    # VALIDATE: confirm Qwen3-VL's processor accepts this content shape (it may
    # want {"type":"image","image":<PIL|path|url>}; older Qwen-VL used exactly this).
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ref},
                {"type": "image", "image": gt},
                {"type": "image", "image": recon},
                {"type": "text", "text": build_instruction()},
            ],
        },
    ]


def answer_prefixes(generated: str, axes=AXES) -> "dict[str, str | None]":
    """For each axis, the prefix of ``generated`` ending right after that axis's
    rating marker ``"<Axis>: "`` -- positioned so the next token is the 1-5 digit.
    Only a marker immediately followed by a 1-5 digit counts (so an axis word used
    in the model's prose is skipped), and the last such match wins. ``None`` when an
    axis's rating is absent. Pure: this is how :meth:`Qwen3VLJudge.score` locates
    the digit position to read after the model's own chain-of-thought."""
    out: "dict[str, str | None]" = {}
    for axis in axes:
        last = None
        for m in re.finditer(re.escape(axis) + r":[ \t]*(?=[1-5])", generated):
            last = m
        out[axis] = generated[: last.end()] if last is not None else None
    return out


def _set_determinism(seed: int = SEED) -> None:
    """Pin every reproducibility knob from the inference recipe. Greedy decoding
    removes sampling noise; these remove kernel/init nondeterminism."""
    import random

    import numpy as np
    import torch
    from transformers import set_seed

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    # warn_only: a few vision ops lack deterministic kernels; we still fix seeds +
    # batch=1 + sdpa, which covers the dominant nondeterminism sources.
    torch.use_deterministic_algorithms(True, warn_only=True)


def _input_device(device: str, embed_device_fn):
    """Where model inputs should be placed. For a concrete device string ("cuda:0")
    it is that device. For ``device_map="auto"`` -- weights sharded across several
    GPUs -- inputs go to the input embedding's device (``embed_device_fn()``, called
    only in this case since it needs the loaded model); accelerate's dispatch hooks
    then forward activations across the shards from there."""
    if device == "auto":
        return embed_device_fn()
    return device


class Qwen3VLJudge:
    """Loads Qwen3-VL once and scores (ref, gt, recon) triples on the three axes.

    Deterministic: bf16, sdpa attention (NOT flash-attn-2), batch=1, fixed seeds,
    greedy -- see the module docstring. FP8 is intentionally unsupported here
    (not loadable via transformers; the FP8 path is vLLM-only and nondeterministic)."""

    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = "cuda:0", dtype: str = "bfloat16"):
        """``device`` is a concrete GPU ("cuda:0") or "auto" to shard the weights
        across every visible GPU (``device_map="auto"``) -- the path for a model too
        large for one card (e.g. Qwen3-VL-32B bf16, ~64GB, across 2x40GB). Sharding is
        layer placement only, so greedy/logit outputs stay deterministic."""
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        _set_determinism()
        self.device = device
        model_path = os.path.expanduser(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=getattr(torch, dtype),  # VALIDATE: some transformers want torch_dtype=
            device_map=device,            # "auto" shards across GPUs via accelerate
            attn_implementation="sdpa",  # reproducibility: NOT flash_attention_2
        ).eval()
        self.input_device = _input_device(
            device, lambda: self.model.get_input_embeddings().weight.device)
        self._digit_ids = self._resolve_digit_ids()

    def _resolve_digit_ids(self) -> list[int]:
        """Vocabulary ids for the next-token digits '1'..'5' after an "<Axis>: "
        prefix. VALIDATE live: digits are usually single tokens, but confirm there
        is no leading-space variant and that each encodes to exactly one id."""
        tok = self.processor.tokenizer
        ids = []
        for d in "12345":
            enc = tok.encode(d, add_special_tokens=False)
            if len(enc) != 1:
                raise RuntimeError(
                    f"digit {d!r} is not a single token ({enc}); resolve the "
                    f"correct answer-position id before scoring"
                )
            ids.append(enc[0])
        return ids

    def _next_token_logits(self, messages: list[dict], answer_prefix: str):
        """Logits for the token that follows ``answer_prefix`` (appended to the
        assistant turn). One forward pass; no generation."""
        import torch
        from qwen_vl_utils import process_vision_info  # VALIDATE: Qwen3-VL util name

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text = text + answer_prefix
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.input_device)
        with torch.no_grad():
            out = self.model(**inputs)
        return out.logits[0, -1, :]  # next-token logits at the digit position

    def _generate(self, messages: list[dict], max_new_tokens: int = 320) -> str:
        """Greedy (deterministic) chain-of-thought generation: the model reasons,
        then emits the three "<Axis>: <digit>" rating lines. Returns only the decoded
        new tokens. Batch=1, no sampling. The reasoning is what lifts the ratings off
        the hedged middle; the probability-weighted digit read happens afterwards."""
        import torch
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.input_device)
        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = gen[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(new, skip_special_tokens=True)[0]

    def score(self, ref, gt, recon) -> AxisScores:
        """Score one (reference, ground-truth, reconstruction) triple. The model first
        reasons (chain-of-thought) and emits its three ratings; then, for each axis, we
        re-read the '1'..'5' logits at that axis's rating position -- conditioned on the
        model's own reasoning -- for the probability-weighted expected value. An axis the
        model failed to rate scores NaN rather than silently defaulting."""
        messages = build_messages(ref, gt, recon)
        generated = self._generate(messages)
        prefixes = answer_prefixes(generated)
        out: dict[str, float] = {}
        for axis in AXES:
            prefix = prefixes.get(axis)
            if prefix is None:
                out[axis] = float("nan")
                continue
            logits = self._next_token_logits(messages, prefix)
            out[axis] = weighted_score([float(logits[i]) for i in self._digit_ids])
        return AxisScores(out["Texture"], out["Geometry"], out["Semantic"])
