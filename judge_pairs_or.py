"""Blind consistency judge over the dumped crops, via OpenRouter (OpenAI-compatible).

One ISOLATED chat.completions call per pair -- no human/Qwen scores and no other
pairs in context. That isolation is the fix for the non-blind caveat. Scores are
the emitted 1-5 integer per axis (argmax); OpenRouter exposes no reliable
per-token logprobs across providers, so the open judge's digit-logit expected
value is not reproduced here (documented difference).

Reuses prompt.py (SYSTEM + build_instruction + AXES) unchanged, so every model in
the sweep gets the identical rubric and three-image framing.

    OPENROUTER_API_KEY=... python judge_pairs_or.py <crops_dir> <out.json> <model_slug>

e.g. model_slug = anthropic/claude-opus-5.5 | openai/gpt-5.1 |
     google/gemini-2.5-pro | qwen/qwen3-vl-235b-a22b-instruct | qwen/qwen3-vl-8b-instruct
"""
import base64
import json
import os
import re
import sys
import time

from openai import OpenAI

from scripts.lr_vcc.mllm_judge.prompt import AXES, SYSTEM, build_instruction

CROPS_DIR, OUT_PATH, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]
MAXTOK = int(os.environ.get("JUDGE_MAX_TOKENS", "3000"))

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    default_headers={"X-Title": "LRVCC consistency judge"},
)


def data_url(path):
    b = base64.standard_b64encode(open(path, "rb").read()).decode("ascii")
    return f"data:image/png;base64,{b}"


def parse_axis(text, axis):
    last = None
    for m in re.finditer(re.escape(axis) + r"\s*:\s*\**\s*([1-5])", text):
        last = m
    return int(last.group(1)) if last else None


def score_pair(ref, gt, recon):
    content = [
        {"type": "text", "text": "Image 1 - REFERENCE (clean object, before the gap):"},
        {"type": "image_url", "image_url": {"url": data_url(ref)}},
        {"type": "text", "text": "Image 2 - GROUND TRUTH (target frame):"},
        {"type": "image_url", "image_url": {"url": data_url(gt)}},
        {"type": "text", "text": "Image 3 - RECONSTRUCTION (to be judged):"},
        {"type": "image_url", "image_url": {"url": data_url(recon)}},
        {"type": "text", "text": build_instruction()},
    ]
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": content}]
    last_err = None
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, max_tokens=MAXTOK,
            )
            text = resp.choices[0].message.content or ""
            return {ax: parse_axis(text, ax) for ax in AXES}, text
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise last_err


def main():
    manifest = json.load(open(os.path.join(CROPS_DIR, "crops_manifest.json")))
    out, failures, empties = [], 0, 0
    for k, m in enumerate(manifest):
        stem = m["stem"]
        paths = [os.path.join(CROPS_DIR, stem + s) for s in ("_ref.png", "_gt.png", "_recon.png")]
        if "error" in m or not all(os.path.exists(p) for p in paths):
            continue
        try:
            scores, text = score_pair(*paths)
        except Exception as e:
            failures += 1
            print(f"[{k + 1}/{len(manifest)}] {stem}: ERROR {e!r}", flush=True)
            continue
        if all(scores[a] is None for a in AXES):
            empties += 1
            print(f"[{k + 1}/{len(manifest)}] {stem}: NO-PARSE head={text[:80]!r}", flush=True)
        out.append({
            "video": m["video"], "i": m["i"], "j": m["j"],
            "texture": scores["Texture"], "geometry": scores["Geometry"],
            "semantic": scores["Semantic"],
        })
        print(f"[{k + 1}/{len(manifest)}] {stem}: {scores}", flush=True)
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"SCORED={len(out)} failures={failures} no_parse={empties} model={MODEL} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
