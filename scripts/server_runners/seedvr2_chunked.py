#!/usr/bin/env python
"""Chunked SeedVR2 processing: split LR videos into 33-frame chunks
(sp_size=2 alignment: (t-1) % 8 == 0), and reassemble per-chunk outputs
into full-length videos, trimming model padding via the manifest.

Modes:
  split:    --input_dir <LR mp4 dir> --chunks_root <root>
  assemble: --chunks_root <root> --outputs_root <root> --final_dir <dir>

Runs in the vsr env (cv2 only).
"""
import argparse
import glob
import json
import os

import cv2

CHUNK = 33  # (33-1) % (4*sp_size=8) == 0


def split(input_dir, chunks_root):
    for path in sorted(glob.glob(os.path.join(input_dir, "*.mp4"))):
        name = os.path.splitext(os.path.basename(path))[0]
        outd = os.path.join(chunks_root, name)
        os.makedirs(outd, exist_ok=True)
        man_path = os.path.join(outd, "manifest.json")
        if os.path.isfile(man_path):
            print(f"[skip] {name} (manifest exists)")
            continue
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        manifest = {"fps": fps, "chunks": {}}
        idx = n_in_chunk = 0
        writer = None
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if writer is None:
                cname = f"{name}__c{idx:05d}"
                writer = cv2.VideoWriter(
                    os.path.join(outd, cname + ".mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                n_in_chunk = 0
            writer.write(fr)
            n_in_chunk += 1
            if n_in_chunk == CHUNK:
                writer.release()
                manifest["chunks"][cname] = n_in_chunk
                writer = None
                idx += 1
        if writer is not None:
            writer.release()
            manifest["chunks"][f"{name}__c{idx:05d}"] = n_in_chunk
        cap.release()
        json.dump(manifest, open(man_path, "w"), indent=1)
        print(f"[split] {name}: {len(manifest['chunks'])} chunks @ {fps:.2f}fps")


def assemble(chunks_root, outputs_root, final_dir):
    os.makedirs(final_dir, exist_ok=True)
    for man_path in sorted(glob.glob(os.path.join(chunks_root, "*", "manifest.json"))):
        name = os.path.basename(os.path.dirname(man_path))
        final_path = os.path.join(final_dir, name + ".mp4")
        if os.path.isfile(final_path) and os.path.getsize(final_path) > 0:
            print(f"[skip] {name}")
            continue
        man = json.load(open(man_path))
        outd = os.path.join(outputs_root, name)
        writer = None
        total = 0
        missing = []
        for cname in sorted(man["chunks"]):
            hits = glob.glob(os.path.join(outd, cname + "*"))
            if not hits:
                missing.append(cname)
                continue
            cap = cv2.VideoCapture(hits[0])
            want = man["chunks"][cname]
            got = 0
            while got < want:
                ok, fr = cap.read()
                if not ok:
                    break
                if writer is None:
                    h, w = fr.shape[:2]
                    writer = cv2.VideoWriter(
                        final_path, cv2.VideoWriter_fourcc(*"mp4v"),
                        man["fps"], (w, h))
                writer.write(fr)
                got += 1
            cap.release()
            total += got
        if writer is not None:
            writer.release()
        status = "OK" if not missing else f"MISSING {len(missing)} chunks"
        print(f"[assemble] {name}: {total} frames -> {final_path} [{status}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["split", "assemble"])
    ap.add_argument("--input_dir")
    ap.add_argument("--chunks_root", required=True)
    ap.add_argument("--outputs_root")
    ap.add_argument("--final_dir")
    a = ap.parse_args()
    if a.mode == "split":
        split(a.input_dir, a.chunks_root)
    else:
        assemble(a.chunks_root, a.outputs_root, a.final_dir)


if __name__ == "__main__":
    main()
