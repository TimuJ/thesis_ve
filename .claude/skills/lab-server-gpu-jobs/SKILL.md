---
name: lab-server-gpu-jobs
description: Use when running GPU work on the shared ZJU lab server over SSH — launching detached evaluation/battery/generation jobs, surviving sshd throttling, pulling and aggregating results, or fixing the VBench runtime — i.e. any "run X on the server" step in this project.
---

# Lab-Server GPU Jobs

## Overview

How to run long GPU jobs on the shared lab server reliably. The server is
heavily shared (load ~10, ~60 users), so the hazards are **sshd throttling**,
**launches that hang the SSH channel**, and a **split results tree**. Jobs must
survive your SSH session dying — always launch detached and verify separately.

## Connection

The exact host / port / key / user are the `ssh` allow-rule in
`.claude/settings.local.json` — read them there (kept out of git). Shape:
`ssh -p <port> -i <key> <user>@<host> 'bash -lc "<remote cmds>"'`.
Env: conda at `~/miniconda3/envs/{vbench,vsr,identity,flashvsr,seedvr,seedvr310}`.

## sshd throttling (the #1 failure)

Under load, sshd resets at handshake: `kex_exchange_identification: Connection
reset by peer`. It is a **throttle, not an outage** (TCP connects; the box is up).
- **Read-only checks:** wrap in a retry loop, print a `GOT` sentinel first, break
  on success: `for a in $(seq 1 6); do out=$(ssh … 'echo GOT; …'); echo "$out" |
  grep -q GOT && { echo "$out"; break; }; done`.
- **If it persists** across two rounds of retries: **stop and report to the
  user** (project rule — do not hammer the box). It has self-cleared before.

## Launching a detached job (avoid the channel hang)

Use `setsid`, redirect all fds, and end with `echo PID; exit 0` — **no trailing
`sleep`/`tail`/`pgrep`** (those hold the channel open and the call hangs):
```
ssh … 'bash -lc "rm -f ~/logs/JOB.done; setsid bash ~/RUNNER args \
  > ~/logs/JOB.nohup 2>&1 < /dev/null & echo LAUNCHED_PID=\$!; exit 0"'
```
Then **verify with a separate read-only SSH** (process alive? log advancing?).
If the launch call still hangs and moves to a local background task, that is fine
— the job is detached and unaffected; `TaskStop` the hung local channel.
Generation (`gen_subset`) **skips existing files** so it is safe to re-run, but
do not blind-retry a *launch* (concurrent writers can race) — verify instead.

## The results-tree gotcha

`generate_all`/`gen_subset` write to the **repo tree**
`~/thesis_ve/results/synthetic_artefacts/<family>/`, but the audit runner and the
existing battery read `~/results/synthetic_artefacts/`. After generating a new
family, **symlink it across**:
`ln -s ~/thesis_ve/results/synthetic_artefacts/<fam> ~/results/synthetic_artefacts/<fam>`.

## VBench runtime fixes (needed for motion_smoothness / AMT)

`huggingface.co` is blocked from the host and VBench's `init_submodules`
hard-codes it (ignores `HF_ENDPOINT`). Once per environment:
```
export HF_ENDPOINT=https://hf-mirror.com
wget -O ~/.cache/vbench/amt_model/amt-s.pth \
  https://hf-mirror.com/lalala125/AMT/resolve/main/amt-s.pth
~/miniconda3/envs/vbench/bin/pip install omegaconf einops
```
Verify the import path (`init_submodules(['motion_smoothness'])`) before a
multi-hour run rather than discovering a missing dep an hour in.

## GPU choice

Pin with `CUDA_VISIBLE_DEVICES=<i>` and `nice`. Check **free memory** (not just
utilization) first: `nvidia-smi --query-gpu=index,memory.free,utilization.gpu
--format=csv,noheader` — a 100%-util GPU with free memory still runs.

## Waiting for completion + pulling results

The harness cannot track a detached server job, so poll with a background Bash
loop (`run_in_background`) that SSH-checks for the done-marker; make it
**gap-robust** (require 2 consecutive dead polls before declaring "stopped", to
survive brief inter-step gaps). These waiters are reaped at session transitions
but write their terminal line first — on resume, just re-check the server state.
Pull result JSONs with an scp retry loop (throttle). Results:
`~/results/vbench_reward_audit/<dim>/<family>/*eval_results*.json`.

## Aggregation

`eval_long` splits each video into ~84 two-second clips and scores per clip; the
clip path carries `<base>_sev<X>`, so aggregate per-clip → per-(base, severity)
mean by parsing the clip's parent dir name.

## Common mistakes

- Trailing `sleep`/`tail` after a `setsid &` launch → channel hangs.
- Forgetting the repo-tree→`~/results` symlink → the audit "can't find" a family.
- Hammering sshd through a throttle instead of backing off and reporting.
- Trusting `pgrep -f X | sed -n 1p` — the launcher's own command line matches the
  pattern; list all matches and look for the real `python … eval_long.py`.
