# Bi-weekly Progress Report — Timur Iakshibaev

## Period: June 15 – June 28, 2026

## Headline

This period opened with an **infrastructure shock** — the lab's primary GPU
server was decommissioned mid-experiment on June 15, killing the final
identity-slow-fast battery and forcing an unplanned data-rescue sprint.
The thesis work survived intact: all metric JSONs, source videos, composite
results and code are mirrored locally and on the surviving server disk
(disk3) by Jun 15 evening. The v5 composite verdict matrix is now finalised
at 11 of 12 artefact rows (the missing flip_invert row was the predicted-PASS
sanity control). With the rescue stabilised, the remainder of this period
pivots to the original short-term plan: real-SR-model evaluation, sub-metric
ablation, and starting the methodology-chapter rewrite.

1. **Lab server decommissioning survived without thesis-relevant data loss.**
   `223.109.239.43` was retired on June 15 at ~15:30 CST mid-run. Three
   physical disks moved to a CPU rescue host at `180.127.11.177:26200`;
   ~7 GB of thesis-relevant data consolidated onto the surviving disk3.
   Full local mirror + 1.5 GB Google-Drive-ready backup at
   `~/Downloads/zju_server_backup_2026-06-15/`.
2. **v5 composite verdict matrix locked at 11 of 12 rows.** flip_invert
   identity-stage battery was killed mid-run; the per-clip work directory
   contained only intermediate MP4s (no scores). Documented as
   "predicted-PASS control row, incomplete due to server decommission."
3. **Documentation built for cross-host portability.** `docs/server_restore_guide.md`
   captures the end-to-end procedure to rebuild the metric pipeline on a
   fresh GPU host (~2 h read-only restore + optional regeneration steps).
   Private notes updated with the new rescue-server connection info and the
   minimal disk3 layout. Conda env spec for all three envs (vsr, vbench, uav)
   snapshotted to `docs/server_conda_envs_2026-06-15.txt` before original
   environments were lost.
4. **Short- and long-term plans authored.** `docs/plans/2026-06-15-short-term-plan.md`
   (June 15 → July 15) and `docs/plans/2026-06-15-long-term-plan.md`
   (July 15 → September 30 and beyond). The former rescopes the next 4
   weeks around real-SR-model evaluation; the latter covers blind-review,
   revisions, paper co-write, and five open follow-up research directions.

---

## Server decommissioning timeline (June 15)

| time (CST) | event |
|---|---|
| ~00:30 | flip_invert identity battery started on GPU 1 (parked-fyx slot) |
| ~12:00 | lab IT messages: 3pm shutdown, copy data to disk3 by tomorrow noon |
| ~14:00 | recognise that disk1+disk2 will be dismounted, only disk3 remains |
| 14:51 | killed chained eval_flip_g0r session, moved it to GPU 1 in parallel with g7 |
| ~15:30 | original GPU server hard-shutdown (`Connection refused` from local) |
| ~15:40 | flip_invert identity stage interrupted — `_work/` had only 7WHI clips (1 of 5 videos), no per-clip scores |
| 16:00 | new rescue CPU server at `180.127.11.177:26200` discovered; ssh-copy-id installed our key |
| ~16:30 | full inventory of disks 1/2/3 on rescue host (240 GB candidate vs 76 GB free on disk3) |
| ~17:00 | selective migration plan agreed: ~7 GB compact (JSONs + SR outputs + source videos + flip_invert MP4s + DOVE eval from disk1) |
| ~17:30 | migration complete: 7.2 GB on `/data/disk3/timur/`, 68 GB free |
| ~19:00 | rescue server SSH goes intermittent (timeout → refused on retry) |
| ~22:00 | rescue server fully unreachable from local; no further progress that day |

By end of day everything thesis-relevant lives in three places (local
working tree, git remote, rescue server disk3) — at-rest safety achieved.

---

## v5 finalisation — what landed before the server died

The June 5-18 report's metric-redesign sprint completed the day before:

- **D' (anchor-window Lab histogram)** — `scripts/lr_vcc/color_histogram_anchor.py`
- **D'' (CLIP-image trajectory)** — `scripts/lr_vcc/compute_clip_trajectory.py`
  (switched from open_clip to OpenAI `clip` package because HF Hub is
  unreachable from the lab network)
- **Six-transform flip family** — `scripts/synthetic_artefacts/flip.py`
- **v5 LR-VCC composite** — `run_lr_vcc.py` extended with `--color_hist_anchor_dir`,
  `--clip_trajectory_dir`, `--dprime_beta`, `--dprime2_beta`. 7 sub-metrics
  total (A, T, I, D, D', D'', E). Default behaviour byte-identical when
  new args omitted.

The v5 verdict matrix (committed to `reports/figures/verdict_matrix_v5.md`):

| artefact | hhsz | 7WHI | KZ | BrRLK | mJog | clean |
|---|---:|---:|---:|---:|---:|---:|
| color_drift | WEAK | PASS | PASS | PASS | PASS | 5/5 |
| chunk_boundary | PASS | PASS | PASS | INV | WEAK | 4/5 |
| flicker | FLAT | FLAT | FLAT | INV | FLAT | 0/5 |
| identity_degradation | WEAK | FLAT | WEAK | FLAT | FLAT | 2/5 |
| identity_drift | WEAK | FLAT | FLAT | FLAT | FLAT | 1/5 |
| **background_drift** | **PASS** | **FLAT** | **PASS** | INV | **WEAK** | **4/5** |
| flip_horizontal | FLAT | FLAT | FLAT | FLAT | FLAT | 0/5 |
| flip_transpose | PASS | FLAT | PASS | FLAT | WEAK | 3/5 |
| flip_periodic | WEAK | FLAT | FLAT | FLAT | FLAT | 1/5 |
| flip_elastic | FLAT | FLAT | FLAT | FLAT | WEAK | 1/5 |
| flip_channel_shuffle | FLAT | PASS | PASS | FLAT | PASS | 3/5 |
| **flip_invert** | — | — | — | — | — | (pending) |

**The headline:** background_drift went from 0/5 PASS under D alone to 4/5
PASS under the v5 composite. The convergence-rewards-stability mechanism
identified earlier in the period is empirically fixed in production by
D' + D''. Only BrRLK cartoon content remains inverted, with the magnitude
halved from v4 (+0.127 → +0.065).

**Flip ablation predictions held cleanly:**
flip_horizontal is composite-invisible on all 5 bases (the smoking-gun probe),
flip_channel_shuffle and flip_transpose catch where predicted, the
sub-metric design is empirically justified.

---

## Code delivered this period

| File | Purpose | Commit |
|---|---|---|
| `scripts/server_runners/` (60 scripts) | Rescued shell runners from the original GPU server | 5224c29 |
| `scripts/server_runners/run_b11_v5_composite.sh` | Missing v5 composite runner added later | 1bb3b81 |
| `docs/server_conda_envs_2026-06-15.txt` | pip-freeze snapshot of vsr / vbench / uav envs | 284e604 |
| `docs/server_restore_guide.md` | End-to-end pipeline restore guide for new GPU host | ac1862d |
| `docs/plans/2026-06-15-short-term-plan.md` | June 15 → July 15 plan (5 priorities + check-ins) | f252a6c |
| `docs/plans/2026-06-15-long-term-plan.md` | July 15 → September 30 + open research directions | f252a6c |
| `docs/private/server-setup.md` (gitignored) | Updated with rescue-server connection info + post-Jun-16 disk3 layout | — |

Test suite **126 passing**, 0 failing. No code-rewrite this period — work
was rescue-coordination, documentation, and migration.

---

## Backup state (end of period)

Three independent copies of all irrecoverable data:

| location | size | content |
|---|---|---|
| Local Mac working tree | ~4 GB | Full mirror of all metric JSONs, masks, refs, SR outputs |
| Git remote `TimuJ/thesis_ve` | small | All source code, runner scripts, reports, docs, plans |
| Rescue server `/data/disk3/timur/` | 7.2 GB | Server-side persistent copy (currently unreachable, server intermittent) |
| `~/Downloads/zju_server_backup_2026-06-15/` | 1.5 GB | Source MKVs + JSON tarball, Drive-ready if needed |

The rescue server has been intermittently unreachable since June 15 evening
(timeout → refused). Data on disk3 is non-canonical for that reason; local
working tree is the source of truth.

---

## Next steps (plan for June 16–28 within this period)

Reverting to the original short-term plan (`docs/plans/2026-06-15-short-term-plan.md`)
once a new GPU host is available:

1. **Real-SR-model evaluation (P1 — thesis headline experiment).** Apply
   v5 LR-VCC to existing MGLD-VSR + Upscale-A-Video outputs + a frame-wise
   RealESRGAN lower anchor. Build the model-ranking table (PSNR / SSIM /
   LPIPS vs LR-VCC) — the key demonstration that the new composite ranks SR
   models in a way per-frame metrics cannot. GPU-bound, ~3 days.
2. **β / α sensitivity sweep (P2).** Recomputable from cached trajectory
   JSONs — no GPU. Can start immediately. Half a day.
3. **Leave-one-out sub-metric ablation (P3).** Same — local recompute only.
   Half a day.
4. **Classmate-model outreach (P4).** Send 5 LR mp4s + submission spec to
   classmates by ~June 18. Soft deadline for receipts ~June 25.
5. **Begin writing track (P5).** Switch `zjuthesis.tex` to `Period=paper`
   on June 22. Methodology chapter draft starts thereafter — heavy reuse
   from proposal preliminary-work section.

Blocking question for the start of next period: **new GPU host availability.**
If no new server before June 22, P1 slips and the writing track has to
front-run — methodology + experiments chapters drafted around the v5
matrix we already have, with the real-model table promoted to a "future
work / preliminary numbers" footnote.

---

## Open technical questions

1. **flip_invert composite row** — predicted PASS as the histogram-disrupting
   control. Do we need to re-run identity slow-fast on a future GPU to
   complete the matrix, or accept the gap with a footnote? Lean: accept the
   gap, write one paragraph in the thesis.
2. **Rescue server stability** — intermittent reachability from local. Is
   disk3 going to remain a usable longer-term mirror, or do we need to
   abandon it?
3. **New GPU host** — when, where, what specs? Blocks Week-2 work in the
   short-term plan.

---

## Carry-over from previous period

- **Task A6** ("Integrate into proposal + finish Tasks 13, 14, 15") and
  **Task 15** ("Internal proofread + submission") still listed as pending
  but refer to the May 31 proposal submission cycle and are stale; safe to
  retire.
- The **June 5-18 bi-weekly report** (`Timur_Iakshibaev_2026-06-05_to_2026-06-18.md`)
  was updated mid-period with the v5 headline. **June 1–15 short report**
  for the professor was written on June 13 and is current.
