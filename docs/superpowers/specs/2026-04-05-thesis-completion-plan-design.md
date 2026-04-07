# Thesis Completion Plan — Video Super-Resolution for Long Videos

**Date:** 2026-04-05
**Last updated:** 2026-04-07
**Status:** In Progress — Phase 1
**Deadline:** July 15, 2026 (blind review submission)
**Final deadline:** September 30, 2026

## Context

Master's thesis at ZJU on 4x super-resolution for long HD videos (>1 minute). Collaboration with PhD student supervisor who specializes in diffusion-based models. Plan is to co-write a paper first, derive thesis from it — but thesis submission does not depend on paper acceptance.

### Current State (April 7, 2026)
- **MGLD-VSR baseline:** VERIFIED (UDM10 PSNR 26.48 vs paper 25.99, VideoLQ NR close to paper)
- **Upscale-A-Video baseline:** PARTIALLY VERIFIED — pipeline confirmed via DOVE cross-validation, but ~5.8 dB gap on RealBasicVSR LQ (degradation mismatch, not a bug). VideoLQ NR evaluation in progress (2/50 clips)
- **Param sweep:** 5 configs tested, ruled out hyperparameters as cause of gap (~0.9 dB variation)
- **Research direction:** "SSM + diffusion for long-video VSR" — specifics TBD after April 9 meeting
- **Dataset:** Sample long-video data from PhD student; full dataset (gov company) promised but delayed
- **Thesis chapters:** Contain old VOS content, need complete rewrite
- **Literature review:** Not started for VSR topic

### Approach
Parallel writing + research (Approach B) with dataset de-risking (Approach C). Start writing intro and lit review before method is confirmed — the "long-video VSR problem" framing is stable regardless of method. Run existing baselines on long videos early to document where they fail.

---

## Phase 1: Foundation (April 5 → April 18) — 2 weeks

**Goals:** Finish baseline verification, deep literature review, get research direction from PhD student.

| Task | Deadline | Status | Notes |
|------|----------|--------|-------|
| Finish UAV UDM10 evaluation | Apr 6 | **DONE** | PSNR 24.94 vs paper 30.79 — degradation mismatch |
| Run UAV on YouHQ40-RealBasicVSR | Apr 8 | **DONE** | PSNR 23.40 vs paper 25.83 — same issue |
| Investigate UAV gap (param sweep + DOVE cross-validation) | Apr 7 | **DONE** | Unplanned but necessary — pipeline verified, gap is degradation |
| Run UAV VideoLQ NR evaluation | Apr 11 | **RUNNING** | 2/50 clips, ~4 days remaining |
| Prepare & present baseline papers at April 9 meeting | Apr 9 | **READY** | 18-slide presentation done |
| Get sample long-video data from PhD student | Apr 9 | PENDING | Meeting tomorrow |
| Ask supervisor if proposal rewrite is required | Apr 9 | PENDING | Meeting tomorrow |
| Run both baselines on long videos — document where they fail | Apr 14 | NOT STARTED | Blocked on sample data from meeting |
| Deep literature review: SSM for video (Mamba, S4), diffusion VSR, long-video methods | Apr 18 | NOT STARTED | Start after Apr 9 meeting |
| Start writing: Introduction chapter (VSR problem, long-video challenge) | Apr 18 | NOT STARTED | Start after lit review |
| Start writing: Literature review chapter | Apr 18 | NOT STARTED | Start after lit review |

**Key output:** Research direction confirmed. "Baselines fail on long videos" documented. Intro + lit review drafts started.

### Phase 1 — Remaining priorities (Apr 8–18)
1. **Apr 9 meeting** — present baselines, get long-video sample data, clarify research direction with PhD student
2. **Apr 9–14** — run baselines on long videos (if data received), start literature review
3. **Apr 14–18** — deep literature review, begin Introduction and Literature Review chapter drafts

---

## Phase 2: Method Design (April 19 → May 2) — 2 weeks

**Goals:** Design the core method with PhD student guidance, finalize contribution claim.

| Task | Deadline | Depends on |
|------|----------|------------|
| Synthesize PhD student's direction into 2-3 candidate architectures | Apr 23 | Phase 1 meeting |
| Discuss candidates with PhD student, pick one | Apr 25 | Above |
| Write detailed method spec (architecture, data flow, training strategy, loss functions) | Apr 30 | Direction confirmed |
| Secure full long-video dataset (follow up on gov data if still missing) | Apr 30 | PhD student |
| Continue writing: Methodology chapter first draft (write as you design) | May 2 | Method spec |
| Proposal rewrite (if required) — derive from method spec + lit review | May 2 | Supervisor answer |

**Risk mitigations:**
- If PhD student's direction is vague: fall back to "SSM temporal backbone replacing attention in existing diffusion VSR pipeline"
- If gov dataset still not available: use public long-video sources (YouTube-VOS long clips, MovieNet, or stitch REDS sequences) as interim benchmark
- If proposal is required: mostly reformatting method spec + lit review, ~2-3 days extra

**Key output:** Method spec document. Methodology chapter draft. Dataset secured or fallback identified.

---

## Phase 3: Implementation + Experiments (May 3 → June 13) — 6 weeks

### Phase 3a: Core Implementation (May 3 → May 23) — 3 weeks

| Task | Deadline | Depends on |
|------|----------|------------|
| Set up training pipeline (dataloader, training loop, logging) | May 9 | Method spec |
| Implement core model architecture | May 16 | Above |
| Train on small dataset subset, verify it converges | May 19 | Above |
| First inference on UDM10 — compare against baselines | May 23 | Above |

### Phase 3b: Full Experiments + Ablations (May 24 → June 13) — 3 weeks

| Task | Deadline | Depends on |
|------|----------|------------|
| Full training run on complete dataset | May 30 | 3a verified |
| Evaluate on all standard benchmarks (UDM10, REDS, Vid4) | Jun 3 | Trained model |
| Long-video evaluation — the core thesis claim | Jun 6 | Trained model + dataset |
| Ablation studies (SSM vs attention, sequence length scaling, etc.) | Jun 10 | Full results |
| Efficiency comparison (FPS, VRAM vs baselines on long videos) | Jun 10 | Above |
| Write: Experiments chapter as results come in | Jun 13 | Results |

**Risk mitigations:**
- If training doesn't converge by May 19: simplify architecture, reduce to proven components, consult PhD student immediately — don't spend more than 1 week debugging
- If results don't beat baselines on short videos: OK if they beat baselines on long videos (the thesis claim)
- If full dataset still unavailable: run long-video experiments on stitched/public sequences, note as limitation

**Key output:** Trained model with results on standard + long-video benchmarks. Experiments chapter draft. Ablation tables.

---

## Phase 4: Writing + Polish (June 14 → July 15) — 4.5 weeks

### Phase 4a: Complete Drafts (June 14 → June 27) — 2 weeks

| Task | Deadline | Depends on |
|------|----------|------------|
| Finalize Introduction chapter (add results references, contributions list) | Jun 18 | Experiments done |
| Finalize Literature Review chapter | Jun 20 | — |
| Finalize Methodology chapter (implementation details, architecture figures) | Jun 24 | Final model |
| Finalize Experiments chapter (all tables, figures, analysis) | Jun 27 | All results |

### Phase 4b: Integration + Review (June 28 → July 15) — 2.5 weeks

| Task | Deadline | Depends on |
|------|----------|------------|
| Write Abstract + Conclusion | Jun 30 | All chapters done |
| Create all figures (architecture diagrams, comparison visuals, charts) | Jul 3 | — |
| Internal review pass — consistency, flow, grammar | Jul 5 | Full draft |
| Send to PhD student + supervisor for review | Jul 6 | Above |
| Address reviewer feedback | Jul 12 | Feedback received |
| Final formatting, BlindReview=true, compile clean | Jul 14 | — |
| **Submit for blind review** | **Jul 15** | — |

**Risk mitigations:**
- If experiments run late: prioritize experiments chapter over polishing intro/lit review
- If supervisor feedback is slow: submit what you have — blind review can be improved for final (Sep 30)
- Buffer: blind review is not final; Sep 30 deadline gives 2.5 months for revisions

---

## Top 3 Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Gov dataset never arrives | No long-video experiments | Use public long-video sources or stitched sequences; document as limitation |
| PhD student direction too vague after Apr 9 | Method design stalls | Fall back to "SSM replaces attention in diffusion VSR"; schedule weekly check-ins |
| Training doesn't converge | No results for thesis | Simplify after 1 week max; worst case, contribution shifts to long-video evaluation benchmark + baseline analysis |

## Immediate Actions (before April 9)

1. Finish UAV UDM10 evaluation (already running)
2. Start literature review on SSMs for video (Mamba, S4, S5)
3. Start writing Introduction chapter ("long-video VSR is unsolved" framing)
4. Get sample long-video data at the April 9 meeting
