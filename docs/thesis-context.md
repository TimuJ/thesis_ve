# Thesis Context & Research Notes

## Situation Overview

Timur changed thesis topic in March 2026 from "Reasoning Video Segmentation" to "Video Super-Resolution for Long Videos". The change was driven by:
- Opportunity to work closely with a PhD student who specializes in diffusion-based models
- Plan to co-write a paper first, then derive the thesis from it
- Extension of study period to September 30, 2026

## Key People

- **Timur Iakshibaev** — Master's student, CS, Zhejiang University
- **PhD student supervisor** — specializes in diffusion-based models, will provide close guidance. Shared arxiv 2505.20171 as reference direction. No specific method details provided yet (as of March 21, 2026).
- **Professor** — thesis advisor, quality bar is CCF-B/ECCV level

## Timeline

| Date | Milestone |
|------|-----------|
| Mar 21, 2026 | Topic change, repo setup |
| ~Apr 2026 | Research direction confirmed with PhD student |
| ~May-Jun 2026 | Paper writing + experiments |
| **Jul 15, 2026** | **Thesis blind review submission** |
| Jul-Aug 2026 | Review period |
| Aug-Sep 2026 | Revisions based on feedback |
| **Sep 30, 2026** | **Final thesis submission** |

## Research Direction

### Reference Paper: arxiv 2505.20171
**"Long-Context State-Space Video World Models"** (Po, Nitzan, Zhang, Chen, Dao, Shechtman, Wetzstein, Huang)

Key ideas:
- State-space models (SSMs) for efficient long-range temporal memory in video
- Block-wise SSM scanning scheme balancing spatial consistency vs temporal memory
- Dense local attention for frame-to-frame coherence
- Leverages SSMs' causal sequence modeling strengths
- Tested on Memory Maze and Minecraft for long-range spatial retrieval

### How This Connects to VSR
- **Long video challenge:** Attention-based VSR methods (VRT, RVRT) hit O(n^2) memory walls on long sequences
- **SSM opportunity:** SSMs provide O(n) temporal modeling — natural fit for >1 minute videos
- **Diffusion quality:** Diffusion models produce highest quality SR but are slow; SSM temporal backbone could make them viable for long videos
- **Potential thesis angle:** SSM-based temporal feature propagation + diffusion-based frame enhancement for long-video VSR

### VSR Landscape (State of the Art as of early 2026)
| Method | Type | Key Feature |
|--------|------|-------------|
| BasicVSR++ | Recurrent CNN | Flow-guided propagation, second-order grid propagation |
| VRT | Transformer | Mutual attention, parallel warping |
| RVRT | Recurrent Transformer | Recurrent + transformer, guided deformable attention |
| PSRT | Transformer | Patch alignment + recurrent transformer |
| Upscale-A-Video | Diffusion | Temporal-aware diffusion for video upscaling |
| StableSR | Diffusion | Stable Diffusion prior for image SR |

## Lessons from Previous Thesis

### What Worked
1. **Modular code structure** — clean `src/` with separate modules for data, evaluation, models
2. **Central path config** — `src/configs/paths.py` with env var overrides for local/GPU portability
3. **Early testing** — pytest tests caught real bugs (contour resampling edge cases)
4. **Ablation-first approach** — running ablations early revealed that linear > Kriging, causing a productive thesis pivot
5. **GPU experiment scripts** — automated bash scripts + README made lab runs reproducible

### What Didn't Work / Lessons
1. **Building before confirming direction** — built full Kriging pipeline before discovering linear interpolation was better. This time: research first, code second.
2. **Topic overcommitment** — committed to Kriging as core contribution too early. Keep contribution claims flexible until experiments confirm.
3. **Font issues** — zjuthesis template needs specific fonts installed; check this early on any new machine.

## Proposal Requirements

A new thesis proposal will need to be written for the VSR topic. The `proposal/` directory contains the previous proposal structure (pdfLaTeX-based) which can be adapted:
- `chapters/introduction_background.tex` — rewrite for VSR background
- `chapters/statement_problem.tex` — reframe for long-video SR challenges
- `chapters/aim_hypothesis.tex` — new hypotheses around SSM + diffusion
- `chapters/research_objectives.tex` — new objectives
- `references.bib` — replace with VSR literature

## Open Questions (to discuss with PhD student)
1. Which diffusion architecture? (latent diffusion, pixel-space, flow-matching)
2. Which SSM variant? (Mamba, Mamba-2, S4, S5)
3. Scale factor? (x2, x4, x8?)
4. Training data strategy for long videos?
5. Is the contribution the SSM temporal module, the full pipeline, or the long-video evaluation?
6. Paper target venue?

## April 9 Meeting Update
- PhD student still exploring i2v (image-to-video) papers — **research direction still TBD**
- Long-video sample data will be collected by another student
- Gov dataset provided but not public — internal tests only
- Proposal rewrite likely needed — **deadline May 31, 2026**
- **Evaluation strategy shift:** align with DOVE benchmark instead of individual paper numbers
- **VBench** needed for human-perception-aligned evaluation (OOM on long videos — beta exists)
