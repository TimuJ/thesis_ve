# Meeting Agenda — April 9, 2026

## Attendees
- Timur (student)
- PhD student supervisor
- Supervisor (if present)

## Presentation
- 18-slide baseline methods presentation (MGLD-VSR + Upscale-A-Video)
- Update slides with UAV RealBasicVSR gap findings + DOVE cross-validation before meeting

## Key Questions to Raise

### 1. Research Direction
- What specific architecture/approach should we pursue?
- Current fallback: "SSM replaces attention in diffusion VSR pipeline" — is this aligned?
- Any specific papers or methods the PhD student wants to build on?

### 2. Long-Video Sample Data
- Can the PhD student provide sample long-video data at this meeting?
- We need it to run baselines on long videos and document failure modes
- This is blocking Task 8 in our plan

### 3. Gov Company Dataset Status
- Any update on the full long-video dataset?
- If not coming soon, we need to commit to a fallback:
  - YouTube-VOS long clips
  - MovieNet
  - Stitched REDS sequences
  - Other public long-video sources?

### 4. Proposal Rewrite
- Is a proposal rewrite required for the new VSR topic?
- If yes, estimated ~2-3 days (reformatting method spec + lit review)

### 5. UAV Degradation Gap (share findings)
- UAV shows 5.8 dB PSNR gap on RealBasicVSR LQ (24.94 vs paper 30.79)
- Pipeline verified via DOVE paper cross-validation (23.22 vs DOVE's 21.72 for UAV)
- Param sweep ruled out hyperparameters (~0.9 dB variation)
- Root cause: degradation mismatch (random seeds)
- Question: is this expected? Do they know what seeds/params UAV authors used?
- Our approach: consistent degradation (seed 42) across all methods for fair comparison

### 6. Thesis Title
- Preliminary Chinese title: 基于状态空间模型的长视频超分辨率方法研究
- English: State-Space Model Based Video Super-Resolution for Long Videos
- Should this change based on confirmed research direction?

## Notes (fill in during meeting)

### Research direction decided:


### Long-video data:


### Proposal rewrite:


### Dataset plan:


### Other action items:

