# Local Work During Server Downtime — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce Introduction chapter full draft, Literature Review chapter (sections 1-2 full, sections 3-4 skeletons), populated ref.bib, and updated thesis metadata during ~1 week server downtime.

**Architecture:** Sequential writing — build `ref.bib` first (citation database), then thesis metadata, then Introduction, then Literature Review sections. Each chapter/subsection compiles independently with `latexmk` before commit. Use `\todo{}` markers for gaps needing PhD student input.

**Tech Stack:** LaTeX (XeLaTeX via latexmk), BibTeX, zjuthesis template

**Spec:** `docs/superpowers/specs/2026-04-16-local-work-during-server-downtime-design.md`

---

## File Map

**Modify:**
- `zjuthesis/body/ref.bib` — Add VSR/diffusion/SSM BibTeX entries (keep existing template refs)
- `zjuthesis/zjuthesis.tex` — Update title/topic/Chinese title fields
- `zjuthesis/body/graduate-eng/introduction.tex` — Full rewrite for VSR topic (currently has VOS content from previous thesis)
- `zjuthesis/body/graduate-eng/literature-review.tex` — Full rewrite (currently template placeholder)

**No new files created.** All work is in existing zjuthesis template files.

---

## Task 1: Verify LaTeX toolchain works

**Files:**
- No modifications

- [ ] **Step 1: Compile current thesis state**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve/zjuthesis
latexmk
```

Expected: produces `out/zjuthesis.pdf` with no fatal errors. Warnings about undefined citations or missing figures are expected and OK.

- [ ] **Step 2: Verify PDF exists**

```bash
ls -lh /Users/ana/Desktop/Timur/thesis_ve/zjuthesis/out/zjuthesis.pdf
```

Expected: file exists and is non-zero size.

- [ ] **Step 3: If compile fails, fix toolchain issue before proceeding**

Common issues:
- Missing XeLaTeX: install via MacTeX
- Missing biber: `brew install biber`
- Missing latexmk: `brew install latexmk`

Do not proceed to Task 2 until `latexmk` succeeds.

---

## Task 2: Add `todonotes` package for gap markers

**Files:**
- Modify: `zjuthesis/zjuthesis.tex`

We'll use `\todo{}` macros throughout chapters to mark gaps needing PhD student input. The `todonotes` package renders these as margin notes.

- [ ] **Step 1: Check if todonotes is already loaded**

```bash
grep -n "todonotes" /Users/ana/Desktop/Timur/thesis_ve/zjuthesis/zjuthesis.tex /Users/ana/Desktop/Timur/thesis_ve/zjuthesis/zjuthesis.cls 2>&1
```

Expected: if no results, we need to add it.

- [ ] **Step 2: If not loaded, add to `zjuthesis.tex` preamble**

Find the `\documentclass{zjuthesis}` line. After it, add:

```latex
\usepackage[colorinlistoftodos,prependcaption,textsize=tiny]{todonotes}
```

Use Edit tool to insert after the `\documentclass{zjuthesis}` line.

- [ ] **Step 3: Verify compilation still works**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve/zjuthesis
latexmk
```

Expected: PDF still produces without errors.

- [ ] **Step 4: Commit**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve
git add zjuthesis/zjuthesis.tex
git commit -m "thesis: add todonotes package for gap markers"
```

---

## Task 3: Update thesis metadata in zjuthesis.tex

**Files:**
- Modify: `zjuthesis/zjuthesis.tex`

Update title, topic, and Chinese title fields for the VSR topic.

- [ ] **Step 1: Read current metadata fields**

Read the file and locate the `\Title`, `\TitleEng`, `\Topic`, `\TopicEng` commands (or similar — field names may vary by template version).

```bash
grep -n "Title\|Topic" /Users/ana/Desktop/Timur/thesis_ve/zjuthesis/zjuthesis.tex | head -20
```

- [ ] **Step 2: Update fields with VSR title**

Use Edit tool to update. Preliminary titles from April 9 meeting agenda:

| Field | Value |
|-------|-------|
| Chinese title | 基于状态空间模型的长视频超分辨率方法研究 |
| English title | State-Space Model Based Video Super-Resolution for Long Videos |
| Topic (CN) | 视频超分辨率 |
| Topic (EN) | Video Super-Resolution |

Note: titles are preliminary; they can be refined later when method is confirmed.

- [ ] **Step 3: Verify compilation**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve/zjuthesis
latexmk
```

Expected: PDF produces. Title page shows new VSR title.

- [ ] **Step 4: Visual check PDF**

Open `zjuthesis/out/zjuthesis.pdf` and verify:
- Title page shows VSR title (not old VOS title)
- Chinese title renders correctly (CJK fonts OK)

- [ ] **Step 5: Commit**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve
git add zjuthesis/zjuthesis.tex
git commit -m "thesis: update metadata for VSR topic (preliminary title)"
```

---

## Task 4: Populate ref.bib with VSR bibliography

**Files:**
- Modify: `zjuthesis/body/ref.bib`

Add VSR, diffusion, and SSM references. Keep existing template refs at top; append new entries. Entries are drafted in `docs/superpowers/plans/2026-04-05-phase1-foundation.md` Task 4 — we verify each before adding.

- [ ] **Step 1: Read the draft BibTeX entries**

Source: `docs/superpowers/plans/2026-04-05-phase1-foundation.md` lines 84-347.

These are the 30 entries to add, organized into categories:
- CNN-based VSR (4)
- Transformer-based VSR (3)
- Diffusion-based VSR (4 — note: one duplicate `yeung`/`yang` mgldvsr, remove the duplicate)
- Image SR foundations (4)
- Diffusion models (3)
- State-space models (4)
- SSM in vision (2)
- Optical flow / alignment (2)
- Evaluation / metrics (2)
- Benchmarks / datasets (4)
- General deep learning (2)

- [ ] **Step 2: Verify each BibTeX entry against official source**

For each entry in the draft, do a quick sanity check:
- Title matches the paper
- Author list is correct
- Venue and year match
- Remove duplicate `yang_mgldvsr_2024` (same paper as `yeung_mgldvsr_2024`, different author romanization)

Use WebFetch if needed to check DBLP or Semantic Scholar for correct entries.

- [ ] **Step 3: Append entries to `zjuthesis/body/ref.bib`**

Use Edit tool. Add after the last existing entry with a clear section header:

```bibtex
% ============================================================
% === VSR Thesis References (added 2026-04-16) ==============
% ============================================================

% --- Video Super-Resolution: CNN-based ---
<entries>

% --- Video Super-Resolution: Transformer-based ---
<entries>

% --- Video Super-Resolution: Diffusion-based ---
<entries>

% --- Image Super-Resolution foundations ---
<entries>

% --- Diffusion Models ---
<entries>

% --- State-Space Models ---
<entries>

% --- SSM in Vision ---
<entries>

% --- Optical Flow / Alignment ---
<entries>

% --- Evaluation / Metrics ---
<entries>

% --- Benchmarks / Datasets ---
<entries>

% --- General Deep Learning ---
<entries>
```

- [ ] **Step 4: Add DOVE paper reference**

DOVE is referenced in our alignment work but not in the Phase 1 draft. Add:

```bibtex
@inproceedings{chen_dove_2025,
  title     = {{DOVE}: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution},
  author    = {Chen, Zheng and Zou, Zichen and Zhang, Kewei and Su, Xiongfei and Yuan, Xin and Guo, Yong and Zhang, Yulun},
  booktitle = {NeurIPS},
  year      = {2025}
}
```

- [ ] **Step 5: Verify compilation**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve/zjuthesis
latexmk
```

Expected: PDF produces. BibTeX warnings about unused entries are OK (citations come with chapters).

- [ ] **Step 6: Commit**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve
git add zjuthesis/body/ref.bib
git commit -m "refs: add VSR/diffusion/SSM bibliography for thesis"
```

---

## Task 5: Write Introduction chapter — Section 1 (Background and Motivation)

**Files:**
- Modify: `zjuthesis/body/graduate-eng/introduction.tex`

Replace the existing VOS-era content with VSR content. This task handles the first section only; subsequent sections in Task 6.

- [ ] **Step 1: Clear existing content**

Use Edit tool to replace the entire file contents with just the chapter heading:

```latex
\chapter{Introduction}

\section{Background and Motivation}

% Section 1 content goes here in next step
```

- [ ] **Step 2: Write Background and Motivation section**

Replace the file with the content below (use Write tool since we're replacing the whole file):

```latex
\chapter{Introduction}

\section{Background and Motivation}

Video super-resolution (VSR) aims to reconstruct high-resolution (HR) video sequences from their
low-resolution (LR) counterparts, and has become a fundamental task in computer vision with
applications ranging from surveillance enhancement to video streaming and film restoration.
Unlike single image super-resolution, VSR methods must exploit temporal correlations
across frames to produce spatially sharp and temporally consistent outputs.

The field has progressed through several paradigm shifts. Early approaches relied on optical
flow-based alignment followed by convolutional fusion~\cite{wang_edvr_2019, chan_basicvsr_2021}.
Transformer-based methods such as VRT~\cite{liang_vrt_2022} and RVRT~\cite{liang_rvrt_2024}
introduced attention mechanisms for implicit alignment, achieving state-of-the-art quality on
standard benchmarks. Most recently, diffusion-based approaches~\cite{zhou_upscale_a_video_2024,
yeung_mgldvsr_2024} have demonstrated superior perceptual quality by leveraging pretrained
generative priors from large-scale image diffusion models.

However, a critical limitation pervades existing methods: \textbf{they are designed for and
evaluated on short video clips}, typically 30--100 frames (1--4 seconds). Real-world video
content --- films, surveillance footage, user-generated content --- routinely spans minutes to
hours. When applied to long videos ($>$1000 frames), current methods face fundamental challenges:

\begin{itemize}
    \item \textbf{Memory scaling:} Attention-based methods exhibit $O(n^2)$ memory complexity
          with respect to sequence length, making them impractical for long sequences on
          consumer or even professional hardware.
    \item \textbf{Temporal consistency degradation:} Recurrent methods accumulate errors over
          time, leading to flickering artifacts and color drift in long sequences.
    \item \textbf{Computational cost:} Diffusion-based methods require multiple denoising steps
          per frame, and their temporal modules add further overhead that scales poorly with
          video length.
\end{itemize}

State-space models (SSMs)~\cite{gu_s4_2022, gu_mamba_2023} offer a promising alternative for
long-range temporal modeling. With $O(n)$ time and memory complexity and the ability to maintain
persistent state across arbitrarily long sequences, SSMs have shown strong results in natural
language processing and, more recently, in vision tasks~\cite{zhu_vmamba_2024, shi_vmambair_2024}.
Recent work on long-context video world models~\cite{po_longcontext_ssm_2025} has demonstrated
that SSMs can effectively model temporal dependencies over thousands of frames, suggesting their
potential for long-video super-resolution. \todo{Verify this claim with PhD student materials;
they may have refined the argument for SSM applicability.}

This thesis investigates the application of state-space models to enable temporally consistent,
memory-efficient video super-resolution for long videos exceeding one minute in duration.
```

- [ ] **Step 3: Verify compilation**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve/zjuthesis
latexmk
```

Expected: PDF produces. All citations resolve (no undefined reference warnings for keys in ref.bib).

- [ ] **Step 4: Check PDF renders correctly**

Open `zjuthesis/out/zjuthesis.pdf` and verify:
- Chapter 1 renders as "Introduction"
- Section 1 renders as "Background and Motivation"
- Citations appear as `[1]`, `[2]` etc. (not `[?]`)
- Margin note from `\todo{}` visible

- [ ] **Step 5: Commit**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve
git add zjuthesis/body/graduate-eng/introduction.tex
git commit -m "thesis: rewrite Introduction §1 (Background and Motivation) for VSR"
```

---

## Task 6: Write Introduction chapter — Section 2 (Research Objectives) and Section 3 (Thesis Organization)

**Files:**
- Modify: `zjuthesis/body/graduate-eng/introduction.tex`

- [ ] **Step 1: Append Research Objectives section**

Use Edit tool to append after the Section 1 content (before the end of the file):

```latex

\section{Research Objectives}

The primary objectives of this research are:

\begin{enumerate}
    \item To analyze the failure modes of existing VSR methods (both diffusion-based and
          attention-based) when applied to long video sequences, quantifying degradation in
          temporal consistency, memory usage, and output quality as sequence length increases.
    \item To design a VSR architecture that leverages state-space models for efficient
          long-range temporal feature propagation, enabling processing of videos with
          $>$1000 frames without quality degradation. \todo{Refine once method is
          finalized with PhD student.}
    \item To evaluate the proposed method on both standard short-video benchmarks (UDM10, REDS,
          Vid4) for comparison with existing work, and on a long-video benchmark for validating
          the core contribution.
\end{enumerate}

\section{Contributions}

\todo{Finalize contributions after method is confirmed. Placeholder:}
The main contributions of this thesis are:
\begin{itemize}
    \item A systematic analysis of existing VSR methods' behavior on long video sequences.
    \item A novel architecture combining state-space temporal modeling with diffusion-based
          spatial super-resolution. \todo{Update once method is finalized.}
    \item Evaluation on both standard benchmarks and a dedicated long-video benchmark.
\end{itemize}

\section{Thesis Organization}

The remainder of this thesis is organized as follows:

\begin{itemize}
    \item \textbf{Chapter 2} reviews related work in video super-resolution, diffusion models,
          and state-space models, establishing the theoretical foundation for the proposed method.
    \item \textbf{Chapter 3} presents the proposed methodology, including the overall architecture,
          the state-space temporal module, and the training strategy.
    \item \textbf{Chapter 4} describes the experimental setup, presents quantitative and qualitative
          results on standard and long-video benchmarks, and provides ablation studies analyzing
          key design choices.
\end{itemize}
```

- [ ] **Step 2: Verify compilation**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve/zjuthesis
latexmk
```

Expected: PDF produces. All 4 sections of Introduction chapter visible.

- [ ] **Step 3: Commit**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve
git add zjuthesis/body/graduate-eng/introduction.tex
git commit -m "thesis: add Introduction §2-§4 (Objectives, Contributions, Organization)"
```

---

## Task 7: Write Literature Review chapter — intro paragraph and VSR CNN subsection

**Files:**
- Modify: `zjuthesis/body/graduate-eng/literature-review.tex`

Replace the existing template placeholder with a full lit review structure. This task handles the chapter intro and the CNN-based VSR subsection only.

- [ ] **Step 1: Replace file with chapter opening and CNN subsection**

Use Write tool to replace file contents:

```latex
\chapter{Literature Review}

This chapter reviews the key research areas that form the foundation of this thesis:
video super-resolution methods, diffusion models for image and video restoration,
and state-space models for efficient sequence modeling.

\section{Video Super-Resolution}

Video super-resolution has evolved through three major paradigms: convolutional neural
network (CNN)-based methods with explicit alignment, transformer-based methods with implicit
alignment, and diffusion-based methods with generative priors.

\subsection{CNN-Based Methods}

Early deep learning approaches to VSR focused on explicit motion compensation through optical
flow estimation followed by convolutional feature fusion. EDVR~\cite{wang_edvr_2019} introduced
deformable convolutions for alignment without explicit flow computation, using a pyramid-based
alignment module and temporal attention fusion. This approach demonstrated that learned alignment
could outperform optical flow-based warping for the super-resolution task.

BasicVSR~\cite{chan_basicvsr_2021} systematically analyzed the essential components of VSR
pipelines, finding that a bidirectional recurrent architecture with optical flow-based
propagation (using SpyNet~\cite{ranjan_spynet_2017}) achieves a strong balance of quality and
efficiency. Its successor, BasicVSR++~\cite{chan_basicvsrpp_2022}, added second-order grid
propagation and flow-guided deformable alignment, establishing new state-of-the-art results on
REDS~\cite{nah_reds_2019} and Vimeo-90K~\cite{xue_vimeo90k_2019}.

RealBasicVSR~\cite{chan_realbasicvsr_2022} addressed the domain gap between synthetic training
data and real-world degradations by introducing a practical two-stage degradation pipeline
(blur, noise, JPEG compression, video codec artifacts) that better simulates real-world
conditions. This degradation model has become the \textit{de facto} standard for evaluating
real-world VSR methods, including the baselines studied in this thesis.

The recurrent nature of CNN-based methods provides $O(n)$ memory complexity with respect to
sequence length, making them theoretically suitable for long videos. However, in practice,
error accumulation in the recurrent state leads to quality degradation over long sequences,
manifesting as temporal flickering and color drift.
```

- [ ] **Step 2: Verify compilation**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve/zjuthesis
latexmk
```

Expected: PDF produces. Citations resolve.

- [ ] **Step 3: Commit**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve
git add zjuthesis/body/graduate-eng/literature-review.tex
git commit -m "thesis: start Literature Review (intro + CNN-based VSR)"
```

---

## Task 8: Write Literature Review — Transformer and Diffusion VSR subsections

**Files:**
- Modify: `zjuthesis/body/graduate-eng/literature-review.tex`

- [ ] **Step 1: Append Transformer-based subsection**

Use Edit tool to append after the CNN-Based Methods subsection:

```latex

\subsection{Transformer-Based Methods}

Transformer-based VSR methods replace explicit flow-based alignment with attention mechanisms
that implicitly model inter-frame correspondences.

VRT~\cite{liang_vrt_2022} applies mutual attention across frames, where queries from one frame
attend to keys and values from neighboring frames. While effective, VRT processes fixed-size
temporal windows, limiting its ability to capture long-range dependencies. RVRT~\cite{liang_rvrt_2024}
addresses this by combining recurrent processing with transformer attention, using guided
deformable attention to focus computation on relevant spatial locations.

PSRT~\cite{shi_rethinking_psrt_2022} rethinks alignment in VSR transformers, proposing
patch-level alignment that reduces computational cost while maintaining quality.

The fundamental limitation of transformer-based methods for long videos is the $O(n^2)$ memory
and computation cost of self-attention. While windowed attention and recurrent designs mitigate
this, they introduce compromises: windowed attention limits the receptive field, and recurrent
designs reintroduce the error accumulation problem of CNN-based methods.
```

- [ ] **Step 2: Append Diffusion-based subsection**

Append after the Transformer subsection:

```latex

\subsection{Diffusion-Based Methods}

Diffusion models~\cite{ho_ddpm_2020, rombach_ldm_2022} have recently been applied to video
super-resolution, leveraging the rich generative priors learned from large-scale image generation.

Upscale-A-Video~\cite{zhou_upscale_a_video_2024} (CVPR 2024) adapts the Stable Diffusion
framework for temporal-consistent video upscaling. It introduces a local-global temporal
strategy: local temporal layers model short-range frame-to-frame consistency, while a
flow-guided recurrent latent propagation module ensures long-range coherence. The method
also employs a text-guided latent refinement step using CLIP features to improve semantic
consistency.

MGLD-VSR~\cite{yeung_mgldvsr_2024} (ECCV 2024) proposes motion-guided latent diffusion for
real-world VSR. Its key innovation is a motion-guided sampling strategy that conditions the
diffusion process on estimated optical flow fields, encouraging the denoising trajectory to
respect inter-frame motion patterns. This explicit motion guidance improves temporal consistency
compared to methods that rely solely on temporal attention. MGLD-VSR also employs a
degradation-aware prompt extractor that adapts the diffusion process to the specific
degradation characteristics of the input.

DOVE~\cite{chen_dove_2025} (NeurIPS 2025) advances this line of work by proposing a one-step
diffusion model for real-world VSR, fine-tuned from a pretrained video diffusion backbone
(CogVideoX). Unlike multi-step approaches, DOVE achieves competitive quality with a single
denoising step, offering a $28\times$ speedup over MGLD-VSR. The authors introduce a
latent-pixel training strategy and construct a high-quality video dataset (HQ-VSR) tailored
for video super-resolution.

Diffusion-based methods demonstrate superior perceptual quality compared to regression-based
approaches, but share a common limitation: their temporal modules are designed for short clips
and do not scale efficiently to long sequences. Upscale-A-Video processes clips of 45 frames;
MGLD-VSR operates on 32-frame segments; DOVE evaluates on 33-frame videos. None of these
methods have been evaluated on videos exceeding 100 frames.
```

- [ ] **Step 3: Verify compilation**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve/zjuthesis
latexmk
```

Expected: PDF produces. All citations resolve.

- [ ] **Step 4: Commit**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve
git add zjuthesis/body/graduate-eng/literature-review.tex
git commit -m "thesis: add Lit Review Transformer + Diffusion VSR subsections"
```

---

## Task 9: Write Literature Review — Diffusion Models section (§2)

**Files:**
- Modify: `zjuthesis/body/graduate-eng/literature-review.tex`

- [ ] **Step 1: Append Diffusion Models section**

Append after the Video Super-Resolution section:

```latex

\section{Diffusion Models}

Denoising Diffusion Probabilistic Models (DDPMs)~\cite{ho_ddpm_2020} learn to reverse a
gradual noising process, generating samples by iteratively denoising from Gaussian noise.
The forward process adds noise over $T$ steps according to a fixed schedule:
\begin{equation}
    q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})
\end{equation}
where $\beta_t$ is the noise schedule. The reverse process is parameterized by a neural
network $\epsilon_\theta$ trained to predict the noise:
\begin{equation}
    p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})
\end{equation}

Latent Diffusion Models (LDMs)~\cite{rombach_ldm_2022} improve efficiency by operating in a
compressed latent space obtained through a pretrained autoencoder. This reduces computational
cost by a factor of $4\times$--$8\times$ while maintaining generation quality, and enables the
use of powerful pretrained models such as Stable Diffusion for downstream tasks including
super-resolution.

DDIM~\cite{song_ddim_2021} provides a deterministic sampling procedure that allows fewer
denoising steps while maintaining quality, which is particularly important for video applications
where per-frame inference cost directly impacts total processing time.

For super-resolution, diffusion models are typically conditioned on the low-resolution input,
either by concatenating it with the noisy latent or by injecting it through cross-attention.
StableSR~\cite{wang_stablesr_2024} demonstrated that pretrained Stable Diffusion provides
strong priors for image SR, achieving state-of-the-art perceptual quality.
```

- [ ] **Step 2: Verify compilation**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve/zjuthesis
latexmk
```

Expected: PDF produces. Math equations render correctly.

- [ ] **Step 3: Commit**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve
git add zjuthesis/body/graduate-eng/literature-review.tex
git commit -m "thesis: add Lit Review Diffusion Models section"
```

---

## Task 10: Write Literature Review — SSM section (§3) skeleton

**Files:**
- Modify: `zjuthesis/body/graduate-eng/literature-review.tex`

This section is a skeleton only — detailed content to be filled in from PhD student materials when available.

- [ ] **Step 1: Append SSM section skeleton**

Append after the Diffusion Models section:

```latex

\section{State-Space Models}

\todo{This section is a skeleton. Detailed content will be expanded based on PhD student's
literature review materials. Preliminary framing below.}

State-space models provide an alternative to attention for sequence modeling, offering
$O(n)$ time and memory complexity with respect to sequence length.

\subsection{Foundations}

\todo{Expand with PhD student materials.}

The Structured State Space sequence model (S4)~\cite{gu_s4_2022} parameterizes a continuous-time
linear system:
\begin{equation}
    h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t), \quad y(t) = \mathbf{C}h(t) + \mathbf{D}x(t)
\end{equation}
where $\mathbf{A} \in \mathbb{R}^{N \times N}$ is the state matrix, and $h(t)$ is the hidden
state. S4 initializes $\mathbf{A}$ with the HiPPO matrix, which provides optimal polynomial
approximation of input history, enabling long-range dependency modeling.

Mamba~\cite{gu_mamba_2023} extends S4 with input-dependent (selective) state transitions,
where the matrices $\mathbf{B}$, $\mathbf{C}$, and the discretization step $\Delta$ are
functions of the input. Mamba-2~\cite{dao_mamba2_2024} further refines this by establishing
a formal connection between SSMs and attention through structured state space duality.

\subsection{SSMs in Vision}

\todo{Expand with PhD student materials.}

Vision Mamba~\cite{zhu_vmamba_2024} adapts the Mamba architecture for visual representation
learning, introducing bidirectional scanning strategies to handle the 2D nature of images.
VMambaIR~\cite{shi_vmambair_2024} applies this to image restoration.

\subsection{Relevance to Long-Video Super-Resolution}

\todo{Refine this argument once research direction is finalized.}

The properties of SSMs align with the requirements of long-video VSR: linear complexity,
persistent memory, and selective processing. To our knowledge, no existing work applies SSMs
as the temporal backbone for diffusion-based video super-resolution. This thesis explores
this combination, hypothesizing that SSM-based temporal propagation can maintain the perceptual
quality of diffusion-based SR while scaling efficiently to long videos.
```

- [ ] **Step 2: Verify compilation**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve/zjuthesis
latexmk
```

Expected: PDF produces. `\todo{}` markers render as margin notes.

- [ ] **Step 3: Commit**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve
git add zjuthesis/body/graduate-eng/literature-review.tex
git commit -m "thesis: add Lit Review SSM section skeleton (pending PhD materials)"
```

---

## Task 11: Write Literature Review — Long-Video Processing section (§4) skeleton

**Files:**
- Modify: `zjuthesis/body/graduate-eng/literature-review.tex`

- [ ] **Step 1: Append Long-Video Processing section skeleton**

Append after the SSM section:

```latex

\section{Long-Video Processing}

\todo{This section awaits empirical evidence from long-video experiments (Task 8 in Phase 1
plan). Fill with concrete numbers once baselines are evaluated on long sequences.}

While VSR methods report strong performance on short clips of 30--100 frames, the community
has given limited attention to videos of longer durations. Most published benchmarks
(UDM10~\cite{yi_udm10_2019}, Vid4~\cite{liu_vid4_2013}, REDS~\cite{nah_reds_2019},
Vimeo-90K~\cite{xue_vimeo90k_2019}) contain clips of 7--100 frames. As a consequence, the
failure modes of existing methods on longer sequences remain largely unexplored.

Relevant work on long-context video modeling exists in adjacent areas. Po et al.~\cite{po_longcontext_ssm_2025}
demonstrated that state-space models can maintain spatial consistency over thousands of frames
in a video world model context. \todo{Add more long-video-specific references as they emerge
from PhD student survey or new publications.}

This thesis contributes to filling this gap by (1) systematically evaluating existing VSR
methods on long sequences to document their failure modes, and (2) proposing an architecture
that scales efficiently to long videos. \todo{Refine once long-video benchmark results are
available.}
```

- [ ] **Step 2: Verify compilation**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve/zjuthesis
latexmk
```

Expected: PDF produces. Literature Review chapter complete with 4 sections.

- [ ] **Step 3: Final PDF visual check**

Open `zjuthesis/out/zjuthesis.pdf` and verify:
- Chapter 1 (Introduction) has 4 sections, all readable
- Chapter 2 (Literature Review) has 4 sections
- Margin notes from `\todo{}` visible throughout
- All citations resolve (no `[?]` entries)
- Table of Contents updated

- [ ] **Step 4: Commit**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve
git add zjuthesis/body/graduate-eng/literature-review.tex
git commit -m "thesis: add Lit Review Long-Video Processing section skeleton"
```

---

## Task 12: Final review and summary commit

**Files:**
- No code changes

- [ ] **Step 1: Final compilation sanity check**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve/zjuthesis
latexmk -C  # clean aux files
latexmk
```

Expected: clean build produces PDF with no fatal errors. Biber warnings about unused bibentries are OK.

- [ ] **Step 2: Word count check**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve/zjuthesis/body/graduate-eng
wc -w introduction.tex literature-review.tex
```

Expected targets (rough): Introduction ~1500-2500 words; Literature Review ~3500-5000 words.

- [ ] **Step 3: Review `\todo{}` markers remaining**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve/zjuthesis
grep -n "todo{" body/graduate-eng/introduction.tex body/graduate-eng/literature-review.tex
```

Expected: several markers in SSM section, Long-Video section, Introduction §4 (Contributions), and Introduction §2 (method details). These are intentional gaps for PhD student materials.

- [ ] **Step 4: Push commits**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve
git log --oneline -15  # verify clean commit history
git push
```

Expected: all commits pushed to GitHub.

- [ ] **Step 5: Update weekly report with progress**

Append to `reports/Timur_Iakshibaev_2026-04-13_to_2026-04-26.md` a section documenting:
- Server downtime context
- Local work completed: ref.bib, metadata, Introduction chapter, Literature Review (2 full sections + 2 skeleton sections)
- Remaining gaps marked as `\todo{}` for when PhD student materials arrive

- [ ] **Step 6: Commit weekly report update**

```bash
cd /Users/ana/Desktop/Timur/thesis_ve
git add reports/Timur_Iakshibaev_2026-04-13_to_2026-04-26.md
git commit -m "docs: update weekly report with local thesis writing progress"
git push
```

---

## Summary

On completion, this plan produces:
- `zjuthesis/body/ref.bib` — ~31 VSR/diffusion/SSM BibTeX entries added
- `zjuthesis/zjuthesis.tex` — Updated metadata for VSR topic, `todonotes` loaded
- `zjuthesis/body/graduate-eng/introduction.tex` — 4-section draft (Background, Objectives, Contributions, Organization)
- `zjuthesis/body/graduate-eng/literature-review.tex` — 4-section draft (VSR, Diffusion Models, SSM skeleton, Long-Video skeleton)
- Weekly report updated
- All commits pushed to GitHub

**Total estimated commits:** 10–12

**When server returns:** resume experimental work in parallel. Fill SSM and Long-Video `\todo{}` markers as PhD student materials and long-video data arrive.
