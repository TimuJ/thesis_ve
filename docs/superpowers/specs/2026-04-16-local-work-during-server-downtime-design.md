# Local Work During Server Downtime — Design

**Date:** 2026-04-16
**Context:** GPU server (223.109.239.43) is down for security investigation (suspected compromise via `zrk` user in another account). Expected downtime: up to ~1 week. We need to make meaningful thesis progress without server access.

## Goal

Produce a usable Introduction chapter draft and a partial Literature Review chapter draft during the server downtime. When the server comes back, we'll have concrete thesis progress that can continue in parallel with experimental work.

## Scope

### In Scope
- Populate `zjuthesis/body/ref.bib` with verified BibTeX entries for VSR/diffusion/SSM papers
- Update thesis metadata in `zjuthesis/zjuthesis.tex` (title, topic)
- Write full Introduction chapter (`zjuthesis/body/graduate-eng/introduction.tex`)
- Write Literature Review chapter (`zjuthesis/body/graduate-eng/literature-review.tex`) — sections on CNN/Transformer/Diffusion VSR and Diffusion Models
- Write skeletons (with `\todo{}` markers) for SSM and Long-Video sections of Literature Review

### Out of Scope
- **Proposal rewrite** — blocked by PhD student materials and lack of finalized research direction
- **Deep SSM literature survey** — PhD student is doing this; we'll integrate their reports later
- **Methodology / Experiments chapters** — research direction not finalized
- **Server-dependent experiments** — server is down

## Architecture

Three components produce the chapter drafts:

### Component 1: `ref.bib` — Citation database
- **Purpose:** Single source of truth for all BibTeX entries used in chapters
- **Starting point:** ~30 entries already drafted in `docs/superpowers/plans/2026-04-05-phase1-foundation.md` (Task 4)
- **Workflow:** Verify each entry against official source (DBLP, Semantic Scholar, or paper's project page) before adding to `ref.bib`
- **Naming scheme:** `author_short_year` (e.g., `chan_basicvsr_2021`, `gu_mamba_2023`)
- **Growth:** Incremental — add entries as chapters need them, not bulk pre-fill

### Component 2: Introduction chapter
- **File:** `zjuthesis/body/graduate-eng/introduction.tex`
- **Purpose:** Frame the VSR + long-video problem, motivate SSM-based approach, state thesis objectives and organization
- **Structure (4 sections):**
  1. **Background and Motivation** — VSR evolution (CNN → Transformer → Diffusion), long-video limitations in existing methods, SSMs as promising alternative
  2. **Research Objectives** — 3 bullets: (a) analyze failure modes of existing VSR on long videos, (b) design SSM-based architecture, (c) evaluate on short + long benchmarks
  3. **Thesis Organization** — Chapter overview (2 → Lit Review, 3 → Methodology, 4 → Experiments)
  4. **Contributions placeholder** — `\todo{finalize after method confirmed}`
- **Starting point:** LaTeX template already drafted in Phase 1 plan Task 6
- **Completeness:** ~80% doable now; SSM-specific claims marked with `\todo{verify with PhD student}`

### Component 3: Literature Review chapter
- **File:** `zjuthesis/body/graduate-eng/literature-review.tex`
- **Purpose:** Survey related work; establish research gap for long-video VSR
- **Structure (4 sections):**
  1. **Video Super-Resolution** — 3 subsections:
     - CNN-based (EDVR, BasicVSR, BasicVSR++, RealBasicVSR)
     - Transformer-based (VRT, RVRT, PSRT)
     - Diffusion-based (Upscale-A-Video, MGLD-VSR, StableSR, DOVE)
  2. **Diffusion Models** — DDPM, LDM, DDIM fundamentals; SR applications
  3. **State-Space Models** — `\todo{expand with PhD student materials}`; skeleton with S4, Mamba, VMambaIR stubs
  4. **Long-Video Processing** — `\todo{fill after long-video data arrives}`; note the research gap
- **Starting point:** LaTeX template already drafted in Phase 1 plan Task 7
- **Completeness:** Sections 1–2 fully writable now; sections 3–4 skeletons with explicit gaps

## Data Flow

```
Paper reading → extract key claims/metrics → write section in .tex
                                         ↓
                         add BibTeX entry to ref.bib (if new)
                                         ↓
                     cite via \cite{key} in section
                                         ↓
                       latexmk → verify PDF compiles
                                         ↓
                              git commit section
```

## Error Handling / Risks

| Risk | Mitigation |
|------|------------|
| LaTeX compilation errors | Compile after each section, not at end |
| Citation key mismatches | Keep ref.bib and chapters in sync; consistent naming scheme |
| Chapter drift from final research direction | Use hedging where uncertain; mark with `\todo{}`; structure is flexible |
| Overlap with PhD student's SSM work | Explicit boundary: we cover VSR & Diffusion; PhD student covers SSM depth |
| Fabricated citations | If unsure about paper details, add `\todo{verify}` marker — never make up numbers |

## Build Sequence

1. **Setup** — Verify LaTeX toolchain works (`cd zjuthesis && latexmk` on current state)
2. **ref.bib** — Populate with ~30 Phase 1 entries; verify each via DBLP/Semantic Scholar
3. **Metadata** — Update title/topic/Chinese title fields in `zjuthesis.tex`
4. **Commit** — `refs: populate VSR/diffusion/SSM bibliography` + `thesis: update metadata for VSR topic`
5. **Introduction** — Write full chapter; compile; commit
6. **Lit Review §1 (VSR)** — Write CNN + Transformer + Diffusion subsections; compile; commit
7. **Lit Review §2 (Diffusion Models)** — Write; compile; commit
8. **Lit Review §3 (SSM)** — Skeleton with `\todo{}` markers; compile; commit
9. **Lit Review §4 (Long-Video)** — Skeleton with `\todo{}` markers; compile; commit

Each step ends with a clean compile and a commit. If a step fails to compile, fix before moving on.

## Testing

- **LaTeX compilation** must succeed after every section: `cd zjuthesis && latexmk` produces PDF with no undefined references
- **Citation check:** every `\cite{key}` in chapters must have matching entry in `ref.bib`
- **PDF visual check:** open `zjuthesis/out/zjuthesis.pdf` periodically to verify formatting
- **Word count targets (rough):** Introduction ~2000 words; Literature Review ~4000 words

## Dependencies and Assumptions

**Assumed available:**
- LaTeX toolchain (XeLaTeX via latexmk) — should work locally on Mac
- `zjuthesis` template already bootstrapped from previous thesis
- Papers cited are accessible (arxiv, official project pages)

**Not assumed:**
- Server access
- PhD student materials (yet)
- Finalized research direction
- Long-video sample data

## Handoff

When server returns or PhD student provides materials:
- Server returns → resume experimental work in parallel (MGLD-VSR synthetic, UAV DOVE alignment, VBench)
- PhD student materials → fill Lit Review §3 (SSM) in place of `\todo{}` markers
- Long-video data → fill Lit Review §4 + enable Task 8 (long-video eval)
- Method finalized → update Introduction §4 (Contributions) + start Methodology chapter

## References

- `docs/superpowers/plans/2026-04-05-phase1-foundation.md` — Has full LaTeX templates for Introduction (Task 6) and Literature Review (Task 7), plus draft ref.bib (Task 4)
- `docs/superpowers/specs/2026-04-05-thesis-completion-plan-design.md` — Overall thesis plan
- `docs/meeting-notes/2026-04-09-meeting-agenda.md` — Research direction discussion
- `docs/private/server-incident-2026-04-16.md` — Server downtime context
