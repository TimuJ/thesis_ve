---
name: findings-note
description: Use when writing up the result of an investigation, probe, audit, or experiment as a note under docs/notes/ — an evidence-grounded record other researchers can interrogate, with method, numbers, honest limitations, and a status line.
---

# Findings Note

## Overview

A `docs/notes/` note records what an investigation found, for future readers (and
your future self) to interrogate. Core principle: **evidence first, honest
always** — every claim carries its number, limitations are stated not buried, and
negative results are reported as results.

## When to use

- After a probe / audit / experiment produces a result worth keeping.
- NOT for progress reporting to people → that's `lab-meeting-report` /
  `supervisor-biweekly-report`. A note is the durable technical record the reports
  cite.

## File

`docs/notes/<YYYY-MM-DD>-<kebab-slug>.md`. Commit with the figures/data it cites.

## Structure (recipe, in order)

1. **Title + Status line** — what is complete vs pending, and where the
   artifacts live (figure paths, server result paths, commit).
2. **What prompted this** — the question, in one short paragraph.
3. **Method** — enough to reproduce: inputs, parameters, tool invocations, how
   raw outputs were aggregated.
4. **Result** — tables with the actual numbers, then the finding in prose.
5. **Reading / interpretation** — the *mechanism* (why the result holds), and
   what it does/doesn't support. State the comparison that matters.
6. **Honest limitations** — a REQUIRED section. Sample size, saturation,
   estimates-vs-measured, single-run, confounds. If a hypothesis did not survive,
   record it here as a negative result.
7. **What this establishes / next** — the durable takeaway and the next step.

## Conventions

- **Flag estimates vs measured values** explicitly (e.g. "~1080p (est. from file
  size)").
- **Quote the number with the claim** — never "it rises" without the Δ.
- **Cross-reference** related notes and the commit that carries the change.
- Keep a negative result visible — the hypothesis that died is often the most
  useful line in the note.

## Common mistakes

- Omitting the limitations section, or softening it into the conclusion.
- Asserting a direction/effect without its magnitude.
- Writing a narrative of *how you worked* instead of *what holds*.

## Examples

- `docs/notes/2026-09-14-vbench-dimension-audit-findings.md` (battery audit)
- `docs/notes/2026-09-14-reduced-reference-probe-findings.md` (probe, with a
  surviving negative result)
