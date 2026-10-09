---
name: supervisor-biweekly-report
description: Use when preparing Timur's biweekly progress report for his supervisor, covering the period since the last supervisor report; produces both a long and a short version.
---

# Supervisor Biweekly Report

## Overview

The biweekly progress report for the supervisor. **Always two files — a long
version and a short version** — covering the period since the previous supervisor
biweekly. Framing is research-progress for a supervisor, not code-level detail
for labmates.

## When to use

- The supervisor asks for the biweekly / progress report.
- NOT the lab/group-meeting recap (different audience) → use `lab-meeting-report`.

## Conventions (project rules — load-bearing)

- **Always produce BOTH files.** Long + short. The short version is
  **summary-only** (no tables, no key-numbers block — narrative paragraphs).
- **Research-outcome framing; NO thesis mentions.**
- **No exact calendar dates in prose** — dates live only in the filenames. Use
  "this period", "since the last report", "next two weeks".
- **Files:** `reports/Timur_Iakshibaev_biweekly_<start>_to_<end>.md` and the same
  name with a `_short.md` suffix. `<start>` is the day after the previous report's
  end (or the day the previous one was sent). Commit and push when done.

## Long version — structure (recipe, in order)

- `# Biweekly Report — Timur Iakshibaev`
- `## The period in three headlines` — three bold-led items, the key developments.
- `## Key numbers` — the headline measured results, compact.
- `## Decisions and framing this period` — framing shifts and open decisions.
- `## Outcomes this period` — what was produced.
- `## Next two weeks` — numbered.
- `## One-line summary for the meeting`

## Short version — structure (recipe, in order)

- `# Progress Report — Timur Iakshibaev`
- `**Topic:** <one line naming the thread>`
- `## Summary` — 3–4 **bold-lead** narrative paragraphs (each opens with a bold
  claim sentence, then supports it). Summary only — no tables.
- `## Next Period` — numbered.

## Common mistakes

- Shipping only the long version — the short is **required**, not optional.
- Putting tables / a key-numbers block in the short version — it is prose-only.
- Hard calendar dates in the prose — keep them in the filename.

## Examples

- Long: `reports/Timur_Iakshibaev_biweekly_2026-09-14_to_2026-10-09.md`
- Short: `reports/Timur_Iakshibaev_biweekly_2026-09-14_to_2026-10-09_short.md`
