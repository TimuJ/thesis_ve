---
name: lab-meeting-report
description: Use when preparing a recap report for a lab / group meeting whose audience is labmates and the lab lead (not the supervisor one-on-one), covering the period since the last lab meeting.
---

# Lab-Meeting Report

## Overview

A technical recap for the weekly/biweekly lab meeting. The audience is labmates
and the lab lead, who will **engage on mechanisms and numbers** — so include
enough technical detail to survive Q&A, and **exclude anything already presented
at a prior meeting** (this is the top differentiator from the supervisor report).

## When to use

- Writing a recap for a lab / group meeting (labmates + lab lead).
- NOT the supervisor's one-on-one progress report → use `supervisor-biweekly-report`.

## Conventions (project rules — load-bearing)

- **Research-outcome framing; no thesis mentions.**
- **No exact calendar dates in prose** — dates live only in the filename. Use
  "since the last meeting", "this period".
- **Exclude already-presented material.** Read the previous report(s) in
  `reports/` and omit the overlap; keep only a 1–2 line "where we left off" for
  orientation. If the period was build-up (no new results), say so up front so the
  room doesn't expect fresh numbers.
- **File:** `reports/Timur_Iakshibaev_biweekly_<start>_to_<end>.md` (repo-root
  `reports/`). Commit and push when done.

## Structure (recipe — the report IS these parts, in order)

1. **Title + audience note** (1–2 lines: who it's for; flag build-up vs results).
2. **Where we left off** (1–2 lines, context only — not a re-presentation).
3. **2–4 numbered sections**, each one topic, each carrying the *mechanism*
   (why a result holds) plus the numbers — not just the claim.
4. **Key numbers** — a compact block the presenter can field questions from.
5. **One-line summary** for the room.

## Technical depth

The audience probes, so state the mechanism, not only the result (e.g. *why* a
metric is blind/inverted, not just that it is). **Flag estimates vs measured
values explicitly** — a technical audience will ask, and an unmarked estimate
presented as fact is the easy miss.

## Example

Most recent exemplar: `reports/Timur_Iakshibaev_biweekly_2026-09-24_to_2026-10-09.md`.
