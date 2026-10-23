# Progress Report — Timur Iakshibaev

**Topic:** choosing and building the real disappear-reappear dataset for the
reduced-reference consistency benchmark.

## Summary

**The evaluation dataset is now chosen and built, which was the step the last report
set up.** The goal was real footage in which a sizeable subject disappears for a while
and reappears, long enough and at a resolution and frame rate usable for
super-resolution. After testing candidates, the source is the full LaSOT tracking
corpus at 30 fps, from which 88 qualifying sequences were extracted and encoded with
their ground truth — the material the first correspondence experiment needs.

**The deciding factor was frame rate, not gap length, and it reframed the choice.** The
most natural-looking candidate offered the longest absences — a subject gone for up to
roughly thirty-five seconds — but only at six frames per second, for both the images
and the annotations, with no higher-rate version available and no way to raise it
except synthetic frame interpolation. Six frames per second is fine for comparing the
subject before and after the gap but makes the frame-to-frame temporal measurement
meaningless, so it was ruled out as super-resolution input. Scanning the full LaSOT
instead gave 88 videos where a sizeable subject genuinely drops out for a few seconds
and returns, at a normal frame rate; accepting shorter gaps in exchange for a usable
frame rate is the right trade for a consistency benchmark.

**The full pool is built and a starter set is already with a collaborator.** All 88
clips are encoded with per-frame bounding-box and absence ground truth, and a
ten-video diverse subset has been handed to a labmate to run the first
super-resolution and consistency pass. A high-resolution alternative was also assessed
and shelved because it is prohibitively large, its download host was unreachable, and
its footage carries usage restrictions.

**One decision from the supervisor is still outstanding and still gates
implementation:** whether the benchmark formally adopts the reduced-reference framing —
measuring consistency against the low-quality input rather than the output's own past.
The dataset work removes the last practical blocker to the first quantitative result,
so this framing decision is now the main thing standing between the specification and
running code.

## Next Period

1. Produce the first quantitative correspondence result on the built footage — degrade,
   super-resolve, and score subject preservation across each gap against the ground
   truth.
2. Settle the reduced-reference adoption decision that gates implementation.
3. Fix the reporting pool (the full set versus a premium subset) and complete the
   collaborator's first pass on the starter videos.
4. Add the collaborator's model as a comparison row once his outputs arrive.
