"""Constants for the cross-gap matched-pair consistency pipeline (A-D-R adaptation).

Fixed values come from the spec; everything marked PROVISIONAL is builder-tunable
(see the spec's "Left to the builder" / "Open decisions") and must only be read
from here, never hard-coded elsewhere.
"""

# Fixed by the spec / upstream pool design.
MIN_GAP_S = 3.0          # gap floor = FlashVSR context window (~3 s)
HEADLINE_GAP_S = 5.0     # preferred headline gap
FLASHVSR_WINDOW_S = 3.0  # the SR model's temporal context the gap must exceed

# PROVISIONAL — tune on GT configuration-match behaviour, never on SR scores.
LAMBDA_AREA = 1.0        # weight on |log(area_a/area_b)| in the selection cost
LAMBDA_ASPECT = 1.0      # weight on |log(aspect_a/aspect_b)| in the selection cost
LAMBDA_SIZE = 0.5        # reward on larger object: +LAMBDA_SIZE*(-log min-area) in cost
A_MIN = 0.01             # min object area fraction per side (<1% ≈ <24 px in /4 LQ)
BIG_AREA_FRAC = 0.05     # object area fraction at/above which size-reliability saturates to 1
C_GT_FLOOR = 0.5         # weak-anchor floor on GT cross-gap similarity
MAX_WINDOW_OFFSET = 150  # frames of present run adjacent to a gap edge (~5 s @30fps)

# PROVISIONAL — multi-pair selection (richer pair set; tune on GT behaviour only).
TOP_K_PER_VIDEO = 5          # target reviewable pairs per video (top-K across all gaps)
MIN_PAIR_SEP_S = 1.0         # two pairs are distinct unless within this on BOTH endpoints
ROMA_POSE_GATE_THRESH = 0.5  # RoMa pose-match score to admit a pair as reviewable (selection gate)

# PROVISIONAL — frame-quality gates from reviewer feedback (pose / blur / edge).
# The reviewer rejected pairs whose two frames show a different pose/angle, a
# motion-blurred frame, or a small object sitting on the frame edge. These gate the
# GT selection (frozen, so every model still sees the same pair) and fold into the
# consistency reliability so residual weak pairs sort down / into the gallery appendix.
LAMBDA_APPEAR = 3.0          # weight on embedding cos-distance in the selection cost (pose match)
MAX_APPEARANCE_DIST = 0.35   # reject a pair whose two frames' embeddings are farther apart (different pose)
LAMBDA_BLUR = 2.0            # weight on the per-frame blur penalty in the selection cost
LAMBDA_BORDER = 2.0          # weight on the per-frame (size-scaled) edge penalty in the selection cost
EDGE_MARGIN_FRAC = 0.03      # a box within this fraction of a frame edge counts as "on the edge"

# PROVISIONAL — Option B: long-range "big temporally-separated" pair (no disappearance).
# Prioritise A-D-R gap pairs; also emit one big early-vs-late pair per clip at reduced
# reliability, so clips whose large object never sits near a gap (e.g. a bus big only at
# the start and end) still yield a big, reviewable pair.
LONG_RANGE_RELIABILITY = 0.5  # reliability multiplier for a non-gap (no-disappearance) pair
GLOBAL_TOPK = 60              # biggest present frames per clip-half considered for the long-range pair
GLOBAL_MIN_SEP_S = MIN_GAP_S  # a long-range pair must span at least this many seconds
