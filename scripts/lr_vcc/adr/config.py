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
A_MIN = 0.01             # min object area fraction per side (<1% ≈ <24 px in /4 LQ)
C_GT_FLOOR = 0.5         # weak-anchor floor on GT cross-gap similarity
MAX_WINDOW_OFFSET = 150  # frames of present run adjacent to a gap edge (~5 s @30fps)
