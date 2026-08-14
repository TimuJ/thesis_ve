"""Pre-registered expectations for the LR-VCC severity battery.

Declared BEFORE any calibration is fitted. The six designed-for families and
the flip control family carry predictions stated in the thesis experiments
chapter: flip_invert (histogram-destroying) is caught everywhere,
flip_horizontal / periodic / elastic are invisible, flip_channel_shuffle is
caught partially.

flip_transpose is UNCONSTRAINED on purpose: it preserves the histogram but
destroys geometry, so the pre-registration is genuinely ambiguous. Assigning it
an expectation after its results were seen would be reading the answer off the
data, so it is excluded from the fit objective and reported only.
"""

RESPOND = "RESPOND"
SILENT = "SILENT"
UNCONSTRAINED = "UNCONSTRAINED"

EXPECTATION = {
    # designed-for long-range families
    "color_drift": RESPOND,
    "background_drift": RESPOND,
    "chunk_boundary": RESPOND,
    "flicker": RESPOND,
    "identity_degradation": RESPOND,
    "identity_drift": RESPOND,
    # flip controls with a positive prediction
    "flip_invert": RESPOND,
    "flip_channel_shuffle": RESPOND,
    # flip controls predicted invisible
    "flip_horizontal": SILENT,
    "flip_periodic": SILENT,
    "flip_elastic": SILENT,
    # ambiguous pre-registration
    "flip_transpose": UNCONSTRAINED,
}

SUB_METRICS = ("appearance", "temporal", "identity", "color_stability",
               "color_slope", "color_hist_anchor", "clip_trajectory")

# Which sub-metrics each family was constructed to excite. Used by failure
# attribution to decide which sub-metric "should have fired" in a given cell.
DESIGNED_FOR = {
    "color_drift": ("color_stability", "color_slope", "color_hist_anchor"),
    "background_drift": ("color_hist_anchor", "clip_trajectory", "appearance"),
    "chunk_boundary": ("temporal", "color_stability"),
    "flicker": ("temporal", "appearance"),
    "identity_degradation": ("identity", "appearance"),
    "identity_drift": ("identity", "clip_trajectory"),
    "flip_invert": ("color_stability", "color_hist_anchor", "clip_trajectory",
                    "appearance"),
    "flip_channel_shuffle": ("color_hist_anchor", "clip_trajectory",
                             "appearance"),
}

SEVERITIES = ("0p02", "0p05", "0p10", "0p20", "0p40")
SEVERITY_VALUES = {"0p02": 0.02, "0p05": 0.05, "0p10": 0.10,
                   "0p20": 0.20, "0p40": 0.40}

BASES = ("7WHI2L_FDNg", "BrRLKMbBTYQ", "KZ8p6b1zJ9U", "hhszUXL1Cu8",
         "mJog8DlRk_4")


def conforms(family, verdict):
    """True/False for constrained families, None for UNCONSTRAINED ones.

    RESPOND conforms on PASS or WEAK (a downward response of at least 0.02).
    SILENT conforms only on FLAT.
    """
    exp = EXPECTATION[family]
    if exp == UNCONSTRAINED:
        return None
    if exp == RESPOND:
        return verdict in ("PASS", "WEAK")
    return verdict == "FLAT"
