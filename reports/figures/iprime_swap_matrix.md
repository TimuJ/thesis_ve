# I′ swap experiment — identity families, same-era composite

Three arms over identical inputs. The two same-era arms differ ONLY in the
identity slot — legacy score vs anchored I′, both derived from the same npz
embedding dumps (same detector pass, same era). Composition: v5 production
parameters, bit-exact recomposer, `_regen` sub-metric inputs. The frozen-v5
column is the published old-era matrix, shown for reference; it differs from
the same-era legacy arm in BOTH era and identity branch (fused vs slow-only),
so only the two same-era arms are a controlled comparison.

delta = lrvcc(0p40) − lrvcc(0p02); PASS ≤ −0.05, WEAK ≤ −0.02, FLAT < +0.02.

### identity_degradation

| base | v5 frozen (old era) | same-era legacy | same-era I′ |
|---|---|---|---|
| 7WHI2L_FDNg | +0.029 INVERTED | −0.004 FLAT | **−0.020 WEAK** |
| BrRLKMbBTYQ | +0.000 FLAT | −0.001 FLAT | −0.001 FLAT |
| KZ8p6b1zJ9U | −0.030 WEAK | −0.027 WEAK | −0.007 FLAT |
| hhszUXL1Cu8 | −0.053 PASS | −0.013 FLAT | **−0.023 WEAK** |
| mJog8DlRk_4 | +0.013 FLAT | −0.008 FLAT | −0.009 FLAT |
| **clean** | 2/5 | 1/5 | **2/5** |

### identity_drift

| base | v5 frozen (old era) | same-era legacy | same-era I′ |
|---|---|---|---|
| 7WHI2L_FDNg | −0.001 FLAT | −0.007 FLAT | −0.007 FLAT |
| BrRLKMbBTYQ | −0.000 FLAT | −0.001 FLAT | −0.001 FLAT |
| KZ8p6b1zJ9U | −0.025 WEAK | −0.021 WEAK | −0.034 WEAK |
| hhszUXL1Cu8 | −0.069 PASS | −0.026 WEAK | −0.031 WEAK |
| mJog8DlRk_4 | −0.003 FLAT | −0.010 FLAT | −0.005 FLAT |
| **clean** | 2/5 | 2/5 | 2/5 |

## Reading

- **At the raw-statistic level I′ is clearly better** (9/10 cells correct
  direction, +0.02–0.08, vs legacy flat/inverted). **At the composite level,
  with v5 parameters, the gain is one cell** on degradation and parity on
  drift. The attenuation is the known composition mechanism from the
  calibration phase: identity carries ~0.2 of the softmax weight, so a raw
  +0.03–0.06 moves the composite by ~0.006–0.012 — under the WEAK threshold.
  And I′ enters here UNSCALED: it has no fitted response parameter yet. This
  experiment is therefore a lower bound; the fitted-β version is what the
  harness integration produces.
- **KZ costs a cell on degradation** (WEAK → FLAT): the anchor is built from
  the corrupted video's own opening clips, and on close-up content attacked
  directly by the corruption, a blurred anchor matches blurred faces — the
  documented anchor-contamination case. Planned handling is an anchor-quality
  reliability gate, not a measurement change.
- **The era shift alone removes the flagship old-era inversion** on
  7WHI2L_FDNg (+0.029 INVERTED → −0.004 FLAT before I′ is even involved).
  The old-era reward-direction evidence conflates era and the fused (slow+
  fast) branch; in the new-era slow-branch replay the legacy measure is flat
  rather than inverted. The composite-level case for I′ is "flat → responding",
  not "inverted → responding", and write-ups should state it that way.

Provenance: `_regen` metric stack + npz embedding dumps (new era, this
server); composition via `scripts/lr_vcc/calibration/recompose.py` at
`PROD_PARAMS`; extraction scripts `iprime_extract.py` / `regen_rows_extract.py`.
