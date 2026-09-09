# Results report

6900 hidden-unit observations across 275 networks; systems: Iranian-17, Western-12; bases: dft; controls: input_shuffled, label_shuffled, none.

## 1. Frequency range recruited (significant units, real task)

```
       system basis  n_sig  n_total  frac_sig  min_freq  max_freq  nyquist  reaches_nyquist  mean_R                  freqs_used
0  Iranian-17   dft   1056     1950     0.542         1        12       12             True   0.888  1|2|3|4|5|6|7|8|9|10|11|12
1  Western-12   dft    260      350     0.743         1         6        6             True   0.950                 1|2|3|4|5|6
```

> Significance is FDR-corrected (Benjamini-Hochberg) within each system x control family, so the thousands of per-unit permutation tests do not inflate the counts.

## 1b. Spectral concentration (A1: single-phase-space dominance)

```
       system  mean_concentration  median_concentration  mean_best_R  n_sig
0  Iranian-17               0.545                 0.542        0.888   1056
1  Western-12               0.905                 0.930        0.950    260
```

> Concentration = share of a unit's total explained spectral power in its single best frequency. It measures the *one-phase-space* claim directly; a unit split across two frequencies keeps a high max |R| but a low concentration.

## 1c. Occupancy of the Iranian-only band 7-12 (A3)

```
       system  n_sig  n_in_7_12  frac_in_7_12  western_ceiling                    note
0  Iranian-17   1056        595         0.563                6       Iranian-only band
1  Western-12    260          0         0.000                6  impossible for 12-grid
```

> Frequencies 7-12 are mathematically impossible for the 12-grid Western system. Iranian units occupying this band cannot be an artefact of the N/2 bound -- the network could have stayed in 1-6 but did not -- so this is the non-mechanical core of the cardinality result.

> Primary result. On the lattice DFT the Iranian system (17 notes on a 24-tone quarter-tone lattice) spans frequencies 1..12 and the Western system (12 notes on a 12-tone lattice) spans 1..6; `reaches_nyquist=True` for both means each recruits the full range its lattice permits, and the ceiling (12 vs 6) is set by lattice cardinality.

## 1d. Nyquist-frequency construct check (k=12 confound)

```
       system  n_sig  n_7_12  frac_7_12  n_at_nyquist_12  frac_at_12  n_7_11  frac_7_11
0  Iranian-17   1056     595      0.563              339       0.321     256      0.242
1  Western-12    260       0      0.000                0       0.000       0      0.000
```

> On the 24-tone lattice, phase space k=12 reduces to (-1)^g: +1 at the twelve chromatic pitch classes, -1 at the five microtonal inflections, with a vanishing sine component. It is therefore identical to a binary chromatic/microtonal indicator and is one-dimensional. Report the band **7-11** as the confound-free high-frequency claim.

## 1e. Task-matched comparison (interval tasks only)

```
       system  n_interval_units  n_in_7_11  frac_in_7_11  ci_low  ci_high  max_freq  fisher_p_vs_other_system
0  Iranian-17               279         86         0.308     NaN      NaN        12              1.508000e-28
1  Western-12               260          0         0.000     NaN      NaN         6              1.508000e-28
```

> The Iranian battery contains Dang tasks with no Western counterpart, so a pooled comparison is confounded with task type. Restricting to the two interval tasks -- built by the identical rule in both systems -- isolates lattice cardinality from task complexity.

## 2. Does overall fit strength differ between systems?

**Network-level Mann-Whitney (mean logit R per network):**

```
                                          test    system_a  n_a  mean_a    system_b  n_b  mean_b  mean_diff       U             p
0  Mann-Whitney U (network-level mean logit R)  Iranian-17  225   1.635  Western-12   50   2.814     -1.179  1418.0  1.345000e-16
```

> `system` is a between-network factor, so all system comparisons are made at the network level (the unit of replication), not the hidden-unit level.

## 3. Control conditions

```
       system         control  mean_R   sd_R  frac_sig     n
0  Iranian-17  input_shuffled   0.780  0.138     0.544  1950
1  Iranian-17  label_shuffled   0.604  0.076     0.000  1950
2  Iranian-17            none   0.778  0.139     0.542  1950
3  Western-12  input_shuffled   0.906  0.104     0.797   350
4  Western-12  label_shuffled   0.653  0.090     0.000   350
5  Western-12            none   0.889  0.118     0.743   350
```


**Real-vs-label-null contrast (network-level paired Wilcoxon, logit R):**

```
       system  n_networks  mean_logitR_real  mean_logitR_labelnull  mean_diff  diff_ci_low  diff_ci_high  wilcoxon_W  rank_biserial    wilcoxon_p
0  Iranian-17         225             1.635                  0.435      1.200        1.124         1.279         0.0            1.0  1.150000e-38
1  Western-12          50             2.814                  0.665      2.149        1.910         2.406         0.0            1.0  1.780000e-15
```

> The decisive control: a large positive `mean_diff` (real minus label-shuffled) with small `wilcoxon_p` means the Fourier structure is task-driven rather than an artefact of analysis flexibility.

## 4. Fourier vs non-Fourier baselines (Wilcoxon)

```
       system                comparison  mean_fourier  mean_ref    wilcoxon_p
0  Iranian-17  best_R vs pitch_height_r         0.778     0.155  0.000000e+00
1  Iranian-17    best_R vs random_r_p95         0.778     0.478  0.000000e+00
2  Western-12  best_R vs pitch_height_r         0.889     0.239  4.090000e-59
3  Western-12    best_R vs random_r_p95         0.889     0.571  8.120000e-59
```

## 5. Cardinality (lattice DFT basis)

```
       system  max_freq  nyquist  mean_norm  frac_top_quartile  n_units  reaches_nyquist
0  Iranian-17        12       12      0.622              0.436     1056             True
1  Western-12         6        6      0.510              0.292      260             True
```


**Normalised best frequency by system (network-level Mann-Whitney):**

```
                                                  test    system_a  n_a  mean_a    system_b  n_b  mean_b  mean_diff       U             p
0  Mann-Whitney U (network-level normalised best freq)  Iranian-17  225   0.635  Western-12   50    0.49      0.145  9140.5  4.734000e-12
```

> `reaches_nyquist = True` for both systems: each recruits the full range its lattice allows (ceiling 12 vs 6, set by lattice cardinality). Beyond the ceiling, the *normalised* best frequency is also higher for the Iranian system (a significant `mean_diff`), indicating the higher-cardinality system recruits proportionally higher frequencies -- report this as a secondary observation supporting the cardinality hypothesis.

## Figures

- `figures/fig1_frequency_range.png` — frequency histograms per system
- `figures/fig2_controls.png` — fit strength by control condition
- `figures/fig3_baselines.png` — Fourier vs pitch-height vs random ceiling
