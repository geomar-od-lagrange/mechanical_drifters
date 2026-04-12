# C5: Parameter sensitivity

Vary drifter parameters within physically plausible ranges and check
whether the Callies et al. defaults are adequate or tuning is needed.

## Parameters to vary

| Parameter | Default | Range | Physical basis |
|---|---|---|---|
| `k_d` (drogue drag) | 154 kg/m | 100–200 | C_D uncertainty (1.0–1.4) |
| `k_b` (buoy drag) | 12 kg/m | 8–20 | C_D and submerged height |
| `m_tilde_d` (drogue added mass) | 101 kg | 80–130 | C_perp uncertainty |
| `l` (pole length) | 3.0 m | 2.5–3.5 | Manufacturing tolerance |

## Approach

1. Use the C4 re-seeded framework (12h segments) for each parameter
   set — gives separation vs lead time per configuration.
2. Latin hypercube or one-at-a-time sampling over the parameter space.
3. Compare against the baseline (Callies defaults) and observed
   drifter tracks.
4. Check whether the alpha formula
   (`alpha = sqrt(k_b) / (sqrt(k_b) + sqrt(k_d))`) from C6 holds
   across the parameter range.

## Depends on

- C4 (re-seeded validation) — for the evaluation framework.
- C6 (along-track validation) — for the alpha baseline.
