---
category: features
created: '2026-09-29'
id: 2026-09-29_hmc-core-sampler
last_updated: '2026-09-29'
owner: Neil D. Lawrence
priority: High
related_cips:
- '0008'
status: Proposed
tags:
- backlog
- features
- hmc
- mcmc
- teaching
title: Implement teachable Neal-style HMC core sampler
---

# Task: Implement teachable Neal-style HMC core sampler

## Description

Deliver the NumPy-first Hamiltonian Monte Carlo core for CIP-0008: kinetic energy with diagonal mass, leapfrog integration, Metropolis accept/reject on $\Delta H$, and a small public API taking `potential(q)` / `grad_potential(q)` and returning samples plus accept-rate / optional $H$ trace.

## Acceptance Criteria

- [ ] Module under `mlai` (e.g. `hmc.py`) with a clear `HamiltonianMonteCarlo` (or equivalent) class
- [ ] Configurable step size, leapfrog steps, and diagonal mass
- [ ] Unit tests on $\mathcal{N}(0,I)$: empirical mean/covariance within tolerance; seed reproducibility
- [ ] Conservation / accept-rate sanity checks for tiny and tuned step sizes
- [ ] Module docstring cites Neal hybrid Monte Carlo and stays readable for a lecture

## Implementation Notes

- Clarity over speed; no NUTS / adaptive HMC in this task
- Export from `mlai` once the API is stable enough for snippets

## Related

- CIP: 0008
- REQ: 0008

## Progress Updates

### 2026-09-29
Task created when CIP-0008 was accepted.
