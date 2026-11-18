---
id: "0009"
title: "Learners can fit sparse and latent Gaussian processes with readable numerics"
status: "Proposed"
priority: "Medium"
created: "2026-11-18"
last_updated: "2026-11-18"
related_tenets:
- vibesafe-mlai-tenets
stakeholders:
- students
- educators
- contributors
tags:
- gaussian-processes
- sparse
- variational
- gplvm
- teaching
- numerics
---

# REQ-0009: Learners can fit sparse and latent Gaussian processes with readable numerics

## Description

Course users can construct a small sparse variational Gaussian process (inducing-point VFE / DTC-style bound) whose objective and parameter gradients are inspectable in the same spirit as the exact `GP` teaching model. The numerics remain stable enough for typical lecture-scale problems without hiding the mathematics behind opaque library calls. The same objective surface can treat latent inputs as trainable parameters, so a minimal GP-LVM-style demonstration sits on the shared “fit by gradients” story already used for linear models and networks.

**Why this matters**: Mathematical Transparency and Educational Focus — sparse and latent GPs are core graduate material; Clarity Over Cleverness — prefer an explicit factorisation students can follow over a dense production sparse-GP stack.

**Who benefits**: Students meeting inducing-point and GP-LVM ideas for the first time; educators who want one optimiser narrative across exact GP, sparse GP, and latent-\(X\) demos; contributors validating against a known stable sparse bound.

## Acceptance Criteria

- [ ] A teachable sparse variational GP exposes an objective consistent with the collapsed VFE / DTCVAR bound (plain DTC optional by omitting the trace correction)
- [ ] Kernel, noise, and inducing locations participate in a uniform parameters / gradients surface usable with the shared optimiser
- [ ] Latent inputs can be included on that surface so a small unsupervised / GP-LVM-style fit is demonstrable
- [ ] Numerical checks (finite differences and/or dense / Gram+chol parity on tiny \(N,M\)) support the teaching claims of correctness and relative stability
- [ ] Lecture-facing material can show the factorisation story without requiring GPy for the core sparse path

## Notes

Exact GP hyperparameter gradients on the shared optimiser surface remain under REQ-0007 / CIP-0007 and are a prerequisite for a coherent “GP family on one loop” narrative. Choice of LQ versus Gram+Cholesky, and the precise reverse-mode design, are HOW (CIP).

## References

- **Related Tenets**: vibesafe-mlai-tenets (mathematical transparency, clarity, educational focus)
- **Related**: REQ-0007 (shared optimization surface)
- **External**: Titsias (2009) VFE; Seeger et al. (2017) auto-differentiating linear algebra §3.3 / §4.3 (BLR via LQ)

## Progress Updates

### 2026-11-18
Requirement proposed alongside CIP-000C (LQ teaching sparse VFE path; builds on CIP-0007 GP surface).
