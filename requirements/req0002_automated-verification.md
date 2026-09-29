---
id: "0002"
title: "Package behaviour is automatically verified against regressions"
status: "Proposed"
priority: "High"
created: "2026-09-29"
last_updated: "2026-09-29"
related_tenets:
- vibesafe-mlai-tenets
stakeholders:
- contributors
- educators
- students
tags:
- testing
- quality
- reproducibility
---

# REQ-0002: Package behaviour is automatically verified against regressions

## Description

Changes to `mlai` can be checked by an automated suite that covers core library behaviour and teaching workflows. Contributors and educators can trust that examples and APIs still work after edits, and students see testing used as a normal part of scientific software practice.

**Why this matters**: Supports Reproducibility and Good Python Practices; testing is itself educational under Educational Focus.

**Who benefits**: Maintainers catching breakages early; educators relying on tutorials; students learning reliable workflows.

## Acceptance Criteria

- [ ] A documented test suite runs locally with a single conventional command
- [ ] Core library units and key tutorial/integration workflows are covered
- [ ] Continuous integration runs the suite on proposed changes
- [ ] Coverage (or equivalent quality signal) is visible and actionable
- [ ] Tests that need optional dependencies fail clearly or skip with documented reasons, without blocking unrelated checks

## Notes

Extracted from CIP-0002. Some tests and CI already exist; the requirement captures the desired outcome, not the pytest-specific design in the CIP.

## References

- **Related Tenets**: vibesafe-mlai-tenets (reproducibility, good Python practices, educational focus)
- **Implements via**: CIP-0002

## Progress Updates

### 2026-09-29
Requirement extracted from CIP-0002.

### 2026-09-29
CIP-0002 accepted and moved to In Progress. Requirement remains Proposed until residual acceptance criteria (optional-deps clarity, plot depth, documented single-command suite) are closer to done.
