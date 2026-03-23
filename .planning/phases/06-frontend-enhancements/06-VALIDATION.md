---
phase: 6
slug: frontend-enhancements
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-23
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | vitest (existing FrontEnd setup) |
| **Config file** | FrontEnd/vite.config.ts |
| **Quick run command** | `cd FrontEnd && npx vitest run --reporter=verbose` |
| **Full suite command** | `cd FrontEnd && npx vitest run` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd FrontEnd && npx vitest run --reporter=verbose`
- **After every plan wave:** Run `cd FrontEnd && npx vitest run`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | UI-01 | integration | Browser console check | ❌ W0 | ⬜ pending |
| 06-01-02 | 01 | 1 | UI-02 | integration | Browser console check | ❌ W0 | ⬜ pending |
| 06-02-01 | 02 | 1 | UI-03, UI-04 | component | `vitest run Evaluation` | ❌ W0 | ⬜ pending |
| 06-02-02 | 02 | 1 | UI-05 | visual | Manual heatmap check | ❌ W0 | ⬜ pending |
| 06-03-01 | 03 | 2 | UI-06, UI-07 | integration | Browser console check | ❌ W0 | ⬜ pending |
| 06-03-02 | 03 | 2 | UI-08 | e2e | No console errors check | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Existing vitest infrastructure covers component tests
- [ ] Console error checking via browser dev tools (manual)

*Existing infrastructure covers basic component testing. Visual/integration tests are manual.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Communication page real data | UI-01 | Requires running SUMO backend | Start backend, run training, check Communication page |
| Training page rewards | UI-02 | Requires active training session | Start training, verify per-client reward charts |
| Evaluation results table | UI-03 | Requires completed evaluation | Run evaluation, check results table renders |
| Transfer matrix heatmap | UI-04, UI-05 | Visual rendering verification | Run evaluation with transfer, check heatmap colors |
| Index page real stats | UI-06, UI-07 | Requires evaluation data | Run evaluation, check Index page updates |
| No console errors | UI-08 | Browser-specific | Navigate all pages, check console |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
