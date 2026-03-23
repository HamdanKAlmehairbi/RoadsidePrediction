---
phase: 7
slug: experiment-campaigns
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-23
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (used in Phase 3 tests) |
| **Config file** | None detected — pytest auto-discovers |
| **Quick run command** | `pytest BackEnd/tests/ -x -q --timeout=30` |
| **Full suite command** | `pytest BackEnd/tests/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest BackEnd/tests/ -x -q --timeout=30`
- **After every plan wave:** Run `pytest BackEnd/tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | CAMP-01, CAMP-07 | unit | `pytest BackEnd/tests/test_campaign.py -x` | ❌ W0 | ⬜ pending |
| 07-01-02 | 01 | 1 | CAMP-01, CAMP-05 | integration | `pytest BackEnd/tests/test_campaign.py -x` | ❌ W0 | ⬜ pending |
| 07-02-01 | 02 | 2 | CAMP-02, CAMP-03, CAMP-04 | unit | `pytest BackEnd/tests/test_ablation_configs.py -x` | ❌ W0 | ⬜ pending |
| 07-02-02 | 02 | 2 | CAMP-02, CAMP-03, CAMP-04 | integration | `pytest BackEnd/tests/test_ablation_configs.py -x` | ❌ W0 | ⬜ pending |
| 07-03-01 | 03 | 2 | CAMP-05, CAMP-06 | unit | `pytest BackEnd/tests/test_analysis.py -x` | ❌ W0 | ⬜ pending |
| 07-03-02 | 03 | 2 | CAMP-06 | unit | `pytest BackEnd/tests/test_analysis.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `BackEnd/tests/test_campaign.py` — smoke test for campaign runner and config dataclass
- [ ] `BackEnd/tests/test_ablation_configs.py` — unit test for ablation config builders
- [ ] `BackEnd/tests/test_analysis.py` — unit test for Wilcoxon, LaTeX table, and chart generation

*Tests are inlined within each plan's tasks — Wave 0 stubs created alongside implementation.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Baseline campaign runs to completion | CAMP-01 | Requires SUMO + multi-hour training | Run `python BackEnd/scripts/run_campaign.py --seeds 1` and verify results JSON |
| FedProx ablation produces lower waiting time | CAMP-02 | Requires training runs | Run ablation script, compare results |
| Cooperative reward ablation valid | CAMP-03 | Requires training runs | Run ablation script, compare results |
| Time-of-day ablation valid | CAMP-04 | Requires training runs | Run ablation script, compare results |
| Charts are publication quality | CAMP-06 | Visual inspection | Review generated PNG/PDF files |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
