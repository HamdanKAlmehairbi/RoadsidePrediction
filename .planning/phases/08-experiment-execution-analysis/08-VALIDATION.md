---
phase: 8
slug: experiment-execution-analysis
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-23
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | BackEnd/pytest.ini |
| **Quick run command** | `cd BackEnd && python -m pytest tests/test_analysis.py -x -q` |
| **Full suite command** | `cd BackEnd && python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd BackEnd && python -m pytest tests/test_analysis.py -x -q`
- **After every plan wave:** Run `cd BackEnd && python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 1 | EXP-01 | integration | `cd BackEnd && python scripts/run_campaign.py --campaign baseline --dry-run` | ✅ | ⬜ pending |
| 08-02-01 | 02 | 2 | EXP-01 | integration | `cd BackEnd && python scripts/run_campaign.py --campaign baseline --topologies grid-3x3 grid-5x5` | ✅ | ⬜ pending |
| 08-03-01 | 03 | 3 | EXP-02 | integration | `cd BackEnd && python scripts/run_extension_ablation.py --ablation fedprox` | ✅ | ⬜ pending |
| 08-03-02 | 03 | 3 | EXP-03 | integration | `cd BackEnd && python scripts/run_extension_ablation.py --ablation cooperative` | ✅ | ⬜ pending |
| 08-03-03 | 03 | 3 | EXP-04 | integration | `cd BackEnd && python scripts/run_extension_ablation.py --ablation time-of-day` | ✅ | ⬜ pending |
| 08-04-01 | 04 | 4 | EXP-05 | unit | `cd BackEnd && python -m pytest tests/test_analysis.py -x -q` | ✅ | ⬜ pending |
| 08-04-02 | 04 | 4 | EXP-05 | integration | `cd BackEnd && python scripts/generate_tables.py` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. Phase 7 delivered all scripts and test files.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Dry-run sanity check | EXP-01 | Requires human judgement on "reasonable" wait times | Review dry-run output for avg_wait_time < 500s |
| Baseline checkpoint review | EXP-01 | User decision gate before ablation commitment | Inspect baseline results JSON before proceeding |
| LaTeX table formatting | EXP-05 | Visual quality check | Open generated .tex files and verify booktabs formatting |
| Chart readability | EXP-05 | Visual quality check | Open generated .png files and verify labels, legends, error bars |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
