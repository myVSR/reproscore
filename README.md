# ReproScore

**Separating Readiness from Outcome in Research Software Reproducibility Assessment**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)

ReproScore is a two-tier scoring framework for assessing the reproducibility of research software repositories. It separates *reproducibility readiness* (what a repository contains) from *reproducibility outcome* (whether the software actually runs), making the distinction explicit and measurable.

**Author**: Sheeba Samuel · [sheeba.samuel@informatik.tu-chemnitz.de](mailto:sheeba.samuel@informatik.tu-chemnitz.de) · Chemnitz University of Technology

---

## Overview

| Score | What it measures | When available |
|---|---|---|
| **RRS** | Static readiness — 26 sub-metrics across 5 categories | Always (no execution needed) |
| **ROS** | Execution outcome — up to 6 sandboxed probes | When execution infrastructure is available |
| **RCS** | Coverage-adaptive composite of RRS + ROS | When any ROS component is available |

The core insight is that a repository can score well on static readiness yet fail to execute (e.g. pinned dependencies with a version conflict), and conversely execute successfully despite minimal static signals. ReproScore makes this *readiness–outcome gap* visible rather than conflating the two quantities.

---

## Repository structure

```
reproscore/
├── ablation_analysis.py        ← Reproduce all evaluation statistics from scores.csv
├── config/
│   └── default_rubric.yaml     ← Category weights (w_i), gate parameters (τ, k), penalties
├── src/
│   ├── scoring/
│   │   ├── rrs.py              ← 26 sub-metric detectors, gate function, category aggregation
│   │   ├── ros.py              ← Execution outcome scoring (6 components)
│   │   ├── rcs.py              ← Composite score with coverage weight α
│   │   └── rubric.py           ← YAML rubric loader and validator
│   └── utils/
│       └── notebook_paths.py   ← Notebook discovery and exclusion filters
├── tests/
│   └── test_scoring.py         ← Unit tests for RRS (no execution required)
└── data/
    └── ablation/
        └── 20260511_101920/    ← Evaluation run (423 repositories)
            ├── scores.csv      ← Per-repository RRS scores + 26 sub-metrics + ground truth
            ├── analysis_results.json
            ├── analysis.log
            ├── provenance.json
            ├── logs/           ← Clone and score logs
            └── repos/          ← Per-repository score provenance (JSON, one file per repo)
```

---

## Quick start

### Install

```bash
git clone https://github.com/myVSR/reproscore.git
cd reproscore
pip install -r requirements.txt
```

### Score a repository

```python
from src.scoring.rrs import RRSScorer

result = RRSScorer().score("/path/to/repo")
print(f"RRS: {result.rrs:.1f}")
for sym, cat in result.category_scores.items():
    print(f"  {sym}: {cat.raw_score:.1f}")
for ev in result.evidence:
    if ev.fix_suggestion:
        print(f"  [{ev.metric_id}] {ev.fix_suggestion}")
```

### Reproduce evaluation statistics

```bash
python ablation_analysis.py
# or point at the bundled run explicitly:
python ablation_analysis.py --run-dir data/ablation/20260511_101920
```

### Run tests

```bash
pytest tests/test_scoring.py -v
```

---

## Scoring model

### RRS — 5 categories, 26 sub-metrics

| Cat | Name | Weight | τ | k | Sub-metrics |
|---|---|---|---|---|---|
| E | Environment specification | 0.30 | 40 | 1.5 | dep_pinning, container_spec, env_bootstrap, python_version_declared |
| A | Data accessibility | 0.25 | 30 | 1.5 | data_description, data_pointer, workflow_orchestration, data_acquisition_script |
| D | Documentation | 0.20 | 20 | 1.2 | doc_structure, install_instructions, usage_examples, inline_explanation_density, execution_entry_point, docstring_coverage, reuse_metadata |
| C | Code portability | 0.15 | 25 | 1.2 | no_absolute_paths, import_resolvability, no_hardcoded_credentials, silent_failure_masking |
| S | Reproducibility signals | 0.10 | 30 | 1.2 | seed_management, notebook_exec_order, test_file_presence, expected_outputs, ci_presence, config_externalised, hardware_requirements |

Within-category weights reflect execution failure pattern analysis; see `src/scoring/rrs.py` `_aggregate_*` docstrings for rationale.

### Gate function

```
g(x, τ, k) = x / 100                    if x ≥ τ
           = (x / τ)^k · (τ / 100)      if x < τ
```

Penalises sub-threshold failures non-linearly. Core categories (E, A) use `k = 1.5`; quality categories (D, C, S) use `k = 1.2`.

### Hard penalties

| Condition | Penalty |
|---|---|
| E < 10 (no environment specification) | −20 pts |
| A < 10 (no data artefacts) | −15 pts |
| seed score < 50 (stochastic ops, no seeds) | −10 pts |

Penalty magnitudes are calibrated to approximately the maximum weight contribution of the penalised category.

### Community rubric

Override any weight or gate parameter via a YAML profile:

```yaml
name: bioinformatics-v1
version: "1.0"
categories:
  E: {weight: 0.35, tau: 40, k: 1.5}
  A: {weight: 0.40, tau: 30, k: 1.5}
  D: {weight: 0.10, tau: 20, k: 1.2}
  C: {weight: 0.05, tau: 25, k: 1.2}
  S: {weight: 0.10, tau: 30, k: 1.2}
```

```python
from src.scoring.rrs import RRSScorer
from src.scoring.rubric import load_rubric

rubric = load_rubric("my_rubric.yaml")
result = RRSScorer(rubric=rubric).score("/path/to/repo")
```

---

## Evaluation dataset

The `data/ablation/20260511_101920/` directory contains results for 423 Python/Jupyter repositories stratified across five execution failure modes:

| Failure mode | n | Description |
|---|---|---|
| success | 84–85 | All notebooks completed without error |
| install_dep | 84–85 | Install-time dependency conflict |
| missing_module | 84–85 | ModuleNotFoundError / ImportError at runtime |
| missing_data | 84–85 | FileNotFoundError / missing input data |
| code_error | 84–85 | TypeError / NameError / SyntaxError |

`scores.csv` contains one row per repository with RRS, all 26 sub-metric scores, category scores, and the ground-truth failure mode label.

`repos/` contains per-repository provenance JSON (sub-metric evidence, file-level detections) for all 423 repositories.

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
