"""
tests/test_scoring.py
=====================
Unit tests for RRS, ROS, and RCS scoring (no Docker required).
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure package root on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Gate function tests
# ---------------------------------------------------------------------------

class TestGateFunction:
    def test_above_threshold_is_linear(self):
        from src.scoring.rrs import gate
        assert gate(50, 40, 1.5) == pytest.approx(0.50)
        assert gate(100, 40, 1.5) == pytest.approx(1.00)
        assert gate(40, 40, 1.5) == pytest.approx(0.40)

    def test_below_threshold_is_compressed(self):
        from src.scoring.rrs import gate
        # g(20, 40, 1.5) should be < 0.20 (compressed)
        assert gate(20, 40, 1.5) < 0.20
        assert gate(0, 40, 1.5) == pytest.approx(0.0)

    def test_table_values_from_paper(self):
        from src.scoring.rrs import gate
        # Table 2 from ReproScore paper
        assert gate(10, 40, 1.5) == pytest.approx(0.050, abs=0.002)
        assert gate(20, 40, 1.5) == pytest.approx(0.141, abs=0.002)
        assert gate(30, 40, 1.5) == pytest.approx(0.260, abs=0.002)

    def test_continuity_at_threshold(self):
        from src.scoring.rrs import gate
        # g should be continuous at tau
        assert gate(40 - 1e-6, 40, 1.5) == pytest.approx(gate(40, 40, 1.5), abs=0.001)


# ---------------------------------------------------------------------------
# Rubric tests
# ---------------------------------------------------------------------------

class TestRubric:
    def test_default_rubric_loads(self):
        from src.scoring.rubric import load_rubric
        rubric = load_rubric()
        assert rubric.name == "default"
        assert abs(sum(v["weight"] for v in rubric.categories.values()) - 1.0) < 0.01

    def test_rubric_validation_fails_bad_weights(self):
        from src.scoring.rubric import Rubric
        bad_rubric = Rubric(
            name="bad", version="1.0",
            categories={
                "E": {"weight": 0.50, "tau": 40, "k": 1.5},
                "A": {"weight": 0.50, "tau": 30, "k": 1.5},
                "D": {"weight": 0.20, "tau": 20, "k": 1.2},  # sum > 1
                "C": {"weight": 0.15, "tau": 25, "k": 1.2},
                "S": {"weight": 0.10, "tau": 30, "k": 1.2},
            },
            penalties={"environment_hard_threshold": 10, "data_hard_threshold": 10,
                       "environment_hard_penalty": 20, "data_hard_penalty": 15,
                       "seed_threshold": 50, "seed_penalty": 10},
            ros_components={
                "I": {"weight": 0.35}, "X": {"weight": 0.30},
                "delta": {"weight": 0.20}, "N": {"weight": 0.10},
                "T": {"weight": 0.05},
            },
            rcs={"alpha_max": 0.70, "alpha_min": 0.10},
        )
        with pytest.raises(ValueError):
            bad_rubric.validate()


# ---------------------------------------------------------------------------
# RRS scorer tests (using a real temp directory)
# ---------------------------------------------------------------------------

class TestRRSScorer:

    @pytest.fixture
    def minimal_repo(self, tmp_path):
        """A minimal Python repo with requirements.txt and README."""
        (tmp_path / "requirements.txt").write_text(
            "numpy==1.24.3\npandas==2.0.1\n"
        )
        (tmp_path / "README.md").write_text(
            "# My Repo\n## Installation\npip install -r requirements.txt\n"
            "## Usage\n```python\nimport numpy\n```\n"
        )
        (tmp_path / "analysis.py").write_text(
            "# Analysis\nimport numpy as np\nnp.random.seed(42)\nx = np.random.rand(10)\n"
        )
        return tmp_path

    @pytest.fixture
    def well_specified_repo(self, tmp_path):
        """A well-specified repo scoring highly on most categories."""
        (tmp_path / "requirements.txt").write_text("numpy==1.24.3\n")
        (tmp_path / "environment.yml").write_text(
            "name: myenv\ndependencies:\n  - python=3.10\n  - numpy=1.24.3\n"
        )
        (tmp_path / "Dockerfile").write_text(
            "FROM python:3.10\nCOPY . .\nRUN pip install -r requirements.txt\n"
        )
        readme = (
            "# My Repo\n## Overview\nA reproducible analysis.\n"
            "## Installation\npip install -r requirements.txt\n"
            "## Usage\n```python\nimport numpy\n```\n"
            "## Requirements\nnumpy, pandas\n"
            "## Examples\nSee examples/ folder.\n"
            "## Data\nData available at https://zenodo.org/record/123456\n"
        )
        (tmp_path / "README.md").write_text(readme)
        (tmp_path / "DATA.md").write_text("Data from Zenodo doi:10.5281/zenodo.123456")
        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        (examples_dir / "example1.py").write_text("import numpy as np\n")
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "ci.yml").write_text(
            "on: push\njobs:\n  test:\n    strategy:\n      matrix:\n        python-version: [3.9, 3.10]\n"
        )
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_analysis.py").write_text(
            '"""Tests."""\nimport numpy as np\n\ndef test_basic():\n    assert True\n'
        )
        (tmp_path / "CONTRIBUTING.md").write_text("# Contributing\n")
        (tmp_path / "data.csv").write_text("a,b,c\n1,2,3\n")
        analysis = (
            '"""Main analysis module."""\n'
            "import numpy as np\n"
            "# set random seed for reproducibility\n"
            "np.random.seed(42)\n"
            "# some computation\n"
            "result = np.random.rand(10)  # random call\n"
        )
        (tmp_path / "analysis.py").write_text(analysis)
        return tmp_path

    @pytest.fixture
    def empty_repo(self, tmp_path):
        return tmp_path

    @pytest.fixture
    def abs_path_repo(self, tmp_path):
        """Repo with hardcoded absolute paths."""
        (tmp_path / "requirements.txt").write_text("numpy==1.24.3\n")
        (tmp_path / "analysis.py").write_text(
            "import numpy as np\n"
            "data = np.loadtxt('/home/alice/data/experiment.csv')\n"
        )
        return tmp_path

    @pytest.fixture
    def declared_imports_repo(self, tmp_path):
        """Repo where all third-party imports are declared in requirements.txt."""
        (tmp_path / "requirements.txt").write_text("numpy==1.24.3\npandas==2.0.1\n")
        (tmp_path / "analysis.py").write_text(
            "import numpy as np\nimport pandas as pd\nimport os\n"
        )
        return tmp_path

    @pytest.fixture
    def out_of_order_notebook_repo(self, tmp_path):
        """Repo containing a notebook with out-of-order execution counts."""
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {"cell_type": "code", "execution_count": 3,
                 "source": ["x = 1"], "outputs": [], "metadata": {}},
                {"cell_type": "code", "execution_count": 1,
                 "source": ["y = 2"], "outputs": [], "metadata": {}},
            ],
        }
        (tmp_path / "notebook.ipynb").write_text(json.dumps(nb))
        return tmp_path

    @pytest.fixture
    def in_order_notebook_repo(self, tmp_path):
        """Repo containing a notebook with in-order execution counts."""
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {"cell_type": "code", "execution_count": 1,
                 "source": ["x = 1"], "outputs": [], "metadata": {}},
                {"cell_type": "code", "execution_count": 2,
                 "source": ["y = 2"], "outputs": [], "metadata": {}},
            ],
        }
        (tmp_path / "notebook.ipynb").write_text(json.dumps(nb))
        return tmp_path

    @pytest.fixture
    def reuse_metadata_repo(self, tmp_path):
        """Repo with LICENSE + CITATION.cff."""
        (tmp_path / "LICENSE").write_text("MIT License\n")
        (tmp_path / "CITATION.cff").write_text(
            "cff-version: 1.2.0\ntitle: My Software\n"
        )
        return tmp_path

    def test_scores_are_in_range(self, minimal_repo):
        from src.scoring.rrs import RRSScorer
        scorer = RRSScorer()
        result = scorer.score(minimal_repo)
        assert 0.0 <= result.rrs <= 100.0

    def test_well_specified_scores_higher_than_empty(self, well_specified_repo, empty_repo):
        from src.scoring.rrs import RRSScorer
        scorer = RRSScorer()
        r_well = scorer.score(well_specified_repo)
        r_empty = scorer.score(empty_repo)
        # well-specified should be at least as good as empty
        assert r_well.rrs >= r_empty.rrs
        # and well-specified should have fewer fix suggestions
        well_fixes = sum(1 for e in r_well.evidence if e.fix_suggestion)
        empty_fixes = sum(1 for e in r_empty.evidence if e.fix_suggestion)
        assert well_fixes <= empty_fixes

    def test_all_five_categories_present(self, minimal_repo):
        from src.scoring.rrs import RRSScorer
        scorer = RRSScorer()
        result = scorer.score(minimal_repo)
        assert set(result.category_scores.keys()) == {"E", "A", "D", "C", "S"}

    def test_26_sub_metrics_in_evidence(self, minimal_repo):
        from src.scoring.rrs import RRSScorer
        scorer = RRSScorer()
        result = scorer.score(minimal_repo)
        assert len(result.evidence) == 26

    def test_empty_repo_hard_penalties_applied(self, empty_repo):
        from src.scoring.rrs import RRSScorer
        scorer = RRSScorer()
        result = scorer.score(empty_repo)
        # E < 10 and A < 10 → both hard penalties
        assert result.penalty_environment == 20.0
        assert result.penalty_data == 15.0

    def test_to_dict_serialisable(self, minimal_repo):
        from src.scoring.rrs import RRSScorer
        scorer = RRSScorer()
        result = scorer.score(minimal_repo)
        d = result.to_dict()
        # Should JSON-serialise without error
        json.dumps(d)

    def test_dep_pinning_detected(self, minimal_repo):
        from src.scoring.rrs import RRSScorer
        scorer = RRSScorer()
        result = scorer.score(minimal_repo)
        ev = next(e for e in result.evidence if e.metric_id == "dep_pinning")
        # Fully pinned requirements.txt without lockfile → 80
        assert ev.raw_score == pytest.approx(80.0)

    def test_dep_pinning_lockfile_is_100(self, tmp_path):
        from src.scoring.rrs import RRSScorer
        (tmp_path / "uv.lock").write_text("# generated by uv\n")
        scorer = RRSScorer()
        result = scorer.score(tmp_path)
        ev = next(e for e in result.evidence if e.metric_id == "dep_pinning")
        assert ev.raw_score == pytest.approx(100.0)

    def test_seed_management_detected(self, well_specified_repo):
        from src.scoring.rrs import RRSScorer
        scorer = RRSScorer()
        result = scorer.score(well_specified_repo)
        ev = next(e for e in result.evidence if e.metric_id == "seed_management")
        # analysis.py has both np.random and np.random.seed → σ=1.0
        assert ev.raw_score == pytest.approx(100.0)

    def test_no_absolute_paths_clean(self, minimal_repo):
        from src.scoring.rrs import RRSScorer
        scorer = RRSScorer()
        result = scorer.score(minimal_repo)
        ev = next(e for e in result.evidence if e.metric_id == "no_absolute_paths")
        assert ev.raw_score == pytest.approx(100.0)

    def test_no_absolute_paths_detects_hardcoded(self, abs_path_repo):
        from src.scoring.rrs import RRSScorer
        scorer = RRSScorer()
        result = scorer.score(abs_path_repo)
        ev = next(e for e in result.evidence if e.metric_id == "no_absolute_paths")
        assert ev.raw_score < 100.0
        assert ev.fix_suggestion != ""

    def test_import_resolvability_fully_declared(self, declared_imports_repo):
        from src.scoring.rrs import RRSScorer
        scorer = RRSScorer()
        result = scorer.score(declared_imports_repo)
        ev = next(e for e in result.evidence if e.metric_id == "import_resolvability")
        # numpy and pandas both declared in requirements.txt
        assert ev.raw_score == pytest.approx(100.0)

    def test_notebook_exec_order_in_order(self, in_order_notebook_repo):
        from src.scoring.rrs import RRSScorer
        scorer = RRSScorer()
        result = scorer.score(in_order_notebook_repo)
        ev = next(e for e in result.evidence if e.metric_id == "notebook_exec_order")
        assert ev.raw_score == pytest.approx(100.0)

    def test_notebook_exec_order_out_of_order(self, out_of_order_notebook_repo):
        from src.scoring.rrs import RRSScorer
        scorer = RRSScorer()
        result = scorer.score(out_of_order_notebook_repo)
        ev = next(e for e in result.evidence if e.metric_id == "notebook_exec_order")
        assert ev.raw_score < 100.0
        assert ev.fix_suggestion != ""

    def test_reuse_metadata_two_files(self, reuse_metadata_repo):
        from src.scoring.rrs import RRSScorer
        scorer = RRSScorer()
        result = scorer.score(reuse_metadata_repo)
        ev = next(e for e in result.evidence if e.metric_id == "reuse_metadata")
        # LICENSE + CITATION.cff = tier 2 → score 70
        assert ev.raw_score == pytest.approx(70.0)

    def test_reuse_metadata_three_files(self, tmp_path):
        from src.scoring.rrs import RRSScorer
        (tmp_path / "LICENSE").write_text("MIT\n")
        (tmp_path / "CITATION.cff").write_text("cff-version: 1.2.0\n")
        (tmp_path / "codemeta.json").write_text('{"@type": "SoftwareSourceCode"}\n')
        scorer = RRSScorer()
        result = scorer.score(tmp_path)
        ev = next(e for e in result.evidence if e.metric_id == "reuse_metadata")
        assert ev.raw_score == pytest.approx(100.0)

    def test_paper_worked_example(self, tmp_path):
        """
        Replicate the paper's worked example:
        E=85, A=70, D=65, C=72, S=80 → RRS ≈ 74.80
        (We verify the formula, not the exact sub-metric detection)
        """
        from src.scoring.rrs import gate
        from src.scoring.rubric import load_rubric
        rubric = load_rubric()
        cfg = rubric.categories

        base = 100.0 * (
            cfg["E"]["weight"] * gate(85, cfg["E"]["tau"], cfg["E"]["k"]) +
            cfg["A"]["weight"] * gate(70, cfg["A"]["tau"], cfg["A"]["k"]) +
            cfg["D"]["weight"] * gate(65, cfg["D"]["tau"], cfg["D"]["k"]) +
            cfg["C"]["weight"] * gate(72, cfg["C"]["tau"], cfg["C"]["k"]) +
            cfg["S"]["weight"] * gate(80, cfg["S"]["tau"], cfg["S"]["k"])
        )
        # No penalties in example: both E≥10 and A≥10, sigma≥0.5
        assert base == pytest.approx(74.80, abs=0.1)


# ---------------------------------------------------------------------------
# ROS tests
# ---------------------------------------------------------------------------

class TestROSScorer:

    def test_no_evidence_returns_none(self):
        from src.scoring.ros import ROSScorer, ExecutionEvidence
        scorer = ROSScorer()
        result = scorer.score(ExecutionEvidence())
        assert result.ros is None
        assert result.available_components == []

    def test_notebooks_only(self):
        from src.scoring.ros import ROSScorer, ExecutionEvidence
        scorer = ROSScorer()
        ev = ExecutionEvidence(notebook_exec_rate=0.60)
        result = scorer.score(ev)
        assert result.ros == pytest.approx(60.0)
        assert "N" in result.available_components

    def test_full_ros(self):
        from src.scoring.ros import ROSScorer, ExecutionEvidence
        scorer = ROSScorer()
        ev = ExecutionEvidence(
            install_success=True,
            execution_success=True,
            output_determinism=90.0,
            notebook_exec_rate=0.60,
            import_success_rate=1.0,
            test_pass_rate=0.80,
        )
        result = scorer.score(ev)
        assert result.ros is not None
        assert 0 <= result.ros <= 100
        assert result.coverage_weight_sum == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# RCS tests
# ---------------------------------------------------------------------------

class TestRCSScorer:

    def test_no_ros_returns_rrs(self):
        from src.scoring.rcs import RCSScorer
        scorer = RCSScorer()
        result = scorer.score(rrs=72.0, ros=None, coverage_weight_sum=0.0)
        assert result.rcs == pytest.approx(72.0)
        assert result.alpha == 0.0

    def test_full_ros_increases_composite(self):
        from src.scoring.rcs import RCSScorer
        scorer = RCSScorer()
        result = scorer.score(rrs=72.0, ros=78.8, coverage_weight_sum=1.0)
        # ROS > RRS so composite should be > RRS
        assert result.rcs > 72.0
        assert result.alpha == pytest.approx(0.70)

    def test_rcs_bounded(self):
        from src.scoring.rcs import RCSScorer
        scorer = RCSScorer()
        for rrs, ros in [(0, 0), (100, 100), (50, 50), (0, 100), (100, 0)]:
            r = scorer.score(rrs=rrs, ros=ros, coverage_weight_sum=1.0)
            assert 0 <= r.rcs <= 100

    def test_paper_table6_notebooks_only(self):
        """Table 6 row: Notebooks only, α=0.10, ROS=60.0, RRS=72.0 → RCS=70.8"""
        from src.scoring.rcs import RCSScorer
        scorer = RCSScorer()
        # coverage_weight_sum for N only = 0.10
        result = scorer.score(rrs=72.0, ros=60.0, coverage_weight_sum=0.10)
        assert result.alpha == pytest.approx(0.10)
        assert result.rcs == pytest.approx(70.8, abs=0.1)

    def test_paper_table6_full_ros(self):
        """Table 6 row: Full ROS, α=0.70, ROS=78.8, RRS=72.0 → RCS=76.7"""
        from src.scoring.rcs import RCSScorer
        scorer = RCSScorer()
        result = scorer.score(rrs=72.0, ros=78.8, coverage_weight_sum=1.0)
        assert result.alpha == pytest.approx(0.70)
        assert result.rcs == pytest.approx(76.7, abs=0.2)
