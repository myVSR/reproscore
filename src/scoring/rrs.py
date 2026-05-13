"""
src/scoring/rrs.py
==================
Reproducibility Readiness Score (RRS) — static analysis of repository artefacts.

26 sub-metrics across five categories:
  E — Environment specification  (4 sub-metrics)
  A — Data accessibility         (4 sub-metrics)
  D — Documentation              (7 sub-metrics)
  C — Code portability           (4 sub-metrics)
  S — Reproducibility signals    (7 sub-metrics)

Can be run standalone without Docker:
    from src.scoring.rrs import RRSScorer
    scorer = RRSScorer()
    result = scorer.score("/path/to/repo")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .rubric import Rubric, load_rubric
from src.utils.notebook_paths import is_excluded_notebook


# ---------------------------------------------------------------------------
# Notebook cell extraction helper
# ---------------------------------------------------------------------------

def _nb_cells(nb_path: Path) -> Tuple[List[str], int, int]:
    """
    Parse a .ipynb and return (code_lines, n_markdown_cells, n_code_cells).
    code_lines: all non-empty lines from code cells.
    Returns ([], 0, 0) on any parse error.
    """
    try:
        data = json.loads(nb_path.read_text(errors="replace"))
    except Exception:
        return [], 0, 0
    code_lines: List[str] = []
    n_md = 0
    n_code = 0
    for cell in data.get("cells", []):
        ct = cell.get("cell_type", "")
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else source
        if ct == "code":
            n_code += 1
            code_lines.extend(text.splitlines())
        elif ct == "markdown":
            if len(text.strip()) >= 10:
                n_md += 1
    return code_lines, n_md, n_code


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SubMetricEvidence:
    """Provenance record for a single sub-metric (maps to PROV-O Activity)."""
    metric_id: str
    category: str
    raw_score: float
    file_path: Optional[str]
    line_range: Optional[Tuple[int, int]] = None
    source_context: Optional[str] = None
    deduction_reason: str = ""
    fix_suggestion: str = ""


@dataclass
class CategoryScore:
    name: str
    symbol: str
    raw_score: float
    gated_score: float
    sub_metrics: Dict[str, SubMetricEvidence] = field(default_factory=dict)


@dataclass
class RRSResult:
    """Full RRS computation result."""
    rrs: float
    category_scores: Dict[str, CategoryScore] = field(default_factory=dict)
    evidence: List[SubMetricEvidence] = field(default_factory=list)
    penalty_environment: float = 0.0
    penalty_data: float = 0.0
    penalty_seed: float = 0.0
    rubric_name: str = "default"

    def to_dict(self) -> dict:
        return {
            "rrs": self.rrs,
            "rubric": self.rubric_name,
            "penalties": {
                "environment": self.penalty_environment,
                "data": self.penalty_data,
                "seed": self.penalty_seed,
            },
            "categories": {
                sym: {
                    "name": cat.name,
                    "raw_score": cat.raw_score,
                    "gated_score": cat.gated_score,
                }
                for sym, cat in self.category_scores.items()
            },
            "evidence": [asdict(e) for e in self.evidence],
        }


# ---------------------------------------------------------------------------
# Gate function (Equation 1 from paper)
# ---------------------------------------------------------------------------

def gate(x: float, tau: float, k: float) -> float:
    """Power-law gate g(x, tau, k). Above tau: linear. Below tau: power-law compression."""
    if x >= tau:
        return x / 100.0
    return (x / tau) ** k * (tau / 100.0)


# ---------------------------------------------------------------------------
# Shared constants for import_resolvability
# ---------------------------------------------------------------------------

_STDLIB = frozenset({
    'abc', 'ast', 'asyncio', 'base64', 'binascii', 'builtins', 'calendar',
    'cmath', 'code', 'collections', 'concurrent', 'configparser', 'contextlib',
    'copy', 'csv', 'dataclasses', 'datetime', 'decimal', 'difflib', 'email',
    'enum', 'errno', 'fractions', 'functools', 'gc', 'glob', 'gzip', 'hashlib',
    'heapq', 'hmac', 'html', 'http', 'importlib', 'inspect', 'io', 'ipaddress',
    'itertools', 'json', 'logging', 'math', 'multiprocessing', 'numbers',
    'operator', 'os', 'pathlib', 'pickle', 'platform', 'pprint', 'queue',
    'random', 're', 'shutil', 'signal', 'socket', 'sqlite3', 'stat', 'string',
    'struct', 'subprocess', 'sys', 'tempfile', 'textwrap', 'threading', 'time',
    'timeit', 'traceback', 'types', 'typing', 'unittest', 'urllib', 'uuid',
    'warnings', 'weakref', 'xml', 'zipfile', 'zlib', '_thread', 'atexit',
    'compileall', 'curses', 'dbm', 'dis', 'doctest', 'encodings', 'filecmp',
    'fileinput', 'ftplib', 'getopt', 'getpass', 'gettext', 'grp', 'imaplib',
    'keyword', 'lib2to3', 'mailbox', 'marshal', 'mmap', 'modulefinder',
    'netrc', 'nntplib', 'optparse', 'pathlib', 'pdb', 'pkgutil', 'posixpath',
    'profile', 'pstats', 'pty', 'pwd', 'py_compile', 'pydoc', 'readline',
    'reprlib', 'resource', 'rlcompleter', 'runpy', 'sched', 'secrets', 'select',
    'selectors', 'shelve', 'shlex', 'site', 'smtplib', 'ssl', 'statistics',
    'stringprep', 'symtable', 'sysconfig', 'syslog', 'tabnanny', 'tarfile',
    'telnetlib', 'termios', 'test', 'token', 'tokenize', 'trace', 'tracemalloc',
    'tty', 'unicodedata', 'venv', 'wsgiref', 'xdrlib', 'xmlrpc', 'zipapp',
    'zipimport', '__future__', 'antigravity', 'this',
    # IPython/Jupyter builtins
    'IPython', 'ipykernel', 'ipywidgets', 'nbformat', 'ipython_genutils',
    'traitlets', 'jedi', 'comm', 'ipympl',
    # Common testing infrastructure (not third-party deps for science repos)
    '',
})

_IMPORT_TO_PKG = {
    'cv2': 'opencv-python', 'PIL': 'pillow', 'sklearn': 'scikit-learn',
    'skimage': 'scikit-image', 'bs4': 'beautifulsoup4', 'yaml': 'pyyaml',
    'dotenv': 'python-dotenv', 'pkg_resources': 'setuptools',
    'tf': 'tensorflow', 'Bio': 'biopython', 'osgeo': 'gdal',
    'Crypto': 'pycryptodome', 'jwt': 'pyjwt', 'dateutil': 'python-dateutil',
    'attr': 'attrs', 'google': 'google-cloud', 'azure': 'azure-storage-blob',
    'wx': 'wxpython', 'gi': 'pygobject', 'gtk': 'pygobject',
    'sklearn_crfsuite': 'sklearn-crfsuite', 'umap': 'umap-learn',
}


def _find_readme(repo: Path) -> Optional[Path]:
    for name in ["README.md", "README.rst", "README.txt", "README"]:
        fp = repo / name
        if fp.exists():
            return fp
    return None


# ---------------------------------------------------------------------------
# Sub-metric detectors — 26 atomic sub-metrics
# ---------------------------------------------------------------------------

class _SubMetricDetectors:

    # =====================================================================
    # Category E: Environment Specification
    # =====================================================================

    @staticmethod
    def dep_pinning(repo: Path) -> SubMetricEvidence:
        """E1: Dependency pinning — tiered: lockfile>full pins>partial>file present>absent."""
        lockfiles = [
            "uv.lock", "poetry.lock", "Pipfile.lock", "pdm.lock",
            "pixi.lock", "renv.lock", "package-lock.json", "yarn.lock",
        ]
        for fname in lockfiles:
            if (repo / fname).exists():
                return SubMetricEvidence(
                    metric_id="dep_pinning", category="E", raw_score=100,
                    file_path=fname,
                    deduction_reason=f"Lockfile found: {fname} (strongest reproducibility guarantee)",
                )

        pin_exact = re.compile(r'==\s*\d+[\.\d]*')
        pin_constrained = re.compile(r'[><=!]=?\s*\d')
        req_files = [
            "requirements.txt", "requirements-dev.txt", "pyproject.toml",
            "setup.cfg", "setup.py", "Pipfile", "environment.yml", "environment.yaml",
        ]
        for fname in req_files:
            fp = repo / fname
            if not fp.exists():
                continue
            content = fp.read_text(errors="replace")
            lines = [
                l.strip() for l in content.splitlines()
                if l.strip()
                and not l.strip().startswith('#')
                and not l.strip().startswith('[')
                and not l.strip().startswith('-')
                and not l.strip().startswith('name:')
                and not l.strip().startswith('channels:')
                and '=' in l or re.search(r'[a-zA-Z]', l)
            ]
            # Filter to lines that look like package declarations
            pkg_lines = [l for l in lines if re.match(r'[a-zA-Z]', l)]
            if not pkg_lines:
                return SubMetricEvidence(
                    metric_id="dep_pinning", category="E", raw_score=20,
                    file_path=fname,
                    deduction_reason=f"{fname} present but contains no package entries",
                    fix_suggestion=f"Add pinned dependencies to {fname} (e.g. numpy==1.24.3).",
                )
            pinned = sum(1 for l in pkg_lines if pin_exact.search(l))
            constrained = sum(1 for l in pkg_lines if pin_constrained.search(l))
            total = len(pkg_lines)
            if pinned >= total:
                return SubMetricEvidence(
                    metric_id="dep_pinning", category="E", raw_score=80,
                    file_path=fname,
                    deduction_reason=f"All {total} dependencies fully pinned (==) in {fname}",
                    fix_suggestion="Use a lockfile (uv.lock, poetry.lock) for stronger guarantees.",
                )
            if pinned > 0 or constrained > 0:
                frac = (pinned + 0.5 * max(0, constrained - pinned)) / total
                score = round(50 + frac * 20)
                return SubMetricEvidence(
                    metric_id="dep_pinning", category="E", raw_score=score,
                    file_path=fname,
                    deduction_reason=f"{pinned}/{total} fully pinned; {constrained}/{total} constrained in {fname}",
                    fix_suggestion=f"Pin all dependencies with == in {fname} or use a lockfile.",
                )
            return SubMetricEvidence(
                metric_id="dep_pinning", category="E", raw_score=20,
                file_path=fname,
                deduction_reason=f"{fname} has {total} entries but no version pins",
                fix_suggestion=f"Pin versions in {fname} (e.g. numpy==1.24.3) or use uv.lock/poetry.lock.",
            )

        return SubMetricEvidence(
            metric_id="dep_pinning", category="E", raw_score=0,
            file_path=None,
            deduction_reason="No dependency specification file found",
            fix_suggestion="Add requirements.txt with pinned versions or a lockfile (uv.lock, poetry.lock).",
        )

    @staticmethod
    def container_spec(repo: Path) -> SubMetricEvidence:
        """E2: Container specification — presence and minimal content check."""
        glob_patterns = [
            "**/Dockerfile", "**/docker-compose.yml", "**/docker-compose.yaml",
            "**/Singularity", "**/*.def",
        ]
        for pat in glob_patterns:
            hits = [p for p in repo.glob(pat) if not is_excluded_notebook(p)]
            if not hits:
                continue
            docker_path = hits[0]
            rel_path = str(docker_path.relative_to(repo))

            if docker_path.suffix in {".yml", ".yaml"} or docker_path.name == "Singularity":
                return SubMetricEvidence(
                    metric_id="container_spec", category="E", raw_score=80,
                    file_path=rel_path,
                    deduction_reason=f"Container spec found: {docker_path.name}",
                )
            if docker_path.suffix == ".def":
                return SubMetricEvidence(
                    metric_id="container_spec", category="E", raw_score=80,
                    file_path=rel_path,
                    deduction_reason=f"Singularity definition found: {docker_path.name}",
                )

            # Parse Dockerfile content
            content = docker_path.read_text(errors="replace")
            has_from = bool(re.search(r'^FROM\s+\S+', content, re.M | re.I))
            has_run = bool(re.search(r'^RUN\s+', content, re.M | re.I))
            pinned_base = bool(re.search(
                r'^FROM\s+\S+:\d+\.\d+[\.\d]*(?:-\S+)?\s*(?:#.*)?$', content, re.M | re.I
            ))

            if has_from and has_run and pinned_base:
                return SubMetricEvidence(
                    metric_id="container_spec", category="E", raw_score=100,
                    file_path=rel_path,
                    deduction_reason="Dockerfile: pinned base image + RUN install commands",
                )
            if has_from and has_run:
                return SubMetricEvidence(
                    metric_id="container_spec", category="E", raw_score=80,
                    file_path=rel_path,
                    deduction_reason="Dockerfile: FROM + RUN present (base image not pinned)",
                    fix_suggestion="Pin the base image tag (e.g. FROM python:3.11.4) for full reproducibility.",
                )
            if has_from:
                return SubMetricEvidence(
                    metric_id="container_spec", category="E", raw_score=50,
                    file_path=rel_path,
                    deduction_reason="Dockerfile: FROM only — no RUN/install commands found",
                    fix_suggestion="Add RUN pip install commands and pin the base image tag.",
                )
            return SubMetricEvidence(
                metric_id="container_spec", category="E", raw_score=20,
                file_path=rel_path,
                deduction_reason="Container spec present but appears empty or malformed",
                fix_suggestion="Add FROM (pinned) and RUN install commands to your Dockerfile.",
            )

        return SubMetricEvidence(
            metric_id="container_spec", category="E", raw_score=0,
            file_path=None,
            deduction_reason="No container specification (Dockerfile, Singularity) found",
            fix_suggestion="Add a Dockerfile with a pinned base image and RUN pip install commands.",
        )

    @staticmethod
    def env_bootstrap(repo: Path) -> SubMetricEvidence:
        """E3: Executable environment bootstrap — one-command env creation scripts."""
        for fname in ["install.sh", "bootstrap.sh", "setup.sh", "env_setup.sh",
                      "create_env.sh", "install_env.sh"]:
            if (repo / fname).exists():
                return SubMetricEvidence(
                    metric_id="env_bootstrap", category="E", raw_score=100,
                    file_path=fname,
                    deduction_reason=f"Environment bootstrap script found: {fname}",
                )

        # Makefile with setup/install/env targets (not run targets)
        makefile = repo / "Makefile"
        if makefile.exists():
            content = makefile.read_text(errors="replace")
            if re.search(
                r'^(?:setup|install|env|environment|deps|dependencies|bootstrap)\s*:',
                content, re.M
            ):
                return SubMetricEvidence(
                    metric_id="env_bootstrap", category="E", raw_score=100,
                    file_path="Makefile",
                    deduction_reason="Makefile with setup/install/env target found",
                )

        # README with one-command bootstrap instructions
        readme = _find_readme(repo)
        if readme:
            content = readme.read_text(errors="replace").lower()
            one_cmd_markers = [
                "pixi install", "uv sync", "conda env create",
                "make setup", "make install", "make env",
                "bash install.sh", "./install.sh", "./bootstrap.sh",
                "./setup.sh", "bash setup.sh",
            ]
            if any(m in content for m in one_cmd_markers):
                return SubMetricEvidence(
                    metric_id="env_bootstrap", category="E", raw_score=80,
                    file_path=readme.name,
                    deduction_reason="One-command environment setup instructions found in README",
                )

        return SubMetricEvidence(
            metric_id="env_bootstrap", category="E", raw_score=0,
            file_path=None,
            deduction_reason="No executable environment bootstrap found",
            fix_suggestion=(
                "Add an install.sh script or Makefile setup target for "
                "one-command environment creation."
            ),
        )

    @staticmethod
    def python_version_declared(repo: Path) -> SubMetricEvidence:
        """E4: Runtime version explicitly declared (Python/R/Julia)."""
        # .python-version (pyenv)
        pv = repo / ".python-version"
        if pv.exists():
            ver = pv.read_text(errors="replace").strip()[:30]
            return SubMetricEvidence(
                metric_id="python_version_declared", category="E", raw_score=100,
                file_path=".python-version",
                deduction_reason=f"Python version declared via pyenv: {ver}",
            )

        # runtime.txt (repo2docker / Heroku)
        rt = repo / "runtime.txt"
        if rt.exists():
            ver = rt.read_text(errors="replace").strip()[:30]
            return SubMetricEvidence(
                metric_id="python_version_declared", category="E", raw_score=100,
                file_path="runtime.txt",
                deduction_reason=f"Runtime version declared: {ver}",
            )

        # python_requires in pyproject.toml / setup.cfg / setup.py
        for fname in ["pyproject.toml", "setup.cfg", "setup.py"]:
            fp = repo / fname
            if fp.exists():
                content = fp.read_text(errors="replace")
                if re.search(r'python_requires\s*=\s*["\']', content):
                    return SubMetricEvidence(
                        metric_id="python_version_declared", category="E", raw_score=100,
                        file_path=fname,
                        deduction_reason=f"python_requires found in {fname}",
                    )

        # environment.yml with explicit python version
        for fname in ["environment.yml", "environment.yaml"]:
            fp = repo / fname
            if fp.exists():
                content = fp.read_text(errors="replace")
                if re.search(r'[-\s]python\s*[=<>][\s\d]', content):
                    return SubMetricEvidence(
                        metric_id="python_version_declared", category="E", raw_score=100,
                        file_path=fname,
                        deduction_reason=f"Python version constraint found in {fname}",
                    )

        # renv.lock with R version
        rl = repo / "renv.lock"
        if rl.exists():
            content = rl.read_text(errors="replace")
            if re.search(r'"R"\s*:\s*\{[^}]*"Version"', content, re.S):
                return SubMetricEvidence(
                    metric_id="python_version_declared", category="E", raw_score=100,
                    file_path="renv.lock",
                    deduction_reason="R version declared in renv.lock",
                )

        # Pinned base image in Dockerfile implies runtime version
        for pat in ["**/Dockerfile"]:
            hits = [p for p in repo.glob(pat) if not is_excluded_notebook(p)]
            for df in hits[:1]:
                content = df.read_text(errors="replace")
                if re.search(
                    r'^FROM\s+(?:python|r-base|rocker/r|julia)\s*:\s*\d+\.\d+',
                    content, re.M | re.I
                ):
                    return SubMetricEvidence(
                        metric_id="python_version_declared", category="E", raw_score=100,
                        file_path=str(df.relative_to(repo)),
                        deduction_reason="Pinned runtime version in Dockerfile FROM",
                    )

        return SubMetricEvidence(
            metric_id="python_version_declared", category="E", raw_score=0,
            file_path=None,
            deduction_reason="No explicit Python/R version declaration found",
            fix_suggestion=(
                "Add a .python-version file (pyenv), runtime.txt, or "
                "python_requires in pyproject.toml."
            ),
        )

    # =====================================================================
    # Category A: Data Accessibility
    # =====================================================================

    @staticmethod
    def data_description(repo: Path) -> SubMetricEvidence:
        """A1: Data description — tiered by content richness."""
        for fname in ["DATA.md", "data/README.md", "data/README.txt",
                      "data/DESCRIPTION.md"]:
            fp = repo / fname
            if fp.exists():
                content = fp.read_text(errors="replace")
                wc = len(content.split())
                if wc >= 100:
                    score, reason = 100, f"Rich data documentation: {fname} ({wc} words)"
                elif wc >= 30:
                    score, reason = 70, f"Brief data documentation: {fname} ({wc} words)"
                else:
                    score, reason = 40, f"Minimal data documentation: {fname} ({wc} words)"
                return SubMetricEvidence(
                    metric_id="data_description", category="A", raw_score=score,
                    file_path=fname, deduction_reason=reason,
                    fix_suggestion="Expand data documentation with dataset name, size, source, and format." if score < 100 else "",
                )

        readme = _find_readme(repo)
        if readme:
            content = readme.read_text(errors="replace")
            match = re.search(
                r'(#+\s*(?:data|dataset|data\s+availability|data\s+description)[^\n]*\n)',
                content, re.I
            )
            if match:
                section = content[match.start():]
                next_h = re.search(r'\n#+\s', section[5:])
                section = section[:next_h.start() + 5] if next_h else section
                wc = len(section.split())
                if wc >= 60:
                    score, reason = 80, f"Substantial data section in README ({wc} words)"
                elif wc >= 20:
                    score, reason = 50, f"Brief data section in README ({wc} words)"
                else:
                    score, reason = 20, f"Minimal data mention in README ({wc} words)"
                return SubMetricEvidence(
                    metric_id="data_description", category="A", raw_score=score,
                    file_path=readme.name, deduction_reason=reason,
                    fix_suggestion="Expand the data section with dataset name, size, source, format." if score < 80 else "",
                )
            lower = content.lower()
            if any(kw in lower for kw in ["dataset", "data availability", "data source"]):
                return SubMetricEvidence(
                    metric_id="data_description", category="A", raw_score=20,
                    file_path=readme.name,
                    deduction_reason="Data keywords in README but no dedicated section",
                    fix_suggestion="Add a '## Data' section describing dataset name, source, format, size.",
                )

        return SubMetricEvidence(
            metric_id="data_description", category="A", raw_score=0,
            file_path=None,
            deduction_reason="No data description found",
            fix_suggestion="Add a DATA.md or '## Data' README section describing dataset source and format.",
        )

    @staticmethod
    def data_pointer(repo: Path) -> SubMetricEvidence:
        """A2: Data pointer — tiered by archival quality. Absorbs open_format signal."""
        archival = [
            ("zenodo.org", 100), ("figshare.com", 100), ("osf.io", 100),
            ("dryad", 100), ("dataverse", 100), ("doi.org/10.", 100),
            ("mendeley.com/datasets", 100), ("data.mendeley.com", 100),
            ("pangaea.de", 100), ("ukdataservice", 100),
        ]
        institutional = [
            ("ebi.ac.uk", 80), ("ncbi.nlm.nih.gov", 80), ("sra.ncbi", 80),
            ("geo.ncbi", 80), ("arrayexpress", 80), ("embl.org", 80),
            ("kaggle.com/datasets", 75), ("huggingface.co/datasets", 80),
            ("archive.ics.uci.edu", 80),
        ]
        release = [("github.com", 60), ("gitlab.com", 60), ("bitbucket.org", 60)]
        all_patterns = archival + institutional + release

        generic_data_url = re.compile(
            r'https?://[^\s\)\]>"\']+\.(?:csv|tsv|json|parquet|hdf5|h5|zip|tar\.gz|gz)\b',
            re.I
        )

        best_score = 0
        best_match = None
        best_file = None

        check_docs = ["DATA.md", "data/README.md"]
        for doc in check_docs:
            fp = repo / doc
            if fp.exists():
                lower = fp.read_text(errors="replace").lower()
                for pat, score in all_patterns:
                    if pat in lower and score > best_score:
                        best_score, best_match, best_file = score, pat, doc

        readme = _find_readme(repo)
        if readme:
            content = readme.read_text(errors="replace")
            lower = content.lower()
            for pat, score in all_patterns:
                if pat in lower and score > best_score:
                    best_score, best_match, best_file = score, pat, readme.name
            if best_score == 0 and generic_data_url.search(content):
                best_score, best_match, best_file = 40, "direct data URL", readme.name

        if best_score > 0:
            return SubMetricEvidence(
                metric_id="data_pointer", category="A", raw_score=best_score,
                file_path=best_file,
                deduction_reason=f"Data pointer: {best_match} (quality tier: {best_score}/100)",
                fix_suggestion="Archive data on Zenodo/Figshare for a persistent DOI (score 100)." if best_score < 100 else "",
            )

        # Local open-format data as last resort (absorbs open_format metric)
        _data_exts = {'.csv', '.tsv', '.json', '.parquet', '.hdf5', '.h5',
                      '.nc', '.zarr', '.npy', '.rds', '.feather', '.arrow'}
        for ext in ['.csv', '.tsv', '.json', '.parquet']:
            hits = [p for p in repo.glob(f"**/*{ext}")
                    if not is_excluded_notebook(p)
                    and not any(part.startswith('.') for part in p.parts)][:1]
            if hits:
                return SubMetricEvidence(
                    metric_id="data_pointer", category="A", raw_score=25,
                    file_path=str(hits[0].relative_to(repo)),
                    deduction_reason=f"Local open-format data file ({ext}) — no external pointer",
                    fix_suggestion="Archive data on Zenodo or Figshare and link from README.",
                )

        return SubMetricEvidence(
            metric_id="data_pointer", category="A", raw_score=0,
            file_path=None,
            deduction_reason="No data pointer found",
            fix_suggestion="Link to a data archive (Zenodo, Figshare, OSF) in your README.",
        )

    @staticmethod
    def workflow_orchestration(repo: Path) -> SubMetricEvidence:
        """A3: Workflow orchestration — pipeline management tool presence."""
        for fname in ["Snakefile", "dvc.yaml", "nextflow.config", "main.nf",
                      "workflow.cwl", "main.wdl", "Snakefile.smk"]:
            if (repo / fname).exists():
                return SubMetricEvidence(
                    metric_id="workflow_orchestration", category="A", raw_score=100,
                    file_path=fname,
                    deduction_reason=f"Workflow orchestration file: {fname}",
                )

        for pat in ["**/*.nf", "**/*.cwl", "**/*.wdl", "**/Snakefile"]:
            hits = [p for p in repo.glob(pat) if not is_excluded_notebook(p)]
            if hits:
                return SubMetricEvidence(
                    metric_id="workflow_orchestration", category="A", raw_score=100,
                    file_path=str(hits[0].relative_to(repo)),
                    deduction_reason=f"Workflow file: {hits[0].name}",
                )

        # DVC configuration
        dvc_files = [p for p in repo.glob("**/*.dvc") if not is_excluded_notebook(p)]
        if dvc_files or (repo / ".dvc").is_dir():
            fp = ".dvc" if (repo / ".dvc").is_dir() else str(dvc_files[0].relative_to(repo))
            return SubMetricEvidence(
                metric_id="workflow_orchestration", category="A", raw_score=100,
                file_path=fp,
                deduction_reason="DVC pipeline configuration found",
            )

        # Makefile (general orchestration)
        if (repo / "Makefile").exists():
            return SubMetricEvidence(
                metric_id="workflow_orchestration", category="A", raw_score=80,
                file_path="Makefile",
                deduction_reason="Makefile found (general workflow orchestration)",
            )

        # Explicit pipeline scripts
        for fname in ["workflow.py", "pipeline.py", "run_pipeline.py", "run_analysis.py"]:
            if (repo / fname).exists():
                return SubMetricEvidence(
                    metric_id="workflow_orchestration", category="A", raw_score=70,
                    file_path=fname,
                    deduction_reason=f"Pipeline script: {fname}",
                )

        return SubMetricEvidence(
            metric_id="workflow_orchestration", category="A", raw_score=0,
            file_path=None,
            deduction_reason="No workflow orchestration tool found",
            fix_suggestion="Add a Snakefile, Makefile, or Nextflow pipeline to orchestrate the analysis.",
        )

    @staticmethod
    def data_acquisition_script(repo: Path) -> SubMetricEvidence:
        """A4: Data acquisition script — automated data download detection."""
        # DVC tracking files (strongest signal)
        dvc_files = [p for p in repo.glob("**/*.dvc") if not is_excluded_notebook(p)]
        if dvc_files:
            return SubMetricEvidence(
                metric_id="data_acquisition_script", category="A", raw_score=100,
                file_path=str(dvc_files[0].relative_to(repo)),
                deduction_reason="DVC data tracking file found — data is versioned and retrievable",
            )

        download_re = re.compile(
            r'(?:'
            r'wget\s+https?://'
            r'|curl\s+.*https?://'
            r'|zenodo_get\b'
            r'|osf\s+clone\b'
            r'|osfclient\b'
            r'|kaggle\s+datasets\s+download\b'
            r'|import\s+kaggle\b'
            r'|requests\.get\s*\([^)]*(?:zenodo|figshare|osf\.io)'
            r'|pooch\.retrieve\b'
            r'|dvc\s+pull\b'
            r')', re.I
        )

        check: List[Path] = []
        for pat in ["**/*.sh", "**/Makefile", "**/makefile"]:
            check.extend(p for p in repo.glob(pat) if not is_excluded_notebook(p))
        check.extend(
            f for f in sorted(repo.glob("**/*.py"))[:30]
            if not is_excluded_notebook(f)
        )
        nb_files = [f for f in sorted(repo.glob("**/*.ipynb"))
                    if not is_excluded_notebook(f)][:20]

        for f in check:
            content = f.read_text(errors="replace")
            if download_re.search(content):
                return SubMetricEvidence(
                    metric_id="data_acquisition_script", category="A", raw_score=100,
                    file_path=str(f.relative_to(repo)),
                    deduction_reason=f"Automated data download found in {f.name}",
                )

        for nb in nb_files:
            code_lines, _, _ = _nb_cells(nb)
            if download_re.search("\n".join(code_lines)):
                return SubMetricEvidence(
                    metric_id="data_acquisition_script", category="A", raw_score=100,
                    file_path=str(nb.relative_to(repo)),
                    deduction_reason=f"Automated data download in notebook {nb.name}",
                )

        return SubMetricEvidence(
            metric_id="data_acquisition_script", category="A", raw_score=0,
            file_path=None,
            deduction_reason="No automated data acquisition found",
            fix_suggestion=(
                "Add a download script (wget/curl), DVC tracking, "
                "or Kaggle/Zenodo/OSF API calls to enable programmatic data access."
            ),
        )

    # =====================================================================
    # Category D: Documentation
    # =====================================================================

    @staticmethod
    def doc_structure(repo: Path) -> SubMetricEvidence:
        """D1: README section coverage — execution-relevant sections only."""
        expected = [
            r"#+\s*(install|installation|setup|getting\s+started)",
            r"#+\s*(usage|run|execute|how\s+to\s+run|quick\s*start)",
            r"#+\s*(expected\s+output|expected\s+result|result|output)",
            r"#+\s*(requirement|dependency|prerequisites|hardware|compute)",
        ]
        readme = _find_readme(repo)
        if not readme:
            return SubMetricEvidence(
                metric_id="doc_structure", category="D", raw_score=0,
                file_path=None,
                deduction_reason="No README file found",
                fix_suggestion="Add a README.md with Installation, Usage/Run, Expected Output, and Requirements sections.",
            )
        content = readme.read_text(errors="replace")
        found = sum(1 for pat in expected if re.search(pat, content, re.I))
        score = (found / len(expected)) * 100
        return SubMetricEvidence(
            metric_id="doc_structure", category="D", raw_score=score,
            file_path=readme.name,
            deduction_reason=f"README has {found}/{len(expected)} execution-relevant sections",
            fix_suggestion=(
                "Add missing sections: Installation, Usage/Run, Expected Output, Hardware Requirements."
                if found < len(expected) else ""
            ),
        )

    @staticmethod
    def install_instructions(repo: Path) -> SubMetricEvidence:
        """D2: Install instructions — tiered by completeness."""
        one_cmd_re = re.compile(
            r'(?:pip\s+install\s+-r\s+\S+|conda\s+env\s+create\s+-f\s+\S+'
            r'|make\s+(?:setup|install|env)\b|pixi\s+install\b|uv\s+sync\b'
            r'|docker\s+build\b|bash\s+install\.sh|./install\.sh)',
            re.I
        )
        any_cmd_re = re.compile(
            r'pip\s+install|conda\s+install|poetry\s+install|pipenv\s+install'
            r'|npm\s+install|python\s+setup\.py|renv::restore',
            re.I
        )

        readme = _find_readme(repo)
        if readme:
            content = readme.read_text(errors="replace")
            if one_cmd_re.search(content):
                return SubMetricEvidence(
                    metric_id="install_instructions", category="D", raw_score=100,
                    file_path=readme.name,
                    deduction_reason="One-command install found in README",
                )
            if any_cmd_re.search(content):
                return SubMetricEvidence(
                    metric_id="install_instructions", category="D", raw_score=70,
                    file_path=readme.name,
                    deduction_reason="Multi-step install instructions found in README",
                    fix_suggestion="Consolidate installation to a single command (e.g. make setup).",
                )
            if any(kw in content.lower() for kw in ["install", "dependencies", "requirements"]):
                return SubMetricEvidence(
                    metric_id="install_instructions", category="D", raw_score=30,
                    file_path=readme.name,
                    deduction_reason="Dependency reference in README but no install command",
                    fix_suggestion="Add a concrete install command (e.g. 'pip install -r requirements.txt').",
                )

        for fname in ["INSTALL", "INSTALL.md", "INSTALL.txt"]:
            if (repo / fname).exists():
                return SubMetricEvidence(
                    metric_id="install_instructions", category="D", raw_score=70,
                    file_path=fname,
                    deduction_reason=f"INSTALL file found: {fname}",
                )

        return SubMetricEvidence(
            metric_id="install_instructions", category="D", raw_score=0,
            file_path=None,
            deduction_reason="No install instructions found",
            fix_suggestion="Add installation instructions to README.md.",
        )

    @staticmethod
    def usage_examples(repo: Path) -> SubMetricEvidence:
        """D3: Usage examples — tiered: runnable command > code snippets > examples dir."""
        readme = _find_readme(repo)
        if readme:
            content = readme.read_text(errors="replace")
            # Tier 1: code fence containing a system command
            sys_cmd_re = re.compile(
                r'```[a-zA-Z]*\s*\n(?:[^\n]*\n)*?'
                r'(?:python\s+\S+\.py|bash\s+\S+\.sh|Rscript\s+\S+\.R'
                r'|make\s+\w+|\./\S+\.sh|nextflow\s+run)',
                re.I
            )
            if sys_cmd_re.search(content):
                return SubMetricEvidence(
                    metric_id="usage_examples", category="D", raw_score=100,
                    file_path=readme.name,
                    deduction_reason="Runnable end-to-end example found in README",
                )
            # Tier 2: any code block
            if "```" in content or "    >>>" in content:
                return SubMetricEvidence(
                    metric_id="usage_examples", category="D", raw_score=60,
                    file_path=readme.name,
                    deduction_reason="Code snippets found in README",
                    fix_suggestion="Add a complete runnable command (e.g. 'python main.py --data data/').",
                )

        # Tier 3: examples directory
        for dirname in ["examples", "example", "demos", "demo", "tutorials"]:
            d = repo / dirname
            if d.is_dir() and any(d.iterdir()):
                return SubMetricEvidence(
                    metric_id="usage_examples", category="D", raw_score=30,
                    file_path=dirname,
                    deduction_reason=f"Examples directory: {dirname}/",
                    fix_suggestion="Add a README code block showing how to run the example.",
                )

        return SubMetricEvidence(
            metric_id="usage_examples", category="D", raw_score=0,
            file_path=None,
            deduction_reason="No usage examples found",
            fix_suggestion="Add a code example to README or an examples/ directory.",
        )

    @staticmethod
    def docstring_coverage(repo: Path) -> SubMetricEvidence:
        """D4: Inline docstring coverage on public functions/classes (no Sphinx/MkDocs)."""
        py_files = [f for f in sorted(repo.glob("**/*.py"))
                    if not is_excluded_notebook(f)][:20]
        if not py_files:
            return SubMetricEvidence(
                metric_id="docstring_coverage", category="D", raw_score=50,
                file_path=None,
                deduction_reason="No Python files found",
            )
        docstring_re = re.compile(
            r'(?:^def\s+[^_]\w*\s*\([^)]*\)\s*:\s*\n\s*"""'
            r'|^class\s+\w+\s*(?:\([^)]*\))?\s*:\s*\n\s*""")',
            re.M
        )
        files_with_docs = sum(
            1 for pf in py_files
            if docstring_re.search(pf.read_text(errors="replace"))
        )
        if files_with_docs == 0:
            return SubMetricEvidence(
                metric_id="docstring_coverage", category="D", raw_score=0,
                file_path=None,
                deduction_reason="No inline docstrings on public functions/classes",
                fix_suggestion="Add docstrings to public functions and classes.",
            )
        score = min(files_with_docs / len(py_files), 1.0) * 100
        return SubMetricEvidence(
            metric_id="docstring_coverage", category="D", raw_score=score,
            file_path=str(py_files[0].relative_to(repo)),
            deduction_reason=f"Docstrings in {files_with_docs}/{len(py_files)} Python files",
        )

    @staticmethod
    def inline_explanation_density(repo: Path) -> SubMetricEvidence:
        """D5: Inline explanation density — unified metric (md/code ratio + comment density)."""
        nb_files = [f for f in sorted(repo.glob("**/*.ipynb"))
                    if not is_excluded_notebook(f)][:20]
        py_files = [f for f in sorted(repo.glob("**/*.py"))
                    if not is_excluded_notebook(f)][:20]
        r_files = [f for f in sorted(repo.glob("**/*.R"))
                   if not is_excluded_notebook(f)][:10]

        has_notebooks = bool(nb_files)
        has_scripts = bool(py_files or r_files)

        if not has_notebooks and not has_scripts:
            return SubMetricEvidence(
                metric_id="inline_explanation_density", category="D", raw_score=50,
                file_path=None,
                deduction_reason="No Python/R files or Jupyter notebooks found",
            )

        nb_score: Optional[float] = None
        script_score: Optional[float] = None

        if has_notebooks:
            total_md, total_code = 0, 0
            for nb in nb_files:
                _, n_md, n_code = _nb_cells(nb)
                total_md += n_md
                total_code += n_code
            if total_code > 0:
                nb_score = min(total_md / total_code / 0.5, 1.0) * 100

        if has_scripts:
            total_lines, total_comments = 0, 0
            for f in py_files + r_files:
                for line in f.read_text(errors="replace").splitlines():
                    s = line.strip()
                    if s:
                        total_lines += 1
                        if s.startswith("#"):
                            total_comments += 1
            if total_lines > 0:
                script_score = min(total_comments / total_lines / 0.20, 1.0) * 100

        if nb_score is not None and script_score is not None:
            score = round(0.6 * nb_score + 0.4 * script_score, 1)
            reason = f"Notebook md/code and script comment density combined: {score:.0f}/100"
        elif nb_score is not None:
            score = round(nb_score, 1)
            reason = f"Notebook narrative density: {score:.0f}/100"
        else:
            score = round(script_score, 1)  # type: ignore[arg-type]
            reason = f"Script comment density: {score:.0f}/100"

        return SubMetricEvidence(
            metric_id="inline_explanation_density", category="D", raw_score=score,
            file_path=None,
            deduction_reason=reason,
            fix_suggestion=(
                "Add markdown cells to notebooks (target: ≥1 per 2 code cells) "
                "and inline comments to scripts (target: ≥20%)."
                if score < 60 else ""
            ),
        )

    @staticmethod
    def execution_entry_point(repo: Path) -> SubMetricEvidence:
        """D6: Execution entry point — unambiguous first command to run."""
        for fname in ["run.sh", "run_all.sh", "start.sh", "execute.sh",
                      "run.py", "main.py"]:
            if (repo / fname).exists():
                return SubMetricEvidence(
                    metric_id="execution_entry_point", category="D", raw_score=100,
                    file_path=fname,
                    deduction_reason=f"Execution entry point: {fname}",
                )

        makefile = repo / "Makefile"
        if makefile.exists():
            content = makefile.read_text(errors="replace")
            if re.search(r'^(?:run|all|execute|analysis|main)\s*:', content, re.M):
                return SubMetricEvidence(
                    metric_id="execution_entry_point", category="D", raw_score=100,
                    file_path="Makefile",
                    deduction_reason="Makefile with run/all/execute target",
                )

        readme = _find_readme(repo)
        if readme:
            content = readme.read_text(errors="replace")
            if re.search(
                r'#+\s*(run|execute|quickstart|quick\s+start|how\s+to\s+run)',
                content, re.I
            ):
                return SubMetricEvidence(
                    metric_id="execution_entry_point", category="D", raw_score=80,
                    file_path=readme.name,
                    deduction_reason="Execution instructions section found in README",
                )

        return SubMetricEvidence(
            metric_id="execution_entry_point", category="D", raw_score=0,
            file_path=None,
            deduction_reason="No clear execution entry point found",
            fix_suggestion=(
                "Add a run.sh, main.py, or '## How to Run' README section "
                "with the first command to execute."
            ),
        )

    @staticmethod
    def reuse_metadata(repo: Path) -> SubMetricEvidence:
        """D7: Reuse metadata — LICENSE + CITATION.cff + codemeta.json.

        Low-weight transparency indicators. Not interpreted as evidence of executability.
        """
        found = []

        license_candidates = [
            "LICENSE", "LICENSE.md", "LICENSE.txt", "LICENSE.rst",
            "LICENCE", "LICENCE.md", "COPYING", "COPYING.md",
        ]
        for fname in license_candidates:
            if (repo / fname).exists():
                found.append("LICENSE")
                break

        if (repo / "CITATION.cff").exists():
            found.append("CITATION.cff")
        if (repo / "codemeta.json").exists():
            found.append("codemeta.json")

        tiers = {3: 100, 2: 70, 1: 40, 0: 0}
        score = tiers[min(len(found), 3)]

        if found:
            return SubMetricEvidence(
                metric_id="reuse_metadata", category="D", raw_score=score,
                file_path=found[0],
                deduction_reason=f"Reuse metadata found: {', '.join(found)}",
            )
        return SubMetricEvidence(
            metric_id="reuse_metadata", category="D", raw_score=0,
            file_path=None,
            deduction_reason="No reuse metadata (LICENSE, CITATION.cff, codemeta.json) found",
            fix_suggestion="Add a LICENSE file and optionally a CITATION.cff for software citation.",
        )

    # =====================================================================
    # Category C: Code Portability
    # =====================================================================

    @staticmethod
    def no_absolute_paths(repo: Path) -> SubMetricEvidence:
        """C1: No user-specific absolute paths — portability failure on another machine."""
        abs_path_re = re.compile(
            r'(?:'
            r'/home/[^/\s"\'\\)>\n]+/'
            r'|/Users/[^/\s"\'\\)>\n]+/'
            r'|C:[/\\]Users[/\\]'
            r'|/root/[^/\s"\'\\)>\n]+'
            r'|/mnt/[a-zA-Z][^/\s"\'\\)>\n]*/'
            r')'
        )
        py_files = [f for f in sorted(repo.glob("**/*.py"))
                    if not is_excluded_notebook(f)][:30]
        r_files = [f for f in sorted(repo.glob("**/*.R"))
                   if not is_excluded_notebook(f)][:15]
        nb_files = [f for f in sorted(repo.glob("**/*.ipynb"))
                    if not is_excluded_notebook(f)][:20]

        all_source = py_files + r_files
        total_files = len(all_source) + len(nb_files)
        if total_files == 0:
            return SubMetricEvidence(
                metric_id="no_absolute_paths", category="C", raw_score=100,
                file_path=None,
                deduction_reason="No source files found",
            )

        files_with_abs = 0
        first_hit: Optional[str] = None
        first_match: Optional[str] = None

        for f in all_source:
            m = abs_path_re.search(f.read_text(errors="replace"))
            if m:
                files_with_abs += 1
                if first_hit is None:
                    first_hit = str(f.relative_to(repo))
                    first_match = m.group(0)

        for nb in nb_files:
            code_lines, _, _ = _nb_cells(nb)
            m = abs_path_re.search("\n".join(code_lines))
            if m:
                files_with_abs += 1
                if first_hit is None:
                    first_hit = str(nb.relative_to(repo))
                    first_match = m.group(0)

        if files_with_abs == 0:
            return SubMetricEvidence(
                metric_id="no_absolute_paths", category="C", raw_score=100,
                file_path=None,
                deduction_reason=f"No user-specific absolute paths in {total_files} source files",
            )

        score = max(0.0, (1.0 - files_with_abs / total_files) * 100)
        return SubMetricEvidence(
            metric_id="no_absolute_paths", category="C", raw_score=score,
            file_path=first_hit,
            deduction_reason=f"Absolute paths in {files_with_abs}/{total_files} files (e.g. {first_match!r})",
            fix_suggestion="Replace hardcoded paths with relative paths or os.environ variables.",
        )

    @staticmethod
    def import_resolvability(repo: Path) -> SubMetricEvidence:
        """C2: Import resolvability — imports cross-referenced against declared deps."""
        import_re = re.compile(
            r'^\s*(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)', re.M
        )
        pkg_name_re = re.compile(
            r'(?:^|[\s\-,\[\(])([a-zA-Z][a-zA-Z0-9_\-\.]{1,})\s*(?:[>=<!,\]\)]|$)',
            re.M
        )

        # Collect declared packages from dependency files
        dep_files = [
            "requirements.txt", "requirements-dev.txt", "requirements/base.txt",
            "setup.py", "setup.cfg", "pyproject.toml", "Pipfile",
            "environment.yml", "environment.yaml",
        ]
        declared: set = set()
        for fname in dep_files:
            fp = repo / fname
            if not fp.exists():
                continue
            content = fp.read_text(errors="replace")
            for m in pkg_name_re.finditer(content):
                name = m.group(1).lower().replace('-', '_').replace('.', '_')
                if len(name) > 1 and not name.isdigit():
                    declared.add(name)

        if not declared:
            return SubMetricEvidence(
                metric_id="import_resolvability", category="C", raw_score=0,
                file_path=None,
                deduction_reason="No dependency specification files found — all third-party imports are implicitly undeclared",
                fix_suggestion="Add requirements.txt or environment.yml to declare dependencies.",
            )

        # Collect imports from .py and notebook code cells
        py_files = [f for f in sorted(repo.glob("**/*.py"))
                    if not is_excluded_notebook(f)][:30]
        nb_files = [f for f in sorted(repo.glob("**/*.ipynb"))
                    if not is_excluded_notebook(f)][:20]

        all_imports: set = set()
        for pf in py_files:
            for m in import_re.finditer(pf.read_text(errors="replace")):
                all_imports.add(m.group(1))
        for nb in nb_files:
            code_lines, _, _ = _nb_cells(nb)
            for m in import_re.finditer("\n".join(code_lines)):
                all_imports.add(m.group(1))

        third_party = {imp for imp in all_imports if imp not in _STDLIB}
        if not third_party:
            return SubMetricEvidence(
                metric_id="import_resolvability", category="C", raw_score=100,
                file_path=None,
                deduction_reason="No third-party imports detected",
            )

        # Cross-reference: import name → PyPI canonical name → declared set
        undeclared = []
        for imp in sorted(third_party):
            pkg = _IMPORT_TO_PKG.get(imp, imp)
            pkg_norm = pkg.lower().replace('-', '_').replace('.', '_')
            imp_norm = imp.lower().replace('-', '_').replace('.', '_')
            if pkg_norm not in declared and imp_norm not in declared:
                undeclared.append(imp)

        score = max(0.0, (1.0 - len(undeclared) / len(third_party)) * 100)
        example = ', '.join(sorted(undeclared)[:5])
        return SubMetricEvidence(
            metric_id="import_resolvability", category="C", raw_score=score,
            file_path=None,
            deduction_reason=(
                f"{len(third_party) - len(undeclared)}/{len(third_party)} imports "
                f"declared in dependency specs"
                + (f"; undeclared: {example}" if undeclared else "")
            ),
            fix_suggestion=(
                f"Add to requirements.txt: {example}" if undeclared else ""
            ),
        )

    @staticmethod
    def no_hardcoded_credentials(repo: Path) -> SubMetricEvidence:
        """C3: No hardcoded credentials — API keys/tokens/passwords in source."""
        cred_re = re.compile(
            r'(?i)(?:api[_-]?key|api[_-]?secret|access[_-]?token|secret[_-]?key'
            r'|password|passwd|auth[_-]?token|private[_-]?key|client[_-]?secret'
            r'|bearer[_-]?token|db[_-]?password)\s*=\s*["\'][^"\']{8,}["\']'
        )
        key_re = re.compile(
            r'(?:AKIA[0-9A-Z]{16}|sk-[a-zA-Z0-9]{32,}|ghp_[a-zA-Z0-9]{36}'
            r'|glpat-[a-zA-Z0-9_\-]{20})'
        )
        _PLACEHOLDERS = ('YOUR_', 'EXAMPLE', 'PLACEHOLDER', 'XXXX', 'TEST_',
                         'CHANGE_ME', 'INSERT_', '<', '>', '...')

        def _has_cred(text: str) -> bool:
            if key_re.search(text):
                return True
            for m in cred_re.finditer(text):
                val = m.group(0).upper()
                if not any(p in val for p in _PLACEHOLDERS):
                    return True
            return False

        py_files = [f for f in sorted(repo.glob("**/*.py"))
                    if not is_excluded_notebook(f)][:30]
        r_files = [f for f in sorted(repo.glob("**/*.R"))
                   if not is_excluded_notebook(f)][:15]
        nb_files = [f for f in sorted(repo.glob("**/*.ipynb"))
                    if not is_excluded_notebook(f)][:20]

        all_source = py_files + r_files
        total_files = len(all_source) + len(nb_files)
        if total_files == 0:
            return SubMetricEvidence(
                metric_id="no_hardcoded_credentials", category="C", raw_score=100,
                file_path=None,
                deduction_reason="No source files found",
            )

        files_with_creds = 0
        first_hit: Optional[str] = None
        for f in all_source:
            if _has_cred(f.read_text(errors="replace")):
                files_with_creds += 1
                if first_hit is None:
                    first_hit = str(f.relative_to(repo))
        for nb in nb_files:
            code_lines, _, _ = _nb_cells(nb)
            if _has_cred("\n".join(code_lines)):
                files_with_creds += 1
                if first_hit is None:
                    first_hit = str(nb.relative_to(repo))

        if files_with_creds == 0:
            return SubMetricEvidence(
                metric_id="no_hardcoded_credentials", category="C", raw_score=100,
                file_path=None,
                deduction_reason=f"No hardcoded credentials in {total_files} source files",
            )
        score = max(0.0, (1.0 - files_with_creds / total_files) * 100)
        return SubMetricEvidence(
            metric_id="no_hardcoded_credentials", category="C", raw_score=score,
            file_path=first_hit,
            deduction_reason=f"Potential hardcoded credentials in {files_with_creds}/{total_files} files",
            fix_suggestion="Replace hardcoded credentials with environment variables (os.environ.get('API_KEY')).",
        )

    @staticmethod
    def silent_failure_masking(repo: Path) -> SubMetricEvidence:
        """C4: Silent failure masking — bare except:pass hides execution errors."""
        bare_except_re = re.compile(
            r'except\s*:\s*\n\s*(?:pass|\.\.\.)\b'
            r'|except\s+Exception\s*(?:as\s+\w+\s*)?:\s*\n\s*(?:pass|\.\.\.)\b',
            re.M
        )
        py_files = [f for f in sorted(repo.glob("**/*.py"))
                    if not is_excluded_notebook(f)][:30]
        nb_files = [f for f in sorted(repo.glob("**/*.ipynb"))
                    if not is_excluded_notebook(f)][:20]

        total_files = len(py_files) + len(nb_files)
        if total_files == 0:
            return SubMetricEvidence(
                metric_id="silent_failure_masking", category="C", raw_score=100,
                file_path=None,
                deduction_reason="No Python source files found",
            )

        files_with_masking = 0
        first_hit: Optional[str] = None
        for f in py_files:
            if bare_except_re.search(f.read_text(errors="replace")):
                files_with_masking += 1
                if first_hit is None:
                    first_hit = str(f.relative_to(repo))
        for nb in nb_files:
            code_lines, _, _ = _nb_cells(nb)
            if bare_except_re.search("\n".join(code_lines)):
                files_with_masking += 1
                if first_hit is None:
                    first_hit = str(nb.relative_to(repo))

        if files_with_masking == 0:
            return SubMetricEvidence(
                metric_id="silent_failure_masking", category="C", raw_score=100,
                file_path=None,
                deduction_reason=f"No bare exception suppression in {total_files} source files",
            )
        score = max(0.0, (1.0 - files_with_masking / total_files) * 100)
        return SubMetricEvidence(
            metric_id="silent_failure_masking", category="C", raw_score=score,
            file_path=first_hit,
            deduction_reason=f"Silent failure masking in {files_with_masking}/{total_files} files",
            fix_suggestion="Replace bare except:pass with specific exception handling that logs or re-raises.",
        )

    # =====================================================================
    # Category S: Reproducibility Signals
    # =====================================================================

    @staticmethod
    def seed_management(repo: Path) -> Tuple[SubMetricEvidence, float]:
        """S1: Deterministic execution controls — σ = |Fseed| / |Frand|."""
        rand_patterns = re.compile(
            r'(np\.random\.|torch\.rand|random\.randint|random\.seed'
            r'|sklearn.*random_state|tf\.random\.|set\.seed\(|sample\()',
            re.I
        )
        seed_patterns = re.compile(
            r'(np\.random\.seed\(|torch\.manual_seed\(|tf\.random\.set_seed\('
            r'|random\.seed\(|set\.seed\(|np\.seed\()',
            re.I
        )
        py_r_files = [
            f for f in sorted(repo.glob("**/*.py")) + sorted(repo.glob("**/*.R"))
            if not is_excluded_notebook(f)
        ][:30]
        nb_files = [f for f in sorted(repo.glob("**/*.ipynb"))
                    if not is_excluded_notebook(f)][:20]

        f_rand: List[str] = []
        f_seed: List[str] = []
        for pf in py_r_files + nb_files:
            if pf.suffix in {".py", ".R"}:
                raw = pf.read_text(errors="replace")
                content = "\n".join(
                    ln for ln in raw.splitlines()
                    if not ln.strip().startswith("#")
                )
            else:
                code_lines, _, _ = _nb_cells(pf)
                content = "\n".join(
                    ln for ln in code_lines
                    if not ln.strip().startswith("#")
                )
            if rand_patterns.search(content):
                rel = str(pf.relative_to(repo))
                f_rand.append(rel)
                if seed_patterns.search(content):
                    f_seed.append(rel)

        sigma = 1.0 if not f_rand else len(f_seed) / len(f_rand)
        score = sigma * 100
        ev = SubMetricEvidence(
            metric_id="seed_management", category="S", raw_score=score,
            file_path=f_seed[0] if f_seed else (f_rand[0] if f_rand else None),
            deduction_reason=(
                f"Deterministic execution controls: {len(f_seed)}/{len(f_rand)} "
                f"stochastic files have seed-setting calls (σ={sigma:.2f})"
            ),
            fix_suggestion=(
                f"Add seed-setting in: {', '.join(f_rand[:3])}."
                if sigma < 1.0 and f_rand else ""
            ),
        )
        return ev, sigma

    @staticmethod
    def notebook_exec_order(repo: Path) -> SubMetricEvidence:
        """S2: Notebook execution order — cells executed in monotonic sequence."""
        nb_files = [f for f in sorted(repo.glob("**/*.ipynb"))
                    if not is_excluded_notebook(f)][:20]
        if not nb_files:
            return SubMetricEvidence(
                metric_id="notebook_exec_order", category="S", raw_score=100,
                file_path=None,
                deduction_reason="No notebooks found — metric not applicable",
            )

        n_total = 0
        n_ordered = 0
        first_disordered: Optional[str] = None

        for nb in nb_files:
            try:
                data = json.loads(nb.read_text(errors="replace"))
            except Exception:
                continue
            exec_counts = [
                cell.get("execution_count")
                for cell in data.get("cells", [])
                if cell.get("cell_type") == "code"
                and cell.get("execution_count") is not None
            ]
            n_total += 1
            if len(exec_counts) < 2:
                n_ordered += 1
                continue
            is_ordered = all(
                exec_counts[i] < exec_counts[i + 1]
                for i in range(len(exec_counts) - 1)
            )
            if is_ordered:
                n_ordered += 1
            elif first_disordered is None:
                first_disordered = str(nb.relative_to(repo))

        if n_total == 0:
            return SubMetricEvidence(
                metric_id="notebook_exec_order", category="S", raw_score=100,
                file_path=None,
                deduction_reason="No executed notebooks found",
            )
        score = (n_ordered / n_total) * 100
        return SubMetricEvidence(
            metric_id="notebook_exec_order", category="S", raw_score=score,
            file_path=first_disordered,
            deduction_reason=f"{n_ordered}/{n_total} notebooks have monotonic execution order",
            fix_suggestion=(
                "Re-run notebooks top-to-bottom (Kernel → Restart & Run All) before committing."
                if score < 100 else ""
            ),
        )

    @staticmethod
    def test_file_presence(repo: Path) -> SubMetricEvidence:
        """S3: Test file presence — min(|T|/2, 1)."""
        test_patterns = [
            "test_*.py", "*_test.py", "tests/**/*.py",
            "test/**/*.py", "testthat.R", "test-*.R",
        ]
        test_files = []
        for pat in test_patterns:
            test_files.extend(repo.glob(f"**/{pat}"))
        test_files = [f for f in test_files if not is_excluded_notebook(f)]
        count = len(set(test_files))
        score = min(count / 2, 1.0) * 100
        return SubMetricEvidence(
            metric_id="test_file_presence", category="S", raw_score=score,
            file_path=str(test_files[0].relative_to(repo)) if test_files else None,
            deduction_reason=f"Found {count} test file(s)",
            fix_suggestion="Add test files (test_*.py) using pytest or unittest." if count == 0 else "",
        )

    @staticmethod
    def expected_outputs(repo: Path) -> SubMetricEvidence:
        """S4: Expected outputs — reference results committed to repo."""
        output_dirs = [
            "results", "outputs", "output", "expected", "reference",
            "expected_outputs", "reference_outputs", "ground_truth", "figures",
        ]
        for dname in output_dirs:
            d = repo / dname
            if d.is_dir():
                files = [f for f in d.iterdir()
                         if f.is_file() and f.name not in {'.gitkeep', '.gitignore'}]
                if files:
                    return SubMetricEvidence(
                        metric_id="expected_outputs", category="S", raw_score=100,
                        file_path=str(d.relative_to(repo)),
                        deduction_reason=f"Reference output directory: {dname}/",
                    )

        readme = _find_readme(repo)
        if readme:
            lower = readme.read_text(errors="replace").lower()
            if re.search(
                r'expected\s+(?:output|result)|sample\s+output|example\s+output'
                r'|reference\s+output', lower
            ):
                return SubMetricEvidence(
                    metric_id="expected_outputs", category="S", raw_score=60,
                    file_path=readme.name,
                    deduction_reason="Expected output section found in README",
                )

        # Committed output files at moderate score
        for ext in ['.png', '.pdf', '.svg']:
            hits = [
                p for p in repo.glob(f"**/*{ext}")
                if not is_excluded_notebook(p)
                and not any(part.startswith('.') for part in p.parts)
            ][:1]
            if hits:
                return SubMetricEvidence(
                    metric_id="expected_outputs", category="S", raw_score=40,
                    file_path=str(hits[0].relative_to(repo)),
                    deduction_reason=f"Output files ({ext}) committed in repo",
                )

        return SubMetricEvidence(
            metric_id="expected_outputs", category="S", raw_score=0,
            file_path=None,
            deduction_reason="No reference output directory or expected results found",
            fix_suggestion="Add a results/ directory with reference outputs to enable verification.",
        )

    @staticmethod
    def ci_presence(repo: Path) -> SubMetricEvidence:
        """S5: CI configuration presence."""
        ci_indicators = [
            ".github/workflows", ".travis.yml", ".circleci",
            ".gitlab-ci.yml", "azure-pipelines.yml", "Jenkinsfile",
        ]
        for ci in ci_indicators:
            if (repo / ci).exists():
                return SubMetricEvidence(
                    metric_id="ci_presence", category="S", raw_score=100,
                    file_path=ci,
                    deduction_reason=f"CI configuration: {ci}",
                )
        return SubMetricEvidence(
            metric_id="ci_presence", category="S", raw_score=0,
            file_path=None,
            deduction_reason="No CI configuration found",
            fix_suggestion="Add a GitHub Actions workflow (.github/workflows/) for CI.",
        )

    @staticmethod
    def config_externalised(repo: Path) -> SubMetricEvidence:
        """S6: Config externalised — experimental parameters in external config files."""
        config_files = [
            "config.yaml", "config.yml", "config.json", "config.toml",
            "params.yaml", "params.yml", "params.json",
            "settings.yaml", "settings.yml", "hyperparams.yaml", "hyperparams.json",
        ]
        for fname in config_files:
            if (repo / fname).exists():
                return SubMetricEvidence(
                    metric_id="config_externalised", category="S", raw_score=100,
                    file_path=fname,
                    deduction_reason=f"External config file: {fname}",
                )

        for config_dir in ["config", "configs", "conf", "params", "settings"]:
            d = repo / config_dir
            if d.is_dir():
                yaml_files = (list(d.glob("*.yaml")) + list(d.glob("*.yml"))
                              + list(d.glob("*.json")))
                if yaml_files:
                    return SubMetricEvidence(
                        metric_id="config_externalised", category="S", raw_score=100,
                        file_path=str(yaml_files[0].relative_to(repo)),
                        deduction_reason=f"Config directory: {config_dir}/",
                    )

        # argparse/click/typer in scripts = CLI-externalised config
        argparse_re = re.compile(r'\bargparse\b|\bArgumentParser\b|\bclick\b|\btyper\b')
        hydra_re = re.compile(r'\bhydra\b|\bomegaconf\b', re.I)
        py_files = [f for f in sorted(repo.glob("**/*.py"))
                    if not is_excluded_notebook(f)][:20]
        for pf in py_files:
            content = pf.read_text(errors="replace")
            if hydra_re.search(content):
                return SubMetricEvidence(
                    metric_id="config_externalised", category="S", raw_score=100,
                    file_path=str(pf.relative_to(repo)),
                    deduction_reason="Hydra/OmegaConf config management detected",
                )
            if argparse_re.search(content):
                return SubMetricEvidence(
                    metric_id="config_externalised", category="S", raw_score=80,
                    file_path=str(pf.relative_to(repo)),
                    deduction_reason="CLI argument parsing (argparse/click/typer) detected",
                )

        return SubMetricEvidence(
            metric_id="config_externalised", category="S", raw_score=0,
            file_path=None,
            deduction_reason="No externalised configuration found",
            fix_suggestion=(
                "Store experimental parameters in config.yaml or use argparse "
                "to enable exact re-run verification."
            ),
        )

    @staticmethod
    def hardware_requirements(repo: Path) -> SubMetricEvidence:
        """S7: Hardware requirements declared when GPU/accelerator dependencies are present."""
        gpu_pkg_re = re.compile(
            r'\b(?:torch|tensorflow|tensorflow.gpu|jax|cupy|rapids|mxnet'
            r'|pytorch.lightning|lightning)\b', re.I
        )
        dep_files = ["requirements.txt", "environment.yml", "environment.yaml",
                     "setup.py", "pyproject.toml"]

        uses_gpu_pkg = False
        for fname in dep_files:
            fp = repo / fname
            if fp.exists():
                content = fp.read_text(errors="replace")
                if gpu_pkg_re.search(content) or re.search(r'cudatoolkit|cuda', content, re.I):
                    uses_gpu_pkg = True
                    break

        if not uses_gpu_pkg:
            return SubMetricEvidence(
                metric_id="hardware_requirements", category="S", raw_score=100,
                file_path=None,
                deduction_reason="No GPU-specific dependencies detected — metric not applicable",
            )

        # Has GPU packages — check if requirements are declared
        cuda_declared_re = re.compile(r'\b(?:cuda|nvidia|gpu|vram|cudatoolkit|cudnn)\b', re.I)
        readme = _find_readme(repo)
        if readme:
            if cuda_declared_re.search(readme.read_text(errors="replace")):
                return SubMetricEvidence(
                    metric_id="hardware_requirements", category="S", raw_score=100,
                    file_path=readme.name,
                    deduction_reason="GPU/CUDA requirements declared in README",
                )

        for fname in ["environment.yml", "environment.yaml"]:
            fp = repo / fname
            if fp.exists():
                content = fp.read_text(errors="replace")
                if re.search(r'cudatoolkit|cuda\s*=', content, re.I):
                    return SubMetricEvidence(
                        metric_id="hardware_requirements", category="S", raw_score=100,
                        file_path=fname,
                        deduction_reason="CUDA version specified in environment spec",
                    )

        return SubMetricEvidence(
            metric_id="hardware_requirements", category="S", raw_score=0,
            file_path=None,
            deduction_reason="GPU dependencies found but no hardware requirements declared",
            fix_suggestion="Declare GPU/CUDA requirements in README (GPU type, CUDA version, memory).",
        )


# ---------------------------------------------------------------------------
# Category aggregators
# ---------------------------------------------------------------------------

def _aggregate_E(sub: Dict[str, SubMetricEvidence]) -> float:
    """Aggregate Environment category.

    Weights reflect reproducibility guarantee strength derived from execution
    failure analysis (Trisovic et al. 2022; Samuel & Mietchen 2024):
      container_spec (0.30): containers provide the strongest guarantee — full
        OS-level isolation; pinned base + RUN commands make the environment
        fully deterministic.
      dep_pinning (0.25): lockfiles or exact version pins are the primary
        mechanism for eliminating version-conflict failures.
      env_bootstrap (0.25): one-command environment creation directly enables
        re-execution by a third party.
      python_version_declared (0.20): runtime version declaration prevents
        implicit Python version mismatches.
    """
    return 100 * (
        0.25 * (sub["dep_pinning"].raw_score / 100) +
        0.30 * (sub["container_spec"].raw_score / 100) +
        0.25 * (sub["env_bootstrap"].raw_score / 100) +
        0.20 * (sub["python_version_declared"].raw_score / 100)
    )


def _aggregate_A(sub: Dict[str, SubMetricEvidence]) -> float:
    """Aggregate Data accessibility category.

    Weights reflect data retrieval reliability (Samuel & Mietchen 2024):
      data_pointer (0.30): archival DOI > institutional URL > platform link;
        permanence directly determines whether input data is retrievable.
      data_acquisition_script (0.30): automated download removes manual steps
        that commonly cause missing-data failures.
      data_description (0.20): documentation clarifies data provenance and
        format, reducing interpretation errors.
      workflow_orchestration (0.20): end-to-end pipeline tools (Snakemake,
        DVC) coordinate data and compute steps reproducibly.
    """
    return 100 * (
        0.20 * (sub["data_description"].raw_score / 100) +
        0.30 * (sub["data_pointer"].raw_score / 100) +
        0.20 * (sub["workflow_orchestration"].raw_score / 100) +
        0.30 * (sub["data_acquisition_script"].raw_score / 100)
    )


def _aggregate_D(sub: Dict[str, SubMetricEvidence]) -> float:
    """Aggregate Documentation category.

    Weights reflect the effort needed to re-execute without author involvement:
      doc_structure (0.25): presence of the four execution-relevant README
        sections (install, run, outputs, requirements) most directly enables
        re-execution by a third party.
      install_instructions (0.20) + usage_examples (0.20): together cover the
        two steps that block re-execution when absent.
      inline_explanation (0.15): notebook md/code ratio and comment density
        reduce reverse-engineering effort.
      execution_entry_point (0.10): a clear first command removes the most
        common barrier for new users.
      docstring_coverage (0.05) + reuse_metadata (0.05): FAIR compliance
        signals; important but not directly execution-blocking.
    """
    return 100 * (
        0.25 * (sub["doc_structure"].raw_score / 100) +
        0.20 * (sub["install_instructions"].raw_score / 100) +
        0.20 * (sub["usage_examples"].raw_score / 100) +
        0.15 * (sub["inline_explanation_density"].raw_score / 100) +
        0.10 * (sub["execution_entry_point"].raw_score / 100) +
        0.05 * (sub["docstring_coverage"].raw_score / 100) +
        0.05 * (sub["reuse_metadata"].raw_score / 100)
    )


def _aggregate_C(sub: Dict[str, SubMetricEvidence]) -> float:
    """Aggregate Code portability category.

    Weights reflect direct causation of cross-machine execution failure
    (Trisovic et al. 2022):
      no_absolute_paths (0.40): machine-specific paths (/home/user/, C:\\Users)
        are the most common portability-breaking defect; highest weight.
      import_resolvability (0.35): undeclared imports directly cause
        ModuleNotFoundError on other machines; score=0 when no dep files exist.
      no_hardcoded_credentials (0.15): credential patterns break execution on
        other machines and pose security risks.
      silent_failure_masking (0.10): bare except:pass hides errors that would
        otherwise surface portability defects earlier.
    """
    return 100 * (
        0.40 * (sub["no_absolute_paths"].raw_score / 100) +
        0.35 * (sub["import_resolvability"].raw_score / 100) +
        0.15 * (sub["no_hardcoded_credentials"].raw_score / 100) +
        0.10 * (sub["silent_failure_masking"].raw_score / 100)
    )


def _aggregate_S(sub: Dict[str, SubMetricEvidence]) -> float:
    """Aggregate Reproducibility signals category.

    Weights reflect the causal chain from execution to identical results
    (Samuel & Mietchen 2024; Pimentel et al. 2019):
      seed_management (0.30): stochastic operations without seeds directly
        prevent result reproduction; highest weight in the category.
      notebook_exec_order (0.20): non-monotonic execution counts indicate
        out-of-order runs that produce state-dependent results.
      test_file_presence (0.18): test suites provide executable verification
        of expected behaviour.
      expected_outputs (0.12): committed reference outputs enable diff-based
        verification.
      ci_presence (0.10): CI configuration enforces re-execution on each
        commit, providing ongoing reproducibility assurance.
      config_externalised (0.06): externalised parameters enable
        parameterised re-runs without source modification.
      hardware_requirements (0.04): GPU/CUDA declarations prevent silent
        hardware-mismatch failures; lowest weight as it is conditional.
    """
    return 100 * (
        0.30 * (sub["seed_management"].raw_score / 100) +
        0.20 * (sub["notebook_exec_order"].raw_score / 100) +
        0.18 * (sub["test_file_presence"].raw_score / 100) +
        0.12 * (sub["expected_outputs"].raw_score / 100) +
        0.10 * (sub["ci_presence"].raw_score / 100) +
        0.06 * (sub["config_externalised"].raw_score / 100) +
        0.04 * (sub["hardware_requirements"].raw_score / 100)
    )


# ---------------------------------------------------------------------------
# RRS Scorer
# ---------------------------------------------------------------------------

class RRSScorer:
    """
    Computes the Reproducibility Readiness Score (RRS) for a local repository.

    Run standalone (no Docker required):
        scorer = RRSScorer()
        result = scorer.score("/path/to/repo")
        print(result.rrs)
    """

    def __init__(self, rubric: Optional[Rubric] = None):
        self.rubric = rubric or load_rubric()
        self._d = _SubMetricDetectors()

    def score(self, repo_path: str | Path) -> RRSResult:
        """Compute RRS for the repository at repo_path. Returns full evidence trail."""
        repo = Path(repo_path)
        if not repo.is_dir():
            raise ValueError(f"Not a directory: {repo}")

        d = self._d

        # --- 26 sub-metrics ---
        sub_E = {
            "dep_pinning":            d.dep_pinning(repo),
            "container_spec":         d.container_spec(repo),
            "env_bootstrap":          d.env_bootstrap(repo),
            "python_version_declared": d.python_version_declared(repo),
        }
        sub_A = {
            "data_description":       d.data_description(repo),
            "data_pointer":           d.data_pointer(repo),
            "workflow_orchestration": d.workflow_orchestration(repo),
            "data_acquisition_script": d.data_acquisition_script(repo),
        }
        sub_D = {
            "doc_structure":              d.doc_structure(repo),
            "install_instructions":       d.install_instructions(repo),
            "usage_examples":             d.usage_examples(repo),
            "docstring_coverage":         d.docstring_coverage(repo),
            "inline_explanation_density": d.inline_explanation_density(repo),
            "execution_entry_point":      d.execution_entry_point(repo),
            "reuse_metadata":             d.reuse_metadata(repo),
        }
        sub_C = {
            "no_absolute_paths":        d.no_absolute_paths(repo),
            "import_resolvability":     d.import_resolvability(repo),
            "no_hardcoded_credentials": d.no_hardcoded_credentials(repo),
            "silent_failure_masking":   d.silent_failure_masking(repo),
        }
        seed_ev, sigma = d.seed_management(repo)
        sub_S = {
            "seed_management":      seed_ev,
            "notebook_exec_order":  d.notebook_exec_order(repo),
            "test_file_presence":   d.test_file_presence(repo),
            "expected_outputs":     d.expected_outputs(repo),
            "ci_presence":          d.ci_presence(repo),
            "config_externalised":  d.config_externalised(repo),
            "hardware_requirements": d.hardware_requirements(repo),
        }

        # --- Category raw scores ---
        raw_E = _aggregate_E(sub_E)
        raw_A = _aggregate_A(sub_A)
        raw_D = _aggregate_D(sub_D)
        raw_C = _aggregate_C(sub_C)
        raw_S = _aggregate_S(sub_S)

        # --- Gate function with rubric parameters ---
        cfg = self.rubric.categories
        gated_E = gate(raw_E, cfg["E"]["tau"], cfg["E"]["k"]) * cfg["E"]["weight"]
        gated_A = gate(raw_A, cfg["A"]["tau"], cfg["A"]["k"]) * cfg["A"]["weight"]
        gated_D = gate(raw_D, cfg["D"]["tau"], cfg["D"]["k"]) * cfg["D"]["weight"]
        gated_C = gate(raw_C, cfg["C"]["tau"], cfg["C"]["k"]) * cfg["C"]["weight"]
        gated_S = gate(raw_S, cfg["S"]["tau"], cfg["S"]["k"]) * cfg["S"]["weight"]

        base = 100.0 * (gated_E + gated_A + gated_D + gated_C + gated_S)

        # --- Hard penalties ---
        p = self.rubric.penalties
        penalty_E = p["environment_hard_penalty"] if raw_E < p["environment_hard_threshold"] else 0.0
        penalty_A = p["data_hard_penalty"] if raw_A < p["data_hard_threshold"] else 0.0
        penalty_seed = p["seed_penalty"] if (sigma * 100) < p["seed_threshold"] else 0.0

        rrs = max(0.0, min(100.0, base - penalty_E - penalty_A - penalty_seed))

        all_evidence = (list(sub_E.values()) + list(sub_A.values()) +
                        list(sub_D.values()) + list(sub_C.values()) +
                        list(sub_S.values()))

        category_scores = {
            "E": CategoryScore("Environment Specification", "E", raw_E,
                               gated_E * 100, sub_E),
            "A": CategoryScore("Data Accessibility", "A", raw_A,
                               gated_A * 100, sub_A),
            "D": CategoryScore("Documentation", "D", raw_D,
                               gated_D * 100, sub_D),
            "C": CategoryScore("Code Portability", "C", raw_C,
                               gated_C * 100, sub_C),
            "S": CategoryScore("Reproducibility Signals", "S", raw_S,
                               gated_S * 100, sub_S),
        }

        return RRSResult(
            rrs=round(rrs, 2),
            category_scores=category_scores,
            evidence=all_evidence,
            penalty_environment=penalty_E,
            penalty_data=penalty_A,
            penalty_seed=penalty_seed,
            rubric_name=self.rubric.name,
        )
