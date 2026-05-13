"""
ablation_batch.py
=================
ReproScore — Clone a stratified sample of repositories, run static RRS
scoring, and write a self-contained provenance-rich run directory.

Author: Sheeba Samuel <sheeba.samuel@informatik.tu-chemnitz.de>

Run directory layout
--------------------
data/ablation/<YYYYMMDD_HHMMSS>/
    provenance.json          — run-level metadata (DB hash, git commit, params)
    scores.csv               — one row per repo; all 26 sub-metrics + ground truth
    logs/
        clone.log            — per-repo clone status
        score.log            — per-repo scoring status
        errors.json          — repos that failed clone or score (if any)
    repos/
        <owner__slug>.json   — per-repo clone + score result with timestamps

Usage
-----
    python ablation_batch.py [--limit N] [--workers W] [--out-parent DIR]
    python ablation_batch.py --skip-clone   # re-score already-cloned repos
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from src.scoring.rrs import RRSScorer

# ── constants ─────────────────────────────────────────────────────────────────

GIGA_DB   = ROOT / "data" / "db.sqlite" # Download from https://doi.org/10.5281/zenodo.8226725
CLONE_DIR = ROOT / "data" / "ablation_repos"

GITHUB_BASE = "https://github.com"

# ── failure-mode classification ───────────────────────────────────────────────

def _classify_reason(reason: str | None) -> str:
    if reason is None:
        return "success"
    if "Install" in reason:
        return "install_dep"
    if "ModuleNotFound" in reason or "ImportError" in reason:
        return "missing_module"
    if "FileNotFound" in reason or "IOError" in reason:
        return "missing_data"
    # Network / external-service failures: resource not reachable at execution time
    if any(e in reason for e in (
        "HTTPError", "ConnectionError", "NoValidConnectionsError",
        "ConnectionRefusedError",
    )):
        return "missing_data"
    # Data-format failures: file exists but is unreadable / malformed
    if any(e in reason for e in (
        "LZMAError", "XLRDError", "JSONDecodeError", "UnicodeDecodeError",
        "UnicodeError",
    )):
        return "missing_data"
    if "Skipping" in reason:
        return "skipped"
    if any(e in reason for e in (
        "TypeError", "NameError", "SyntaxError", "AttributeError",
        "RuntimeError", "Exception", "AssertionError", "ValueError",
        "KeyError", "IndexError", "CalledProcessError",
        "OSError", "UsageError", "StdinNotImplementedError",
        "IndentationError", "ZeroDivisionError", "TclError",
        "SystemError", "OptionError", "ExecutableNotFound",
        "Unknown exception",
        "error",   # catches bare 'error' and 'error: OpenCV...' etc.
    )):
        return "code_error"
    return "other"


# ── sample builder ────────────────────────────────────────────────────────────

def build_sample(conn: sqlite3.Connection, limit: int,
                 date_cutoff: str | None = "2021-01-01") -> list[dict]:
    """
    Build a perfectly balanced stratified sample of GigaScience repos.

    Filters applied:
      - Python notebooks only: at least one notebook with language='python'
        or a Python-family kernel (python*, conda*, py*, 'Python [Root]').
      - Article published on or after date_cutoff (ISO date string).
        Pass None or "" to disable the date filter.
      - Has at least one execution record (mode 3 or 5).

    Guarantees exactly equal class sizes:
      per_class = min(available_per_mode_minimum, limit // 5)
    so the returned sample has exactly per_class × 5 rows.
    """
    date_clause = (
        f"AND a.published_date >= '{date_cutoff}'"
        if date_cutoff else ""
    )

    rows = conn.execute(f"""
        WITH python_repos AS (
            -- Repos that have at least one Python notebook.
            -- Covers language='python', Python 2/3 kernels, named conda envs.
            SELECT DISTINCT repository_id
            FROM notebooks
            WHERE language = 'python'
               OR kernel LIKE 'python%'
               OR kernel LIKE 'conda%'
               OR kernel LIKE 'py%'
               OR kernel = 'Python [Root]'
        ),
        exec_agg AS (
            SELECT
                repository_id,
                COUNT(*)                                         AS total_exec_count,
                SUM(CASE WHEN reason IS NULL THEN 1 ELSE 0 END) AS success_nb_count
            FROM executions
            WHERE mode IN (3, 5)
            GROUP BY repository_id
        ),
        dom_reason AS (
            -- Pick the dominant FAILURE reason across all notebooks.
            -- NULL (success) gets priority 99 so it only wins when every
            -- notebook succeeded.
            SELECT
                repository_id,
                reason,
                ROW_NUMBER() OVER (
                    PARTITION BY repository_id
                    ORDER BY CASE
                        WHEN reason LIKE '%Install%'           THEN 1
                        WHEN reason LIKE '%ModuleNotFound%'
                          OR reason LIKE '%ImportError%'       THEN 2
                        WHEN reason LIKE '%FileNotFound%'
                          OR reason LIKE '%IOError%'           THEN 3
                        WHEN reason LIKE '%Skipping%'          THEN 4
                        WHEN reason IS NOT NULL                THEN 5
                        ELSE 99
                    END
                ) AS rn
            FROM executions
            WHERE mode IN (3, 5)
        )
        SELECT
            r.id,
            r.domain,
            r.repository,
            r.notebooks_count,
            r.requirements_count,
            r.setups_count,
            r."commit"                AS giga_commit,
            ea.total_exec_count,
            ea.success_nb_count,
            dr.reason                 AS dominant_reason,
            a.published_date          AS article_published_date
        FROM repositories r
        JOIN exec_agg   ea ON ea.repository_id = r.id
        JOIN dom_reason dr ON dr.repository_id = r.id AND dr.rn = 1
        JOIN python_repos pr ON pr.repository_id = r.id
        JOIN article     a  ON a.id = r.article_id
        WHERE r.notebooks_count > 0
          AND r.domain = 'github.com'
          {date_clause}
        ORDER BY r.id
    """).fetchall()

    cols = ["repo_id", "domain", "repo", "nb_count", "req_count",
            "setup_count", "giga_commit", "total_exec_count",
            "success_nb_count", "dominant_reason", "article_published_date"]
    records = [dict(zip(cols, r)) for r in rows]

    for rec in records:
        if rec["success_nb_count"] == rec["total_exec_count"]:
            rec["failure_mode"] = "success"
        else:
            rec["failure_mode"] = _classify_reason(rec["dominant_reason"])

    usable_modes = {"success", "install_dep", "missing_module", "missing_data", "code_error"}
    records = [r for r in records if r["failure_mode"] in usable_modes]

    # Compute available per mode and guarantee equal class sizes.
    available = {m: sum(1 for r in records if r["failure_mode"] == m)
                 for m in usable_modes}
    requested_per_class = max(limit // len(usable_modes), 10)
    per_class = min(requested_per_class, min(available.values()))

    seen, sample = set(), []
    for mode in sorted(usable_modes):
        count = 0
        for rec in records:
            if rec["failure_mode"] == mode and rec["repo"] not in seen:
                sample.append(rec)
                seen.add(rec["repo"])
                count += 1
                if count >= per_class:
                    break

    return sample, available, per_class


# ── provenance helpers ────────────────────────────────────────────────────────

def _db_fingerprint(db_path: Path) -> dict:
    """MD5 of first 4 MB + file size — fast enough for a 1.4 GB file."""
    h = hashlib.md5()
    with open(db_path, "rb") as f:
        h.update(f.read(4 * 1024 * 1024))
    stat = db_path.stat()
    return {
        "path": str(db_path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "md5_first4mb": h.hexdigest(),
    }


def _git_head(repo_root: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=repo_root,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def build_provenance(run_dir: Path, args: argparse.Namespace,
                     sample: list[dict], available: dict, per_class: int) -> dict:
    return {
        "schema": "reproscore-ablation-provenance/1.0",
        "run_id": run_dir.name,
        "started_utc": datetime.now(tz=timezone.utc).isoformat(),
        "finished_utc": None,
        "parameters": {
            "limit": args.limit,
            "workers": args.workers,
            "skip_clone": args.skip_clone,
            "python_only": True,
            "date_cutoff": args.date_cutoff or None,
            "per_class_selected": per_class,
            "available_per_mode": available,
        },
        "software": {
            "reproscore_git_commit": _git_head(ROOT),
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        "ground_truth_db": _db_fingerprint(GIGA_DB),
        "sample_size": len(sample),
        "failure_mode_counts": dict(Counter(r["failure_mode"] for r in sample)),
        "output_files": {
            "scores_csv":  "scores.csv",
            "clone_log":   "logs/clone.log",
            "score_log":   "logs/score.log",
            "per_repo_dir": "repos/",
        },
        "zenodo_note": (
            "This directory is intended for Zenodo archival. "
            "scores.csv is the primary dataset; repos/ contains per-repository "
            "provenance records with individual timestamps; logs/ contains "
            "full execution traces."
        ),
    }


# ── clone ─────────────────────────────────────────────────────────────────────

def clone_repo(rec: dict, dest: Path, timeout: int = 120) -> tuple[bool, str, str]:
    """Shallow-clone. Returns (ok, cloned_commit_sha, error_msg)."""
    url = f"{GITHUB_BASE}/{rec['repo']}"
    if dest.exists():
        return True, _get_cloned_commit(dest), ""
    try:
        r = subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode == 0:
            return True, _get_cloned_commit(dest), ""
        return False, "", r.stderr.strip()[:200]
    except subprocess.TimeoutExpired:
        shutil.rmtree(dest, ignore_errors=True)
        return False, "", "timeout"
    except Exception as exc:
        return False, "", str(exc)[:200]


def _get_cloned_commit(dest: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ── score ─────────────────────────────────────────────────────────────────────

def score_repo(repo_path: Path) -> dict:
    """Run RRSScorer; return flat dict of all sub-metric + category scores."""
    scorer = RRSScorer()
    result = scorer.score(repo_path)

    row: dict = {
        "rrs":       round(result.rrs, 2),
        "penalty_E": round(result.penalty_environment, 2),
        "penalty_A": round(result.penalty_data, 2),
        "penalty_seed": round(result.penalty_seed, 2),
    }
    for sym, cat in result.category_scores.items():
        row[f"cat_{sym}_raw"]   = round(cat.raw_score, 2)
        row[f"cat_{sym}_gated"] = round(cat.gated_score, 2)
    for ev in result.evidence:
        row[f"sub_{ev.metric_id}"] = round(ev.raw_score, 2)
    return row


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ablation batch scorer")
    parser.add_argument("--limit",      type=int, default=500,
                        help="Max repos to process (default 500)")
    parser.add_argument("--workers",    type=int, default=8,
                        help="Parallel clone workers (default 8)")
    parser.add_argument("--out-parent", default=str(ROOT / "data" / "ablation"),
                        help="Parent dir for timestamped run output")
    parser.add_argument("--skip-clone", action="store_true",
                        help="Skip cloning; only score already-cloned repos")
    parser.add_argument("--date-cutoff", default="2021-01-01",
                        help="Exclude articles published before this ISO date "
                             "(default 2021-01-01). Pass '' to disable.")
    args = parser.parse_args()
    args.date_cutoff = args.date_cutoff.strip() or None

    # ── run directory (timestamped) ───────────────────────────────────────────
    run_ts  = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_parent) / run_ts
    log_dir  = run_dir / "logs"
    repo_dir = run_dir / "repos"
    for d in (run_dir, log_dir, repo_dir, CLONE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ── logging ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    root_logger = logging.getLogger()

    def _add_file_handler(log_path: Path, logger_name: str) -> logging.Logger:
        lg = logging.getLogger(logger_name)
        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lg.addHandler(fh)
        return lg

    logger       = logging.getLogger("ablation")
    clone_logger = _add_file_handler(log_dir / "clone.log", "ablation.clone")
    score_logger = _add_file_handler(log_dir / "score.log", "ablation.score")

    # ── sample ────────────────────────────────────────────────────────────────
    logger.info(f"Run directory : {run_dir}")
    logger.info(f"Database      : {GIGA_DB}  ({GIGA_DB.stat().st_size:,} bytes)")
    logger.info(f"Filters       : python_only=True  date_cutoff={args.date_cutoff!r}")
    logger.info(f"Building sample (limit={args.limit}) …")

    conn = sqlite3.connect(str(GIGA_DB))
    sample, available, per_class = build_sample(conn, args.limit, args.date_cutoff)
    conn.close()

    logger.info(f"Available per mode (after filters):")
    for m, n in sorted(available.items()):
        logger.info(f"  {m:<20} {n:4d}")
    logger.info(f"Per-class selected : {per_class}  (total {per_class * 5})")
    logger.info(f"Sample: {len(sample)} repos")
    for rec in sample:
        logger.info(f"  {rec['repo']:<55}  mode={rec['failure_mode']}")

    # ── provenance stub (updated at end) ─────────────────────────────────────
    prov = build_provenance(run_dir, args, sample, available, per_class)
    prov_path = run_dir / "provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2))

    # ── clone ─────────────────────────────────────────────────────────────────
    # slug → (ok, commit, clone_ts_start, clone_ts_end, error)
    clone_results: dict[str, tuple] = {}

    if not args.skip_clone:
        logger.info("Cloning repos …")

        def _do_clone(rec):
            slug = rec["repo"]
            dest = CLONE_DIR / slug.replace("/", "__")
            ts_start = datetime.now(tz=timezone.utc).isoformat()
            ok, commit, err = clone_repo(rec, dest)
            ts_end = datetime.now(tz=timezone.utc).isoformat()
            clone_logger.info(
                f"{'OK  ' if ok else 'FAIL'} {slug}  "
                f"commit={commit or 'n/a'}  err={err or 'none'}"
            )
            return slug, ok, commit, ts_start, ts_end, err

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_do_clone, rec): rec for rec in sample}
            for fut in as_completed(futs):
                slug, ok, commit, ts_s, ts_e, err = fut.result()
                clone_results[slug] = (ok, commit, ts_s, ts_e, err)
                logger.info(f"  {'✓' if ok else '✗'} {slug}  {commit or err}")
    else:
        logger.info("Skipping clone (--skip-clone)")
        for rec in sample:
            dest = CLONE_DIR / rec["repo"].replace("/", "__")
            if dest.exists():
                clone_results[rec["repo"]] = (True, _get_cloned_commit(dest), None, None, "")
            else:
                clone_results[rec["repo"]] = (False, "", None, None, "not found")

    # ── score ─────────────────────────────────────────────────────────────────
    logger.info("Scoring repos …")
    rows, errors = [], []

    for i, rec in enumerate(sample, 1):
        slug = rec["repo"]
        dest = CLONE_DIR / slug.replace("/", "__")
        ok, commit, clone_ts_s, clone_ts_e, clone_err = clone_results.get(
            slug, (False, "", None, None, "missing from clone results")
        )

        if not ok or not dest.exists():
            score_logger.warning(f"SKIP {slug}: {clone_err}")
            errors.append({"repo": slug, "stage": "clone", "error": clone_err})
            continue

        score_ts_s = datetime.now(tz=timezone.utc).isoformat()
        logger.info(f"  [{i}/{len(sample)}] scoring {slug} …")
        t0 = time.monotonic()
        try:
            score_data = score_repo(dest)
        except Exception as exc:
            score_logger.warning(f"SCORE_ERROR {slug}: {exc}")
            errors.append({"repo": slug, "stage": "score", "error": str(exc)})
            continue
        elapsed = round(time.monotonic() - t0, 2)
        score_ts_e = datetime.now(tz=timezone.utc).isoformat()

        rrs_val = score_data.get("rrs", float("nan"))
        score_logger.info(
            f"OK  {slug}  rrs={rrs_val:.1f}  "
            f"mode={rec['failure_mode']}  elapsed={elapsed}s"
        )

        # ── per-repo provenance JSON ──────────────────────────────────────
        repo_prov = {
            "schema": "reproscore-repo-result/1.0",
            "repo": slug,
            "domain": rec["domain"],
            "clone_url": f"https://{rec['domain']}/{slug}",
            "cloned_commit": commit,
            "clone_timestamp_start_utc": clone_ts_s,
            "clone_timestamp_end_utc":   clone_ts_e,
            "score_timestamp_start_utc": score_ts_s,
            "score_timestamp_end_utc":   score_ts_e,
            "score_elapsed_seconds": elapsed,
            "ground_truth": {
                "source": "GigaScience execution database",
                "db_run_id": prov["run_id"],
                "giga_repo_id":            rec["repo_id"],
                "giga_repo_commit":        rec.get("giga_commit"),
                "article_published_date":  rec.get("article_published_date"),
                "dominant_reason":         rec["dominant_reason"],
                "failure_mode":            rec["failure_mode"],
                "total_exec_count":        rec["total_exec_count"],
                "success_nb_count":        rec["success_nb_count"],
                "nb_count":                rec["nb_count"],
            },
            "scores": score_data,
        }
        slug_safe = slug.replace("/", "__")
        (repo_dir / f"{slug_safe}.json").write_text(json.dumps(repo_prov, indent=2))

        # ── flat CSV row ──────────────────────────────────────────────────
        row = {
            "repo":                    slug,
            "giga_repo_id":            rec["repo_id"],
            "clone_url":               f"https://{rec['domain']}/{slug}",
            "cloned_commit":           commit,
            "score_timestamp_utc":     score_ts_s,
            "article_published_date":  rec.get("article_published_date"),
            "nb_count":                rec["nb_count"],
            "dominant_reason":         rec["dominant_reason"],
            "failure_mode":            rec["failure_mode"],
            "success_nb_count":        rec["success_nb_count"],
            "total_exec_count":        rec["total_exec_count"],
            "score_seconds":           elapsed,
        }
        row.update(score_data)
        rows.append(row)
        logger.info(f"    RRS={rrs_val:.1f}  mode={rec['failure_mode']}  ({elapsed}s)")

    # ── write scores.csv ──────────────────────────────────────────────────────
    scores_path = run_dir / "scores.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(scores_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        logger.info(f"\nWrote {len(rows)} rows → {scores_path}")
    else:
        logger.error("No repos scored — scores.csv not written.")

    # ── write errors ──────────────────────────────────────────────────────────
    if errors:
        err_path = log_dir / "errors.json"
        err_path.write_text(json.dumps(errors, indent=2))
        logger.warning(f"Errors ({len(errors)}) → {err_path}")

    # ── finalise provenance ───────────────────────────────────────────────────
    prov["finished_utc"]        = datetime.now(tz=timezone.utc).isoformat()
    prov["scored_count"]        = len(rows)
    prov["error_count"]         = len(errors)
    prov["failure_mode_scored"] = dict(Counter(r["failure_mode"] for r in rows))
    prov_path.write_text(json.dumps(prov, indent=2))
    logger.info(f"Provenance → {prov_path}")

    # ── summary ───────────────────────────────────────────────────────────────
    logger.info("\nFailure mode breakdown (scored):")
    for mode, cnt in Counter(r["failure_mode"] for r in rows).most_common():
        logger.info(f"  {mode:<20} {cnt}")
    logger.info(f"\nRun complete: {run_dir}")


if __name__ == "__main__":
    main()
