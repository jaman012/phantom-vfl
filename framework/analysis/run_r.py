"""
Run Track B (framework/analysis/track_b.R) with Rscript.

  check_r()      find the Rscript command and read the versions of R and of the
                 R package emmeans; raise RError when one of them is missing
  run_track_b()  run track_b.R on a long table, copy its screen output to the
                 screen (and so to the run log), check its exit code and its
                 output files, and return the omnibus row and the pairwise table

Rscript runs with the options --vanilla (no user profile, no saved workspace)
and a time limit.
"""

import shutil
import subprocess
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parent / "track_b.R"
OMNIBUS_FILE = "track_b_omnibus.csv"
PAIRWISE_FILE = "track_b_pairwise.csv"
SESSION_FILE = "r_session.txt"
TIME_LIMIT = 600  # seconds
VERSION_CODE = ('cat(paste(R.version$major, R.version$minor, sep = "."), "\\n", sep = ""); '
                'if (requireNamespace("emmeans", quietly = TRUE)) '
                'cat(as.character(packageVersion("emmeans")), "\\n", sep = "") else cat("missing\\n")')


class RError(ValueError):
    """Rscript, the R package emmeans or a Track B run failed."""


def check_r(command):
    """Return {"rscript": path, "r": version, "emmeans": version}. Raise RError when R or emmeans is missing."""
    path = shutil.which(command)
    if path is None:
        raise RError(f"Track B needs R: the command '{command}' was not found. Install R, set analysis.rscript "
                     f"to the Rscript command, or set analysis.track_b to false (Track A only).")
    try:
        result = subprocess.run([path, "--vanilla", "-e", VERSION_CODE], capture_output=True, text=True,
                                timeout=120, check=False)
    except subprocess.TimeoutExpired as error:
        raise RError(f"'{path}' did not answer within 120 s") from error
    lines = result.stdout.split()
    if result.returncode != 0 or len(lines) != 2:
        raise RError(f"'{path}' failed (exit code {result.returncode}): {result.stderr.strip()[-500:]}")
    if lines[1] == "missing":
        raise RError("Track B needs the R package emmeans. Install it in R with install.packages(\"emmeans\"), "
                     "or set analysis.track_b to false (Track A only).")
    return {"rscript": path, "r": lines[0], "emmeans": lines[1]}


def run_track_b(rscript, long_csv, output_dir, alpha, configs):
    """Run track_b.R. Return (omnibus row as a dict, pairwise table). Raise RError on a fault."""
    output_dir = Path(output_dir)
    for name in (OMNIBUS_FILE, PAIRWISE_FILE, SESSION_FILE):
        if (output_dir / name).exists():
            (output_dir / name).unlink()
    command = [rscript, "--vanilla", str(SCRIPT), str(long_csv), str(output_dir), repr(float(alpha)),
               ",".join(configs)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=TIME_LIMIT, check=False)
    except subprocess.TimeoutExpired as error:
        raise RError(f"track_b.R did not finish within {TIME_LIMIT} s") from error
    for line in (result.stdout + result.stderr).splitlines():
        if line.strip():
            print(f"  R: {line}")
    if result.returncode != 0:
        raise RError(f"track_b.R failed with exit code {result.returncode} (see the R lines above)")
    missing = [name for name in (OMNIBUS_FILE, PAIRWISE_FILE, SESSION_FILE) if not (output_dir / name).exists()]
    if missing:
        raise RError(f"track_b.R did not write {missing}")
    omnibus = pd.read_csv(output_dir / OMNIBUS_FILE, keep_default_na=False, na_values=["NA"])
    pairwise = pd.read_csv(output_dir / PAIRWISE_FILE)
    m = len(configs)
    if len(omnibus) != 1 or len(pairwise) != m * (m - 1) // 2:
        raise RError(f"track_b.R wrote {len(omnibus)} omnibus row(s) and {len(pairwise)} pairs "
                     f"(want 1 and {m * (m - 1) // 2})")
    return omnibus.iloc[0].to_dict(), pairwise
