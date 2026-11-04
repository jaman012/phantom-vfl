"""
Run log: a copy of the screen output of a run in two text files.

Terminal keeps only the newest lines of a long run, so a warning can scroll
away before a user reads it. run_log() copies every line that the run prints
(normal output and error output) to a log file, and every warning line and
error line to a warnings file:

  <data folder>/logs/<name>_<date>-<time>.log            all screen output
  <data folder>/logs/<name>_<date>-<time>_warnings.txt   warnings and errors only

Rules:
- A warning line starts with WARNING_PREFIX ("[!]"); spaces before it are
  allowed. No other line of the framework uses this prefix.
- A Python warning (for example a pandas FutureWarning) is printed as a
  warning line.
- An error that stops the run is printed with ERROR_PREFIX ("[ERROR]"), so it
  goes to both files; its traceback goes to the log file only. The error then
  stops the run as before.
- At the end, the run prints the number of warnings and errors and the paths
  of both files.
- Each run log gets new file names; an old log is never overwritten.
- The warnings file writes the context of the run after the prefix, for
  example "[!] (seed 42, k20) Client_A: ...". A stage sets the context with
  set_context(); the log file keeps the lines as printed.

The runner opens one run log for the whole run. A stage script opens its own
run log only when it runs alone from its command line. A run log that opens
while another run log is open does nothing, so a run never makes two logs.
If the data folder does not exist, the run has no run log; the error that
follows shows on the screen.
"""

import datetime
import sys
import traceback
import warnings
from contextlib import contextmanager
from pathlib import Path

LOGS_FOLDER = "logs"
WARNING_PREFIX = "[!]"
ERROR_PREFIX = "[ERROR]"

_active = None  # the open run log of this process, or None


class RunLogFiles:
    """The two open files of one run log, and the counts of warning and error lines."""

    def __init__(self, log_path, warnings_path):
        self.log_path = Path(log_path)
        self.warnings_path = Path(warnings_path)
        self.log = open(self.log_path, "w", encoding="utf-8")
        self.warnings = open(self.warnings_path, "w", encoding="utf-8")
        self.warning_count = 0
        self.error_count = 0
        self.context = ""

    def check_line(self, text):
        """
        Copy one complete screen line to the warnings file if it is a warning or
        an error line, with the context after the prefix.
        """
        stripped = text.lstrip()
        if stripped.startswith(WARNING_PREFIX):
            prefix = WARNING_PREFIX
            self.warning_count += 1
        elif stripped.startswith(ERROR_PREFIX):
            prefix = ERROR_PREFIX
            self.error_count += 1
        else:
            return
        rest = stripped[len(prefix):].lstrip()
        context = f" ({self.context})" if self.context else ""
        self.warnings.write(f"{prefix}{context} {rest}\n")
        self.warnings.flush()

    def close(self):
        self.log.close()
        self.warnings.close()


class _Tee:
    """A text stream that writes to a screen stream and to the log file, and checks each complete line."""

    def __init__(self, stream, files):
        self._stream = stream
        self._files = files
        self._pending = ""

    def write(self, text):
        self._stream.write(text)
        self._files.log.write(text)
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._files.check_line(line)
        return len(text)

    def flush(self):
        self._stream.flush()
        self._files.log.flush()

    def finish(self):
        """Check the last line if it has no line end."""
        if self._pending:
            self._files.check_line(self._pending)
            self._pending = ""

    def __getattr__(self, name):
        return getattr(self._stream, name)


def set_context(text):
    """
    Set the context of the open run log (for example 'seed 42, k20'). The
    warnings file writes it in each warning and error line. Without an open
    run log this does nothing.
    """
    if _active is not None:
        _active.context = str(text)


def _show_python_warning(message, category, filename, lineno, file=None, line=None):
    """Print a Python warning as a warning line (replaces warnings.showwarning while a run log is open)."""
    print(f"{WARNING_PREFIX} Python warning: {category.__name__}: {message} ({filename}, line {lineno})")


def _new_paths(logs_dir, name):
    """New log and warnings paths; a number is added if the names exist."""
    base = f"{name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    candidate, number = base, 1
    while (logs_dir / f"{candidate}.log").exists() or (logs_dir / f"{candidate}_warnings.txt").exists():
        number += 1
        candidate = f"{base}_{number}"
    return logs_dir / f"{candidate}.log", logs_dir / f"{candidate}_warnings.txt"


def _error_text(error):
    if isinstance(error, KeyboardInterrupt):
        return "stopped by the user (Ctrl-C)"
    if isinstance(error, SystemExit):
        return f"stopped (exit {error.code})"
    return f"{type(error).__name__}: {error}"


@contextmanager
def run_log(data_dir, name):
    """
    Copy the screen output of the code in the with-block to a log file and a
    warnings file in <data_dir>/logs/.

    Args:
        data_dir: data folder of the run (it must exist)
        name: first part of the file names, for example 'run' or 'stage1'

    Yields:
        the RunLogFiles of the open run log, or None if no run log opened
        (another run log is open, or the data folder does not exist)
    """
    global _active
    if _active is not None:
        yield None
        return
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        print(f"No run log: the data folder {data_dir} does not exist.")
        yield None
        return
    logs_dir = data_dir / LOGS_FOLDER
    logs_dir.mkdir(exist_ok=True)
    files = RunLogFiles(*_new_paths(logs_dir, name))
    screen_out, screen_err, show_warning = sys.stdout, sys.stderr, warnings.showwarning
    tee_out, tee_err = _Tee(screen_out, files), _Tee(screen_err, files)
    sys.stdout, sys.stderr = tee_out, tee_err
    warnings.showwarning = _show_python_warning
    _active = files
    print(f"Run log: {files.log_path}")
    try:
        yield files
    except BaseException as error:
        if not (isinstance(error, SystemExit) and error.code in (None, 0)):
            for line in _error_text(error).splitlines() or [""]:
                print(f"{ERROR_PREFIX} {line}")
            sys.stdout.flush()
            files.log.write(traceback.format_exc())
        raise
    finally:
        tee_out.finish()
        tee_err.finish()
        print(f"Warnings: {files.warning_count}; errors: {files.error_count}. "
              f"Warnings file: {files.warnings_path}")
        print(f"Full log: {files.log_path}")
        tee_out.flush()
        tee_err.flush()
        sys.stdout, sys.stderr = screen_out, screen_err
        warnings.showwarning = show_warning
        _active = None
        files.close()
