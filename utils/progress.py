"""Small console progress reporter with elapsed time and ETA."""

import sys
import time


def format_duration(seconds):
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return "{}h {:02d}m".format(hours, minutes)
    if minutes:
        return "{}m {:02d}s".format(minutes, seconds)
    return "{}s".format(seconds)


def format_bytes(value):
    value = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return "{:.1f} {}".format(value, unit)
        value /= 1024


class ProgressReporter:
    """Render throttled progress updates without owning the work being tracked."""

    def __init__(self, label, total=None, unit="steps", interval=1.0):
        self.label = label
        self.total = total if total and total > 0 else None
        self.unit = unit
        self.interval = interval
        self.started_at = time.monotonic()
        self.last_update = 0.0
        self.displayed = False
        self.current = 0

    def update(self, current, detail=None, force=False):
        self.current = current
        now = time.monotonic()
        if not force and now - self.last_update < self.interval:
            return

        elapsed = now - self.started_at
        rate = current / elapsed if elapsed > 0 else 0.0
        eta = (self.total - current) / rate if self.total and rate > 0 else None

        if self.unit == "bytes":
            amount = format_bytes(current)
            if self.total:
                percent = min(100.0, current * 100.0 / self.total)
                amount = "{} / {} ({:.1f}%)".format(
                    amount, format_bytes(self.total), percent)
            rate_text = "{} /s".format(format_bytes(rate))
        else:
            amount = "{} {}".format(current, self.unit)
            if self.total:
                percent = min(100.0, current * 100.0 / self.total)
                amount = "{} / {} {} ({:.1f}%)".format(
                    current, self.total, self.unit, percent)
            rate_text = "{:.2f} {}/s".format(rate, self.unit)

        message = "{}: {} | {} elapsed | ETA {} | {}".format(
            self.label, amount, format_duration(elapsed),
            format_duration(eta), rate_text)
        if detail:
            message += " | {}".format(detail)
        sys.stdout.write("\r" + message)
        sys.stdout.flush()
        self.last_update = now
        self.displayed = True

    def newline(self):
        if self.displayed:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.displayed = False

    def finish(self, detail=None):
        if self.total is not None:
            self.update(self.total, detail=detail, force=True)
        self.newline()
