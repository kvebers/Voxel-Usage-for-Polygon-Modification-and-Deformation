import csv
import time
from contextlib import contextmanager, nullcontext


class Profiler:
    def __init__(self):
        self._timings: dict[str, list] = {}
        self._counts: dict[str, list] = {}
        self._timing_order: list[str] = []
        self._count_order: list[str] = []

    @contextmanager
    def section(self, name: str):
        self._ensure_timing(name)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._timings[name].append(time.perf_counter() - t0)

    def record(self, name: str, dt_seconds: float):
        self._ensure_timing(name)
        self._timings[name].append(dt_seconds)

    def count(self, name: str, n: int):
        if name not in self._counts:
            self._counts[name] = []
            self._count_order.append(name)
        self._counts[name].append(int(n))

    def _ensure_timing(self, name: str):
        if name not in self._timings:
            self._timings[name] = []
            self._timing_order.append(name)

    def save_csv(self, path: str = "profile_frames.csv"):
        if not self._timings and not self._counts:
            print("Profiler: no data collected.")
            return

        n_frames = max(
            (max(len(v) for v in self._timings.values()) if self._timings else 0),
            (max(len(v) for v in self._counts.values()) if self._counts else 0),
        )

        timing_cols = [f"{n}_ms" for n in self._timing_order]
        count_cols = [f"{n}_n" for n in self._count_order]

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame"] + timing_cols + count_cols)
            for i in range(n_frames):
                row = [i]
                for name in self._timing_order:
                    s = self._timings[name]
                    row.append(f"{s[i] * 1000:.4f}" if i < len(s) else "")
                for name in self._count_order:
                    s = self._counts[name]
                    row.append(str(s[i]) if i < len(s) else "")
                w.writerow(row)

            totals = ["TOTAL"]
            for name in self._timing_order:
                totals.append(f"{sum(self._timings[name]) * 1000:.4f}")
            for name in self._count_order:
                totals.append(str(sum(self._counts[name])))
            w.writerow(totals)

        print(f"Profile saved → {path}  ({n_frames} frames)")

    def save_summary_csv(self, path: str = "profile_setup.csv"):
        if not self._timings:
            print("Profiler: no data collected.")
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stage", "ms"])
            for name in self._timing_order:
                total_ms = sum(self._timings[name]) * 1000.0
                w.writerow([name, f"{total_ms:.4f}"])
        print(f"Profile saved → {path}  ({len(self._timing_order)} stages)")


class NullProfiler:
    def section(self, _name: str):
        return nullcontext()

    def record(self, _name: str, _dt: float):
        pass

    def count(self, _name: str, _n: int):
        pass

    def save_csv(self, _path: str = "profile_frames.csv"):
        pass

    def save_summary_csv(self, _path: str = "profile_setup.csv"):
        pass
