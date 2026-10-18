"""Background GPU sampler built on nvidia-smi.

Reads from outside the server process, so it works for any engine (HF, vLLM,
SGLang). It reports memory the *driver* sees. That matters for vLLM, which
reserves ~90% of VRAM up front for its KV cache, so "used" memory looks huge
even when idle; PyTorch's own counters would miss that.
"""

import subprocess
import threading
import time

_QUERY = ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
          "--format=csv,noheader,nounits"]


def parse_nvidia_smi(output: str) -> list[dict]:
    rows = []
    for line in output.strip().splitlines():
        index, used, total, util = (field.strip() for field in line.split(","))
        rows.append({"gpu": int(index), "mem_used_mib": float(used),
                     "mem_total_mib": float(total), "util_pct": float(util)})
    return rows


class GPUMonitor:
    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def __enter__(self) -> "GPUMonitor":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(_QUERY, capture_output=True, text=True, timeout=5, check=True).stdout
            except (FileNotFoundError, subprocess.SubprocessError):
                return  # no NVIDIA GPU (e.g. laptop): report nothing
            self.samples.extend(parse_nvidia_smi(out))
            self._stop.wait(self.interval_s)

    def summary(self) -> dict[str, dict]:
        """Peak memory and mean 'utilization' per GPU.

        Remember: utilization.gpu is the fraction of time a kernel was running,
        not how much of the GPU's compute was used (see MFU/MBU).
        """
        by_gpu: dict[int, list[dict]] = {}
        for s in self.samples:
            by_gpu.setdefault(s["gpu"], []).append(s)
        return {
            str(gpu): {
                "peak_mem_mib": max(s["mem_used_mib"] for s in rows),
                "mem_total_mib": rows[0]["mem_total_mib"],
                "mean_util_pct": sum(s["util_pct"] for s in rows) / len(rows),
            }
            for gpu, rows in by_gpu.items()
        }
