"""Detect RAM / VRAM and decide whether a model fits.

Supported detection:
  - Apple Silicon: unified memory (GPU budget = Metal working-set limit)
  - NVIDIA (any OS): nvidia-smi
  - AMD on Linux: /sys/class/drm/*/device/mem_info_vram_total
  - Any GPU on Windows: display-adapter registry key (qwMemorySize)
Unknown or new hardware: pass --vram-gb / --ram-gb to override detection.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

import psutil

GB = 1024**3


@dataclass
class Gpu:
    name: str
    vram_gb: float


@dataclass
class Hardware:
    os: str
    cpu: str
    cores: int
    ram_gb: float
    ram_available_gb: float
    gpus: list[Gpu] = field(default_factory=list)
    unified: bool = False  # Apple Silicon style: GPU uses system RAM
    gpu_budget_gb: float = 0.0  # memory a model may occupy on the GPU

    @property
    def vram_gb(self) -> float:
        return sum(g.vram_gb for g in self.gpus)

    def placement(self, need_gb: float, allow_partial: bool = True) -> str | None:
        """Where a model needing `need_gb` would run, or None if it does not fit."""
        if self.unified:
            return "GPU (unified)" if need_gb <= self.gpu_budget_gb else None
        if self.gpus and need_gb <= self.gpu_budget_gb:
            return "GPU"
        ram_usable = self.ram_gb * 0.85 - 2.0  # leave room for the OS
        if self.gpus and allow_partial and need_gb <= self.gpu_budget_gb + ram_usable:
            return "GPU+CPU"
        if need_gb <= ram_usable:
            return "CPU"
        return None

    def describe(self) -> str:
        gpus = ", ".join(f"{g.name} ({g.vram_gb:.0f} GB)" for g in self.gpus) or "none detected"
        mem = f"{self.ram_gb:.0f} GB RAM"
        if self.unified:
            mem += f" unified, GPU budget {self.gpu_budget_gb:.0f} GB"
        return f"{self.os} | {self.cpu} ({self.cores} cores) | {mem} | GPU: {gpus}"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _cpu_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or platform.processor()
    if system == "Linux":
        try:
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if system == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        if out.strip():
            return out.strip().splitlines()[0]
    return platform.processor() or platform.machine()


def _os_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    if system == "Linux":
        try:
            info = platform.freedesktop_os_release()
            return info.get("PRETTY_NAME", "Linux")
        except OSError:
            return f"Linux {platform.release()}"
    return f"{system} {platform.release()}"


def _nvidia() -> list[Gpu]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.strip().splitlines():
        name, mib = (p.strip() for p in line.rsplit(",", 1))
        gpus.append(Gpu(name, float(mib) / 1024))
    return gpus


def _amd_linux() -> list[Gpu]:
    gpus = []
    for path in glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"):
        try:
            vram = int(open(path).read().strip()) / GB
        except (OSError, ValueError):
            continue
        if vram >= 1:  # skip tiny iGPU carve-outs
            gpus.append(Gpu("AMD GPU", vram))
    return gpus


def _windows_registry() -> list[Gpu]:
    # Win32_VideoController.AdapterRAM is capped at 4 GB, so read the driver registry instead.
    script = (
        "Get-ItemProperty 'HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\"
        "{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' -ErrorAction SilentlyContinue | "
        "Where-Object { $_.'HardwareInformation.qwMemorySize' } | "
        "Select-Object DriverDesc, 'HardwareInformation.qwMemorySize' | ConvertTo-Json"
    )
    out = _run(["powershell", "-NoProfile", "-Command", script])
    try:
        rows = json.loads(out) if out.strip() else []
    except json.JSONDecodeError:
        return []
    if isinstance(rows, dict):
        rows = [rows]
    gpus = []
    for r in rows:
        vram = int(r["HardwareInformation.qwMemorySize"]) / GB
        if vram >= 1 and "NVIDIA" not in r.get("DriverDesc", ""):  # NVIDIA comes from nvidia-smi
            gpus.append(Gpu(r.get("DriverDesc", "GPU"), vram))
    return gpus


def _apple_gpu_budget(ram_gb: float) -> float:
    # Metal lets the GPU wire roughly 2/3 (<=36 GB) or 3/4 (>36 GB) of RAM unless
    # the user raised it with `sudo sysctl iogpu.wired_limit_mb=...`.
    wired = _run(["sysctl", "-n", "iogpu.wired_limit_mb"]).strip()
    if wired.isdigit() and int(wired) > 0:
        return int(wired) / 1024
    return ram_gb * (0.75 if ram_gb > 36 else 0.67)


def detect(vram_override: float | None = None, ram_override: float | None = None) -> Hardware:
    vm = psutil.virtual_memory()
    ram_gb = ram_override or vm.total / GB
    hw = Hardware(
        os=_os_name(), cpu=_cpu_name(), cores=psutil.cpu_count(logical=False) or os.cpu_count() or 0,
        ram_gb=ram_gb, ram_available_gb=vm.available / GB,
    )
    system, machine = platform.system(), platform.machine()
    if system == "Darwin" and machine == "arm64":
        hw.unified = True
        hw.gpus = [Gpu("Apple GPU", ram_gb)]
        hw.gpu_budget_gb = vram_override or _apple_gpu_budget(ram_gb)
        return hw

    gpus = _nvidia()
    if system == "Linux":
        gpus += _amd_linux()
    elif system == "Windows":
        gpus += _windows_registry()
    if vram_override:
        gpus = [Gpu(gpus[0].name if gpus else "GPU (manual)", vram_override)]
    hw.gpus = gpus
    hw.gpu_budget_gb = max(0.0, hw.vram_gb - 0.5 * len(gpus))  # driver/context overhead per GPU
    return hw
