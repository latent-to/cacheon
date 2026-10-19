"""Device and identity fixtures for the end-to-end harness.

Anything a caller varies per test stays an argument; a second device model is a
named profile here rather than a mode flag on one builder.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib

from cacheon.eval.device_state import GPUConfiguration


def sha(label: str) -> str:
    """A deterministic stand-in digest, named by what it stands for."""

    return hashlib.sha256(label.encode()).hexdigest()


def gpu(index: int = 0, model: str = "b300") -> GPUConfiguration:
    """One device of ``model``. The second model is not decoration.

    A generic evaluator that resolves identity from sealed inputs has to be
    exercised on more than one device profile, or "generic" is untested. The
    end-to-end harness runs the H100 profile for exactly that reason.
    """

    if model == "h100":
        return replace(
            gpu(index), name="NVIDIA H100 80GB HBM3", memory_total_mib=81_559,
            power_limit_mw=700_000, max_graphics_clock_mhz=1_980,
            max_memory_clock_mhz=2_619,
        )
    return GPUConfiguration(
        physical_id=index,
        uuid=f"GPU-00000000-{index:04x}-0000-0000-{index:012x}",
        pci_bus_id=f"00000000:{index + 1:02x}:00.0",
        name="NVIDIA B300 SXM6 AC",
        memory_total_mib=288_000,
        driver_version="600.10.01",
        power_limit_mw=1_000_000,
        compute_mode="Default",
        persistence_mode="Enabled",
        application_graphics_clock_mhz=None,
        application_memory_clock_mhz=None,
        max_graphics_clock_mhz=2_500,
        max_memory_clock_mhz=5_000,
    )
