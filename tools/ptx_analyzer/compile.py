"""nvcc orchestration: compile a kernel.cu into PTX + cubin + SASS."""

from __future__ import annotations
import subprocess
from pathlib import Path


def compile_ptx(out_dir: Path, source: Path, arch: str,
                defines: dict[str, str | int],
                cwd: Path) -> tuple[Path, str]:
    """Compile to PTX. Returns (ptx_path, ptxas_stderr)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ptx_path = out_dir / "kernel.ptx"
    cmd = [
        "nvcc", f"-arch={arch}", "-O3",
        *(f"-D{k}={v}" for k, v in defines.items()),
        "-ptx", "--ptxas-options=-v",
        str(source), "-o", str(ptx_path),
    ]
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc -ptx failed:\n{result.stderr}")
    (out_dir / "ptxas_ptx_verbose.txt").write_text(result.stderr)
    return ptx_path, result.stderr


def compile_cubin(out_dir: Path, source: Path, arch: str,
                  defines: dict[str, str | int],
                  cwd: Path) -> tuple[Path | None, str]:
    """Compile to cubin (for SASS extraction). Returns (cubin_path, stderr)."""
    cubin_path = out_dir / "kernel.cubin"
    cmd = [
        "nvcc", f"-arch={arch}", "-O3",
        *(f"-D{k}={v}" for k, v in defines.items()),
        "--cubin", "--ptxas-options=-v",
        str(source), "-o", str(cubin_path),
    ]
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        return None, result.stderr
    (out_dir / "ptxas_cubin_verbose.txt").write_text(result.stderr)
    return cubin_path, result.stderr


def dump_sass(cubin: Path, out_path: Path) -> Path | None:
    """cuobjdump --dump-sass cubin > out_path."""
    try:
        with open(out_path, "w") as f:
            subprocess.run(["cuobjdump", "--dump-sass", str(cubin)],
                           stdout=f, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out_path
