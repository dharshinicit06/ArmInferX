"""ArmInferX — repair helper for the llama-cpp-python DLL issue on Windows.

Symptom (verified on the Windows dev machine, llama-cpp-python 0.3.34):

    RuntimeError: Failed to load shared library
    '...\\site-packages\\llama_cpp\\lib\\llama.dll'
    The error is: FileNotFoundError: Could not find module
    '...\\llama_cpp\\lib\\llama.dll' (or one of its dependencies)

Root cause
----------
The llama.cpp DLLs shipped in the wheel are valid x86-64 binaries, but
``ggml-base.dll`` and ``ggml-cpu.dll`` import ``VCOMP140.DLL`` — the MSVC
**OpenMP** runtime. On machines where the Microsoft Visual C++ 2015-2022
Redistributable is missing (or was removed), that DLL is absent from
``C:\\Windows\\System32`` and *every* llama_cpp DLL fails to load — even
though the DLL files themselves are present.

Fix
---
The proper fix is installing the official "Microsoft Visual C++ 2015-2022
Redistributable (x64)". This script automates the alternative: it locates an
existing x86-64 ``vcomp140.dll`` on the machine and copies it next to the
package's other DLLs (a per-venv, no-admin fix). It is idempotent: if
``import llama_cpp`` already works, it exits without changing anything.

Run from the repo root (use the project venv):

    backend\\.venv\\Scripts\\python.exe scripts\\fix_llama_dll.py

Exit codes: 0 = llama_cpp importable (already fixed, or fixed by this run);
1 = could not fix automatically (instructions printed).
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import shutil
import sys
from pathlib import Path


def _pkg_lib_dir() -> Path | None:
    spec = importlib.util.find_spec("llama_cpp")
    if spec is None or spec.submodule_search_locations is None:
        return None
    lib = Path(list(spec.submodule_search_locations)[0]) / "lib"
    return lib if lib.is_dir() else None


def _import_ok() -> bool:
    try:
        import llama_cpp  # noqa: F401
        return True
    except Exception:
        return False


def _find_vcomp140() -> Path | None:
    """Return a plausible x86-64 vcomp140.dll, or None."""
    # 1) System32 — the canonical 64-bit system DLL location.
    system32 = Path(r"C:\Windows\System32\vcomp140.dll")
    if system32.is_file():
        return system32

    # 2) Visual Studio VC++ redistributable folders (both Program Files
    #    variants). Prefer files under an x64/OpenMP path.
    hits: list[Path] = []
    for var in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(var)
        if not base:
            continue
        vs = Path(base) / "Microsoft Visual Studio"
        if not vs.is_dir():
            continue
        try:
            for candidate in vs.rglob("vcomp140.dll"):
                parts = {p.lower() for p in candidate.parts}
                if "x64" in parts or "openmp" in parts:
                    hits.append(candidate)
        except OSError:
            continue
    if hits:
        return hits[0]

    # 3) Bounded top-level app scan (depth-limited) for bundled copies
    #    (several common apps ship a stock Microsoft OpenMP runtime).
    for var in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(var)
        if not base:
            continue
        root = Path(base)
        if not root.is_dir():
            continue
        try:
            for app in root.iterdir():
                bin_dir = app / "bin"
                if bin_dir.is_dir() and (bin_dir / "vcomp140.dll").is_file():
                    return bin_dir / "vcomp140.dll"
        except OSError:
            continue
    return None


def main() -> int:
    print("ArmInferX — llama-cpp-python DLL repair helper")
    print("=" * 62)

    if sys.platform != "win32":
        print("Not a Windows host — nothing to do (this fix is Windows-only).")
        return 0 if _import_ok() else 1

    if _import_ok():
        import llama_cpp
        version = getattr(llama_cpp, "__version__", "?")
        print(f"OK: llama_cpp already imports — version {version}. No changes made.")
        return 0

    lib = _pkg_lib_dir()
    if lib is None:
        print("ERROR: could not locate llama_cpp/lib. Is llama-cpp-python")
        print("       installed in THIS venv?")
        return 1

    print(f"llama_cpp import FAILED. Inspecting: {lib}")
    target = lib / "vcomp140.dll"
    if target.is_file():
        print("vcomp140.dll is already present next to the package DLLs, but the")
        print("import still failed. Possible causes:")
        print("  - the MSVC runtime (vcruntime140.dll) is missing system-wide;")
        print("  - the wheel install is corrupt.")
        print("Try reinstalling:")
        print("  backend\\.venv\\Scripts\\python.exe -m pip install --force-reinstall llama-cpp-python==0.3.34")
        return 1

    print("The llama.cpp DLLs import VCOMP140.DLL (MSVC OpenMP runtime), which")
    print("is missing on this machine. Searching for an x86-64 copy...")
    src = _find_vcomp140()
    if src is None:
        print()
        print("No vcomp140.dll found on this machine. The proper fix is to install")
        print("the official Microsoft Visual C++ 2015-2022 Redistributable (x64):")
        print("  https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("(restart your shell after installing, then re-run this script).")
        return 1

    print(f"Found: {src}")
    shutil.copy2(src, target)
    print(f"Copied to: {target}")

    try:
        os.add_dll_directory(str(lib))
        ctypes.CDLL(str(lib / "ggml-base.dll"))
        print("OK: ggml-base.dll now loads — the dependency is resolved.")
    except OSError as exc:  # pragma: no cover - depends on host state
        print(f"WARN: still cannot load ggml-base.dll: {exc}")
        return 1

    if _import_ok():
        import llama_cpp
        version = getattr(llama_cpp, "__version__", "?")
        print(f"OK: import llama_cpp now works — version {version}.")
        print("Fix applied to this venv only. A freshly recreated venv needs it")
        print("again unless the VC++ redistributable is installed system-wide.")
        return 0

    print("Still failing after the copy. Install the VC++ redistributable and")
    print("retry, or reinstall llama-cpp-python:")
    print("  backend\\.venv\\Scripts\\python.exe -m pip install --force-reinstall llama-cpp-python==0.3.34")
    return 1


if __name__ == "__main__":
    sys.exit(main())
