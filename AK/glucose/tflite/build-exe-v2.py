#!/usr/bin/env python3
"""
build-exe.py
Batch-compile all .py files in the source folder into Nuitka standalone executables.

SOURCE (.py):
/home/apc-3/PycharmProjects/PythonProjectAK/TFLite-Conversion-Module/Code/4.Model-converter

OUTPUT (binaries):
/home/apc-3/PycharmProjects/PythonProjectAK/TFLite-Conversion-Module/Code/4.Model-converter-bin
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


# ===================== CONFIG =====================
SOURCE_DIR = Path(
    "/home/apc-3/PycharmProjects/PythonProjectAK/"
    "TFLite-Conversion-Module/Code/4.Model-converter"
)

# ✅ You requested this as the output root
OUTPUT_ROOT = Path(
    "/home/apc-3/PycharmProjects/PythonProjectAK/"
    "TFLite-Conversion-Module/Code/4.Model-converter-bin"
)

# Where to place the build script itself (optional organization)
SCRIPT_OUT_DIR = OUTPUT_ROOT / "script"
BIN_OUT_DIR = OUTPUT_ROOT / "bin-all-v2"

ONEFILE_MODE = False  # False recommended for Linux + heavy deps

SKIP_FILES = {
    "build-exe.py",
    "__init__.py",
}

EXTRA_NUITKA_ARGS = [
    # "--static-libpython=no",   # needed for conda envs (avoids libpython-static requirement)
    # "lto=no",
    # "--clang",
]
# ==================================================


def run(cmd: list[str], cwd: Path | None = None):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_patchelf():
    if shutil.which("patchelf") is None:
        print("❌ Missing system dependency: patchelf")
        print("Nuitka standalone mode on Linux requires patchelf.")
        print("\nInstall it once with:")
        print("  sudo apt update && sudo apt install -y patchelf")
        sys.exit(1)


def ensure_nuitka():
    try:
        import nuitka  # noqa: F401
        return
    except Exception:
        print("⚠️ Nuitka not found in this environment. Installing via pip...")
        run([sys.executable, "-m", "pip", "install", "-U", "pip"])
        run([sys.executable, "-m", "pip", "install", "-U", "nuitka"])


def prepare_output_dirs():
    if not SOURCE_DIR.exists():
        print(f"❌ SOURCE_DIR not found: {SOURCE_DIR}")
        sys.exit(1)

    SCRIPT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    BIN_OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_scripts() -> list[Path]:
    return [
        p for p in sorted(SOURCE_DIR.glob("*.py"))
        if p.name not in SKIP_FILES
    ]


def build_script(script: Path):
    name = script.stem
    out_dir = BIN_OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "nuitka",
        str(script),

        "--standalone",
        "--follow-imports",
        "--assume-yes-for-downloads",

        "--static-libpython=no",
        # "--clang",
        # "lto=no",

        f"--output-dir={out_dir}",
        f"--output-filename={name}",
    ]

    if ONEFILE_MODE:
        cmd.append("--onefile")

    cmd.extend(EXTRA_NUITKA_ARGS)

    run(cmd, cwd=SOURCE_DIR)
    print(f"✅ Built: {name}")
    print(f"📁 Output: {out_dir}")


def main():
    print("==== Nuitka Batch Builder ====")
    print(f"Python      : {sys.executable}")
    print(f"Source dir  : {SOURCE_DIR}")
    print(f"Output root : {OUTPUT_ROOT}")
    print(f"Mode        : {'onefile' if ONEFILE_MODE else 'standalone'}")

    # Step A (checks) + Step B (build) in one run
    ensure_patchelf()
    ensure_nuitka()
    prepare_output_dirs()

    scripts = collect_scripts()
    if not scripts:
        print("⚠️ No .py scripts found.")
        return

    print(f"\n🔍 Found {len(scripts)} script(s):")
    for s in scripts:
        print(f"  - {s.name}")

    failures = 0
    for script in scripts:
        try:
            build_script(script)
        except subprocess.CalledProcessError:
            failures += 1
            print(f"❌ Build failed: {script.name}")

    print("\n========== SUMMARY ==========")
    print(f"Source folder : {SOURCE_DIR}")
    print(f"Output folder : {BIN_OUT_DIR}")
    print(f"Success       : {len(scripts) - failures}")
    print(f"Failed        : {failures}")

    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
