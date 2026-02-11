#!/usr/bin/env python3
"""
build_exe.py
Batch-compile all .py files in the source folder into Nuitka standalone executables.

SOURCE (.py):
/home/apc-3/PycharmProjects/PythonProjectAK/TFLite-Conversion-Module/Code/4.Model-converter

OUTPUT (binaries):
/home/apc-3/PycharmProjects/PythonProjectAK/TFLite-Conversion-Module/Code/4.Model-converter-bin
"""

import sys
import shutil
import subprocess
from pathlib import Path

# ===================== CONFIG =====================
SOURCE_DIR = Path(
    "/home/apc-3/PycharmProjects/PythonProjectAK/"
    "TFLite-Conversion-Module/Code/4.Model-converter"
)

OUTPUT_DIR = Path(
    "/home/apc-3/PycharmProjects/PythonProjectAK/"
    "TFLite-Conversion-Module/Code/4.Model-converter-bin/bin-all"
)

ONEFILE_MODE = False   # True → single-file exe, False → standalone folder (recommended)

SKIP_FILES = {
    # "build_exe.py",
    # "__init__.py",
}

EXTRA_NUITKA_ARGS = ["--static-libpython=no"]
# ==================================================


def run(cmd: list[str], cwd: Path):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_nuitka():
    try:
        import nuitka  # noqa
    except Exception:
        print("❌ Nuitka is not installed.")
        print("Install with: pip install -U nuitka")
        sys.exit(1)


def prepare_output_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_scripts():
    return [
        p for p in sorted(SOURCE_DIR.glob("*.py"))
        if p.name not in SKIP_FILES
    ]


def build_script(script: Path):
    name = script.stem
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "nuitka",
        str(script),

        "--standalone",
        "--follow-imports",
        "--assume-yes-for-downloads",

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
    if not SOURCE_DIR.exists():
        print(f"❌ SOURCE_DIR not found: {SOURCE_DIR}")
        sys.exit(1)

    ensure_nuitka()
    prepare_output_dirs()

    scripts = collect_scripts()
    if not scripts:
        print("⚠️ No .py scripts found.")
        return

    print(f"🔍 Found {len(scripts)} script(s):")
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
    print(f"Output folder : {OUTPUT_DIR}")
    print(f"Mode          : {'onefile' if ONEFILE_MODE else 'standalone'}")
    print(f"Success       : {len(scripts) - failures}")
    print(f"Failed        : {failures}")

    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
