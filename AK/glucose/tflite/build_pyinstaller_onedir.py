#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

# ================= CONFIG =================

SOURCE_DIR = Path(
    "/home/apc-3/PycharmProjects/PythonProjectAK/"
    "TFLite-Conversion-Module/Code/4.Model-converter"
)

# ✅ REQUIRED output directory
DISTPATH = Path(
    "/home/apc-3/PycharmProjects/PythonProjectAK/"
    "TFLite-Conversion-Module/Code/4.Model-converter-bin-pyinstaller/bin"
)

WORKPATH = DISTPATH / "build"
SPECPATH = DISTPATH / "spec"

# Scripts to convert (subprocess tools)
SCRIPTS = [
    # "4.1.SP-tf-to-tflite.py",
    # "4.2.SP-tflite-to-h.py",
    # "4.3.tfl-operations.py",
    # "4.4.extract-wavelength.py",
    # "4.5.generate-preprocess.py",
    "4.0.SP-master-v2.py",  # pathed master script with the pyinstaller script 4.1-4.5
]

# Extra options per script (TensorFlow needs this)
EXTRA_OPTS = {
    # "4.1.SP-tf-to-tflite.py": ["--collect-all", "tensorflow"],
}

# ==========================================


def run(cmd):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_pyinstaller():
    try:
        import PyInstaller  # noqa
    except Exception:
        print("Installing PyInstaller...")
        run([sys.executable, "-m", "pip", "install", "-U",
             "pyinstaller", "pyinstaller-hooks-contrib"])


def build_one(script_name: str):
    script_path = SOURCE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    exe_name = script_path.stem

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onedir",
        "--clean",
        "--noconfirm",
        f"--name={exe_name}",
        f"--distpath={DISTPATH}",
        f"--workpath={WORKPATH}",
        f"--specpath={SPECPATH}",
    ]

    # TensorFlow-heavy script needs collection
    cmd += EXTRA_OPTS.get(script_name, [])

    cmd.append(str(script_path))
    run(cmd)


def main():
    if not SOURCE_DIR.exists():
        print(f"❌ SOURCE_DIR not found: {SOURCE_DIR}")
        sys.exit(1)

    DISTPATH.mkdir(parents=True, exist_ok=True)
    WORKPATH.mkdir(parents=True, exist_ok=True)
    SPECPATH.mkdir(parents=True, exist_ok=True)

    ensure_pyinstaller()

    ok, fail = 0, 0
    for script in SCRIPTS:
        try:
            build_one(script)
            ok += 1
            print(f"✅ Built: {script}")
        except Exception as e:
            fail += 1
            print(f"❌ Failed: {script}")
            print("   ", e)

    print("\n========== SUMMARY ==========")
    print(f"Source folder : {SOURCE_DIR}")
    print(f"Output folder : {DISTPATH}")
    print(f"Success       : {ok}")
    print(f"Failed        : {fail}")

    if fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
