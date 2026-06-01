from pathlib import Path
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parent


def collect_scripts() -> list[Path]:
    scripts: list[Path] = []

    datapoints_dir = BASE_DIR / "Datapoints"
    if datapoints_dir.exists():
        scripts.extend(sorted(datapoints_dir.glob("*.py")))

    coverage_dir = BASE_DIR / "Coverage"
    if coverage_dir.exists():
        scripts.extend(sorted(coverage_dir.rglob("*.py")))

    return [script for script in scripts if script.name != Path(__file__).name]


for script in collect_scripts():
    print(f"Starting {script.relative_to(BASE_DIR)}...")
    subprocess.run([sys.executable, str(script)], cwd=str(script.parent), check=True)