"""
ScholarMind — Setup & Run Script
One-command setup: install deps, start Endee, ingest data, launch dashboard.
Usage:
    python setup.py install    # Install dependencies
    python setup.py ingest     # Run data ingestion
    python setup.py run        # Launch Streamlit dashboard
    python setup.py eval       # Run evaluation benchmarks
    python setup.py test       # Run smoke tests
    python setup.py all        # Full pipeline: install → ingest → run
"""

import subprocess
import sys
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd: str, desc: str):
    """Run a shell command with description."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=PROJECT_DIR)
    if result.returncode != 0:
        print(f"\n❌ Failed: {desc}")
        sys.exit(1)
    print(f"✓ {desc}")


def install():
    """Install Python dependencies."""
    run_cmd(f"{sys.executable} -m pip install -r requirements.txt", "Installing Python dependencies")


def start_endee():
    """Start Endee via Docker Compose."""
    run_cmd("docker compose up -d", "Starting Endee server (Docker)")
    print("\n⏳ Waiting 5 seconds for Endee to initialize...")
    import time
    time.sleep(5)
    run_cmd("docker ps --filter name=endee-server --format \"{{.Status}}\"", "Checking Endee status")


def ingest():
    """Run data ingestion pipeline."""
    run_cmd(f"{sys.executable} ingest.py", "Running data ingestion → Endee")


def run_app():
    """Launch Streamlit dashboard."""
    run_cmd(f"{sys.executable} -m streamlit run app.py", "Launching ScholarMind Dashboard")


def run_eval():
    """Run evaluation benchmarks."""
    run_cmd(f"{sys.executable} eval.py", "Running retrieval evaluation")


def run_tests():
    """Run smoke tests."""
    run_cmd(f"{sys.executable} -m pytest test_scholarmind.py -v", "Running smoke tests")


def show_help():
    """Print usage."""
    print(__doc__)


if __name__ == "__main__":
    commands = {
        "install": install,
        "start": start_endee,
        "ingest": ingest,
        "run": run_app,
        "eval": run_eval,
        "test": run_tests,
        "all": lambda: (install(), start_endee(), ingest(), run_app()),
        "help": show_help,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        show_help()
        sys.exit(0)

    commands[sys.argv[1]]()
