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


def get_venv_python() -> str:
    """Get the Python executable path inside the venv."""
    venv_dir = os.path.join(PROJECT_DIR, "venv")
    if sys.platform == "win32":
        python_path = os.path.join(venv_dir, "Scripts", "python")
    else:
        python_path = os.path.join(venv_dir, "bin", "python")
    if os.path.exists(python_path) or os.path.exists(python_path + ".exe"):
        return f'"{python_path}"'
    # Fallback to system python if venv doesn't exist yet
    print("⚠️  venv not found — using system Python. Run 'python setup.py install' first.")
    return sys.executable


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
    """Create virtual environment and install Python dependencies."""
    venv_dir = os.path.join(PROJECT_DIR, "venv")
    if not os.path.exists(venv_dir):
        run_cmd(f"{sys.executable} -m venv venv", "Creating virtual environment")

    # Determine pip path inside venv
    if sys.platform == "win32":
        pip_path = os.path.join(venv_dir, "Scripts", "pip")
    else:
        pip_path = os.path.join(venv_dir, "bin", "pip")

    run_cmd(f"\"{pip_path}\" install -r requirements.txt", "Installing Python dependencies into venv")


def start_endee():
    """Start Endee via Docker Compose."""
    run_cmd("docker compose up -d", "Starting Endee server (Docker)")
    print("\n⏳ Waiting 5 seconds for Endee to initialize...")
    import time
    time.sleep(5)
    run_cmd("docker ps --filter name=endee-server --format \"{{.Status}}\"", "Checking Endee status")


def ingest():
    """Run data ingestion pipeline."""
    run_cmd(f"{get_venv_python()} ingest.py", "Running data ingestion → Endee")


def run_app():
    """Launch Streamlit dashboard."""
    run_cmd(f"{get_venv_python()} -m streamlit run app.py", "Launching ScholarMind Dashboard")


def run_eval():
    """Run evaluation benchmarks."""
    run_cmd(f"{get_venv_python()} eval.py", "Running retrieval evaluation")


def run_tests():
    """Run smoke tests."""
    run_cmd(f"{get_venv_python()} -m pytest test_scholarmind.py -v", "Running smoke tests")


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
