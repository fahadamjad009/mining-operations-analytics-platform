import subprocess
import sys

def run(cmd: list[str]):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    # Mining mode pipeline
    run([sys.executable, "-m", "src.miningops.generate_data"])
    run([sys.executable, "-m", "src.miningops.kpis"])
    run([sys.executable, "-m", "src.miningops.train"])
    run([sys.executable, "-m", "src.miningops.kpi_snapshot"])

    # NASA mode pipeline
    run([sys.executable, "-m", "src.miningops.nasa_ingest"])
    run([sys.executable, "-m", "src.miningops.nasa_train"])

    # Reporting (plots + HTML)
    run([sys.executable, "-m", "src.miningops.report"])

    print("\n✅ All done. Open: reports/mining_ops_report.html")

if __name__ == "__main__":
    main()
