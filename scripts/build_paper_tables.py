from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "reporting/build_paper_tables.py"), run_name="__main__")
