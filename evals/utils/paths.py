from pathlib import Path

# Project root path
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

ROUTER_GROUND_TRUTH: str = str(PROJECT_ROOT / "evals" / "files" / "router_ground_truth.json")
READ_BOOKS_GROUND_TRUTH: str = str(PROJECT_ROOT / "evals" / "files" / "read_books_ground_truth.json")
RECOMMEND_BOOKS_GROUND_TRUTH: str = str(PROJECT_ROOT / "evals" / "files" / "recommend_books_ground_truth.json")
