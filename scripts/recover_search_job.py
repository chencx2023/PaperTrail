"""Keep the original search-only recovery command available."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.recover_job import main as _recover_main, parse_args


def main(argv: list[str] | None = None) -> int:
    return _recover_main(argv, allowed_kinds=("search",))


if __name__ == "__main__":
    raise SystemExit(main())
