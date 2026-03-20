from __future__ import annotations

import argparse
import logging

from .config import get_settings
from .embeddings import build_and_save_indexes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval index artifacts.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild indexes even if the current artifacts are already up to date.",
    )
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    build_and_save_indexes(force=args.force)


if __name__ == "__main__":
    main()
