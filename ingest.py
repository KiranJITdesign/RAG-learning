#!/usr/bin/env python3
"""Root wrapper for Internal Ops Co-Pilot ingestion."""
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent / "schemesaathi"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from schemesaathi.ingest import main

if __name__ == "__main__":
    main()
