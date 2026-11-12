#!/usr/bin/env python3
"""Root wrapper for Internal Ops Co-Pilot CLI."""
import sys
from pathlib import Path

# Add schemesaathi package directory to path
PACKAGE_DIR = Path(__file__).resolve().parent / "schemesaathi"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

# Import and execute the main CLI
from schemesaathi.ask import main

if __name__ == "__main__":
    sys.exit(main())
