#!/usr/bin/env python3
"""Create a UTF-8 text document without replacing an existing file."""
import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit("usage: write-document.py PATH CONTENT")
try:
    with Path(sys.argv[1]).open("x", encoding="utf-8") as output:
        output.write(sys.argv[2])
except (OSError, ValueError) as error:
    sys.exit(f"cannot create document: {error}")
print(f"Created {sys.argv[1]}")
