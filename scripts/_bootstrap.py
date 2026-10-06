"""Lets the scripts in this folder import the project package without installing it."""

import os
import sys
from pathlib import Path

# Let PyTorch run operations that Apple's GPU (MPS) doesn't support yet on the CPU instead of
# failing. It has to be set before torch is imported, so it lives here.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
