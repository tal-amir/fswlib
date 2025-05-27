# examples/setup_fsw_import.py

import sys
import pathlib

# Point to project_root/src
project_root = pathlib.Path(__file__).resolve().parents[2]
src_dir = project_root / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Explicitly import fsw so it’s exposed to users of this module
import fsw
_ = fsw.__name__  # To avoid an "unused import" warning