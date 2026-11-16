from pathlib import Path
import sys
from report import build
build(Path(sys.argv[1]).resolve())
