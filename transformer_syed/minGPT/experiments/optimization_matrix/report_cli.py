from pathlib import Path
import sys
from report import build
from overview import build as overview
out=Path(sys.argv[1]).resolve()
build(out)
overview(out.parent)
