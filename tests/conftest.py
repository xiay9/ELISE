import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'code', ROOT / 'code/elise'):
    sys.path.insert(0, str(path))
