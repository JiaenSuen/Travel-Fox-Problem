from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfp.reporting import rebuild_all_reports


if __name__ == "__main__":
    outputs = rebuild_all_reports(ROOT)
    for readme, image in outputs:
        print(f"updated {readme.relative_to(ROOT)} | {image.relative_to(ROOT)}")
