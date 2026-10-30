import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Keep pretrained weight downloads off C:
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
