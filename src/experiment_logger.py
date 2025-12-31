import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def save_experiment(record: Dict[str, Any], out_dir: Path) -> Path:
    """
    Persist a single experiment run to a JSON file.

    Parameters
    ----------
    record:
        Serializable dictionary with experiment information.
    out_dir:
        Directory where the JSON will be saved. Created if missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"experiment_{stamp}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return path
