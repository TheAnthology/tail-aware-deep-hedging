from pathlib import Path
import json
import numpy as np


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_weights(model, path):
    """NPZ includes every model array, including calibrated BN buffers."""
    path = Path(path)
    if not model.built and hasattr(model, "initialize"):
        model.initialize()
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            model.set_weights([data[f"arr_{i}"] for i in range(len(data.files))])
    else:
        model.load_weights(path)


def save_weights(model, path):
    np.savez_compressed(path, *model.get_weights())
