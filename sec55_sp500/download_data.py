"""Acquire the attributed public snapshot; reject silent data-version changes."""

import argparse
import hashlib
import shutil
import ssl
import urllib.request
from pathlib import Path
from . import prepare_data

URL = "https://www.openintro.org/data/csv/sp500_1950_2018.csv"
SHA256 = "71835dbbd3b35058dc673db07b6500b9146034dde765225120a203ef1b2ae667"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="Use an already downloaded original snapshot"
    )
    args = parser.parse_args()
    raw = prepare_data.RAW
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        if args.source:
            content = args.source.read_bytes()
        else:
            context = ssl.create_default_context()
            try:
                import certifi

                context = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                pass
            with urllib.request.urlopen(URL, timeout=60, context=context) as response:
                content = response.read()
        if hashlib.sha256(content).hexdigest() != SHA256:
            raise ValueError(
                "Snapshot hash changed: review provenance before replacing the study data"
            )
        raw.write_bytes(content)
    if hashlib.sha256(raw.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Existing raw snapshot does not match the recorded study")
    prepare_data.main()


if __name__ == "__main__":
    main()
