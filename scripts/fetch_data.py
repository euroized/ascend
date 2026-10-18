"""Fetch the pinned input data and verify every file against data/manifest.json.

    python scripts/fetch_data.py                      # connectome (core), ~104 MB
    python scripts/fetch_data.py --group provenance   # v630 data + annotations, used by map_stimuli.py
    python scripts/fetch_data.py --verify             # check files already present, download nothing
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend import data  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="core", choices=sorted(data.load_manifest()["groups"]))
    ap.add_argument("--verify", action="store_true", help="only verify files already present")
    args = ap.parse_args()

    ok = True
    for e in data.entries(args.group):
        if args.verify:
            good = data.verify(e)
            ok &= good
            print(f"{'OK ' if good else 'BAD'} {e['name']}")
        else:
            print(f"fetching {e['name']} ({e['bytes'] / 1e6:.1f} MB) ...", flush=True)
            p = data.fetch(e)
            print(f"OK  {p}  sha256={e['sha256'][:16]}...")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
