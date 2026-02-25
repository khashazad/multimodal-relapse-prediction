import zipfile
import re
from pathlib import Path

ZIP_PATH = Path("data/original/track1/track_1_annotations.zip")
DATASET_PATH = Path("data/original/track1")

def main():
    pattern = re.compile(r"^(P\d+)/(test_\d+)/relapses\.csv$")
    updated = []

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for name in zf.namelist():
            m = pattern.match(name)
            if not m:
                continue
            patient, seq = m.group(1), m.group(2)
            dest = DATASET_PATH / patient / seq / "relapses.csv"
            if not dest.parent.exists():
                print(f"SKIP (folder missing): {dest}")
                continue
            data = zf.read(name)
            dest.write_bytes(data)
            updated.append(str(dest))

    print(f"Updated {len(updated)} test relapses.csv files:")
    for p in sorted(updated):
        print(f"  {p}")

if __name__ == "__main__":
    main()
