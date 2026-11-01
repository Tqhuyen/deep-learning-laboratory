"""Download and normalise SST-2 into two-column TSV files.

Usage:

    python data/prepare_sst2.py

Writes ``data/sst2/train.tsv`` and ``data/sst2/dev.tsv`` with the header
``sentence<TAB>label``. Raw sentences are kept exactly as distributed; the
tokenizer in the experiment does the lowercasing and splitting.

Source: GLUE SST-2, https://gluebenchmark.com/tasks (public research benchmark).
Nothing is downloaded implicitly by a training run; this script is the explicit
preparation step, as required by ``docs/conventions.md``.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"


def download(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for split in ("train", "dev"):
            member = f"SST-2/{split}.tsv"
            text = archive.read(member).decode("utf-8")
            rows = []
            for line in text.splitlines():
                if not line or line.startswith("sentence\t"):
                    continue
                sentence, label = line.rsplit("\t", 1)
                rows.append((sentence, label))
            target = destination / f"{split}.tsv"
            target.write_text(
                "sentence\tlabel\n" + "\n".join(f"{s}\t{l}" for s, l in rows) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {target} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parent / "sst2",
        help="directory for train.tsv and dev.tsv",
    )
    args = parser.parse_args()
    download(args.destination)


if __name__ == "__main__":
    main()
