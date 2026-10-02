# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rechunk the three Nemotron Terminal files that PyArrow cannot stream as one row group."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.june_tpu_67b_a2b.moe.sft_67b_a2b_2stage import NEMOTRON_TERMINAL_DATASET

RECHUNK_FILES = {
    "dataset_adapters/code.parquet",
    "dataset_adapters/math.parquet",
    "dataset_adapters/swe.parquet",
}


def _source_files() -> list[str]:
    groups = NEMOTRON_TERMINAL_DATASET.parquet_files or {}
    return sorted(path for paths in groups.values() for path in paths)


def parquet_content_digest(path: Path, *, batch_size: int = 256) -> tuple[int, str]:
    """Hash logical rows in order, independent of Parquet row-group layout."""
    digest = hashlib.sha256()
    rows = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            encoded = json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            rows += 1
    return rows, digest.hexdigest()


def rechunk_parquet(source: Path, destination: Path, *, row_group_size: int = 1024) -> dict[str, object]:
    """Rewrite physical row groups and prove that logical rows are unchanged."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(source)
    writer: pq.ParquetWriter | None = None
    try:
        for batch in parquet.iter_batches(batch_size=row_group_size):
            if writer is None:
                writer = pq.ParquetWriter(destination, batch.schema, compression="zstd")
            writer.write_table(pa.Table.from_batches([batch]), row_group_size=row_group_size)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"Refusing to create an empty derived Parquet file from {source}.")

    source_rows, source_digest = parquet_content_digest(source)
    derived_rows, derived_digest = parquet_content_digest(destination)
    if (source_rows, source_digest) != (derived_rows, derived_digest):
        destination.unlink(missing_ok=True)
        raise ValueError(f"Logical content changed while rechunking {source}.")
    return {
        "source": str(source),
        "derived": str(destination),
        "rows": source_rows,
        "content_sha256": source_digest,
        "row_group_size": row_group_size,
    }


def prepare_dataset(source_root: Path, output_root: Path, *, row_group_size: int = 1024) -> dict[str, object]:
    """Materialize the pinned 29-file selection, rewriting only the three oversized files."""
    files = _source_files()
    if len(files) != 29:
        raise ValueError(f"Reference selection has {len(files)} files; expected 29.")
    records: list[dict[str, object]] = []
    for relative in files:
        source = source_root / relative
        destination = output_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative in RECHUNK_FILES:
            records.append(rechunk_parquet(source, destination, row_group_size=row_group_size))
        else:
            destination.unlink(missing_ok=True)
            try:
                os.link(source.resolve(), destination)
            except OSError:
                shutil.copy2(source, destination)

    list_path = output_root / "parquet-list.txt"
    list_path.write_text("".join(f"{output_root / relative}\n" for relative in files))
    manifest = {
        "dataset_id": NEMOTRON_TERMINAL_DATASET.hf_dataset_id,
        "dataset_revision": "a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9",
        "files": files,
        "rechunked": records,
    }
    (output_root / "rechunk_provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--row-group-size", type=int, default=1024)
    args = parser.parse_args()
    manifest = prepare_dataset(args.source_root, args.output_root, row_group_size=args.row_group_size)
    print(f"files={len(manifest['files'])}")
    print(f"rechunked={len(manifest['rechunked'])}")
    print("SNOWBALL_NEMOTRON_RECHUNK_OK")


if __name__ == "__main__":
    main()
