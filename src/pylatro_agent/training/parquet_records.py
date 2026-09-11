"""Bounded-memory access to Parquet teacher demonstrations."""
from __future__ import annotations

import bisect
import json
import operator
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..reward import RewardConfig


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Parquet records require pyarrow; install pylatro[agent].") from exc
    return pa, pq


class ParquetRecords(Sequence):
    """Read a file or dataset directory, caching at most two decoded row groups.

    Returned observations have their original NumPy shapes/dtypes. The published
    dataset's additional ``step`` column is omitted. Metadata is embedded in new
    files, or read from metadata/schema.json in a downloaded dataset snapshot.
    """

    def __init__(self, path: str | Path, *, reward_config: RewardConfig):
        from .model_generate import _validate_record_metadata

        _, pq = _arrow()
        path = Path(path)
        paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
        if not paths:
            raise ValueError(f"No Parquet files found in {path}")
        self._files = []
        self._groups = []
        self._ends = []
        self._cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        expected_schema = None
        total = 0
        for file_path in paths:
            file = pq.ParquetFile(file_path)
            schema = file.schema_arrow
            embedded = (schema.metadata or {}).get(b"pylatro")
            if embedded:
                metadata = json.loads(embedded)
            else:
                candidates = [file_path.parent / "metadata/schema.json",
                              file_path.parent.parent / "metadata/schema.json"]
                sidecar = next((p for p in candidates if p.is_file()), None)
                if sidecar is None:
                    raise ValueError("Parquet records require embedded metadata or the dataset's metadata/schema.json")
                metadata = json.loads(sidecar.read_text())
            _validate_record_metadata(metadata, reward_config)
            if expected_schema is not None and not expected_schema.equals(schema, check_metadata=False):
                raise ValueError("Parquet shards have incompatible record schemas")
            expected_schema = schema
            if "obs" not in schema.names or "action" not in schema.names:
                raise ValueError("Parquet dataset is missing observations or actions")
            file_index = len(self._files)
            self._files.append(file)
            for group in range(file.num_row_groups):
                count = file.metadata.row_group(group).num_rows
                if not count:
                    continue
                self._groups.append((file_index, group))
                total += count
                self._ends.append(total)

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def _group(self, index: int) -> list[dict[str, Any]]:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        pa, _ = _arrow()
        file_index, group = self._groups[index]
        table = self._files[file_index].read_row_group(group).combine_chunks()
        columns = {name: table[name].to_pylist() for name in table.column_names if name not in {"obs", "step"}}
        observations = table["obs"].chunk(0)
        if observations.null_count:
            raise ValueError("Null observations are not valid training records")
        arrays = {}
        for field in observations.type:
            array = observations.field(field.name)
            shape = [len(table)]
            while pa.types.is_fixed_size_list(array.type):
                shape.append(array.type.list_size)
                if array.null_count:
                    raise ValueError(f"Null observation array: {field.name}")
                array = array.flatten()
            if pa.types.is_list(array.type) or array.null_count:
                raise ValueError(f"Observation arrays must have fixed shapes and no nulls: {field.name}")
            arrays[field.name] = array.to_numpy(zero_copy_only=False).reshape(shape)
        records = [{**{key: values[i] for key, values in columns.items()},
                    "obs": {key: values[i] for key, values in arrays.items()}}
                   for i in range(len(table))]
        self._cache[index] = records
        if len(self._cache) > 2:
            self._cache.popitem(last=False)
        return records

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        group = bisect.bisect_right(self._ends, index)
        start = self._ends[group - 1] if group else 0
        return self._group(group)[index - start]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for group in range(len(self._groups)):
            yield from self._group(group)

    def shuffled_indices(self) -> np.ndarray:
        """Shuffle groups and decisions within groups, retaining read locality."""
        pieces = []
        for group in np.random.permutation(len(self._groups)):
            start = self._ends[group - 1] if group else 0
            pieces.append(start + np.random.permutation(self._ends[group] - start))
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def save_parquet_records(records, path: Path, *, metadata: dict[str, Any]) -> None:
    """Write standard records as Zstandard-compressed Parquet with metadata."""
    pa, pq = _arrow()
    if not records:
        raise ValueError("Cannot infer a Parquet observation schema from empty records")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    writer = None
    try:
        for start in range(0, len(records), 256):
            batch = records[start:start + 256]
            obs_arrays = []
            names = list(batch[0]["obs"])
            for name in names:
                values = np.stack([r["obs"][name] for r in batch])
                array = pa.array(values.reshape(-1))
                for length in reversed(values.shape[1:]):
                    array = pa.FixedSizeListArray.from_arrays(array, length)
                obs_arrays.append(array)
            columns = {key: pa.array([r[key] for r in batch]) for key in batch[0] if key != "obs"}
            columns["obs"] = pa.StructArray.from_arrays(obs_arrays, names=names)
            table = pa.table(columns).replace_schema_metadata({b"pylatro": json.dumps(metadata).encode()})
            if writer is None:
                writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
            writer.write_table(table)
        writer.close()
        writer = None
        temp.replace(path)
    finally:
        if writer is not None:
            writer.close()
        temp.unlink(missing_ok=True)
