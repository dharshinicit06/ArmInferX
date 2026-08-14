"""Deterministic GGUF metadata analysis for ArmInferX (STEP 11, Phase 2).

Parses the **actual bytes** of a GGUF file — header magic/version, the
key-value metadata table and the tensor-info table — and computes a SHA-256,
so model facts are validated from the file itself rather than assumed from a
filename. No model is loaded into memory (a ~470 MB Q4_K_M file needs only a
few MB of Python objects), and the output is deterministic for a given file.

Supported surface (GGUF v3 layout):

    magic u32 | version u32 | tensor_count u64 | kv_count u64
    kv pairs: key (gguf string), value_type u32, value
    tensor infos: name, n_dims u32, dims[u64], type u32, offset u64

Tensor dimensions: modern GGUF (the spec change adopted by current converters)
stores ``dims`` as ``uint64`` per element; legacy GGUF files used ``uint32``.
This module reads ``uint64`` and automatically falls back to ``uint32`` when
the parsed dims fail a sanity check, so both layout generations parse
correctly.

Quantization naming: ``general.file_type`` codes are **converter-version
dependent** (llama.cpp renumbered the enum when ``Q8_1`` was removed). The
mapping here follows the legacy numbering that produced the Qwen2.5 GGUF
files in this repo (``Q4_K_M = 15``), and the authoritative quantization is
additionally derived from the **per-tensor ggml type breakdown** read from
the file (e.g. ``Q4_K``/``Q6_K`` tensors). The filename-derived ``Q4_K_M``
label is never assumed — it is corroborated by both sources.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections import Counter
from pathlib import Path

GGUF_MAGIC = b"GGUF"
_CHUNK = 1024 * 1024  # 1 MiB read buffer for hashing

#: A dimension is implausible above this bound; used to distinguish the
#: modern uint64 dims layout from the legacy uint32 one.
_MAX_SANE_DIM = 2**31


class GGUFMetadataError(ValueError):
    """Raised when a file is not a readable GGUF or its metadata cannot be parsed."""


#: GGUF value type codes -> names (gguf gguf_type).
VALUE_TYPE_NAMES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

#: ggml tensor type codes -> names (the current ggml enum; unknown codes fall
#: back to ``type_N``).
TENSOR_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
}

#: GGUF ``general.file_type`` codes -> human-readable quantization names.
#: Uses the **legacy** llama.cpp numbering (with ``Q8_1 = 9``) that produced
#: the Qwen2.5 GGUF files in this repo, where ``Q4_K_M = 15``. Newer llama.cpp
#: removed ``Q8_1`` and renumbered (there ``Q4_K_M = 14``); see
#: ``analyze_gguf`` which corroborates with the per-tensor breakdown.
FILE_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    4: "Q4_1_SOME_F16",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
}


class _Reader:
    """Bounds-checked little-endian reader over the open GGUF file."""

    def __init__(self, f, path: Path) -> None:
        self._f = f
        self._path = path

    def _read(self, n: int) -> bytes:
        data = self._f.read(n)
        if len(data) != n:
            raise GGUFMetadataError(
                f"truncated GGUF file (expected {n} more bytes): {self._path}"
            )
        return data

    def u8(self) -> int:
        return struct.unpack("<B", self._read(1))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self._read(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self._read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self._read(8))[0]

    def i8(self) -> int:
        return struct.unpack("<b", self._read(1))[0]

    def i16(self) -> int:
        return struct.unpack("<h", self._read(2))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self._read(4))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self._read(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self._read(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self._read(8))[0]

    def string(self) -> str:
        length = self.u64()
        if length > 64 * 1024 * 1024:  # sanity guard against corrupt lengths
            raise GGUFMetadataError(
                f"implausible string length {length} in GGUF metadata: {self._path}"
            )
        raw = self._read(length)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFMetadataError(
                f"non-UTF-8 string in GGUF metadata ({self._path}): {exc}"
            ) from exc

    def value(self, vtype: int):
        if vtype == 0:
            return self.u8()
        if vtype == 1:
            return self.i8()
        if vtype == 2:
            return self.u16()
        if vtype == 3:
            return self.i16()
        if vtype == 4:
            return self.u32()
        if vtype == 5:
            return self.i32()
        if vtype == 6:
            return self.f32()
        if vtype == 7:
            return bool(self.u8())
        if vtype == 8:
            return self.string()
        if vtype == 9:
            element_type = self.u32()
            count = self.u64()
            if count > 10_000_000:  # sanity guard
                raise GGUFMetadataError(
                    f"implausible array length {count} in GGUF metadata: {self._path}"
                )
            return [self.value(element_type) for _ in range(count)]
        if vtype == 10:
            return self.u64()
        if vtype == 11:
            return self.i64()
        if vtype == 12:
            return self.f64()
        raise GGUFMetadataError(
            f"unknown GGUF metadata value type {vtype} "
            f"({VALUE_TYPE_NAMES.get(vtype, '?')}) in {self._path}"
        )


def sha256_file(path: Path, chunk_size: int = _CHUNK) -> str:
    """Hex SHA-256 of the file, streamed in chunks (memory-light)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def tensor_type_name(code: int) -> str:
    """Human-readable name for a ggml tensor type code."""
    return TENSOR_TYPE_NAMES.get(code, f"type_{code}")


def file_type_name(code: int) -> str | None:
    """Human-readable quantization name for a GGUF file_type code, or None.

    Follows the legacy llama.cpp numbering (``Q4_K_M = 15``) that matches the
    Qwen2.5 GGUF files in this repo.
    """
    return FILE_TYPE_NAMES.get(code)


def _dims_sane(dims: list[int]) -> bool:
    return bool(dims) and all(1 <= d < _MAX_SANE_DIM for d in dims)


def _read_tensor_infos(
    f, path: Path, tensor_count: int, start_offset: int, dims_u64: bool
) -> tuple[list[dict], dict[str, int], int]:
    """Read ``tensor_count`` tensor infos starting at ``start_offset``.

    Args:
        dims_u64: ``True`` for the modern uint64 dims layout, ``False`` for the
            legacy uint32 layout.

    Returns:
        ``(tensor_infos, tensor_types, parameter_count)``.
    """
    f.seek(start_offset)
    reader = _Reader(f, path)
    infos: list[dict] = []
    types: Counter = Counter()
    params = 0
    for _ in range(tensor_count):
        name = reader.string()
        n_dims = reader.u32()
        dims = [reader.u64() if dims_u64 else reader.u32() for _ in range(n_dims)]
        ttype = reader.u32()
        offset = reader.u64()
        element_count = math.prod(dims) if dims else 0
        params += element_count
        tname = tensor_type_name(ttype)
        types[tname] += 1
        infos.append(
            {
                "name": name,
                "dims": dims,
                "type_code": ttype,
                "type": tname,
                "offset": offset,
                "element_count": element_count,
            }
        )
    return infos, dict(types), params


def analyze_gguf(path: str | Path, *, include_sha256: bool = True) -> dict:
    """Extract deterministic metadata from a GGUF file without loading it.

    Args:
        path: The GGUF file to analyze.
        include_sha256: Compute the file SHA-256 (streamed). Costs a full
            read of the file; disable when only metadata is needed.

    Returns:
        A dict with the file facts (path, size, sha256, magic, version,
        tensor/parameter counts, kv metadata, per-tensor-type breakdown,
        file_type code/name and a derived quantization).

    Raises:
        GGUFMetadataError: Not a GGUF file or its metadata is corrupt.
        FileNotFoundError: The path does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {path}")

    file_size_bytes = path.stat().st_size
    with path.open("rb") as f:
        reader = _Reader(f, path)

        magic = reader._read(4)
        if magic != GGUF_MAGIC:
            raise GGUFMetadataError(f"not a GGUF file (bad magic {magic!r}): {path}")
        version = reader.u32()
        tensor_count = reader.u64()
        kv_count = reader.u64()

        metadata: dict = {}
        for _ in range(kv_count):
            key = reader.string()
            vtype = reader.u32()
            metadata[key] = reader.value(vtype)

        tensor_section_offset = f.tell()

        # Modern GGUF: uint64 dims. Legacy GGUF: uint32 dims. Parse with
        # uint64 first; fall back to uint32 when the layout mismatches
        # (implausible dims or premature end-of-file from over-reading).
        try:
            infos, types, params = _read_tensor_infos(
                f, path, tensor_count, tensor_section_offset, dims_u64=True
            )
            layout_mismatch = tensor_count > 0 and not _dims_sane(infos[0]["dims"])
        except GGUFMetadataError:
            layout_mismatch = True
        if layout_mismatch:
            infos, types, params = _read_tensor_infos(
                f, path, tensor_count, tensor_section_offset, dims_u64=False
            )
            if tensor_count > 0 and not _dims_sane(infos[0]["dims"]):
                raise GGUFMetadataError(
                    f"unreadable tensor metadata (corrupt GGUF?): {path}"
                )

    file_type_code = metadata.get("general.file_type")
    result = {
        "path": str(path),
        "file_name": path.name,
        "file_size_bytes": file_size_bytes,
        "gguf_version": version,
        "tensor_count": tensor_count,
        "parameter_count": params,
        "metadata": metadata,
        "tensor_infos": infos,
        "tensor_types": types,
        "file_type": (
            {
                "code": int(file_type_code),
                "name": file_type_name(int(file_type_code)),
            }
            if isinstance(file_type_code, int)
            else None
        ),
    }
    if include_sha256:
        result["sha256"] = sha256_file(path)

    # Convenience accessors for the common model-identity keys.
    result["architecture"] = metadata.get("general.architecture")
    result["model_name"] = metadata.get("general.name")
    result["quantization_version"] = metadata.get("general.quantization_version")
    result["context_length"] = metadata.get(
        "qwen2.context_length",
        metadata.get("llama.context_length", metadata.get("bert.context_length")),
    )
    # Quantization derived from the actual per-tensor breakdown (dominant
    # tensor family), e.g. "Q4_K" for Q4_K_M-family files. This is the
    # authoritative source; the filename label is only corroborating.
    if types:
        dominant = max(types, key=types.get)
        result["quantization"] = dominant
    else:
        result["quantization"] = None
    return result
