"""CPU layout descriptors and bounded device caches for QwenPI metadata.

The descriptor types in this module deliberately contain only immutable Python
values.  They can therefore be built in dataloader workers, pickled by
``DataLoader``, and used as cache keys without touching a CUDA tensor.

The cache value types may contain tensors, but this module does not know about
Transformers model classes.  Keeping that boundary here prevents the metadata
cache from becoming coupled to a particular Hugging Face implementation.
"""

from __future__ import annotations

import os
import operator
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch


_FALSE_ENV_VALUES = frozenset({"", "0", "false", "no", "off"})
_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def env_bool(name: str, default: bool = False) -> bool:
    """Read a conventional boolean environment variable.

    Values are case-insensitive and surrounding whitespace is ignored.
    Unknown non-empty values retain the long-standing shell convention of
    being truthy.  This keeps values such as ``enabled`` backward compatible
    while, importantly, treating ``0`` and ``false`` as disabled.
    """

    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _FALSE_ENV_VALUES:
        return False
    if normalized in _TRUE_ENV_VALUES:
        return True
    return True


def metadata_cache_enabled() -> bool:
    """Whether the structured QwenPI metadata cache is enabled."""

    return env_bool("STARVLA_METADATA_CACHE", default=True)


GridTHW = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class SampleLayout:
    """Immutable metadata needed by one text/vision sample."""

    seq_len: int
    attention_mask_u8: bytes
    mm_token_types_u8: bytes
    image_grids: tuple[GridTHW, ...]
    image_token_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class VisionLayout:
    """Batch-wide values derived solely from ``image_grid_thw``."""

    grids: tuple[GridTHW, ...]
    spatial_merge_size: int
    split_sizes: tuple[int, ...]
    max_seqlen: int


@dataclass(frozen=True, slots=True)
class QwenBatchLayout:
    """Worker-produced, hashable layout descriptor for one Qwen batch."""

    schema_version: int
    batch_size: int
    seq_len: int
    image_offsets: tuple[int, ...]
    samples: tuple[SampleLayout, ...]
    vision: VisionLayout
    flat_image_token_indices: tuple[int, ...]


_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


def _require_cpu_tensor(
    inputs: Mapping[str, torch.Tensor],
    name: str,
    *,
    dimensions: int,
    allow_bool: bool = False,
) -> torch.Tensor:
    try:
        tensor = inputs[name]
    except KeyError as exc:
        raise ValueError(f"qwen_inputs is missing required tensor {name!r}") from exc
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"qwen_inputs[{name!r}] must be a torch.Tensor")
    if tensor.device.type != "cpu":
        raise ValueError(
            f"qwen_inputs[{name!r}] must be on CPU when its layout is built; "
            f"got device {tensor.device}"
        )
    if tensor.ndim != dimensions:
        raise ValueError(
            f"qwen_inputs[{name!r}] must have {dimensions} dimensions; "
            f"got shape {tuple(tensor.shape)}"
        )
    valid_dtypes = _INTEGER_DTYPES | ({torch.bool} if allow_bool else set())
    if tensor.dtype not in valid_dtypes:
        raise ValueError(
            f"qwen_inputs[{name!r}] must have an integer"
            f"{' or bool' if allow_bool else ''} dtype; got {tensor.dtype}"
        )
    return tensor


def _as_nonnegative_count(value: object, index: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"image_counts[{index}] must be an integer, not bool")
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"image_counts[{index}] must be an integer; got {value!r}") from exc
    if count < 0:
        raise ValueError(f"image_counts[{index}] must be non-negative; got {count}")
    return count


def _count_contiguous_runs(positions: tuple[int, ...]) -> int:
    if not positions:
        return 0
    return 1 + sum(current != previous + 1 for previous, current in zip(positions, positions[1:]))


def build_qwen_batch_layout(
    qwen_inputs: Mapping[str, torch.Tensor],
    image_counts: Sequence[int],
    image_token_id: int,
    spatial_merge_size: int,
) -> QwenBatchLayout:
    """Build and validate a Qwen image/text batch descriptor on the CPU.

    The global ``image_grid_thw`` tensor is consumed in sample order using
    explicit prefix offsets.  This is intentionally not inferred as
    ``num_images // batch_size`` because Qwen batches may contain a different
    number of images per sample.
    """

    input_ids = _require_cpu_tensor(qwen_inputs, "input_ids", dimensions=2)
    attention_mask = _require_cpu_tensor(
        qwen_inputs, "attention_mask", dimensions=2, allow_bool=True
    )
    mm_token_type_ids = _require_cpu_tensor(qwen_inputs, "mm_token_type_ids", dimensions=2)
    image_grid_thw = _require_cpu_tensor(qwen_inputs, "image_grid_thw", dimensions=2)

    batch_size, seq_len = input_ids.shape
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError(
            f"input_ids must have positive batch and sequence dimensions; got {(batch_size, seq_len)}"
        )
    expected_shape = (batch_size, seq_len)
    if tuple(attention_mask.shape) != expected_shape:
        raise ValueError(
            "attention_mask shape must match input_ids: "
            f"expected {expected_shape}, got {tuple(attention_mask.shape)}"
        )
    if tuple(mm_token_type_ids.shape) != expected_shape:
        raise ValueError(
            "mm_token_type_ids shape must match input_ids: "
            f"expected {expected_shape}, got {tuple(mm_token_type_ids.shape)}"
        )
    if image_grid_thw.shape[1] != 3:
        raise ValueError(
            "image_grid_thw must have shape [num_images, 3]; "
            f"got {tuple(image_grid_thw.shape)}"
        )

    if isinstance(image_counts, torch.Tensor):
        if image_counts.device.type != "cpu":
            raise ValueError("image_counts tensor must be on CPU")
        if image_counts.ndim != 1:
            raise ValueError("image_counts tensor must be one-dimensional")
        raw_image_counts: Sequence[int] = image_counts.tolist()
    else:
        raw_image_counts = image_counts
    if len(raw_image_counts) != batch_size:
        raise ValueError(
            f"image_counts must have one value per sample: expected {batch_size}, "
            f"got {len(raw_image_counts)}"
        )
    counts = tuple(_as_nonnegative_count(value, i) for i, value in enumerate(raw_image_counts))

    if isinstance(image_token_id, bool):
        raise ValueError("image_token_id must be an integer, not bool")
    if isinstance(spatial_merge_size, bool):
        raise ValueError("spatial_merge_size must be an integer, not bool")
    try:
        image_token_id_int = operator.index(image_token_id)
        merge_size = operator.index(spatial_merge_size)
    except TypeError as exc:
        raise ValueError("image_token_id and spatial_merge_size must be integers") from exc
    if merge_size <= 0:
        raise ValueError(f"spatial_merge_size must be positive; got {merge_size}")

    image_offsets_list = [0]
    for count in counts:
        image_offsets_list.append(image_offsets_list[-1] + count)
    image_offsets = tuple(image_offsets_list)
    num_images = image_grid_thw.shape[0]
    if image_offsets[-1] != num_images:
        raise ValueError(
            "sum(image_counts) must equal image_grid_thw rows: "
            f"got {image_offsets[-1]} and {num_images}"
        )

    grids_list: list[GridTHW] = []
    split_sizes_list: list[int] = []
    for grid_index, raw_grid in enumerate(image_grid_thw.tolist()):
        t, h, w = (int(value) for value in raw_grid)
        if t <= 0 or h <= 0 or w <= 0:
            raise ValueError(
                f"image_grid_thw[{grid_index}] must contain positive values; "
                f"got {(t, h, w)}"
            )
        if h % merge_size or w % merge_size:
            raise ValueError(
                f"image_grid_thw[{grid_index}] spatial dimensions {(h, w)} must "
                f"be divisible by spatial_merge_size={merge_size}"
            )
        grids_list.append((t, h, w))
        split_sizes_list.append(t * (h // merge_size) * (w // merge_size))
    grids = tuple(grids_list)
    split_sizes = tuple(split_sizes_list)

    samples: list[SampleLayout] = []
    flat_image_token_indices: list[int] = []
    input_rows = input_ids.tolist()
    attention_rows = attention_mask.tolist()
    mm_rows = mm_token_type_ids.tolist()

    for sample_index, (ids_row, attention_row, mm_row) in enumerate(
        zip(input_rows, attention_rows, mm_rows)
    ):
        attention_values = tuple(int(value) for value in attention_row)
        invalid_attention_values = {value for value in attention_values if value not in (0, 1)}
        if invalid_attention_values:
            raise ValueError(
                f"attention_mask[{sample_index}] must contain only 0/1 values; "
                f"got {sorted(invalid_attention_values)}"
            )

        mm_values = tuple(int(value) for value in mm_row)
        invalid_mm_values = {value for value in mm_values if not 0 <= value <= 255}
        if invalid_mm_values:
            raise ValueError(
                f"mm_token_type_ids[{sample_index}] values must fit in uint8; "
                f"got {sorted(invalid_mm_values)}"
            )
        unsupported_modalities = {
            token_type
            for token_type, is_valid in zip(mm_values, attention_values)
            if is_valid and token_type not in (0, 1)
        }
        if unsupported_modalities:
            raise ValueError(
                "the QwenPI metadata descriptor currently supports text/image "
                f"tokens only; sample {sample_index} contains modality ids "
                f"{sorted(unsupported_modalities)}"
            )

        valid_image_positions = tuple(
            position
            for position, (token_id, is_valid) in enumerate(zip(ids_row, attention_values))
            if is_valid and int(token_id) == image_token_id_int
        )
        masked_image_positions = tuple(
            position
            for position, (token_id, is_valid) in enumerate(zip(ids_row, attention_values))
            if not is_valid and int(token_id) == image_token_id_int
        )
        if masked_image_positions:
            raise ValueError(
                f"sample {sample_index} contains image placeholder tokens at "
                f"masked positions {masked_image_positions}; the image merge "
                "consumes every placeholder regardless of attention padding"
            )
        valid_mm_image_positions = tuple(
            position
            for position, (token_type, is_valid) in enumerate(zip(mm_values, attention_values))
            if is_valid and token_type == 1
        )
        if valid_mm_image_positions != valid_image_positions:
            raise ValueError(
                f"sample {sample_index} has inconsistent image-token metadata: "
                f"valid mm_token_type_ids==1 positions {valid_mm_image_positions} do not equal "
                f"valid input_ids==image_token_id positions {valid_image_positions}"
            )

        image_count = counts[sample_index]
        run_count = _count_contiguous_runs(valid_image_positions)
        if run_count != image_count:
            raise ValueError(
                f"sample {sample_index} has {run_count} contiguous image-token runs "
                f"but image_counts[{sample_index}] is {image_count}"
            )

        grid_start = image_offsets[sample_index]
        grid_stop = image_offsets[sample_index + 1]
        sample_grids = grids[grid_start:grid_stop]
        expected_image_tokens = sum(split_sizes[grid_start:grid_stop])
        if len(valid_image_positions) != expected_image_tokens:
            raise ValueError(
                f"sample {sample_index} has {len(valid_image_positions)} valid image "
                f"tokens, but its grids require {expected_image_tokens}"
            )

        samples.append(
            SampleLayout(
                seq_len=seq_len,
                attention_mask_u8=bytes(attention_values),
                mm_token_types_u8=bytes(mm_values),
                image_grids=sample_grids,
                image_token_positions=valid_image_positions,
            )
        )
        flat_image_token_indices.extend(
            sample_index * seq_len + position for position in valid_image_positions
        )

    max_seqlen = max((h * w for _, h, w in grids), default=0)
    return QwenBatchLayout(
        schema_version=1,
        batch_size=batch_size,
        seq_len=seq_len,
        image_offsets=image_offsets,
        samples=tuple(samples),
        vision=VisionLayout(
            grids=grids,
            spatial_merge_size=merge_size,
            split_sizes=split_sizes,
            max_seqlen=max_seqlen,
        ),
        flat_image_token_indices=tuple(flat_image_token_indices),
    )


K = TypeVar("K")
V = TypeVar("V")
_MISSING = object()


class BoundedLRU(Generic[K, V]):
    """A small true-LRU cache with explicit hit/miss/eviction counters."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool):
            raise ValueError(f"capacity must be a positive integer; got {capacity!r}")
        try:
            normalized_capacity = operator.index(capacity)
        except TypeError as exc:
            raise ValueError(f"capacity must be a positive integer; got {capacity!r}") from exc
        if normalized_capacity <= 0:
            raise ValueError(f"capacity must be a positive integer; got {capacity!r}")
        self.capacity = normalized_capacity
        self._values: OrderedDict[K, V] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: K, default: V | None = None) -> V | None:
        """Return a value and promote it to most-recently-used."""

        value = self._values.get(key, _MISSING)
        if value is _MISSING:
            self.misses += 1
            return default
        self.hits += 1
        self._values.move_to_end(key)
        return value  # type: ignore[return-value]

    def put(self, key: K, value: V) -> None:
        """Insert a value, evicting the least-recently-used item if needed."""

        if key in self._values:
            self._values[key] = value
            self._values.move_to_end(key)
            return
        self._values[key] = value
        if len(self._values) > self.capacity:
            self._values.popitem(last=False)
            self.evictions += 1

    def get_or_create(self, key: K, factory: Callable[[], V]) -> V:
        """Return a promoted hit, or create and insert one value on a miss."""

        value = self._values.get(key, _MISSING)
        if value is not _MISSING:
            self.hits += 1
            self._values.move_to_end(key)
            return value  # type: ignore[return-value]
        self.misses += 1
        created = factory()
        self.put(key, created)
        return created

    def __len__(self) -> int:
        return len(self._values)

    def stats(self) -> dict[str, int]:
        """Return a stable snapshot suitable for logging."""

        return {
            "capacity": self.capacity,
            "size": len(self),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


@dataclass(frozen=True, slots=True)
class MRoPEEntry:
    position_ids: torch.Tensor
    rope_deltas: torch.Tensor


@dataclass(frozen=True, slots=True)
class VisionEntry:
    idx: torch.Tensor
    weights: torch.Tensor
    permutation: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int


class QwenMetadataCache:
    """Per-model/per-rank owner of all GPU-resident metadata caches."""

    def __init__(
        self,
        *,
        mrope_capacity: int = 4096,
        vision_capacity: int = 16,
        image_indices_capacity: int = 64,
        grid_tensors_capacity: int = 16,
        causal_masks_capacity: int = 256,
    ) -> None:
        self.mrope_samples: BoundedLRU[object, MRoPEEntry] = BoundedLRU(mrope_capacity)
        self.vision_batches: BoundedLRU[object, VisionEntry] = BoundedLRU(vision_capacity)
        self.image_indices: BoundedLRU[object, torch.Tensor] = BoundedLRU(image_indices_capacity)
        self.grid_tensors: BoundedLRU[object, torch.Tensor] = BoundedLRU(grid_tensors_capacity)
        self.causal_masks: BoundedLRU[object, torch.Tensor] = BoundedLRU(causal_masks_capacity)


def device_key(value: torch.Tensor | torch.device | str) -> tuple[str, int | None]:
    """Return a hashable device component for keys containing GPU tensors."""

    device = value.device if isinstance(value, torch.Tensor) else torch.device(value)
    return device.type, device.index


@dataclass(frozen=True, slots=True)
class ActiveQwenMetadata:
    """Metadata state visible to patched Qwen components during one forward."""

    layout: QwenBatchLayout
    cache: QwenMetadataCache


_ACTIVE_QWEN_METADATA: ContextVar[ActiveQwenMetadata | None] = ContextVar(
    "active_qwen_metadata", default=None
)


def get_active_qwen_metadata() -> ActiveQwenMetadata | None:
    """Return the innermost active Qwen metadata context, if one exists."""

    return _ACTIVE_QWEN_METADATA.get()


@contextmanager
def activate_qwen_metadata(
    layout: QwenBatchLayout, cache: QwenMetadataCache
) -> Iterator[ActiveQwenMetadata]:
    """Activate metadata for a forward; nested and exceptional exits are safe."""

    active = ActiveQwenMetadata(layout=layout, cache=cache)
    token = _ACTIVE_QWEN_METADATA.set(active)
    try:
        yield active
    finally:
        _ACTIVE_QWEN_METADATA.reset(token)
