"""Immutable QwenPI layouts and their per-rank device caches."""

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


_FALSE_ENV_VALUES = {"", "0", "false", "no", "off"}


def env_bool(name: str, default: bool = False) -> bool:
    """Read a shell-style boolean; unknown non-empty values remain truthy."""
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() not in _FALSE_ENV_VALUES


def metadata_cache_enabled() -> bool:
    return env_bool("STARVLA_METADATA_CACHE", default=True)


GridTHW = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class SampleLayout:
    """Minimal immutable metadata needed by one text/vision sample."""

    attention_mask_u8: bytes
    image_token_positions: tuple[int, ...]
    image_grids: tuple[GridTHW, ...]

    @property
    def seq_len(self) -> int:
        return len(self.attention_mask_u8)


@dataclass(frozen=True, slots=True)
class VisionLayout:
    """Derived batch-wide view of the per-sample image grids."""

    grids: tuple[GridTHW, ...]
    spatial_merge_size: int

    @property
    def split_sizes(self) -> tuple[int, ...]:
        merge_size = self.spatial_merge_size
        return tuple(
            t * (h // merge_size) * (w // merge_size) for t, h, w in self.grids
        )

    @property
    def max_seqlen(self) -> int:
        return max((h * w for _, h, w in self.grids), default=0)


@dataclass(frozen=True, slots=True)
class QwenBatchLayout:
    """Worker-produced descriptor containing only canonical layout facts."""

    samples: tuple[SampleLayout, ...]
    spatial_merge_size: int

    @property
    def batch_size(self) -> int:
        return len(self.samples)

    @property
    def seq_len(self) -> int:
        return self.samples[0].seq_len if self.samples else 0

    @property
    def vision(self) -> VisionLayout:
        return VisionLayout(
            grids=tuple(
                grid for sample in self.samples for grid in sample.image_grids
            ),
            spatial_merge_size=self.spatial_merge_size,
        )

    @property
    def flat_image_token_indices(self) -> tuple[int, ...]:
        seq_len = self.seq_len
        return tuple(
            sample_index * seq_len + position
            for sample_index, sample in enumerate(self.samples)
            for position in sample.image_token_positions
        )


def build_causal_mask_from_layout(
    sample: SampleLayout, device: torch.device | str
) -> torch.Tensor:
    """Build one sample's bool SDPA mask from its immutable CPU layout."""

    valid_keys = torch.tensor(
        tuple(sample.attention_mask_u8), dtype=torch.bool, device=device
    )
    positions = torch.arange(sample.seq_len, device=device)
    lower_triangle = positions[:, None] >= positions[None, :]
    return (lower_triangle & valid_keys[None, :]).unsqueeze(0).unsqueeze(0)


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _require_cpu_tensor(
    inputs: Mapping[str, torch.Tensor],
    name: str,
    *,
    dimensions: int,
    allow_bool: bool = False,
) -> torch.Tensor:
    tensor = inputs.get(name)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"qwen_inputs[{name!r}] must be a torch.Tensor")
    if tensor.device.type != "cpu":
        raise ValueError(f"qwen_inputs[{name!r}] must be on CPU, got {tensor.device}")
    if tensor.ndim != dimensions:
        raise ValueError(
            f"qwen_inputs[{name!r}] must be {dimensions}D, got {tuple(tensor.shape)}"
        )
    valid_dtypes = _INTEGER_DTYPES | ({torch.bool} if allow_bool else set())
    if tensor.dtype not in valid_dtypes:
        raise ValueError(f"qwen_inputs[{name!r}] has unsupported dtype {tensor.dtype}")
    return tensor


def _contiguous_run_lengths(positions: tuple[int, ...]) -> tuple[int, ...]:
    if not positions:
        return ()
    lengths: list[int] = []
    run_length = 1
    for previous, current in zip(positions, positions[1:]):
        if current == previous + 1:
            run_length += 1
        else:
            lengths.append(run_length)
            run_length = 1
    lengths.append(run_length)
    return tuple(lengths)


def build_qwen_batch_layout(
    qwen_inputs: Mapping[str, torch.Tensor],
    image_counts: Sequence[int],
    image_token_id: int,
    spatial_merge_size: int,
) -> QwenBatchLayout:
    """Build a validated image/text descriptor in a DataLoader worker."""
    input_ids = _require_cpu_tensor(qwen_inputs, "input_ids", dimensions=2)
    attention_mask = _require_cpu_tensor(
        qwen_inputs, "attention_mask", dimensions=2, allow_bool=True
    )
    mm_token_type_ids = _require_cpu_tensor(qwen_inputs, "mm_token_type_ids", dimensions=2)
    image_grid_thw = _require_cpu_tensor(qwen_inputs, "image_grid_thw", dimensions=2)
    if image_grid_thw.dtype != torch.long:
        raise ValueError(
            "qwen_inputs['image_grid_thw'] must have dtype torch.int64 for "
            f"cached reconstruction; got {image_grid_thw.dtype}"
        )

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

    raw_counts = image_counts.tolist() if isinstance(image_counts, torch.Tensor) else image_counts
    if len(raw_counts) != batch_size:
        raise ValueError(
            f"image_counts must have one value per sample: expected {batch_size}, "
            f"got {len(raw_counts)}"
        )
    try:
        image_token_id_int = operator.index(image_token_id)
        merge_size = operator.index(spatial_merge_size)
        counts = tuple(operator.index(value) for value in raw_counts)
    except TypeError as exc:
        raise ValueError("image counts, token id and merge size must be integers") from exc
    if isinstance(image_token_id, bool) or isinstance(spatial_merge_size, bool):
        raise ValueError("image_token_id and spatial_merge_size must be integers")
    if image_token_id_int < 0 or merge_size <= 0 or any(count < 0 for count in counts):
        raise ValueError("image counts/token id must be non-negative and merge size positive")

    image_offsets_list = [0]
    for count in counts:
        image_offsets_list.append(image_offsets_list[-1] + count)
    image_offsets = tuple(image_offsets_list)
    if image_offsets[-1] != image_grid_thw.shape[0]:
        raise ValueError(
            "sum(image_counts) must equal image_grid_thw rows: "
            f"got {image_offsets[-1]} and {image_grid_thw.shape[0]}"
        )

    grids = tuple(tuple(map(int, row)) for row in image_grid_thw.tolist())
    for grid_index, (t, h, w) in enumerate(grids):
        if t <= 0 or h <= 0 or w <= 0:
            raise ValueError(f"image_grid_thw[{grid_index}] must be positive")
        if h % merge_size or w % merge_size:
            raise ValueError(
                f"image_grid_thw[{grid_index}] must be divisible by merge size {merge_size}"
            )
    split_sizes = VisionLayout(grids, merge_size).split_sizes

    samples: list[SampleLayout] = []
    for sample_index, (ids_row, attention_row, mm_row) in enumerate(
        zip(input_ids.tolist(), attention_mask.tolist(), mm_token_type_ids.tolist())
    ):
        mask = tuple(map(int, attention_row))
        mm_types = tuple(map(int, mm_row))
        if set(mask) - {0, 1}:
            raise ValueError(f"attention_mask[{sample_index}] must contain only 0/1")
        if {kind for kind, valid in zip(mm_types, mask) if valid} - {0, 1}:
            raise ValueError("QwenPI metadata supports image/text tokens only")

        image_positions = tuple(
            position for position, token_id in enumerate(ids_row)
            if int(token_id) == image_token_id_int
        )
        masked_image_positions = tuple(
            position for position in image_positions if not mask[position]
        )
        if masked_image_positions:
            raise ValueError(
                f"sample {sample_index} has image tokens at masked positions "
                f"{masked_image_positions}"
            )
        mm_image_positions = tuple(
            position for position, (kind, valid) in enumerate(zip(mm_types, mask))
            if valid and kind == 1
        )
        if mm_image_positions != image_positions:
            raise ValueError(
                f"sample {sample_index} image token ids and modality ids disagree"
            )

        start, stop = image_offsets[sample_index : sample_index + 2]
        sample_grids = grids[start:stop]
        run_lengths = _contiguous_run_lengths(image_positions)
        expected_run_lengths = split_sizes[start:stop]
        if run_lengths != expected_run_lengths:
            raise ValueError(
                f"sample {sample_index} image-token runs have lengths {run_lengths}, "
                f"but its grids require {expected_run_lengths}"
            )

        samples.append(
            SampleLayout(
                attention_mask_u8=bytes(mask),
                image_token_positions=image_positions,
                image_grids=sample_grids,
            )
        )
    return QwenBatchLayout(tuple(samples), merge_size)


K = TypeVar("K")
V = TypeVar("V")


class BoundedLRU(Generic[K, V]):
    """A small bounded true-LRU cache."""

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

    def get(self, key: K, default: V | None = None) -> V | None:
        """Return a value and promote it to most-recently-used."""
        if key not in self._values:
            return default
        self._values.move_to_end(key)
        return self._values[key]

    def put(self, key: K, value: V) -> None:
        """Insert a value, evicting the least-recently-used item if needed."""

        self._values.pop(key, None)
        self._values[key] = value
        if len(self._values) > self.capacity:
            self._values.popitem(last=False)

    def get_or_create(self, key: K, factory: Callable[[], V]) -> V:
        """Return a promoted hit, or create and insert one value on a miss."""
        if key in self._values:
            return self.get(key)  # type: ignore[return-value]
        value = factory()
        self.put(key, value)
        return value

    def __len__(self) -> int:
        return len(self._values)


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
