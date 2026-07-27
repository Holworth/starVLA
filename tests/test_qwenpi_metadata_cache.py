from __future__ import annotations

import dataclasses
import pickle
from types import SimpleNamespace

import pytest
import torch

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI
from starVLA.model.qwenpi_metadata import (
    BoundedLRU,
    QwenMetadataCache,
    activate_qwen_metadata,
    build_causal_mask_from_layout,
    build_qwen_batch_layout,
    device_key,
    env_bool,
    get_active_qwen_metadata,
)


IMAGE_TOKEN_ID = 99
SPATIAL_MERGE_SIZE = 2
SEQ_LEN = 18


def _nonuniform_qwen_inputs() -> tuple[dict[str, torch.Tensor], tuple[int, int]]:
    input_ids = torch.zeros((2, SEQ_LEN), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    mm_token_type_ids = torch.zeros_like(input_ids)
    input_ids[0, 11:] = torch.tensor(
        [10, 11, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 12, 13, 14]
    )
    attention_mask[0, 11:] = 1
    mm_token_type_ids[0, 13:15] = 1
    input_ids[1, 4:] = torch.tensor(
        [20, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 21, 22, IMAGE_TOKEN_ID,
         IMAGE_TOKEN_ID, 23, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID,
         IMAGE_TOKEN_ID, 24, 25]
    )
    attention_mask[1, 4:] = 1
    mm_token_type_ids[1, [5, 6, 9, 10, 12, 13, 14, 15]] = 1
    image_grid_thw = torch.tensor(
        [[1, 2, 4], [1, 4, 2], [2, 2, 2], [1, 4, 4]],
        dtype=torch.long,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "image_grid_thw": image_grid_thw,
    }, (1, 3)


def _clone_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in inputs.items()}


def _select_inputs(
    inputs: dict[str, torch.Tensor],
    sample_rows: tuple[int, ...],
    grid_rows: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    sample_index = torch.tensor(sample_rows, dtype=torch.long)
    grid_index = torch.tensor(grid_rows, dtype=torch.long)
    return {
        name: value.index_select(
            0, grid_index if name == "image_grid_thw" else sample_index
        )
        for name, value in inputs.items()
    }


def _build_layout(
    inputs: dict[str, torch.Tensor], image_counts: tuple[int, ...]
):
    return build_qwen_batch_layout(
        inputs,
        image_counts=image_counts,
        image_token_id=IMAGE_TOKEN_ID,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
    )


@pytest.mark.parametrize(
    ("raw_value", "default", "expected"),
    [(None, False, False), (None, True, True), ("0", True, False),
     ("false", True, False), (" FALSE ", True, False)],
)
def test_env_bool(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str | None,
    default: bool,
    expected: bool,
) -> None:
    if raw_value is None:
        monkeypatch.delenv("STARVLA_FUSED_TEXT_STACK", raising=False)
    else:
        monkeypatch.setenv("STARVLA_FUSED_TEXT_STACK", raw_value)
    assert env_bool("STARVLA_FUSED_TEXT_STACK", default=default) is expected


def test_descriptor_tracks_nonuniform_images_left_padding_and_flat_indices() -> None:
    inputs, image_counts = _nonuniform_qwen_inputs()
    layout = _build_layout(inputs, image_counts)

    assert (layout.batch_size, layout.seq_len) == (2, SEQ_LEN)
    assert layout.spatial_merge_size == SPATIAL_MERGE_SIZE
    assert layout.samples[0].image_grids == ((1, 2, 4),)
    assert layout.samples[1].image_grids == ((1, 4, 2), (2, 2, 2), (1, 4, 4))
    assert layout.samples[0].image_token_positions == (13, 14)
    assert layout.samples[1].image_token_positions == (5, 6, 9, 10, 12, 13, 14, 15)
    assert layout.samples[0].attention_mask_u8 == bytes([0] * 11 + [1] * 7)
    assert layout.samples[1].attention_mask_u8 == bytes([0] * 4 + [1] * 14)
    assert layout.vision.split_sizes == (2, 2, 2, 4)
    assert layout.vision.max_seqlen == 16
    assert layout.flat_image_token_indices == (13, 14, 23, 24, 27, 28, 30, 31, 32, 33)


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if dataclasses.is_dataclass(value):
        value = tuple(getattr(value, field.name) for field in dataclasses.fields(value))
    return isinstance(value, (tuple, list)) and any(map(_contains_tensor, value))


def test_descriptor_is_hashable_picklable_and_contains_no_tensor() -> None:
    inputs, image_counts = _nonuniform_qwen_inputs()
    layout = _build_layout(inputs, image_counts)

    restored = pickle.loads(pickle.dumps(layout))
    assert restored == layout
    assert hash(restored) == hash(layout)
    assert not _contains_tensor(restored)


@pytest.mark.parametrize(
    ("invalid_case", "error"),
    [
        ("grid-count", r"sum\(image_counts\)"),
        ("token-grid-count", "grids require"),
        ("per-image-run-length", "runs have lengths"),
        ("masked-image-token", "masked positions"),
        ("image-grid-dtype", "dtype torch.int64"),
        ("non-cpu", "must be on CPU"),
    ],
)
def test_descriptor_rejects_invalid_layouts(
    invalid_case: str, error: str
) -> None:
    inputs, image_counts = _nonuniform_qwen_inputs()
    inputs = _clone_inputs(inputs)
    if invalid_case == "grid-count":
        image_counts = (1, 2)
    elif invalid_case == "token-grid-count":
        inputs["image_grid_thw"][0] = torch.tensor([1, 4, 4])
    elif invalid_case == "per-image-run-length":
        inputs["input_ids"][1, 7] = IMAGE_TOKEN_ID
        inputs["mm_token_type_ids"][1, 7] = 1
        inputs["input_ids"][1, 9] = 21
        inputs["mm_token_type_ids"][1, 9] = 0
    elif invalid_case == "masked-image-token":
        inputs["input_ids"][0, 0] = IMAGE_TOKEN_ID
    elif invalid_case == "image-grid-dtype":
        inputs["image_grid_thw"] = inputs["image_grid_thw"].to(torch.int32)
    elif invalid_case == "non-cpu":
        inputs["input_ids"] = torch.empty(
            inputs["input_ids"].shape, dtype=torch.long, device="meta"
        )
    with pytest.raises(ValueError, match=error):
        _build_layout(inputs, image_counts)


def test_collator_uses_stock_path_for_non_int64_image_grid(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NO_ALBUMENTATIONS_UPDATE", "1")
    from starVLA.dataloader.lerobot_datasets import QwenVLPreprocessCollate

    inputs, image_counts = _nonuniform_qwen_inputs()
    inputs["image_grid_thw"] = inputs["image_grid_thw"].to(torch.int32)
    collator = QwenVLPreprocessCollate(processor=SimpleNamespace())
    with caplog.at_level("WARNING"):
        assert collator._build_metadata(inputs, image_counts) is None
    assert "cached grid reconstruction requires torch.int64" in caplog.text


def test_bounded_lru_promotes_hits_and_evicts_the_true_lru() -> None:
    cache: BoundedLRU[str, int] = BoundedLRU(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get_or_create("a", lambda: pytest.fail("unexpected factory")) == 1
    assert cache.get_or_create("d", lambda: 4) == 4
    assert cache.get("c") is None
    assert (cache.get("a"), cache.get("d")) == (1, 4)
    assert len(cache) == 2


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_bounded_lru_rejects_invalid_capacity(capacity: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        BoundedLRU(capacity)  # type: ignore[arg-type]


def test_active_metadata_context_is_nested_and_exception_safe() -> None:
    inputs, image_counts = _nonuniform_qwen_inputs()
    layout = _build_layout(inputs, image_counts)
    outer_cache = QwenMetadataCache()
    inner_cache = QwenMetadataCache()
    assert get_active_qwen_metadata() is None
    with activate_qwen_metadata(layout, outer_cache) as outer:
        assert get_active_qwen_metadata() is outer
        with pytest.raises(RuntimeError, match="sentinel"):
            with activate_qwen_metadata(layout, inner_cache) as inner:
                assert get_active_qwen_metadata() is inner
                raise RuntimeError("sentinel")
        assert get_active_qwen_metadata() is outer
    assert get_active_qwen_metadata() is None


def test_qwenpi_rejects_layout_with_model_merge_size_mismatch() -> None:
    inputs, image_counts = _nonuniform_qwen_inputs()
    layout = _build_layout(inputs, image_counts)
    model = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(spatial_merge_size=4)
            )
        )
    )
    framework = SimpleNamespace(qwen_vl_interface=SimpleNamespace(model=model))
    with pytest.raises(RuntimeError, match="spatial_merge_size does not match"):
        Qwen_PI._validate_metadata_layout(framework, layout, inputs)


class _FakeQwen35Model:
    config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=SPATIAL_MERGE_SIZE)
    )

    def __init__(self) -> None:
        model_type = pytest.importorskip(
            "transformers.models.qwen3_5.modeling_qwen3_5"
        ).Qwen3_5Model
        for name in ("get_vision_position_ids", "get_rope_index"):
            setattr(self, name, getattr(model_type, name).__get__(self, type(self)))
        self.rope_deltas: torch.Tensor | None = None


class _FakeQwenPI:
    _build_mrope_entry = Qwen_PI._build_mrope_entry

    def __init__(self, inner: _FakeQwen35Model) -> None:
        self._metadata_cache = QwenMetadataCache(mrope_capacity=8)
        self.qwen_vl_interface = SimpleNamespace(model=SimpleNamespace(model=inner))


def _inject_mrope(
    framework: _FakeQwenPI,
    inner: _FakeQwen35Model,
    inputs: dict[str, torch.Tensor],
    image_counts: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, object]:
    layout = _build_layout(inputs, image_counts)
    qwen_inputs = _clone_inputs(inputs)
    reference_positions, reference_deltas = inner.get_rope_index(
        input_ids=inputs["input_ids"],
        mm_token_type_ids=inputs["mm_token_type_ids"],
        image_grid_thw=inputs["image_grid_thw"],
        attention_mask=inputs["attention_mask"],
    )
    inner.rope_deltas = torch.full_like(reference_deltas, 12345)
    Qwen_PI._inject_cached_position_ids(framework, qwen_inputs, layout)
    assert torch.equal(qwen_inputs["position_ids"], reference_positions)
    assert inner.rope_deltas is not None
    assert torch.equal(inner.rope_deltas, reference_deltas)
    return reference_positions, reference_deltas, layout


def test_qwen35_mrope_inject_handles_mixed_hits_and_reordered_batches() -> None:
    base_inputs, _ = _nonuniform_qwen_inputs()
    inner = _FakeQwen35Model()
    framework = _FakeQwenPI(inner)
    sample_zero = _select_inputs(base_inputs, (0,), (0,))
    _inject_mrope(framework, inner, sample_zero, (1,))
    assert len(framework._metadata_cache.mrope_samples) == 1
    _, full_deltas, full_layout = _inject_mrope(framework, inner, base_inputs, (1, 3))
    assert full_deltas[0].item() != full_deltas[1].item()
    assert len(framework._metadata_cache.mrope_samples) == 2
    reordered_inputs = _select_inputs(
        base_inputs, (1, 0), (1, 2, 3, 0)
    )
    _, reordered_deltas, reordered_layout = _inject_mrope(
        framework, inner, reordered_inputs, (3, 1)
    )
    assert reordered_layout.samples == tuple(reversed(full_layout.samples))
    assert torch.equal(reordered_deltas, full_deltas.flip(0))
    assert len(framework._metadata_cache.mrope_samples) == 2
    _, restored_deltas, _ = _inject_mrope(framework, inner, base_inputs, (1, 3))
    assert torch.equal(restored_deltas, full_deltas)


def _inputs_with_attention(
    base_inputs: dict[str, torch.Tensor], mode: str
) -> dict[str, torch.Tensor]:
    inputs = _clone_inputs(base_inputs)
    if mode == "all-visible":
        inputs["attention_mask"].fill_(1)
    elif mode == "mixed":
        inputs["attention_mask"][0].fill_(1)
    return inputs


def _cached_causal_mask(layout, cache: QwenMetadataCache) -> torch.Tensor:
    dev_key = device_key("cpu")
    return torch.cat(
        [
            cache.causal_masks.get_or_create(
                (sample.attention_mask_u8, dev_key),
                lambda sample=sample: build_causal_mask_from_layout(
                    sample, device="cpu"
                ),
            )
            for sample in layout.samples
        ],
        dim=0,
    )


def _hf_explicit_causal_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    masking_utils = pytest.importorskip("transformers.masking_utils")
    batch_size, seq_len = attention_mask.shape
    reference = masking_utils.sdpa_mask(
        batch_size=batch_size,
        cache_position=torch.arange(seq_len),
        kv_length=seq_len,
        attention_mask=attention_mask,
        allow_is_causal_skip=False,
    )
    assert reference is not None
    return reference.bool()


def test_production_causal_mask_handles_visible_padded_mixed_and_key_order() -> None:
    base_inputs, _ = _nonuniform_qwen_inputs()
    mixed_inputs = _inputs_with_attention(base_inputs, "mixed")
    reordered_mixed_inputs = _select_inputs(
        mixed_inputs, (1, 0), (1, 2, 3, 0)
    )
    cases = (
        ("all-visible", _inputs_with_attention(base_inputs, "all-visible"), (1, 3)),
        ("left-padded", base_inputs, (1, 3)),
        ("mixed", mixed_inputs, (1, 3)),
        ("mixed-reordered", reordered_mixed_inputs, (3, 1)),
    )
    cache = QwenMetadataCache(causal_masks_capacity=8)
    actual_masks = {}
    for name, inputs, image_counts in cases:
        layout = _build_layout(inputs, image_counts)
        actual = _cached_causal_mask(layout, cache)
        reference = _hf_explicit_causal_mask(inputs["attention_mask"])
        assert actual.dtype == torch.bool, name
        assert torch.equal(actual, reference), name
        actual_masks[name] = actual
    assert torch.equal(
        actual_masks["mixed-reordered"], actual_masks["mixed"].flip(0)
    )
    assert len(cache.causal_masks) == 3
