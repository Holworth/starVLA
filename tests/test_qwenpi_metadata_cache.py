from __future__ import annotations

import dataclasses
import pickle
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from starVLA.model.qwenpi_metadata import (
    BoundedLRU,
    MRoPEEntry,
    QwenMetadataCache,
    activate_qwen_metadata,
    build_qwen_batch_layout,
    env_bool,
    get_active_qwen_metadata,
)


IMAGE_TOKEN_ID = 99
SPATIAL_MERGE_SIZE = 2
SEQ_LEN = 18


def _nonuniform_qwen_inputs() -> tuple[dict[str, torch.Tensor], tuple[int, int]]:
    """A left-padded batch with one image in sample 0 and three in sample 1."""

    input_ids = torch.zeros((2, SEQ_LEN), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    mm_token_type_ids = torch.zeros_like(input_ids)

    # sample 0: 11 pads, 2 text, 2 image, 3 text
    input_ids[0, 11:] = torch.tensor(
        [10, 11, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 12, 13, 14]
    )
    attention_mask[0, 11:] = 1
    mm_token_type_ids[0, 13:15] = 1

    # sample 1: 4 pads, then text/image/text/image/text/image/text.  Keeping
    # images separated is important: Qwen consumes one grid per image run.
    input_ids[1, 4:] = torch.tensor(
        [
            20,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            21,
            22,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            23,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            24,
            25,
        ]
    )
    attention_mask[1, 4:] = 1
    mm_token_type_ids[1, [5, 6, 9, 10, 12, 13, 14, 15]] = 1

    # With merge size 2 these grids produce 2, 2, 2, and 4 LLM tokens.
    image_grid_thw = torch.tensor(
        [
            [1, 2, 4],
            [1, 4, 2],
            [2, 2, 2],
            [1, 4, 4],
        ],
        dtype=torch.long,
    )
    return (
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "image_grid_thw": image_grid_thw,
        },
        (1, 3),
    )


def _build_layout():
    qwen_inputs, image_counts = _nonuniform_qwen_inputs()
    layout = build_qwen_batch_layout(
        qwen_inputs,
        image_counts,
        IMAGE_TOKEN_ID,
        SPATIAL_MERGE_SIZE,
    )
    return qwen_inputs, layout


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("0", False),
        ("false", False),
        (" FALSE ", False),
        ("1", True),
        ("true", True),
    ],
)
def test_env_bool_understands_shell_boolean_values(
    monkeypatch: pytest.MonkeyPatch, raw_value: str, expected: bool
) -> None:
    monkeypatch.setenv("QWENPI_TEST_BOOL", raw_value)
    assert env_bool("QWENPI_TEST_BOOL", default=not expected) is expected


def test_env_bool_uses_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QWENPI_TEST_BOOL", raising=False)
    assert env_bool("QWENPI_TEST_BOOL", default=False) is False
    assert env_bool("QWENPI_TEST_BOOL", default=True) is True


def test_descriptor_tracks_nonuniform_images_and_left_padding() -> None:
    _, layout = _build_layout()

    assert layout.schema_version == 1
    assert (layout.batch_size, layout.seq_len) == (2, SEQ_LEN)
    assert layout.image_offsets == (0, 1, 4)
    assert layout.samples[0].image_grids == ((1, 2, 4),)
    assert layout.samples[1].image_grids == (
        (1, 4, 2),
        (2, 2, 2),
        (1, 4, 4),
    )
    assert layout.samples[0].image_token_positions == (13, 14)
    assert layout.samples[1].image_token_positions == (5, 6, 9, 10, 12, 13, 14, 15)
    assert layout.samples[0].attention_mask_u8 == bytes([0] * 11 + [1] * 7)
    assert layout.samples[1].attention_mask_u8 == bytes([0] * 4 + [1] * 14)
    assert layout.vision.split_sizes == (2, 2, 2, 4)
    assert layout.vision.max_seqlen == 16
    assert layout.flat_image_token_indices == (
        13,
        14,
        23,
        24,
        27,
        28,
        30,
        31,
        32,
        33,
    )


def _assert_contains_no_tensor(value: Any) -> None:
    assert not isinstance(value, torch.Tensor)
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            _assert_contains_no_tensor(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_contains_no_tensor(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_contains_no_tensor(key)
            _assert_contains_no_tensor(item)


def test_descriptor_is_hashable_picklable_and_contains_no_tensor() -> None:
    _, layout = _build_layout()

    restored = pickle.loads(pickle.dumps(layout))
    assert restored == layout
    assert hash(restored) == hash(layout)
    _assert_contains_no_tensor(restored)


def test_descriptor_rejects_grid_count_mismatch() -> None:
    qwen_inputs, _ = _nonuniform_qwen_inputs()

    with pytest.raises(ValueError, match=r"sum\(image_counts\)"):
        build_qwen_batch_layout(
            qwen_inputs,
            (1, 2),
            IMAGE_TOKEN_ID,
            SPATIAL_MERGE_SIZE,
        )


def test_descriptor_rejects_image_token_grid_mismatch() -> None:
    qwen_inputs, image_counts = _nonuniform_qwen_inputs()
    qwen_inputs["image_grid_thw"] = qwen_inputs["image_grid_thw"].clone()
    qwen_inputs["image_grid_thw"][0] = torch.tensor([1, 4, 4])

    with pytest.raises(ValueError, match="grids require"):
        build_qwen_batch_layout(
            qwen_inputs,
            image_counts,
            IMAGE_TOKEN_ID,
            SPATIAL_MERGE_SIZE,
        )


def test_descriptor_rejects_masked_image_placeholder() -> None:
    qwen_inputs, image_counts = _nonuniform_qwen_inputs()
    qwen_inputs["input_ids"] = qwen_inputs["input_ids"].clone()
    qwen_inputs["input_ids"][0, 0] = IMAGE_TOKEN_ID

    with pytest.raises(ValueError, match="masked positions"):
        build_qwen_batch_layout(
            qwen_inputs,
            image_counts,
            IMAGE_TOKEN_ID,
            SPATIAL_MERGE_SIZE,
        )


def test_descriptor_rejects_non_cpu_metadata_before_readback() -> None:
    qwen_inputs, image_counts = _nonuniform_qwen_inputs()
    qwen_inputs["input_ids"] = torch.empty(
        qwen_inputs["input_ids"].shape, dtype=torch.long, device="meta"
    )

    with pytest.raises(ValueError, match="must be on CPU"):
        build_qwen_batch_layout(
            qwen_inputs,
            image_counts,
            IMAGE_TOKEN_ID,
            SPATIAL_MERGE_SIZE,
        )


def test_bounded_lru_promotes_hits_and_evicts_the_true_lru() -> None:
    cache: BoundedLRU[str, int] = BoundedLRU(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.get("a") == 1  # "b" is now the least-recently-used key.
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.stats() == {
        "capacity": 2,
        "size": 2,
        "hits": 3,
        "misses": 1,
        "evictions": 1,
    }


def test_bounded_lru_get_or_create_stats_and_capacity_validation() -> None:
    cache: BoundedLRU[str, object] = BoundedLRU(capacity=1)
    value = object()

    assert cache.get_or_create("key", lambda: value) is value
    assert cache.get_or_create("key", lambda: pytest.fail("factory called on hit")) is value
    assert cache.stats() == {
        "capacity": 1,
        "size": 1,
        "hits": 1,
        "misses": 1,
        "evictions": 0,
    }
    for invalid_capacity in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            BoundedLRU(invalid_capacity)  # type: ignore[arg-type]


def test_active_metadata_context_is_nested_and_exception_safe() -> None:
    _, outer_layout = _build_layout()
    inner_inputs, _ = _nonuniform_qwen_inputs()
    inner_inputs["input_ids"] = inner_inputs["input_ids"].clone()
    inner_inputs["input_ids"][0, 11] = 777
    inner_layout = build_qwen_batch_layout(
        inner_inputs,
        (1, 3),
        IMAGE_TOKEN_ID,
        SPATIAL_MERGE_SIZE,
    )
    outer_cache = QwenMetadataCache()
    inner_cache = QwenMetadataCache()

    assert get_active_qwen_metadata() is None
    with activate_qwen_metadata(outer_layout, outer_cache) as outer:
        assert get_active_qwen_metadata() is outer
        with pytest.raises(RuntimeError, match="sentinel"):
            with activate_qwen_metadata(inner_layout, inner_cache) as inner:
                assert get_active_qwen_metadata() is inner
                raise RuntimeError("sentinel")
        assert get_active_qwen_metadata() is outer
    assert get_active_qwen_metadata() is None


class _FakeQwen35Model:
    """Only the two attributes used by the unbound Transformers RoPE helper."""

    config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=SPATIAL_MERGE_SIZE)
    )


def _qwen35_rope_index(
    qwen_inputs: dict[str, torch.Tensor],
    *,
    rows: tuple[int, ...],
    grids: tuple[tuple[int, int, int], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    modeling_qwen3_5 = pytest.importorskip(
        "transformers.models.qwen3_5.modeling_qwen3_5"
    )
    fake = _FakeQwen35Model()
    fake.get_vision_position_ids = modeling_qwen3_5.Qwen3_5Model.get_vision_position_ids.__get__(
        fake, _FakeQwen35Model
    )
    selected_rows = torch.tensor(rows, dtype=torch.long)
    return modeling_qwen3_5.Qwen3_5Model.get_rope_index(
        fake,
        input_ids=qwen_inputs["input_ids"].index_select(0, selected_rows),
        mm_token_type_ids=qwen_inputs["mm_token_type_ids"].index_select(
            0, selected_rows
        ),
        image_grid_thw=torch.tensor(grids, dtype=torch.long),
        attention_mask=qwen_inputs["attention_mask"].index_select(0, selected_rows),
    )


def test_qwen35_per_sample_mrope_cache_matches_full_batch_and_keeps_deltas() -> None:
    qwen_inputs, layout = _build_layout()
    full_positions, full_deltas = _qwen35_rope_index(
        qwen_inputs,
        rows=(0, 1),
        grids=layout.vision.grids,
    )

    cache = QwenMetadataCache(mrope_capacity=8)

    def compute_sample(sample_index: int) -> MRoPEEntry:
        positions, deltas = _qwen35_rope_index(
            qwen_inputs,
            rows=(sample_index,),
            grids=layout.samples[sample_index].image_grids,
        )
        return MRoPEEntry(position_ids=positions, rope_deltas=deltas)

    # Seed only sample 0, then assemble a mixed hit/miss batch.
    cache.mrope_samples.put(layout.samples[0], compute_sample(0))
    mixed_entries: list[MRoPEEntry] = []
    for sample_index, sample_layout in enumerate(layout.samples):
        mixed_entries.append(
            cache.mrope_samples.get_or_create(
                sample_layout, lambda i=sample_index: compute_sample(i)
            )
        )
    assert torch.equal(
        torch.cat([entry.position_ids for entry in mixed_entries], dim=1),
        full_positions,
    )
    assert torch.equal(
        torch.cat([entry.rope_deltas for entry in mixed_entries], dim=0),
        full_deltas,
    )
    assert full_deltas[0].item() != full_deltas[1].item()

    # A later batch may reorder samples after arbitrary model state was left
    # behind by another forward.  Per-sample entries must restore and reorder
    # rope_deltas together with position_ids, rather than reusing stale state.
    stale_model_delta = torch.full_like(full_deltas, 12345)
    reordered_entries = [
        cache.mrope_samples.get(layout.samples[1]),
        cache.mrope_samples.get(layout.samples[0]),
    ]
    assert all(entry is not None for entry in reordered_entries)
    reordered_positions = torch.cat(
        [entry.position_ids for entry in reordered_entries if entry is not None],
        dim=1,
    )
    reordered_deltas = torch.cat(
        [entry.rope_deltas for entry in reordered_entries if entry is not None],
        dim=0,
    )
    reordered_full_positions, reordered_full_deltas = _qwen35_rope_index(
        qwen_inputs,
        rows=(1, 0),
        grids=layout.samples[1].image_grids + layout.samples[0].image_grids,
    )
    assert not torch.equal(stale_model_delta, reordered_deltas)
    assert torch.equal(reordered_positions, reordered_full_positions)
    assert torch.equal(reordered_deltas, reordered_full_deltas)
    assert cache.mrope_samples.stats() == {
        "capacity": 8,
        "size": 2,
        "hits": 3,
        "misses": 1,
        "evictions": 0,
    }


@pytest.mark.parametrize(
    "attention_mask",
    [
        pytest.param(torch.ones((1, 6), dtype=torch.long), id="all-ones"),
        pytest.param(
            torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.long),
            id="left-padding",
        ),
        pytest.param(
            torch.tensor(
                [[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]],
                dtype=torch.long,
            ),
            id="mixed",
        ),
    ],
)
def test_cached_causal_mask_formula_matches_transformers_sdpa(
    attention_mask: torch.Tensor,
) -> None:
    masking_utils = pytest.importorskip("transformers.masking_utils")
    batch_size, seq_len = attention_mask.shape
    cache_position = torch.arange(seq_len)
    hf_mask = masking_utils.sdpa_mask(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=seq_len,
        attention_mask=attention_mask,
        allow_is_causal_skip=False,
    )

    query_index = torch.arange(seq_len).view(seq_len, 1)
    key_index = torch.arange(seq_len).view(1, seq_len)
    causal = key_index <= query_index
    cached_formula = causal.view(1, 1, seq_len, seq_len) & attention_mask.bool().view(
        batch_size, 1, 1, seq_len
    )

    assert hf_mask is not None
    # HF preserves the integral dtype of a 0/1 attention mask here; SDPA and
    # the cache use the equivalent boolean representation.
    assert cached_formula.dtype == torch.bool
    assert torch.equal(cached_formula, hf_mask.bool())
