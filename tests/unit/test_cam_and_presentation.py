# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for forward-only ViT CAM projection and the dashboard view-model builder."""

from __future__ import annotations

import torch
import torch.nn as nn

from taxon_vision.inference.cam import _project_vit_features
from taxon_vision.service.presentation import build_top_candidates

EMBED_DIM = 8
GRID = 14


class _VitStub(nn.Module):
    """Minimal backbone exposing ``num_prefix_tokens`` like timm Vision Transformers."""

    def __init__(self, num_prefix_tokens: int) -> None:
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens


def test_vit_projection_strips_cls_and_restores_grid() -> None:
    """(B, 1+196, D) tokens yield a (B, 14, 14) map computed only from the patch tokens."""
    tokens = torch.zeros(2, 1 + GRID * GRID, EMBED_DIM)
    tokens[:, 0, :] = 100.0  # class token must not influence the map
    tokens[:, 1:, 0] = 1.0
    weights = torch.zeros(EMBED_DIM)
    weights[0] = 2.0
    cam = _project_vit_features(tokens, _VitStub(num_prefix_tokens=1), weights)
    assert cam is not None
    assert cam.shape == (2, GRID, GRID)
    assert torch.allclose(cam, torch.full((2, GRID, GRID), 2.0))


def test_vit_projection_honours_register_tokens() -> None:
    """Additional register tokens declared via ``num_prefix_tokens`` are stripped as well."""
    prefix = 5
    tokens = torch.zeros(1, prefix + GRID * GRID, EMBED_DIM)
    tokens[:, :prefix, :] = 50.0
    tokens[:, prefix:, 1] = 3.0
    weights = torch.zeros(EMBED_DIM)
    weights[1] = 1.0
    cam = _project_vit_features(tokens, _VitStub(num_prefix_tokens=prefix), weights)
    assert cam is not None
    assert cam.shape == (1, GRID, GRID)
    assert torch.allclose(cam, torch.full((1, GRID, GRID), 3.0))


def test_vit_projection_applies_relu() -> None:
    """Negative class evidence is clipped to zero."""
    tokens = torch.zeros(1, 1 + 4, EMBED_DIM)
    tokens[:, 1:, 0] = -1.0
    weights = torch.zeros(EMBED_DIM)
    weights[0] = 1.0
    cam = _project_vit_features(tokens, _VitStub(num_prefix_tokens=1), weights)
    assert cam is not None
    assert torch.count_nonzero(cam) == 0


def test_vit_projection_rejects_non_square_patch_count() -> None:
    """A patch count that is not a perfect square yields None instead of a wrong reshape."""
    tokens = torch.zeros(1, 1 + 10, EMBED_DIM)
    assert _project_vit_features(tokens, _VitStub(num_prefix_tokens=1), torch.zeros(EMBED_DIM)) is None


def test_vit_projection_rejects_dimension_mismatch() -> None:
    """A head whose width differs from the token width cannot be projected."""
    tokens = torch.zeros(1, 1 + 4, EMBED_DIM)
    assert _project_vit_features(tokens, _VitStub(num_prefix_tokens=1), torch.zeros(EMBED_DIM + 1)) is None


def test_top_candidates_ranked_with_conformal_membership() -> None:
    """Candidates are sorted by probability, truncated to top-k and flagged by set membership."""
    catalog = [
        {"id": 11, "scientific_name": "Apis mellifera", "common_name": "Honey bee"},
        {"id": 22, "scientific_name": "Bombus terrestris", "common_name": "Buff-tailed bumblebee"},
        {"id": 33, "scientific_name": "Vespa crabro", "common_name": "European hornet"},
    ]
    result = build_top_candidates(
        probabilities=[0.1, 0.6, 0.3],
        catalog=catalog,
        conformal_set=["Bombus terrestris", "Vespa crabro"],
        top_k=2,
    )
    assert [c["taxon_id"] for c in result] == [22, 33]
    assert [c["confidence_pct"] for c in result] == [60.0, 30.0]
    assert all(c["in_conformal_set"] for c in result)


def test_top_candidates_handles_missing_ids_and_large_k() -> None:
    """Catalogue entries without an id map to 0 and top_k larger than K returns all taxa."""
    catalog = [{"scientific_name": "A a", "common_name": "A"}, {"scientific_name": "B b", "common_name": "B"}]
    result = build_top_candidates([0.4, 0.6], catalog, [], top_k=10)
    assert len(result) == 2
    assert result[0]["scientific_name"] == "B b"
    assert result[0]["taxon_id"] == 0
    assert not result[0]["in_conformal_set"]
