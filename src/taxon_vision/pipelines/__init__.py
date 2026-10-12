# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Automated pipeline workflows for data extraction and model training."""

from __future__ import annotations

from taxon_vision.pipelines.extract_features import (
    FeatureExtractionConfig,
    build_taxon_index_map,
    create_stratified_embedding_split,
    extract_batch_representations,
    get_evaluation_transform,
    run_feature_extraction,
    stream_and_extract_representations,
)

__all__ = [
    "FeatureExtractionConfig",
    "build_taxon_index_map",
    "create_stratified_embedding_split",
    "extract_batch_representations",
    "get_evaluation_transform",
    "run_feature_extraction",
    "stream_and_extract_representations",
]
