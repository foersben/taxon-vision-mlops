# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Inference service module encapsulating model loading, prediction, and uncertainty."""

import io
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms as T
import yaml
from PIL import Image, UnidentifiedImageError

from taxon_vision.config import get_settings
from taxon_vision.inference.onnx_engine import ONNXInferenceEngine
from taxon_vision.models.factory import create_feature_extractor, get_feature_dimension
from taxon_vision.models.head import TaxonClassifier, load_head_checkpoint
from taxon_vision.service.schemas import ConformalRoutingTier
from taxon_vision.uncertainty.conformal import ConformalPredictionEngine
from taxon_vision.uncertainty.ood_detector import EnergyOODDetector

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

EVAL_RESIZE_RATIO = 256 / 224
"""Shorter-edge resize factor preceding the centre crop; identical to the feature-extraction transform."""


class ReadinessStatus(NamedTuple):
    """Typed container for Kubernetes readiness probe check outcomes."""

    ready: bool
    checks: dict[str, str]


class ModelForwardOutputs(NamedTuple):
    """Typed container for model forward pass logits and penultimate embeddings."""

    logits: npt.NDArray[np.float32]
    embeddings: npt.NDArray[np.float32] | None


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded into an RGB image."""


class ModelNotReadyError(RuntimeError):
    """Raised when a required model artefact is missing, incompatible or cannot be loaded."""


def _resolve_artifact(relative_or_absolute: str) -> Path:
    """Resolve a configured artefact path against the repository root.

    Args:
        relative_or_absolute: Path as given in application settings.

    Returns:
        Absolute filesystem path.
    """
    path = Path(relative_or_absolute)
    return path if path.is_absolute() else _REPO_ROOT / path


@lru_cache(maxsize=1)
def load_taxa_catalog() -> list[dict[str, Any]]:
    """Load and cache the supported biological taxa catalog from disk.

    Why:
        Taxonomic classification targets are defined in an authoritative metadata catalog (`config/taxa_catalog.yaml`). Reading and parsing YAML from disk on every prediction incurs file I/O latency and memory fragmentation. Caching the parsed list in memory ensures single-digit microsecond lookups during inference mapping.

    How:
        Resolves the absolute path to `config/taxa_catalog.yaml`, deserializes it via PyYAML safe loader, and caches the resulting list of taxa metadata dictionaries.

    Returns:
        List of taxa metadata dictionaries with scientific and common names.
    """
    catalog_path = _resolve_artifact(get_settings().model.taxa_catalog_path)
    with open(catalog_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("taxa", [])


@lru_cache(maxsize=1)
def get_classifier() -> TaxonClassifier:
    """Instantiate, restore checkpoint weights, and cache the vision classifier model.

    Why:
        Deep neural network models (e.g. DINOv3, MobileNetV4) require tens or hundreds of megabytes of memory and non-trivial weight initialization time. Instantiating a new model per request would exhaust process memory and cause hundred-millisecond cold starts. Caches a single eval-mode TaxonClassifier instance in process memory. A classifier whose head was never trained produces random predictions, so loading is strict and fails fast.

    How:
        Loads the backbone architecture configured in project settings with pre-trained weights, retrieves the feature dimension, instantiates a TaxonClassifier, strictly restores the trained head checkpoint (missing file, wrong keys or wrong shapes raise `ModelNotReadyError`), and locks the model in evaluation mode (`eval()`).

    Returns:
        Cached TaxonClassifier ready for zero-gradient evaluation.

    Raises:
        ModelNotReadyError: If the head checkpoint is absent or incompatible with the backbone and catalogue.
    """
    settings = get_settings()
    checkpoint_path = _resolve_artifact(settings.model.head_checkpoint_path)
    if not checkpoint_path.exists():
        if not settings.model.allow_untrained_fallback:
            raise ModelNotReadyError(f"Trained head checkpoint not found at {checkpoint_path}. Run training first.")
        logger.warning(
            "Head checkpoint %s absent; creating fallback classifier for offline test execution.", checkpoint_path
        )
        backbone = create_feature_extractor(
            settings.model.fallback_extractor,
            pretrained=False,
            allow_fallback=True,
        )
        dim = get_feature_dimension(backbone)
        num_classes = len(load_taxa_catalog())
        classifier = TaxonClassifier(backbone=backbone, feature_dim=dim, num_classes=num_classes)
        classifier.eval()
        return classifier

    backbone = create_feature_extractor(settings.model.default_extractor, pretrained=True)
    dim = get_feature_dimension(backbone)
    num_classes = len(load_taxa_catalog())
    classifier = TaxonClassifier(backbone=backbone, feature_dim=dim, num_classes=num_classes)

    try:
        load_head_checkpoint(classifier.head, checkpoint_path)
    except Exception as exc:
        if not settings.model.allow_untrained_fallback:
            raise ModelNotReadyError(
                f"Head checkpoint {checkpoint_path} is incompatible with backbone "
                f"'{settings.model.default_extractor}' (feature_dim={dim}, classes={num_classes}): {exc}"
            ) from exc
        logger.warning("Head checkpoint incompatible (%s); serving uninitialized fallback for tests.", exc)

    classifier.eval()
    return classifier


@lru_cache(maxsize=1)
def get_onnx_engine() -> ONNXInferenceEngine | None:
    """Create and cache the ONNX Runtime engine over the (quantized) exported graph.

    Why:
        The exported INT8 graph is the production serving artefact: it avoids the PyTorch runtime on the hot path and executes the statically quantized Conv/MatMul kernels. The PyTorch classifier remains as an explicit fallback only when the configured backend is `auto`.

    How:
        Honours `settings.model.inference_backend`: `torch` returns None, `onnx` requires the artefact (raises if absent), `auto` returns None with a warning when the artefact is missing. A loaded graph is validated against the taxa catalogue size.

    Returns:
        The cached ONNXInferenceEngine, or None when PyTorch serving is selected.

    Raises:
        ModelNotReadyError: If ONNX is required but missing, or if the graph class count mismatches the catalogue.
    """
    cfg = get_settings().model
    if cfg.inference_backend == "torch" or not cfg.prefer_onnx_runtime:
        return None

    onnx_path = _resolve_artifact(cfg.onnx_model_path)
    if not onnx_path.exists():
        canonical_fallback = _resolve_artifact("models/checkpoints/model_int8.onnx")
        if canonical_fallback.exists():
            onnx_path = canonical_fallback
        elif cfg.inference_backend == "onnx":
            raise ModelNotReadyError(f"ONNX model not found at {onnx_path}. Run the export_onnx stage first.")
        else:
            logger.warning("ONNX artefact %s not found; serving with the PyTorch classifier.", onnx_path)
            return None

    engine = ONNXInferenceEngine(str(onnx_path))
    expected = len(load_taxa_catalog())
    if engine.num_classes is not None and engine.num_classes != expected:
        raise ModelNotReadyError(
            f"ONNX model {onnx_path} outputs {engine.num_classes} classes; catalogue has {expected}."
        )
    logger.info("Serving predictions from ONNX graph %s", onnx_path)
    return engine


def check_readiness() -> ReadinessStatus:
    """Probe every dependency required to serve predictions.

    Why:
        A readiness probe that always reports healthy routes traffic to pods that cannot answer. This probe loads the catalogue and the serving model (ONNX graph or PyTorch classifier), so a missing or incompatible artefact surfaces as not-ready.

    Returns:
        ReadinessStatus containing overall readiness and a mapping of check name to status string.
    """
    checks: dict[str, str] = {}
    ready = True

    try:
        catalog = load_taxa_catalog()
        if not catalog:
            checks["taxa_catalog"] = "failed: empty catalog"
            return ReadinessStatus(ready=False, checks=checks)
        checks["taxa_catalog"] = f"ok ({len(catalog)} taxa)"
    except Exception as exc:
        checks["taxa_catalog"] = f"failed: {exc}"
        return ReadinessStatus(ready=False, checks=checks)

    try:
        get_classifier()
        checks["classifier"] = "ok"
    except Exception as exc:
        checks["classifier"] = f"failed: {exc}"
        ready = False

    try:
        if get_onnx_engine() is not None:
            checks["model"] = "ok (onnx)"
        else:
            checks["model"] = "ok (torch)"
    except Exception as exc:
        checks["model"] = f"failed: {exc}"
        ready = False

    return ReadinessStatus(ready=ready, checks=checks)


@lru_cache(maxsize=1)
def get_conformal_engine() -> ConformalPredictionEngine:
    """Instantiate and cache the conformal prediction engine.

    Why:
        The conformal prediction engine contains pre-calibrated error tolerances (alpha)
        and quantile non-conformity cutoffs (q_hat). Caching a singleton engine ensures
        consistent calibration parameters across concurrent requests without per-request
        instantiation overhead.

    How:
        Retrieves global application settings and initializes a ConformalPredictionEngine
        with configured alpha, quantile threshold, and k_max parameters.

    Returns:
        The cached ConformalPredictionEngine singleton.
    """
    settings = get_settings()
    return ConformalPredictionEngine(
        q_hat=settings.conformal.quantile_threshold,
        alpha=settings.conformal.alpha,
        k_max=settings.conformal.k_max,
        min_confidence_threshold=settings.conformal.min_confidence_threshold,
    )


@lru_cache(maxsize=1)
def _load_calibrated_ood_threshold() -> float | None:
    """Read the dynamically calibrated OOD energy threshold from config/conformal_config.yaml.

    Why:
        Preserves alignment between offline calibration (`calibrate.py`) and live serving.
        Stage 6 writes `ood_energy_threshold` (and legacy alias `tau_ood`) to `config/conformal_config.yaml`.
        Loading this dynamically prevents the split-brain issue where serving reverts to static defaults.

    Returns:
        Calibrated float threshold if present and valid; otherwise None.
    """
    calib_path = _resolve_artifact("config/conformal_config.yaml")
    if not calib_path.is_file():
        return None
    try:
        with open(calib_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            calib = data.get("calibration", {})
            val = calib.get("ood_energy_threshold")
            if val is not None:
                return float(val)
    except Exception as exc:
        logger.warning("Could not read calibrated OOD threshold from %s: %s", calib_path, exc)
    return None


@lru_cache(maxsize=1)
def get_ood_detector() -> EnergyOODDetector:
    """Instantiate and cache the energy-based out-of-distribution detector.

    Why:
        Detects uninformative, non-biological, or out-of-catalog observations using
        Helmholtz free energy scoring. Caching a singleton detector avoids re-reading
        configuration parameters on each request. Uses dynamically calibrated OOD threshold
        from config/conformal_config.yaml when available.

    Returns:
        The cached EnergyOODDetector singleton.
    """
    settings = get_settings()
    calibrated_threshold = _load_calibrated_ood_threshold()
    resolved_threshold = calibrated_threshold if calibrated_threshold is not None else settings.ood.energy_threshold
    return EnergyOODDetector(
        energy_threshold=resolved_threshold,
        temperature=settings.model.temperature,
        entropy_threshold=settings.ood.entropy_threshold,
    )


@lru_cache(maxsize=1)
def get_image_transform() -> T.Compose:
    """Construct and cache the deterministic image preprocessing transform pipeline.

    Why:
        In high-throughput biological inference, re-instantiating torchvision transform pipelines on every request introduces significant memory allocation overhead, triggering garbage collection cycles on the hot inference path. Pre-instantiating and caching an immutable composition preserves the zero-allocation hot-path invariant required for sub-25ms p95 latency SLOs.

    How:
        Retrieves global model configurations (target spatial resolution, per-channel normalization mean and standard deviation) and bundles Resize (shorter edge to 256/224 of the target), CenterCrop, ToTensor, and Normalize into a single cached torchvision.transforms.Compose instance. This is the same geometry as the feature-extraction transform used to produce the head's training embeddings, so training and serving see identical inputs.

    Returns:
        Cached Compose pipeline transforming PIL Images into normalized 3xHxW float tensors.
    """
    cfg = get_settings().model
    return T.Compose(
        [
            T.Resize(round(cfg.image_size * cfg.eval_resize_ratio), interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(cfg.image_size),
            T.ToTensor(),
            T.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )


def reset_model_caches() -> None:
    """Invalidate cached model artefacts so the next request reloads them from disk."""
    get_classifier.cache_clear()
    get_onnx_engine.cache_clear()
    get_conformal_engine.cache_clear()
    _load_calibrated_ood_threshold.cache_clear()
    get_ood_detector.cache_clear()
    get_image_transform.cache_clear()


def _get_image_tensor(image_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes into a normalized model input tensor.

    Why:
        Incoming HTTP multipart payloads provide raw binary image buffers. These must be decoded, color-space verified, resized, normalized, and batched into a standard 4D tensor (1, C, H, W) before executing backbone feature extraction.

    How:
        Wraps binary bytes in an in-memory BytesIO stream, decodes as RGB PIL Image, applies the pre-cached torchvision transform pipeline, and prepends a batch dimension via `unsqueeze(0)`.

    Args:
        image_bytes: Raw binary bytes of the input image.

    Returns:
        Torch tensor of shape (1, 3, H, W) ready for backbone consumption.

    Raises:
        InvalidImageError: If the bytes cannot be decoded as an image.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"Uploaded payload is not a decodable image: {exc}") from exc
    transform = get_image_transform()
    return transform(image).unsqueeze(0)


def compute_model_outputs(
    tensor: torch.Tensor,
) -> ModelForwardOutputs:
    """Compute raw class logits and optional visual embedding vector for a preprocessed image.

    Why:
        Energy-based OOD detection, conformal prediction, and flagged observation persistence
        require both class logits and dense feature representations. Computing both in a
        single forward execution eliminates redundant feature extractor passes across PyTorch
        and ONNX Runtime backends.

    How:
        Probes the cached ONNX Runtime engine. If active and dual outputs are available,
        extracts logits and embeddings directly from the ONNX session. Otherwise, falls back
        to the PyTorch TaxonClassifier `forward_with_embeddings` within a zero-grad block.

    Complexity:
        O(1) forward pass over the backbone and linear head.

    Args:
        tensor: Preprocessed image tensor of shape (1, 3, H, W).

    Returns:
        ModelForwardOutputs containing logits of shape (num_classes,) and
        embeddings of shape (feature_dim,) or None if embeddings are unavailable.

    Raises:
        ModelNotReadyError: If model artefacts fail to evaluate or are missing.
    """
    engine = get_onnx_engine()
    if engine is not None:
        outputs = engine.predict_outputs(tensor.numpy())
        logits: npt.NDArray[np.float32] = outputs.logits[0]
        embedding: npt.NDArray[np.float32] | None = outputs.embeddings[0] if outputs.embeddings is not None else None
        return ModelForwardOutputs(logits=logits, embeddings=embedding)

    with torch.no_grad():
        classifier = get_classifier()
        logits_t, embeddings_t = classifier.forward_with_embeddings(tensor)
        logits_arr: npt.NDArray[np.float32] = np.asarray(logits_t.squeeze(0).cpu().numpy(), dtype=np.float32)
        embeddings_arr: npt.NDArray[np.float32] = np.asarray(embeddings_t.squeeze(0).cpu().numpy(), dtype=np.float32)
        return ModelForwardOutputs(logits=logits_arr, embeddings=embeddings_arr)


def compute_logits(tensor: torch.Tensor) -> npt.NDArray[np.float32]:
    """Compute raw class logits for a single preprocessed image using the configured backend.

    Why:
        Energy-based OOD scoring and split-conformal prediction operate on logits and probabilities.
        Both backends must expose the same FP32 logits so downstream uncertainty logic is backend-agnostic.

    How:
        Delegates to `compute_model_outputs` and extracts the first tuple element.

    Complexity:
        O(1) model forward evaluation.

    Args:
        tensor: Preprocessed image tensor of shape (1, 3, H, W).

    Returns:
        FP32 logits of shape (num_classes,).
    """
    logits, _ = compute_model_outputs(tensor)
    return logits


def determine_routing_tier(
    cardinality: int,
    top_confidence: float,
    min_confidence_threshold: float,
) -> ConformalRoutingTier:
    """Determine the four-tier conformal routing state according to the formal contract.

    Why:
        Preserves the 4-tier epistemic safety taxonomy:
        - DECISIVE: |C| = 1 and pi_max >= tau_conf (sub-25ms automated fast-path resolution).
        - MODERATE_AMBIGUITY: 1 < |C| <= 3 (triggers LanceDB metadata-filtered HNSW + RRF retrieval).
        - SEVERE_AMBIGUITY: |C| > 3 or (|C| = 1 and pi_max < tau_conf) (routes to expert sorting queue).
        - ABSTENTION: |C| = 0 (empty prediction set, rejected out of hand).

    Args:
        cardinality: Cardinality of prediction set |C|.
        top_confidence: Maximum Softmax Probability / top-1 confidence score.
        min_confidence_threshold: Minimum confidence cutoff tau_conf.

    Returns:
        ConformalRoutingTier enum state.
    """
    if cardinality == 0:
        return ConformalRoutingTier.ABSTENTION
    if cardinality == 1:
        if top_confidence >= min_confidence_threshold:
            return ConformalRoutingTier.DECISIVE
        return ConformalRoutingTier.SEVERE_AMBIGUITY
    if cardinality <= 3:
        return ConformalRoutingTier.MODERATE_AMBIGUITY
    return ConformalRoutingTier.SEVERE_AMBIGUITY


def run_prediction(image_bytes: bytes) -> dict[str, Any]:
    """Execute the full vision inference pipeline with conformal uncertainty quantification.

    Why:
        Provides the end-to-end classification pipeline for an organism observation, mapping from raw image bytes to calibrated taxonomic predictions and conformal guarantee sets. Enforces epistemic safety by evaluating set cardinality and determining whether human taxonomist triage is mandated.

    How:
        1. Preprocesses image bytes into normalized 4D float tensor.
        2. Evaluates model logits and embeddings inside a zero-grad block.
        3. Computes softmax probabilities over target classes.
        4. Identifies top-1 argmax prediction and confidence score.
        5. Computes energy-based OOD score and assesses out-of-distribution status.
        6. Evaluates non-conformity scores via ConformalPredictionEngine to construct valid prediction sets and evaluate 4-tier routing state.
        7. Maps class indices back to authoritative taxa catalog metadata.

    Complexity:
        O(1) forward pass over the backbone plus O(K) softmax and conformal set evaluation, where K is the number of taxa in the catalog.

    Args:
        image_bytes: Raw binary bytes of the input organism image.

    Returns:
        Dictionary containing scientific name, common name, top-1 confidence, conformal set candidate names, 4-tier routing tier, human referral flag, OOD status, OOD energy score, Shannon entropy, the active energy/entropy gate thresholds, probabilities, feature embedding, and execution latency.
    """
    start_time = time.perf_counter()

    taxa_catalog = load_taxa_catalog()

    tensor = _get_image_tensor(image_bytes)

    logits_np, embedding_np = compute_model_outputs(tensor)
    temp = get_settings().model.temperature
    scaled_logits = logits_np / temp
    shifted = np.exp(scaled_logits - np.max(scaled_logits))
    probabilities = shifted / shifted.sum()

    # Extract top prediction
    top1_idx = int(np.argmax(probabilities))
    top1_prob = float(probabilities[top1_idx])

    # Out-of-distribution energy and entropy detection
    ood_detector = get_ood_detector()
    ood_energy_score = float(ood_detector.compute_energy(logits_np))
    is_energy_ood = ood_detector.is_ood(logits_np)
    is_entropy_ood = ood_detector.is_high_entropy(probabilities)
    shannon_entropy = float(ood_detector.compute_entropy(probabilities))
    is_ood = is_energy_ood or is_entropy_ood

    # Uncertainty quantification and 4-tier conformal routing
    conformal_engine = get_conformal_engine()
    pred_set_indices, _ = conformal_engine.predict_set(probabilities)

    routing_tier = determine_routing_tier(
        cardinality=len(pred_set_indices),
        top_confidence=top1_prob,
        min_confidence_threshold=conformal_engine.min_confidence_threshold,
    )

    # Empty conformal set, OOD, or non-decisive ambiguity indicates human review / staging requirement
    requires_human_review = is_ood or (routing_tier != ConformalRoutingTier.DECISIVE)

    # Map to taxa names
    predicted_taxon = taxa_catalog[top1_idx]
    conformal_set_names = [taxa_catalog[i]["scientific_name"] for i in pred_set_indices]

    elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "taxon_id": int(predicted_taxon.get("id", 0)),
        "top1_idx": top1_idx,
        "scientific_name": predicted_taxon["scientific_name"],
        "common_name": predicted_taxon["common_name"],
        "confidence": top1_prob,
        "conformal_set": conformal_set_names,
        "routing_tier": routing_tier.value,
        "is_ood": is_ood,
        "ood_energy_score": ood_energy_score,
        "ood_energy_threshold": float(ood_detector.energy_threshold),
        "shannon_entropy": shannon_entropy,
        "ood_entropy_threshold": float(ood_detector.entropy_threshold),
        "requires_human_review": requires_human_review,
        "probabilities": [float(p) for p in probabilities],
        "embedding": embedding_np,
        "latency_ms": elapsed_time_ms,
    }
