-- SPDX-FileCopyrightText: 2026 Benjamin Förster
-- SPDX-License-Identifier: MIT
-- Prediction telemetry and drift evaluation run tables.

CREATE TABLE IF NOT EXISTS predictions (
    id               BIGSERIAL PRIMARY KEY,
    observation_id   UUID        NOT NULL UNIQUE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version    TEXT,
    extractor        TEXT        NOT NULL,
    top1_taxon       TEXT,
    top1_confidence  REAL,
    conformal_set    TEXT[]      NOT NULL,
    set_size         SMALLINT    NOT NULL,
    ood_energy       REAL,
    is_ood           BOOLEAN     NOT NULL,
    requires_review  BOOLEAN     NOT NULL,
    latency_ms       REAL        NOT NULL,
    embedding        REAL[]      -- Raw pre-normalised activations (length 384 for ViT-S)
);

CREATE INDEX IF NOT EXISTS predictions_created_at_idx ON predictions (created_at DESC);

CREATE TABLE IF NOT EXISTS drift_runs (
    id                BIGSERIAL PRIMARY KEY,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    window_size       INTEGER          NOT NULL,
    w1_distance       DOUBLE PRECISION NOT NULL,
    w1_threshold      DOUBLE PRECISION NOT NULL,
    mmd2              DOUBLE PRECISION NOT NULL,
    mmd2_p_value      DOUBLE PRECISION NOT NULL,
    drift_detected    BOOLEAN          NOT NULL,
    mlflow_run_id     TEXT
);

CREATE INDEX IF NOT EXISTS drift_runs_created_at_idx ON drift_runs (created_at DESC);
