-- SPDX-FileCopyrightText: 2026 Benjamin Förster
-- SPDX-License-Identifier: MIT
-- Initialises application roles and databases for TaxonVision MLOps.

-- 1. Create Roles
CREATE ROLE telemetry LOGIN PASSWORD 'taxon_telemetry_pass';
CREATE ROLE optuna    LOGIN PASSWORD 'taxon_optuna_pass';
CREATE ROLE airflow   LOGIN PASSWORD 'taxon_airflow_pass';
CREATE ROLE mlflow    LOGIN PASSWORD 'taxon_mlflow_pass';

-- 2. Create Databases
CREATE DATABASE telemetry OWNER telemetry;
CREATE DATABASE optuna    OWNER optuna;
CREATE DATABASE airflow   OWNER airflow;
CREATE DATABASE mlflow    OWNER mlflow;

-- 3. Grant Permissions
GRANT ALL PRIVILEGES ON DATABASE telemetry TO telemetry;
GRANT ALL PRIVILEGES ON DATABASE optuna TO optuna;
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;
