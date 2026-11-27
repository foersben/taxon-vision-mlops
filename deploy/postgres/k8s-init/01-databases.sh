#!/bin/sh
set -eu
psql -v ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname "${POSTGRES_DB}" \
  -v telemetry_pw="${TELEMETRY_DB_PASSWORD}" \
  -v optuna_pw="${OPTUNA_DB_PASSWORD}" \
  -v airflow_pw="${AIRFLOW_DB_PASSWORD}" <<'EOSQL'
CREATE ROLE telemetry LOGIN PASSWORD :'telemetry_pw';
CREATE ROLE optuna    LOGIN PASSWORD :'optuna_pw';
CREATE ROLE airflow   LOGIN PASSWORD :'airflow_pw';

CREATE DATABASE telemetry OWNER telemetry;
CREATE DATABASE optuna    OWNER optuna;
CREATE DATABASE airflow   OWNER airflow;

GRANT ALL PRIVILEGES ON DATABASE telemetry TO telemetry;
GRANT ALL PRIVILEGES ON DATABASE optuna    TO optuna;
GRANT ALL PRIVILEGES ON DATABASE airflow   TO airflow;
EOSQL
