pipeline {
  agent {
    kubernetes {
      cloud 'k3s'
      defaultContainer 'ml-runner'
      yaml '''
apiVersion: v1
kind: Pod
metadata:
  namespace: jenkins-ci
  labels:
    app.kubernetes.io/name: taxon-vision-jenkins-agent
spec:
  serviceAccountName: jenkins-agent
  containers:
  - name: ml-runner
    image: ghcr.io/prefix-dev/pixi:latest
    imagePullPolicy: IfNotPresent
    command: ["sleep"]
    args: ["99d"]
    tty: true
    volumeMounts:
    - name: rattler-cache
      mountPath: /root/.cache/rattler/cache
    - name: host-manifests
      mountPath: /var/lib/taxon-vision/manifests
    - name: host-reference
      mountPath: /var/lib/taxon-vision/reference
    - name: host-checkpoints
      mountPath: /var/lib/taxon-vision/checkpoints
    resources:
      limits:
        memory: "8Gi"
        cpu: "4"
      requests:
        memory: "512Mi"
        cpu: "200m"
  volumes:
  - name: rattler-cache
    hostPath:
      path: /var/lib/taxon-vision/rattler-cache
      type: DirectoryOrCreate
  - name: host-manifests
    hostPath:
      path: /var/lib/taxon-vision/manifests
      type: DirectoryOrCreate
  - name: host-reference
    hostPath:
      path: /var/lib/taxon-vision/reference
      type: DirectoryOrCreate
  - name: host-checkpoints
    hostPath:
      path: /var/lib/taxon-vision/checkpoints
      type: DirectoryOrCreate
'''
    }
  }

  parameters {
    booleanParam(name: 'RETRAIN_MODEL', defaultValue: false, description: 'Force GPU feature extraction and model retraining on deployment')
    booleanParam(name: 'TUNE_HYPERPARAMETERS', defaultValue: false, description: 'Run Optuna Bayesian hyperparameter optimization during training')
    booleanParam(name: 'INGEST_DATA', defaultValue: false, description: 'Unfreeze DVC ingestion stage and fetch fresh observations from iNaturalist')
    string(name: 'EXTRACTOR', defaultValue: 'dinov3', description: 'Vision foundation backbone extractor')
  }

  options {
    buildDiscarder(logRotator(numToKeepStr: '15', daysToKeepStr: '14'))
    timeout(time: 240, unit: 'MINUTES')
    timestamps()
    disableConcurrentBuilds(abortPrevious: true)
  }

  triggers {
    // Runs at 02:00 UTC on the 1st day of every month
    cron('0 2 1 * *')
  }

  environment {
    DAGSHUB_TOKEN = credentials('dagshub-token')
    AWS_ACCESS_KEY_ID = credentials('dagshub-token')
    AWS_SECRET_ACCESS_KEY = credentials('dagshub-token')
    MLFLOW_TRACKING_USERNAME = 'foersben'
    MLFLOW_TRACKING_PASSWORD = credentials('dagshub-token')
    MLFLOW_TRACKING_URI = 'https://dagshub.com/foersben/taxon-vision-mlops.mlflow'
    PYTHONUNBUFFERED = '1'
    OMP_NUM_THREADS = '4'
    MKL_NUM_THREADS = '4'
    OPENBLAS_NUM_THREADS = '4'
    NUMEXPR_NUM_THREADS = '4'
    VECLIB_MAXIMUM_THREADS = '4'
  }

  stages {
    stage('Prepare Toolchain') {
      steps {
        container('ml-runner') {
          sh '''
            echo ">>> Setting up toolchain in ephemeral agent..."
            which git >/dev/null 2>&1 || pixi global install git
            which kubectl >/dev/null 2>&1 || pixi global install kubernetes-client
            which curl >/dev/null 2>&1 || pixi global install curl
            export PATH="/root/.pixi/bin:$PATH"
            git config --global --add safe.directory "*"
            pixi --version
            pixi install --frozen -e ci-dev
            echo ">>> Pulling DVC tracked datasets and model checkpoints from DagsHub..."
            pixi run --frozen -e ci-dev dvc pull --force \
              data/manifests/raw_manifest.parquet \
              data/manifests/dataset_manifest.parquet \
              data/embeddings/train_embeddings.pt \
              data/embeddings/val_embeddings.pt \
              data/embeddings/cal_embeddings.pt \
              data/embeddings/test_embeddings.pt \
              data/reference/reference_train_embeddings.pt \
              models/checkpoints/head.pt \
              models/checkpoints/model_int8.onnx || echo "WARNING: DVC pull skipped or failed; continuing with workspace cache."
          '''
        }
      }
    }

    stage('Static Quality & Invariants') {
      parallel {
        stage('Lint & Format') {
          steps {
            container('ml-runner') {
              sh '''
                pixi run --frozen -e ci-dev ruff check .
                pixi run --frozen -e ci-dev ruff format --check .
              '''
            }
          }
        }
        stage('Type Checking') {
          steps {
            container('ml-runner') {
              sh '''
                pixi run --frozen -e ci-dev mypy src scripts
              '''
            }
          }
        }
        stage('Compliance & OKF Audits') {
          steps {
            container('ml-runner') {
              sh '''
                pixi run --frozen -e ci-dev python -m taxon_vision.pipelines.audit_license
                pixi run --frozen -e ci-dev python scripts/governance/validate_okf.py
              '''
            }
          }
        }
      }
    }

    stage('Unit Tests') {
      steps {
        container('ml-runner') {
          sh '''
            pixi run --frozen -e ci-dev pytest tests/unit/ \
              --cov=src/taxon_vision \
              --cov-report= \
              -o "addopts="
          '''
        }
      }
    }

    stage('Integration & Conformal Invariants') {
      steps {
        container('ml-runner') {
          sh '''
            pixi run --frozen -e ci-dev pytest tests/integration/ tests/invariants/ \
              --cov=src/taxon_vision \
              --cov-append \
              --cov-report=xml:reports/coverage/coverage.xml \
              --cov-fail-under=80 \
              -x -q -o "addopts="
          '''
        }
      }
    }

    stage('Documentation Strict Build') {
      steps {
        container('ml-runner') {
          sh '''
            pixi run --frozen -e ci-dev python scripts/governance/visualize_okf.py
            pixi run --frozen -e ci-dev zensical build --strict
          '''
        }
      }
    }

    stage('Model Training & Checkpoint (GPU)') {
      options {
        timeout(time: 150, unit: 'MINUTES')
      }
      environment {
        // INGEST_DATA = "${params.INGEST_DATA || currentBuild.getBuildCauses().any { it.toString().contains('TimerTrigger') }}"
        INGEST_DATA = "${params.INGEST_DATA}"
        TUNE_HYPERPARAMETERS = "${params.TUNE_HYPERPARAMETERS}"
        EXTRACTOR = "${params.EXTRACTOR}"
      }
      when {
        beforeAgent true
        anyOf {
          triggeredBy 'TimerTrigger'
          expression { params.INGEST_DATA == true }
          expression { params.RETRAIN_MODEL == true }
          allOf {
            branch 'develop'
            anyOf {
              changeset "config/taxa_catalog.yaml"
              changeset "data/**"
              changeset "config/**"
              changeset "params.yaml"
              changeset "dvc.yaml"
              changeset "src/taxon_vision/models/**"
              changeset "src/taxon_vision/pipelines/**"
            }
          }
        }
      }
      agent {
        kubernetes {
          cloud 'k3s'
          defaultContainer 'ml-runner'
          yamlFile 'deploy/k8s/jenkins-gpu-pod.yaml'
        }
      }
      steps {
        checkout scm
        container('ml-runner') {
          withCredentials([usernamePassword(credentialsId: 'seaweedfs-s3-credentials', usernameVariable: 'SEAWEEDFS_ACCESS_KEY_ID', passwordVariable: 'SEAWEEDFS_SECRET_ACCESS_KEY')]) {
            sh '''
              # Enable ingestion on scheduled monthly runs or taxa catalog modifications
              if echo "$BUILD_CAUSE" | grep -q "TIMER" || [ "$INGEST_DATA" = "true" ]; then
                export INGEST_DATA=true
              fi

              if git diff --name-only HEAD~1 HEAD 2>/dev/null | grep -q "config/taxa_catalog.yaml"; then
                echo ">>> config/taxa_catalog.yaml was modified; forcing INGEST_DATA=true"
                export INGEST_DATA=true
              fi

              if [ -s "/var/lib/taxon-vision/manifests/catalog_expansions.yaml" ] && grep -q "inat_taxon_id:" "/var/lib/taxon-vision/manifests/catalog_expansions.yaml"; then
                echo ">>> Pending active-learning catalogue expansions found; forcing INGEST_DATA=true"
                export INGEST_DATA=true
              fi

              bash scripts/ci/jenkins_gpu_train.sh
            '''
          }
        }
      }
    }

    stage('Deploy to Kubernetes') {
      when {
        beforeAgent true
        anyOf {
          branch 'main'
          branch 'develop'
        }
      }
      steps {
        container('ml-runner') {
          withCredentials([usernamePassword(credentialsId: 'seaweedfs-s3-credentials', usernameVariable: 'SEAWEEDFS_ACCESS_KEY_ID', passwordVariable: 'SEAWEEDFS_SECRET_ACCESS_KEY')]) {
            sh '''
              export PATH="/root/.pixi/bin:$PATH"
              cleanup_dvc_local() {
                pixi run --frozen -e ci-dev dvc config --local --unset remote.local-seaweedfs.access_key_id >/dev/null 2>&1 || true
                pixi run --frozen -e ci-dev dvc config --local --unset remote.local-seaweedfs.secret_access_key >/dev/null 2>&1 || true
              }
              trap cleanup_dvc_local EXIT

              echo ">>> Executing zero-downtime rolling deployment to Kubernetes cluster..."
              kubectl version --client
              git fetch origin "${BRANCH_NAME}" && git reset --hard "origin/${BRANCH_NAME}" || true

              if [ -f "/var/lib/taxon-vision/checkpoints/head.pt" ] && [ -f "/var/lib/taxon-vision/checkpoints/model_int8.onnx" ] && [ -f "/var/lib/taxon-vision/reference/cal_embeddings.pt" ]; then
                echo ">>> Host volume /var/lib/taxon-vision/checkpoints/ already holds active checkpoints; preserving host checkpoints."
              else
                # Hydrate model checkpoints and calibrated reference embeddings from DVC (prefer in-cluster SeaweedFS cache, fallback to DagsHub origin)
                echo ">>> Hydrating missing model checkpoints and reference embeddings from DVC remotes..."
                K8S_S3_KEY="$(kubectl get secret seaweedfs-s3-secret -n taxon-vision -o jsonpath='{.data.AWS_ACCESS_KEY_ID}' 2>/dev/null | base64 -d 2>/dev/null || true)"
                K8S_S3_SECRET="$(kubectl get secret seaweedfs-s3-secret -n taxon-vision -o jsonpath='{.data.AWS_SECRET_ACCESS_KEY}' 2>/dev/null | base64 -d 2>/dev/null || true)"
                if [ -n "${K8S_S3_KEY}" ] && [ -n "${K8S_S3_SECRET}" ]; then
                  SEAWEEDFS_ACCESS_KEY_ID="${K8S_S3_KEY}"
                  SEAWEEDFS_SECRET_ACCESS_KEY="${K8S_S3_SECRET}"
                fi
                PULLED=false
                if kubectl get service seaweedfs -n taxon-vision >/dev/null 2>&1 && [ -n "${SEAWEEDFS_ACCESS_KEY_ID:-}" ] && [ -n "${SEAWEEDFS_SECRET_ACCESS_KEY:-}" ]; then
                  pixi run --frozen -e ci-dev dvc config --local remote.local-seaweedfs.access_key_id "${SEAWEEDFS_ACCESS_KEY_ID}"
                  pixi run --frozen -e ci-dev dvc config --local remote.local-seaweedfs.secret_access_key "${SEAWEEDFS_SECRET_ACCESS_KEY}"
                  echo ">>> Attempting hydration from in-cluster SeaweedFS (local-seaweedfs)..."
                  if env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY pixi run --frozen -e ci-dev dvc pull --force -r local-seaweedfs \
                      models/checkpoints/head.pt \
                      models/checkpoints/model_int8.onnx \
                      data/embeddings/cal_embeddings.pt; then
                    PULLED=true
                    echo ">>> Checkpoints and reference embeddings successfully hydrated from local-seaweedfs."
                  fi
                fi

                if [ "$PULLED" != "true" ]; then
                  echo ">>> Pulling checkpoints and reference embeddings from DagsHub archive (origin)..."
                  pixi run --frozen -e ci-dev dvc pull --force -r origin \
                    models/checkpoints/head.pt \
                    models/checkpoints/model_int8.onnx \
                    data/embeddings/cal_embeddings.pt || \
                  pixi run --frozen -e ci-dev dvc pull --force \
                    models/checkpoints/head.pt \
                    models/checkpoints/model_int8.onnx \
                    data/embeddings/cal_embeddings.pt || \
                  echo ">>> Warning: Remote artifacts not found in DVC; falling back to existing host volume checkpoints."
                fi

                # Synchronize checkpoints, co-versioned runtime configs, and calibrated reference embeddings atomically for non-root API pod (UID 10001)
                if [ -d "/var/lib/taxon-vision/checkpoints" ] && [ -f "models/checkpoints/head.pt" ]; then
                  echo ">>> Atomically staging checkpoints and co-versioned configs to host volume /var/lib/taxon-vision/checkpoints/..."
                  for ckpt in models/checkpoints/* config/taxa_catalog.yaml config/conformal_config.yaml; do
                      [ -f "$ckpt" ] || continue
                      fname=$(basename "$ckpt")
                      cp -f "$ckpt" "/var/lib/taxon-vision/checkpoints/.${fname}.tmp"
                      chmod a+r "/var/lib/taxon-vision/checkpoints/.${fname}.tmp"
                      mv -f "/var/lib/taxon-vision/checkpoints/.${fname}.tmp" "/var/lib/taxon-vision/checkpoints/${fname}"
                  done
                fi

                if [ -d "/var/lib/taxon-vision/reference" ] && [ -f "data/embeddings/cal_embeddings.pt" ]; then
                  echo ">>> Atomically staging cal_embeddings.pt to host volume /var/lib/taxon-vision/reference/..."
                  cp -f "data/embeddings/cal_embeddings.pt" "/var/lib/taxon-vision/reference/.cal_embeddings.pt.tmp"
                  chmod a+r "/var/lib/taxon-vision/reference/.cal_embeddings.pt.tmp"
                  mv -f "/var/lib/taxon-vision/reference/.cal_embeddings.pt.tmp" "/var/lib/taxon-vision/reference/cal_embeddings.pt"
                fi
              fi

            # Synchronise Airflow 3 DAGs & helper dependencies into ConfigMaps before applying Airflow
            set --
            for f in airflow/dags/*.py; do
              [ -f "$f" ] && set -- "$@" "--from-file=$f"
            done
            kubectl create configmap airflow-dags "$@" -n taxon-vision --dry-run=client -o yaml | kubectl apply -f -
            kubectl create configmap airflow-taxon-deps \
              --from-file=feedback.py=src/taxon_vision/active_learning/feedback.py \
              --from-file=label_studio_sync.py=src/taxon_vision/active_learning/label_studio_sync.py \
              --from-file=triage_queue.py=src/taxon_vision/active_learning/triage_queue.py \
              --from-file=taxa_catalog.yaml=config/taxa_catalog.yaml \
              -n taxon-vision --dry-run=client -o yaml | kubectl apply -f -

            # Apply shared backing services, log aggregation, and dashboards
            kubectl apply -f deploy/k8s/postgres.yaml
            kubectl apply -f deploy/k8s/loki-promtail.yaml
            kubectl apply -f deploy/k8s/monitoring-prometheus-grafana.yaml
            kubectl apply -f deploy/k8s/grafana-dashboards.yaml
            kubectl apply -f deploy/k8s/mlops-storage-and-optuna.yaml
            kubectl apply -f deploy/k8s/evidently-ui.yaml
            kubectl apply -f deploy/k8s/airflow.yaml
            kubectl apply -f deploy/k8s/traefik-ingress.yaml

            kubectl rollout restart deployment/airflow-apiserver deployment/airflow-scheduler -n taxon-vision
            kubectl rollout status deployment/airflow-apiserver -n taxon-vision --timeout=180s
            kubectl rollout status deployment/airflow-scheduler -n taxon-vision --timeout=180s

            # Determine target overlay from Git branch
            if [ "${BRANCH_NAME}" = "main" ]; then
              TARGET_ENV="production"
              TARGET_NS="taxon-vision"
            else
              TARGET_ENV="staging"
              TARGET_NS="taxon-vision-staging"
            fi

            echo ">>> Deploying ${TARGET_ENV} overlay to namespace ${TARGET_NS} via Kustomize..."
            kubectl apply -k deploy/k8s/overlays/${TARGET_ENV}

            # Wait for GHCR SHA image tag published by GitHub Actions to eliminate stale image pull race
            SHORT_SHA=$(git rev-parse --short=7 HEAD)
            COMMIT_MSG=$(git log -1 --pretty=%B || true)
            GHCR_IMAGE="ghcr.io/foersben/taxon-vision-mlops"
            IMAGE_READY=false
            if ! echo "${COMMIT_MSG}" | grep -qiE "skip ci|ci skip"; then
              echo ">>> Waiting for GHCR image ${GHCR_IMAGE}:sha-${SHORT_SHA} (up to 360s)..."
              WAITED=0
              while [ "${WAITED}" -lt 360 ]; do
                GHCR_TOK=$(curl -fsSL "https://ghcr.io/token?service=ghcr.io&scope=repository:foersben/taxon-vision-mlops:pull" | grep -o '"token":"[^"]*"' | cut -d'"' -f4 || true)
                if [ -n "${GHCR_TOK}" ]; then
                  CODE=$(curl -s -o /dev/null -w "%{http_code}" \
                    -H "Authorization: Bearer ${GHCR_TOK}" \
                    -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.oci.image.manifest.v1+json" \
                    "https://ghcr.io/v2/foersben/taxon-vision-mlops/manifests/sha-${SHORT_SHA}" || echo "000")
                  if [ "${CODE}" = "200" ]; then
                    IMAGE_READY=true
                    echo ">>> Confirmed GHCR image sha-${SHORT_SHA} is published."
                    break
                  fi
                fi
                sleep 10
                WAITED=$((WAITED + 10))
              done
            fi

            if [ "${IMAGE_READY}" = "true" ]; then
              kubectl set image deployment/taxon-vision-serving api-service="${GHCR_IMAGE}:sha-${SHORT_SHA}" -n ${TARGET_NS}
            else
              echo ">>> Using branch overlay image tag and restarting deployment to load updated host-mounted artefacts..."
              kubectl rollout restart deployment/taxon-vision-serving -n ${TARGET_NS}
            fi

            # Verify deployment health
            kubectl rollout status deployment/taxon-vision-serving -n ${TARGET_NS} --timeout=300s
            bash scripts/ci/verify_deployment.sh ${TARGET_NS}
            echo ">>> Deployment successfully verified for ${TARGET_ENV}."
            '''
          }
        }
      }
    }
  }

  post {
    always {
      script {
        if (env.WORKSPACE != null) {
          try {
            cleanWs notFailBuild: true
          } catch (Throwable t) {
            echo "Workspace cleanup skipped: ${t.message}"
          }
        }
      }
    }
  }
}
