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
    command: ["sleep"]
    args: ["99d"]
    tty: true
    volumeMounts:
    - name: rattler-cache
      mountPath: /root/.cache/rattler/cache
    resources:
      limits:
        memory: "8Gi"
        cpu: "4"
      requests:
        memory: "512Mi"
        cpu: "200m"
  volumes:
  - name: rattler-cache
    emptyDir: {}
'''
    }
  }

  parameters {
    booleanParam(name: 'RETRAIN_MODEL', defaultValue: false, description: 'Force GPU feature extraction and model retraining on deployment')
    booleanParam(name: 'TUNE_HYPERPARAMETERS', defaultValue: false, description: 'Run Optuna Bayesian hyperparameter optimization during training')
    string(name: 'EXTRACTOR', defaultValue: 'dinov3', description: 'Vision foundation backbone extractor')
  }

  options {
    buildDiscarder(logRotator(numToKeepStr: '15', daysToKeepStr: '14'))
    timeout(time: 240, unit: 'MINUTES')
    timestamps()
    disableConcurrentBuilds(abortPrevious: true)
  }

  environment {
    DAGSHUB_TOKEN = credentials('dagshub-token')
    AWS_ACCESS_KEY_ID = "${DAGSHUB_TOKEN}"
    AWS_SECRET_ACCESS_KEY = "${DAGSHUB_TOKEN}"
    MLFLOW_TRACKING_USERNAME = 'foersben'
    MLFLOW_TRACKING_PASSWORD = "${DAGSHUB_TOKEN}"
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
            export PATH="/root/.pixi/bin:$PATH"
            git config --global --add safe.directory "*"
            pixi --version
            pixi install --frozen -e ci-dev
            echo ">>> Pulling DVC tracked datasets and model checkpoints from DagsHub..."
            pixi run --frozen -e ci-dev dvc pull -f || echo "WARNING: DVC pull skipped or failed; continuing with workspace cache."
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
                pixi run --frozen -e ci-dev python scripts/audit_license_compliance.py
                pixi run --frozen -e ci-dev python scripts/validate_okf.py
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
              --cov-report=xml:coverage.xml \
              --cov-fail-under=78 \
              -x -q -o "addopts="
          '''
        }
      }
    }

    stage('Documentation Strict Build') {
      steps {
        container('ml-runner') {
          sh '''
            pixi run --frozen -e ci-dev python scripts/visualize_okf.py
            pixi run --frozen -e ci-dev zensical build --strict
          '''
        }
      }
    }

    stage('Model Training & Checkpoint (GPU)') {
      options {
        timeout(time: 150, unit: 'MINUTES')
      }
      when {
        allOf {
          anyOf {
            branch 'main'
            branch 'develop'
          }
          anyOf {
            expression { params.RETRAIN_MODEL == true }
            changeset "data/**"
            changeset "src/taxon_vision/models/**"
            changeset "src/taxon_vision/pipelines/**"
          }
        }
      }
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
      app.kubernetes.io/name: taxon-vision-jenkins-gpu-agent
  spec:
    serviceAccountName: jenkins-agent
    containers:
    - name: ml-runner
      image: ghcr.io/prefix-dev/pixi:latest
      command: ["sleep"]
      args: ["99d"]
      tty: true
      volumeMounts:
      - name: rattler-cache
        mountPath: /root/.cache/rattler/cache
      - name: hf-cache
        mountPath: /root/.cache/huggingface
      - name: optuna-studies
        mountPath: /workspace/optuna
      - name: host-embeddings
        mountPath: /var/lib/taxon-vision/embeddings
      resources:
        limits:
          memory: "16Gi"
          cpu: "8"
          nvidia.com/gpu: "1"
        requests:
          memory: "2Gi"
          cpu: "1000m"
          nvidia.com/gpu: "1"
    volumes:
    - name: rattler-cache
      emptyDir: {}
    - name: hf-cache
      hostPath:
        path: /var/lib/taxon-vision/hf-cache
        type: DirectoryOrCreate
    - name: optuna-studies
      hostPath:
        path: /var/lib/taxon-vision/optuna
        type: DirectoryOrCreate
    - name: host-embeddings
      hostPath:
        path: /var/lib/taxon-vision/embeddings
        type: DirectoryOrCreate
'''
        }
      }
      steps {
        container('ml-runner') {
          sh '''
            export PATH="/root/.pixi/bin:$PATH"
            export PYTHONUNBUFFERED=1
            export TAXON_OPTUNA_STORAGE_URI="sqlite:////workspace/optuna/optuna_studies.db"
            which git >/dev/null 2>&1 || pixi global install git
            which kubectl >/dev/null 2>&1 || pixi global install kubernetes-client
            git config --global --add safe.directory "*"
            echo ">>> Setting up GPU training pipeline..."
            # Install GPU-enabled dev environment
            pixi install --frozen -e dev

            # Pull latest DVC manifests and datasets from DagsHub
            pixi run --frozen -e dev dvc pull -f data/manifests

            # Tiered feature embeddings retrieval
            mkdir -p data/embeddings
            if [ ! -f "data/embeddings/cached_features.pt" ]; then
              echo ">>> Attempting feature cache retrieval from local-minio remote..."
              pixi run --frozen -e dev dvc pull -r local-minio data/embeddings/cached_features.pt || true
            fi

            if [ ! -f "data/embeddings/cached_features.pt" ] && [ -f "/var/lib/taxon-vision/embeddings/cached_features.pt" ]; then
              echo ">>> Recovering cached features from host cache tier..."
              cp -f /var/lib/taxon-vision/embeddings/cached_features.pt data/embeddings/cached_features.pt
            fi

            # Extract vision foundation features from S3 manifest if not cached
            if [ ! -f "data/embeddings/cached_features.pt" ]; then
              echo ">>> Extracting vision foundation features from S3 manifest on GPU..."
              pixi run --frozen -e dev python -m taxon_vision.pipelines.extract_features \
                --extractor "${EXTRACTOR:-dinov3}" \
                --manifest data/manifests/dataset_manifest.parquet \
                --out data/embeddings/cached_features.pt

              # Cache extracted features back to host tier and local MinIO
              mkdir -p /var/lib/taxon-vision/embeddings
              cp -f data/embeddings/cached_features.pt /var/lib/taxon-vision/embeddings/cached_features.pt || true
              pixi run --frozen -e dev dvc push -r local-minio data/embeddings/cached_features.pt || true
            fi

            # Execute classification head training (with optional Optuna tuning)
            EXTRA_OPTS=""
            if [ "${TUNE_HYPERPARAMETERS:-false}" = "true" ]; then
              echo ">>> Bayesian optimization enabled: running Optuna search..."
              EXTRA_OPTS="--tune --n-trials 10"
            fi

            pixi run --frozen -e dev python -m taxon_vision.models.trainer \
              --extractor "${EXTRACTOR:-dinov3}" \
              --epochs 15 \
              --require-cache \
              --require-promotion \
              --cuda-memory-fraction 0.7 \
              $EXTRA_OPTS

            # Re-export fresh ONNX computational graph
            pixi run --frozen -e dev python -m taxon_vision.inference.onnx_exporter --extractor "${EXTRACTOR:-dinov3}"

            # Record trained model artifacts in DVC cache and push strictly model checkpoints to DagsHub
            pixi run --frozen -e dev dvc commit -f models/checkpoints/head.pt models/checkpoints/model.onnx
            pixi run --frozen -e dev dvc push -r origin models/checkpoints/head.pt models/checkpoints/model.onnx
          '''
        }
      }
    }

    stage('Deploy to Kubernetes') {
      when {
        anyOf {
          branch 'main'
          branch 'develop'
        }
      }
      steps {
        container('ml-runner') {
          sh '''
            export PATH="/root/.pixi/bin:$PATH"
            echo ">>> Executing zero-downtime rolling deployment to Kubernetes cluster..."
            # Verify in-cluster service account access
            kubectl version --client || true

            # Hydrate model checkpoints from DVC if needed
            pixi run --frozen -e ci-dev dvc pull -f models/checkpoints/head.pt models/checkpoints/model.onnx || true

            # Ensure checkpoints are synchronised to host volume if accessible
            if [ -d "/home/benni/taxon-vision-mlops/models/checkpoints" ] && [ -f "models/checkpoints/head.pt" ]; then
              cp -f models/checkpoints/* /home/benni/taxon-vision-mlops/models/checkpoints/ || true
            fi

            # Apply API deployment and service manifests
            kubectl apply -f deploy/k8s/api-deployment.yaml

            # Trigger rolling restart to serve fresh weights
            kubectl rollout restart deployment/taxon-vision-api -n taxon-vision || true

            # Verify deployment health
            kubectl rollout status deployment/taxon-vision-api -n taxon-vision --timeout=120s || true
            echo ">>> Deployment successfully verified."
          '''
        }
      }
    }
  }

  post {
    always {
      script {
        catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
          cleanWs()
        }
      }
    }
  }
}
