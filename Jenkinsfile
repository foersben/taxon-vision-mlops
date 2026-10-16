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
            pixi install --frozen -e dev

            # Configure local SeaweedFS credentials for DVC cache
            pixi run --frozen -e dev dvc config --local remote.local-seaweedfs.access_key_id taxonadmin
            pixi run --frozen -e dev dvc config --local remote.local-seaweedfs.secret_access_key taxonadmin

            # Prime cached embeddings from host volume tier if available
            mkdir -p data/embeddings
            if [ ! -f "data/embeddings/cached_features.pt" ]; then
              if [ -f "/var/lib/taxon-vision/embeddings/cached_features.pt" ]; then
                echo ">>> Found cached features on host tier; priming data/embeddings/cached_features.pt..."
                cp -f /var/lib/taxon-vision/embeddings/cached_features.pt data/embeddings/cached_features.pt
              elif kubectl get service seaweedfs -n taxon-vision >/dev/null 2>&1; then
                echo ">>> Attempting feature cache retrieval from in-cluster SeaweedFS (local-seaweedfs)..."
                pixi run --frozen -e dev dvc pull -r local-seaweedfs data/embeddings/cached_features.pt || echo "Cache pull from local-seaweedfs skipped; will generate via DVC pipeline."
              fi
            fi

            # Execute unified 7-stage DVC pipeline
            echo ">>> Executing unified 7-stage DVC pipeline..."
            EXTRA_DVC_OPTS=""
            if [ "${TUNE_HYPERPARAMETERS:-false}" = "true" ]; then
              echo ">>> Bayesian optimization enabled: running with training.tune=true..."
              EXTRA_DVC_OPTS="-S training.tune=true"
            fi
            if [ -n "${EXTRACTOR}" ] && [ "${EXTRACTOR}" != "dinov3" ]; then
              EXTRA_DVC_OPTS="${EXTRA_DVC_OPTS} -S model.extractor=${EXTRACTOR}"
            fi

            pixi run --frozen -e dev dvc repro ${EXTRA_DVC_OPTS}

            # Persist generated features to host volume cache
            if [ -d "/var/lib/taxon-vision/embeddings" ] && [ -f "data/embeddings/cached_features.pt" ]; then
              cp -f data/embeddings/cached_features.pt /var/lib/taxon-vision/embeddings/cached_features.pt
            fi

            # Push embedding cache to local SeaweedFS if reachable
            if kubectl get service seaweedfs -n taxon-vision >/dev/null 2>&1; then
              echo ">>> Synchronizing feature cache to in-cluster SeaweedFS (local-seaweedfs)..."
              pixi run --frozen -e dev dvc push -r local-seaweedfs data/embeddings/cached_features.pt
            fi

            # Push release checkpoints and INT8 models to DagsHub S3 remote (origin)
            echo ">>> Pushing release checkpoints and INT8 models to DagsHub S3 remote (origin)..."
            pixi run --frozen -e dev dvc push -r origin models/checkpoints/head.pt models/checkpoints/model.onnx models/model_int8.onnx
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
            kubectl version --client

            # Configure SeaweedFS credentials in case models are in local cache
            if kubectl get service seaweedfs -n taxon-vision >/dev/null 2>&1; then
              pixi run --frozen -e ci-dev dvc config --local remote.local-seaweedfs.access_key_id taxonadmin
              pixi run --frozen -e ci-dev dvc config --local remote.local-seaweedfs.secret_access_key taxonadmin
            fi

            # Hydrate model checkpoints from DVC
            echo ">>> Hydrating model checkpoints from DVC remotes..."
            pixi run --frozen -e ci-dev dvc pull -r origin models/checkpoints/head.pt models/checkpoints/model.onnx models/model_int8.onnx || \
            pixi run --frozen -e ci-dev dvc pull models/checkpoints/head.pt models/checkpoints/model.onnx models/model_int8.onnx || \
            echo ">>> Warning: Remote checkpoints not found in DVC; falling back to existing host volume checkpoints."

            # Synchronize checkpoints to host volume if available
            if [ -d "/home/benni/taxon-vision-mlops/models/checkpoints" ] && [ -f "models/checkpoints/head.pt" ]; then
              cp -f models/checkpoints/* /home/benni/taxon-vision-mlops/models/checkpoints/
            fi
            if [ -d "/home/benni/taxon-vision-mlops/models" ] && [ -f "models/model_int8.onnx" ]; then
              cp -f models/model_int8.onnx /home/benni/taxon-vision-mlops/models/
            fi

            # Apply API deployment and service manifests
            kubectl apply -f deploy/k8s/api-deployment.yaml

            # Trigger rolling restart to serve fresh weights
            kubectl rollout restart deployment/taxon-vision-api -n taxon-vision

            # Verify deployment health
            kubectl rollout status deployment/taxon-vision-api -n taxon-vision --timeout=120s
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
