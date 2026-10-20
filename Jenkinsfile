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
        INGEST_DATA = "${params.INGEST_DATA || currentBuild.getBuildCauses().any { it.toString().contains('TimerTrigger') }}"
        TUNE_HYPERPARAMETERS = "${params.TUNE_HYPERPARAMETERS}"
        EXTRACTOR = "${params.EXTRACTOR}"
      }
      when {
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
          sh '''
            # If config/taxa_catalog.yaml was modified in recent commits, force INGEST_DATA=true
            if git diff --name-only HEAD~1 HEAD 2>/dev/null | grep -q "config/taxa_catalog.yaml"; then
              echo ">>> config/taxa_catalog.yaml was modified; forcing INGEST_DATA=true"
              export INGEST_DATA=true
            fi

            bash scripts/ci/jenkins_gpu_train.sh
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

            # Hydrate model checkpoints from DVC (prefer in-cluster SeaweedFS edge cache, fallback to DagsHub origin)
            echo ">>> Hydrating model checkpoints from DVC remotes..."
            PULLED=false
            if kubectl get service seaweedfs -n taxon-vision >/dev/null 2>&1; then
              pixi run --frozen -e ci-dev dvc config --local remote.local-seaweedfs.access_key_id taxonadmin
              pixi run --frozen -e ci-dev dvc config --local remote.local-seaweedfs.secret_access_key taxonadmin
              echo ">>> Attempting checkpoint pull from in-cluster SeaweedFS (local-seaweedfs)..."
              if pixi run --frozen -e ci-dev dvc pull --force -r local-seaweedfs \
                  models/checkpoints/head.pt \
                  models/checkpoints/model_int8.onnx; then
                PULLED=true
                echo ">>> Checkpoints successfully hydrated from local-seaweedfs."
              fi
            fi

            if [ "$PULLED" != "true" ]; then
              echo ">>> Pulling checkpoints from DagsHub archive (origin)..."
              pixi run --frozen -e ci-dev dvc pull --force -r origin \
                models/checkpoints/head.pt \
                models/checkpoints/model_int8.onnx || \
              pixi run --frozen -e ci-dev dvc pull --force \
                models/checkpoints/head.pt \
                models/checkpoints/model_int8.onnx || \
              echo ">>> Warning: Remote checkpoints not found in DVC; falling back to existing host volume checkpoints."
            fi

            # Synchronize checkpoints to host volume if available and ensure read permissions for non-root API pod (UID 10001)
            if [ -d "/var/lib/taxon-vision/checkpoints" ] && [ -f "models/checkpoints/head.pt" ]; then
              echo ">>> Atomically staging checkpoints to host volume /var/lib/taxon-vision/checkpoints/..."
              for ckpt in models/checkpoints/*; do
                  fname=$(basename "$ckpt")
                  cp -f "$ckpt" "/var/lib/taxon-vision/checkpoints/.${fname}.tmp"
                  chmod a+r "/var/lib/taxon-vision/checkpoints/.${fname}.tmp"
                  mv -f "/var/lib/taxon-vision/checkpoints/.${fname}.tmp" "/var/lib/taxon-vision/checkpoints/${fname}"
              done
            fi

            # Apply API deployment and service manifests
            kubectl apply -f deploy/k8s/api-deployment.yaml

            # Trigger rolling restart to serve fresh weights
            kubectl rollout restart deployment/taxon-vision-api -n taxon-vision

            # Verify deployment health
            kubectl rollout status deployment/taxon-vision-api -n taxon-vision --timeout=120s
            bash scripts/ci/verify_deployment.sh taxon-vision
            echo ">>> Deployment successfully verified."
          '''
        }
      }
    }
  }

  post {
    always {
      script {
        try {
          cleanWs notFailBuild: true
        } catch (Throwable t) {
          echo "Workspace cleanup skipped: ${t.message}"
        }
      }
    }
  }
}
