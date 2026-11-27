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
    image: ghcr.io/prefix-dev/pixi:0.84.2
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
    booleanParam(name: 'SKIP_CODE_GATES', defaultValue: false, description: 'Skip static quality, test, and docs gates (intended exclusively for automated Airflow drift retrains and scheduled runs)')
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
      when {
        beforeAgent true
        not {
          anyOf {
            triggeredBy 'TimerTrigger'
            expression { params.SKIP_CODE_GATES == true }
          }
        }
      }
      parallel {
        stage('Lint & Format') {
          steps {
            container('ml-runner') {
              sh 'pixi run --frozen -e ci-dev just ci-lint'
            }
          }
        }
        stage('Type Checking') {
          steps {
            container('ml-runner') {
              sh 'pixi run --frozen -e ci-dev just ci-typecheck'
            }
          }
        }
        stage('Compliance & OKF Audits') {
          steps {
            container('ml-runner') {
              sh 'pixi run --frozen -e ci-dev just ci-audit'
            }
          }
        }
      }
    }

    stage('Unit Tests') {
      when {
        beforeAgent true
        not {
          anyOf {
            triggeredBy 'TimerTrigger'
            expression { params.SKIP_CODE_GATES == true }
          }
        }
      }
      steps {
        container('ml-runner') {
          sh 'pixi run --frozen -e ci-dev just ci-test-unit'
        }
      }
    }

    stage('Integration & Conformal Invariants') {
      when {
        beforeAgent true
        not {
          anyOf {
            triggeredBy 'TimerTrigger'
            expression { params.SKIP_CODE_GATES == true }
          }
        }
      }
      steps {
        container('ml-runner') {
          sh 'pixi run --frozen -e ci-dev just ci-test-integration'
        }
      }
    }

    stage('Documentation Strict Build') {
      when {
        beforeAgent true
        not {
          anyOf {
            triggeredBy 'TimerTrigger'
            expression { params.SKIP_CODE_GATES == true }
          }
        }
      }
      steps {
        container('ml-runner') {
          sh 'pixi run --frozen -e ci-dev just ci-docs'
        }
      }
    }

    stage('Model Training & Checkpoint (GPU)') {
      options {
        timeout(time: 150, unit: 'MINUTES')
      }
      environment {
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
              changeset "config/default_config.yaml"
              changeset "config/conformal_config.yaml"
              changeset "config/taxa_catalog.yaml"
              changeset "data/**"
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
          yamlFile 'deploy/k8s/optional/jenkins-gpu-pod.yaml'
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
            sh 'bash scripts/ci/deploy_k8s.sh "${BRANCH_NAME}"'
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
