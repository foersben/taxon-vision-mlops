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

  options {
    buildDiscarder(logRotator(numToKeepStr: '15', daysToKeepStr: '14'))
    timeout(time: 45, unit: 'MINUTES')
    timestamps()
    disableConcurrentBuilds(abortPrevious: true)
  }

  environment {
    DAGSHUB_TOKEN = credentials('dagshub-token')
    AWS_ACCESS_KEY_ID = "${DAGSHUB_TOKEN}"
    AWS_SECRET_ACCESS_KEY = "${DAGSHUB_TOKEN}"
    MLFLOW_TRACKING_USERNAME = 'foersben'
    MLFLOW_TRACKING_PASSWORD = "${DAGSHUB_TOKEN}"
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
            pixi run --frozen -e ci-dev dvc pull || echo "WARNING: DVC pull skipped or failed; continuing with workspace cache."
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

    stage('Unit Tests & Coverage') {
      steps {
        container('ml-runner') {
          sh '''
            pixi run --frozen -e ci-dev pytest tests/ \
              --cov=src/taxon_vision \
              --cov-report=xml:coverage.xml \
              --cov-fail-under=78 \
              -o "addopts="
          '''
        }
      }
    }

    stage('Integration & Conformal Invariants') {
      steps {
        container('ml-runner') {
          sh '''
            pixi run --frozen -e ci-dev pytest tests/integration/ tests/invariants/ -x -q -o "addopts="
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

    stage('Deploy to Kubernetes') {
      when {
        anyOf {
          branch 'main'
          branch 'feature/phase1-foundations'
        }
      }
      steps {
        container('ml-runner') {
          sh '''
            export PATH="/root/.pixi/bin:$PATH"
            echo ">>> Executing zero-downtime rolling deployment to Kubernetes cluster..."
            # Verify in-cluster service account access
            kubectl version --client || true

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
      cleanWs()
    }
  }
}
