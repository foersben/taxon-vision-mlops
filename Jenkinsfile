pipeline {
  agent {
    kubernetes {
      cloud 'k3s'
      defaultContainer 'ml-runner'
      yaml '''
apiVersion: v1
kind: Pod
metadata:
  labels:
    app.kubernetes.io/name: taxon-vision-jenkins-agent
spec:
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
        memory: "16Gi"
        cpu: "8"
      requests:
        memory: "4Gi"
        cpu: "2"
  volumes:
  - name: rattler-cache
    hostPath:
      path: /home/benni/.cache/rattler/cache
      type: DirectoryOrCreate
'''
    }
  }

  options {
    buildDiscarder(logRotator(numToKeepStr: '15', daysToKeepStr: '14'))
    timeout(time: 45, unit: 'MINUTES')
    timestamps()
    disableConcurrentBuilds(abortPrevious: true)
  }

  stages {
    stage('Prepare Toolchain') {
      steps {
        container('ml-runner') {
          sh '''
            echo ">>> Setting up toolchain in ephemeral agent..."
            which git >/dev/null 2>&1 || pixi global install git
            git config --global --add safe.directory "*"
            pixi --version
            pixi install --frozen -e ci-dev
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
  }

  post {
    always {
      cleanWs()
    }
  }
}
