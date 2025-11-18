# Flower framework with YOLO

플라워 프레임워크와 YOLO V8 모델을 사용하여 간단한 연합학습 구현

```
-연구개발과제명: 일상생활 공간에서 자율행동체의 복합작업 성공률 향상을 위한 자율행동체 엣지 AI SW 기술 개발
-
-세부 개발 카테고리
-● 지속적 지능 고도화를 위한 자율적 흐름제어 학습 프레임워크 기술 분석 및 설계
-- 기밀성 데이터 활용 지능 고도화를 위한 엣지와 클라우드 분산 협업 학습 프레임워크 기술
-- 엣지와 클라우드 협력 학습 간 최적 자원 활용 및 지속적 지능 배포를 위한 자율적 학습흐름제어 기술
-
-개발 내용 
-- 엣지와 클라우드 분산 협업을 위한 지속적 지능 배포 프레임워크 
-- 자율행동체 엣지 기반 클러스터링 솔루션 및 분산 학습 프레임워크 개발
-```

## Dev Environment

- [Python](https://www.python.org/) 3.10.12
- [Flower Framework](https://flower.ai/) 1.15.0
- [YOLO](https://docs.ultralytics.com/) V8

## Dependency Installation

### 1. Create venv

```bash
python3 -m venv venv

. venv/bin/activate
```

### 2. Install Packages

```bash
pip install -r requirements.txt
```

## Run Flower with Deployment Engine

플라워 배포 기능을 사용하여 연합학습 진행

### 1. Set Project toml [pyproject.toml](./pyproject.toml)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "flwr_yolo"
version = "0.0.1"
description = "Federated Learning with PyTorch, YOLO and Flower"
license = "Apache-2.0"
requires-python = ">=3.10"
readme = "README.md"
dependencies = [
    "fire>=0.7.1",
    "flwr[simulation]>=1.15.0",
    "torch==2.5.1",
    "torchvision==0.20.1",
    "ultralytics==8.3.220",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "lge"

[tool.flwr.app.components]
serverapp = "flwr_yolo.server:app"
clientapp = "flwr_yolo.client:app"

[tool.flwr.app.config]
min-available-clients = 2
num-server-rounds = 3
fraction-fit = 0.5
fraction-evaluate = 1.0
epochs = 30
learning-rate = 0.1
batch-size = 32

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-deployment]
address = "127.0.0.1:9093"
insecure = true

[tool.flwr.federations.former]
address = "10.231.172.246:9093"
insecure = true

[tool.flwr.federations.local-simulation]
options.num-supernodes = 10

[tool.flwr.federations.local-simulation-gpu]
options.num-supernodes = 10
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 0.2
```

### 2. Run Flower superlink

```bash
flower-superlink --insecure
```

### 3. Run Flower supernode

```bash
# Clinet 1
flower-supernode \
     --insecure \
     --superlink 127.0.0.1:9092 \
     --clientappio-api-address 127.0.0.1:9094 \
     --node-config 'client_id=0'

# Client 2
flower-supernode \
     --insecure \
     --superlink 127.0.0.1:9092 \
     --clientappio-api-address 127.0.0.1:9095 \
     --node-config 'client_id=1'
```

### 4. Run Flower Federation

```bash
flwr run . local-deployment --stream
```
