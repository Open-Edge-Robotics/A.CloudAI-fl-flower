# Federated Learning for Object Detection

This is a simple example of federated learning for object detection using the Flower framework.

## Requirements

- [Python](https://www.python.org/) 3.12
- [Flower Framework](https://flower.ai/) 1.11.0
- [PyTorch](https://pytorch.org/) 2.5.0

## Set Virtual Environment

### Install Pyenv

```bash
# Install pyenv
curl https://pyenv.run | bash
```

### Install Python and Create Virtual Environment

```bash
# Install python 3.12.2 using pyenv
pyenv install 3.12.2
pyenv local 3.12.2

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Prepare Data

- Client need to download the PennFudan dataset

```bash
cd src/flower_obj

wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip -P data
cd data && unzip PennFudanPed.zip
```

## Server

```bash
cd src/flower_obj

python server.py
```

## Client

### Set Server Address

```bash
export SERVER_IP=192.168.5.10
export SERVER_PORT=8080
```

### Run Client

```bash
cd src/flower_obj

python client.py
```
