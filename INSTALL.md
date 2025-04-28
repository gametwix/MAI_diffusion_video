# Инструкция по установке

## Требования к системе

- Python 3.10+
- CUDA 11.8+
- PyTorch 2.0+
- TensorRT 8.6+
- ONNX 1.14+
- ONNX Runtime 1.15+
- Stable Fast 0.0.3dev

## Установка CUDA и cuDNN

1. Установите CUDA 11.8:
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

2. Установите cuDNN 8.6:
```bash
wget https://developer.nvidia.com/compute/cudnn/secure/8.6.0/local_installers/11.8/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
tar -xf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

## Установка PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Установка TensorRT

1. Скачайте TensorRT 8.6:
```bash
wget https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-repo-ubuntu2004-cuda11.8-trt8.6.1.6-ga-20231013_1-1_amd64.deb
```

2. Установите TensorRT:
```bash
sudo dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.8-trt8.6.1.6-ga-20231013_1-1_amd64.deb
sudo apt-get update
sudo apt-get install tensorrt
```

## Установка ONNX и ONNX Runtime

```bash
pip install onnx==1.14.0
pip install onnxruntime-gpu==1.15.0
```

## Установка Stable Fast

```bash
pip install stable-fast==0.0.3dev
```

## Установка проекта

1. Клонируйте репозиторий:
```bash
git clone https://github.com/your-username/video-transformation.git
cd video-transformation
```

2. Установите зависимости:
```bash
pip install -e .
```

## Проверка установки

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import tensorrt; print(tensorrt.__version__)"
python -c "import onnx; print(onnx.__version__)"
python -c "import onnxruntime; print(onnxruntime.__version__)"
python -c "import stable_fast; print(stable_fast.__version__)"
```
