Видео-трансформация в реальном времени

Этот проект представляет собой демонстрацию трансформации видео в реальном времени с использованием диффузионных моделей.

# Требования

- Python 3.10+
- CUDA 11.8+
- PyTorch 2.0+
- TensorRT 8.6+
- ONNX 1.14+
- ONNX Runtime 1.15+
- Stable Fast 0.0.3dev

# Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/your-username/video-transformation.git
cd video-transformation
```

2. Установите зависимости:
```bash
pip install -e .
```

# Использование

## Демо с веб-камерой

```bash
python demo_w_video/main.py
```

## Демо с видеофайлом

```bash
python demo_w_video/main.py
```

# Ускорение

Проект поддерживает два метода ускорения:

1. TensorRT
2. Stable Fast

Для использования TensorRT:
```bash
python demo_w_video/main.py --acceleration tensorrt
```

Для использования Stable Fast:
```bash
python demo_w_video/main.py --acceleration sfast
```
