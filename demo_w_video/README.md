# Демо с видеофайлом

Это демонстрация трансформации видео с использованием предварительно записанного видеофайла.

## Требования

- Python 3.10+
- CUDA 11.8+
- PyTorch 2.0+
- TensorRT 8.6+
- ONNX 1.14+
- ONNX Runtime 1.15+
- Stable Fast 0.0.3dev

## Установка

1. Установите зависимости:
```bash
pip install -e .
```

2. Скачайте веса моделей:
```bash
python download_weights.py
```

## Использование

### Запуск демо

```bash
python main.py --input_video path/to/your/video.mp4
```

### Параметры запуска

- `--input_video`: Путь к входному видеофайлу (обязательный параметр)
- `--output_video`: Путь для сохранения выходного видео (по умолчанию "output.mp4")
- `--acceleration`: Выбор метода ускорения (tensorrt или sfast)
- `--width`: Ширина выходного видео (по умолчанию 512)
- `--height`: Высота выходного видео (по умолчанию 512)
- `--prompt`: Текст для генерации (по умолчанию "A man is talking")

### Примеры

1. Запуск с TensorRT:
```bash
python main.py --input_video input.mp4 --acceleration tensorrt
```

2. Запуск с Stable Fast:
```bash
python main.py --input_video input.mp4 --acceleration sfast
```

3. Запуск с пользовательским промптом:
```bash
python main.py --input_video input.mp4 --prompt "Your custom prompt here"
```

## Примечания

- Для достижения наилучшей производительности рекомендуется использовать GPU NVIDIA RTX 4090 или лучше
- При использовании TensorRT требуется предварительная компиляция моделей
- При использовании Stable Fast рекомендуется установить xformers и Triton для максимальной производительности
- Поддерживаются форматы видео: MP4, AVI, MOV 