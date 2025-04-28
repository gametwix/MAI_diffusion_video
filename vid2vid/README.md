## Начало работы с видео-трансформацией

### Подготовка

Рекомендуется использовать [gdown](https://github.com/wkentaro/gdown) для подготовки данных и моделей.
```bash
# Установка gdown
pip install gdown
pip install --upgrade gdown
```

Скачайте тестовые видео.

```bash
cd vid2vid
gdown https://drive.google.com/drive/folders/1q963FU9I4I8ml9_SeaW4jLb4kY3VkNak -O demo_selfie --folder
```

(Рекомендуется) Скачайте веса LORA для лучшей стилизации.

```bash
# Убедитесь, что вы находитесь в директории vid2vid
gdown https://drive.google.com/drive/folders/1D7g-dnCQnjjogTPX-B3fttgdrp9nKeKw -O lora_weights --folder
```

| Ключевые слова                                            | Веса LORA     |
|----------------------------------------------------------|------------------|
| 'pixelart' ,  'pixel art' ,  'Pixel art' ,  'PixArFK'    | PixelArtRedmond15V-PixelArt-PIXARFK.safetensors |
| 'lowpoly', 'low poly', 'Low poly'                        | low_poly.safetensors |
| 'Claymation', 'claymation'                               | Claymation.safetensors |
| 'crayons', 'Crayons', 'crayons doodle', 'Crayons doodle' | doodle.safetensors |
| 'sketch', 'Sketch', 'pencil drawing', 'Pencil drawing'   | Sketch_offcolor.safetensors |
| 'oil painting', 'Oil painting'                           | bichu-v0612.safetensors |

### Оценка

```bash
# Оценка одного видео
python main.py --input ./demo_selfie/jeff_1.mp4 --prompt "Elon Musk is giving a talk."
python main.py --input ./demo_selfie/jeff_1.mp4 --prompt "Claymation, a man is giving a talk."
```

```bash
# Оценка пакета видео
python batch_eval.py --json_file ./demo_selfie/eval_jeff_celebrity.json # Редактирование лиц
python batch_eval.py --json_file ./demo_selfie/eval_jeff_lorastyle.json # Стилизация
```

ПРИМЕЧАНИЕ: Опция `--acceleration tensorrt` НЕ ПОДДЕРЖИВАЕТСЯ! Я пытался ускорить модель с помощью TensorRT, но из-за динамической природы банка признаков это не удалось.

Вы также можете использовать веб-интерфейс, предоставляемый Gradio. Для этого нужно установить `gradio` с помощью `pip install gradio` и запустить его с помощью `python main_gr.py`

### Исследование с помощью команд

```bash
# Не использовать банк признаков, модель вернется к покадровой обработке
python main.py --input ./demo_selfie/jeff_1.mp4 --prompt "Claymation, a man is giving a talk." --use_cached_attn False --output_dir outputs_streamdiffusion
```

```bash
# Укажите силу шума. Чем выше noise_strength, тем больше шума добавляется к начальным кадрам.
# Более высокая сила обычно приводит к лучшим эффектам редактирования, но может снизить согласованность. По умолчанию 0.4.
python main.py --input ./demo_selfie/jeff_1.mp4 --prompt "Claymation, a man is giving a talk." --noise_strength 0.8 --output_dir outputs_strength
```

```bash
# Укажите шаги диффузии. Больше шагов обычно приводит к более высокому качеству, но меньшей скорости.
# По умолчанию 4.
python main.py --input ./demo_selfie/jeff_1.mp4 --prompt "Claymation, a man is giving a talk." --diffusion_steps 1 --output_dir outputs_steps
```

### Распространенные ошибки

#### Проблема ImportError
- **Сообщение об ошибке**: `ImportError: cannot import name 'packaging' from 'pkg_resources'`.

**Возможное решение**:  
Понизьте версию пакета setuptools. Вы можете сделать это, выполнив следующую команду в терминале:

```bash
pip install setuptools==69.5.1
```