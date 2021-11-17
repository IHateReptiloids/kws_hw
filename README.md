## Установка
```
pip install -r requirements.txt
```
## Выполненная работа
1. Воспроизведена модель с семинара. Код обучения находится в `baseline.ipynb`, веса модели сохранены в файле `checkpoints/baseline.pth`. Качество на валидации примерно `3e-5`.
2. Реализован стриминг. Модель находится в файле `src/models/streaming_crnn.py`. Также приведен пример его работы в `streaming_demo.ipynb`.