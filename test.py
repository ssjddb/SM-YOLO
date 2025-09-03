import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('')
    model.val(data='',
              #split='val',
              imgsz=640,
              batch=64,
              workers=64,
              device=0,
              # iou=0.7,
              # rect=False,
              # save_json=True,
              project='',
              name='',
              )