import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('')
    model.track(source='',
                imgsz=640,
                project='',
                name='',
                save=True
                )