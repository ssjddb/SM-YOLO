from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    model = YOLO('yaml/v8-n-SKS-MFF.yaml')   
    train_config = {
        'cfg': 'ultralytics/cfg/default.yaml',
        'data': '',
        'project': '',
        'name': '',
        'epochs': 500,  
        'imgsz': 640,  
        'patience': 100,
        'workers': 64,
        'batch': 64,
        'cache': False,
        'single_cls': False,  
        'device': 0, 
    }

    model.train(**train_config)