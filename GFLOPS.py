import torch
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('') # choose your yaml file
    model.model.eval()
    model.info(detailed=True)
    try:
        model.profile(imgsz=[640, 640])
    except Exception as e:
        print(e)
        pass
    print('after fuse:', end='')
    model.fuse()