#Yolo 3 implementation in pytorch

Implementing Yolo 3 using architecture provided by jpreddie
Test using pretained weights for object detection

##Todos
- Fix batch bug
- Update to use loaders
- fix cuda bug

## Setup/config

```
mkdir config
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -P config/
wget https://pjreddie.com/media/files/yolov3.weights -P config/
wget https://raw.githubusercontent.com/ogechionuoha/files/master/classnames.coco -P config/

```

## Detection 

Use:

```
python detectwithyolo.py --configpath config/yolov3.cfg --weightpath config/yolov3.weights --images images
```

