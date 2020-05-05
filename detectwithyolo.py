import argparse
import torch
import pandas as pd
from helper import *
from darknet import DarkNet

def arg_parse():
    """
    Parse arguments to the detection module
    
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Implementation')
    
    parser.add_argument("--shownet", dest='shownet', help = "Print model flag", default = 0, type=str)
    parser.add_argument("--images", dest='images', help = "Image/Directory to perform detection upon", default = "images", type=str)
    parser.add_argument("--outputs", dest='outputdir', help = "Image/Directory to store detections", default = "outputs", type=str)
    parser.add_argument("--batch", dest="batch", help = "Batch size", default = 1, type=int)
    parser.add_argument("--confidence", dest="confidence", help="Objectness threshold to filter predictions[0-1]", default = 0.5)
    parser.add_argument("--iou", dest="iou", help="IoU Threshhold[0-1]", default = 0.4)
    parser.add_argument("--configpath", dest='configpath', help="Config file", default="config/yolov3.cfg", type = str)
    parser.add_argument("--weightpath", dest ='weightpath', help="weights path", default="config/yolov3.weights", type = str)
    parser.add_argument("--resolution", dest='resolution', help="Input image resolution used to control speed/accuracy", default="416", type=int)
    
    return parser.parse_args()

args = arg_parse()
images = args.images
outputdir = args.outputdir
batch_size = 1 #int(args.batch) gives an error now.. 
confidence = float(args.confidence)
iou_treshold= float(args.iou)
weight_path = args.weightpath
config_path = args.configpath
input_dim = args.resolution

classnames = load_classnames('config/classnames.coco')
num_classes = 80

model = DarkNet(config_path=config_path)
model.load_weights(weight_path=weight_path)
model.to(model.device)
model.eval()

#load images 
imgs, imglist = get_imglist(images)
data_size = len(imgs)

#TODO: change this to use image loader
#PyTorch Variables for images
img_batches = list(map(preprocess_image, imgs, [input_dim for x in range(data_size)]))

#List containing dimensions of original images
img_dim_list = [(x.shape[1], x.shape[0]) for x in imgs]
img_dim_list = torch.FloatTensor(img_dim_list).repeat(1,2)

if torch.cuda.is_available():
    img_dim_list = img_dim_list.to('cuda:0')
    
extra = 0 
if (data_size % batch_size):
    extra = 1

if batch_size != 1:
  num_batches = len(imgs) // batch_size + extra            
  img_batches = [torch.cat((img_batches[i*batch_size : min((i + 1)*batch_size,
                       len(img_batches))]))  for i in range(num_batches)]
                       

finaloutput = None

for i, batch in enumerate(img_batches):
    #forward pass 
    if torch.cuda.is_available():
        batch = batch.to("cuda:0")

    with torch.no_grad():
      prediction = model(Variable(batch))
    
    #get bounding boxes
    prediction = get_true_detections(prediction, num_classes=80, conf_treshold=confidence, iou_treshold=iou_treshold)

    if prediction is None:
        for im_num, image in enumerate(imgs[i*batch_size : min((i + 1)*batch_size, data_size)]):
            img_id = i*batch_size + im_num
        continue

    prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist 

    if finaloutput is None: 
        finaloutput = prediction
    else:
        finaloutput = torch.cat((finaloutput,prediction))

    for img_num, image in enumerate(imgs[i*batch_size: min((i +  1)*batch_size, data_size)]):
        img_id = i*batch_size + img_num
        objs = [classnames[int(x[-1])] for x in finaloutput if int(x[0]) == img_id]
        

finaloutput = rescale(finaloutput, img_dim_list, input_dim)

results = list(map(lambda x: writeimg(x, imgs, classnames), finaloutput))

obj_paths = pd.Series(imglist).apply(lambda x: f"{outputdir}/out_{x.split('/')[-1]}")

outputdir = os.path.join(os.path.realpath('.'), outputdir)

if not os.path.exists(outputdir):
    os.makedirs(outputdir)

writeimages = list(map(cv2.imwrite, obj_paths, imgs))


print('Detection Complete!')
print(f'Check results in {outputdir}')