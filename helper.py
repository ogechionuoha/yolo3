#Helper functions
import os
import torch
import cv2 
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

def get_imglist(inputpath):
    '''
    Loads all the .png, .jpg and .jpeg images in inputpath.
    If inputpath is an image it is loaded
    
    :params inputpath: (str) path to image/images folder
    :retunrs imgs: (list) loaded images
    :retunrs imglist: (list) list of images (full path)
    '''
    inputpath = os.path.join(os.path.realpath('.'), inputpath)
    imglist = []
    if os.path.isdir(inputpath):
      images = os.listdir(inputpath)
      imglist = [os.path.join(inputpath, img) for img in images if (img.split('.'))[-1].lower() in ['png','jpg','jpeg']]
    else:
      if (inputpath.split('.'))[-1].lower() in ['png','jpg','jpeg']:
          imglist = [inputpath]
          
    imgs = [cv2.imread(img) for img in imglist]
      
    return imgs, imglist
    
def preprocess_image(img, input_dim):
    """
    Preprocess an image -> resize, change channels from x,y,channel to channel,x,y then normalise
    
    :params img: input image
    :params input_dim: (int) dimension to resize input image
    :returns img: (torch Variable)
    """

    img = cv2.resize(img, (input_dim, input_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def get_bounding_boxes(output, classes, anchors, input_dim, device='cpu'):
    #print(classes, anchors, input_dim)
    #get relevant dimensions
    batch_size = output.size(0)
    stride = input_dim // output.size(2)
    grid_dim = input_dim // stride
    len_bounding_box = 5 + classes
    num_anchors = len(anchors)
    
    output = output.view(batch_size, num_anchors*len_bounding_box, grid_dim**2)
    output = output.transpose(1,2).contiguous()
    output = output.view(batch_size, grid_dim*grid_dim*num_anchors, len_bounding_box)

    anchors = [(anchor[0]/stride, anchor[1]/stride) for anchor in anchors]

    output = calculate_box_dim(output, grid_dim, anchors,stride, device)
    
    return output

def load_classnames(path):
  '''
  Load class names of the coco data set from path.
  
  :returns: names (list)
  '''
  names = []
  #check if path exists
  with open(path, 'r') as classnames:
    names = classnames.readlines()

  names = [name.rstrip('\n') for name in names]
  
  return names
  

#TODO: Change outputs to y_hat
def calculate_box_dim(outputs, grid_size, anchors, stride, device, num_classes = 80):
    outputs[:,:,0] = torch.sigmoid(outputs[:,:,0]) #sig(bx)
    outputs[:,:,1] = torch.sigmoid(outputs[:,:,1]) #by
    outputs[:,:,4] = torch.sigmoid(outputs[:,:,4])

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    x_offset = x_offset.to(device)
    y_offset = y_offset.to(device)

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,len(anchors)).view(-1,2).unsqueeze(0)

    outputs[:,:,:2] += x_y_offset

    anchors = torch.FloatTensor(anchors)
    anchors = anchors.to(device)

    anchors = anchors.repeat(grid_size**2, 1).unsqueeze(0)
    outputs[:,:,2:4] = torch.exp(outputs[:,:,2:4])*anchors

    #apply sigmod to class score
    outputs[:,:,5: 5 + num_classes] = torch.sigmoid((outputs[:,:, 5 : 5 + num_classes]))


    #scale box according to stride
    outputs[:,:,:4] *= stride

    return outputs

   
def calculate_iou(box1, box2):
    """
    Calculate the intersection over union (IoU) of two bounding boxes.
    box1 and box2 are represented by 4 coordinates each:
    (top_left_x, top_left_y, bottom_left_x, bottom_left_y)
    from: https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch/
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the coordinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
 
    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou


def unique_classes(tensor):
  tensor_np = tensor.cpu().numpy()
  unique_np = np.unique(tensor_np)
  unique_tensor = torch.from_numpy(unique_np)
  
  tensor_res = tensor.new(unique_tensor.shape)
  tensor_res.copy_(unique_tensor)
  
  return tensor_res
  

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas


def in_class_max(img_prediction, class_i):
  """
  Returns the bounding box predictions within a particular class (class_i)
  """
  #extract the detections for class_i
  class_mask = img_prediction*(img_prediction[:,-1] == class_i).float().unsqueeze(1)
  class_mask_ind = torch.nonzero(class_mask[:,-2], as_tuple= True)
  image_pred_class = img_prediction[class_mask_ind].view(-1,7)

  #sort the detections by objectness score (desc)
  conf_sort_index = torch.sort(image_pred_class[:,4], descending = True)[1]
  image_pred_class = image_pred_class[conf_sort_index]

  return image_pred_class


def get_box_corners(detection):
  """
    Convert bounding box ceter coordinates, width and height to top left and bottom right coordinates.
    
    returns list of corners and updated bounding box detections."""
  box_corners = detection.new(detection.shape)
  box_corners[:,:,0] = (detection[:,:,0] - detection[:,:,2]/2) #top left corner x
  box_corners[:,:,1] = (detection[:,:,1] - detection[:,:,3]/2) #top left corner y
  box_corners[:,:,2] = (detection[:,:,0] + detection[:,:,2]/2) #bottom right corner x
  box_corners[:,:,3] = (detection[:,:,1] + detection[:,:,3]/2) #bottom right corner y
  detection[:,:,:4] = box_corners[:,:,:4]

  return box_corners, detection

def rescale(finaloutput, img_dim_list, input_dim):
    """
    Rescale final detections to original image dimensions 
    """
    img_dim_list = torch.index_select(img_dim_list, 0, finaloutput[:,0].long())
    scaling_factor = torch.min(input_dim/img_dim_list,1)[0].view(-1,1)
    finaloutput[:,[1,3]] -= (input_dim - scaling_factor*img_dim_list[:,0].view(-1,1))/2
    finaloutput[:,[2,4]] -= (input_dim - scaling_factor*img_dim_list[:,1].view(-1,1))/2                       
    
    finaloutput[:,1:5] /= scaling_factor
    for i in range(finaloutput.shape[0]):
        finaloutput[i, [1,3]] = torch.clamp(finaloutput[i, [1,3]], 0.0, img_dim_list[i,0])
        finaloutput[i, [2,4]] = torch.clamp(finaloutput[i, [2,4]], 0.0, img_dim_list[i,1])
        
    return finaloutput

def get_true_detections(detections, num_classes, conf_treshold=0.5, iou_treshold=0.4):
  """
  Perform Non Max suppresion
  """
  output = None

  #filter using confidence treshold
  conf_mask = (detections[:,:,4] > conf_treshold).float().unsqueeze(2)
  detections = detections*conf_mask
  
  #get box_corners for iou calculations in non max suppression
  box_corners, detections = get_box_corners(detections)

  #iterate through each image the batch sinc eprocess cannot be vectorised
  batch_size = detections.size(0)

  for ind in range(batch_size):
    img_detections = detections[ind]
    
    #get class with max score in predictions list
    max_class, max_score = torch.max(img_detections[:,5:5+num_classes], 1)
    max_class = max_class.float().unsqueeze(1)
    max_score = max_score.float().unsqueeze(1)
    seq = (img_detections[:,:5], max_class, max_score)
    img_detections = torch.cat(seq, 1)

    #remove zeroed detections
    non_zero_ind = torch.nonzero(img_detections[:,4], as_tuple= False)
    
    
    try:
      image_detections_ = img_detections[non_zero_ind.squeeze(),:].view(-1,7)
    except:
      continue
    
    if image_detections_.shape[0] == 0:
      continue

    # get classes found for this image to be used for nonmax suppression
    img_classes = unique_classes(image_detections_[:,-1])

    for class_i in img_classes:
      class_i_pred = in_class_max(image_detections_, class_i)
      count = class_i_pred.size(0)

      for i in range(count):
        #Get the IOUs of all boxes that come after the one we are looking at 
        #in the loop
        try:
            ious = calculate_iou(class_i_pred[i].unsqueeze(0), class_i_pred[i+1:])
        except ValueError:
            break
        except IndexError:
            break

        #Zero out all the detections that have IoU > treshhold
        iou_mask = (ious < iou_treshold).float().unsqueeze(1)
        class_i_pred[i+1:] *= iou_mask       

        #Remove non-zero entries
        non_zero_ind = torch.nonzero(class_i_pred[:,4], as_tuple=True)
        class_i_pred = class_i_pred[non_zero_ind].view(-1,7)

      batch_ind = class_i_pred.new(class_i_pred.size(0), 1).fill_(ind)      
      #Repeat the batch_id for as many detections of class_i in the image
      seq = batch_ind, class_i_pred

      if output is None:
          output = torch.cat(seq,1)
      else:
          out = torch.cat(seq,1)
          output = torch.cat((output,out))

    return output
    
def writeimg(x, results, classnames, color= (255, 0, 0),):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(classnames[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img