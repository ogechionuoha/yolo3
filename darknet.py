#Darknet implementation for Yolo

import torch
import numpy as np
import torch.nn as nn
from darknethelper import *
from helper import *

class DarkNet(nn.Module):
  """
  Implementation of the darknet architecture based on the network architecture provided by pjreddie
  Architecture config is found here: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg

  """
  def __init__(self, config_path='config/yolov3.cfg', use_cuda=True):
    super().__init__()
    self.layers = parse_config(config_path)
    self.net, self.network_config = self._create_network()
    self.device = 'cpu'
    if use_cuda and torch.cuda.is_available():
      self.device = 'cuda:0'

  def _get_weights(self, path):
    with open(path, 'rb') as weight_file:
      head = np.fromfile(weight_file, dtype=np.int32, count = 5)
      self.head = torch.from_numpy(head)
      self.images_seen = self.head[3]
      weights = np.fromfile(weight_file, dtype=np.float32)

    return weights
  
  def _create_network(self):
    """
    Create a layer using details provided by layer_config.
    
    :params layer_config: (dict) Dictionary of layer configurations
    :return net: (ModuleList) network modules
    :return network_config: (list) configurations and hyper parameters of the network
    """
    layer_config = self.layers
    network_config = {}
    #use ModuleList to hold modules
    net = nn.ModuleList()

    filter_list = []
    
    for key in sorted(layer_config):
        
        config,num,layer_type = layer_config[key], *key
        
        if num > 0:
            module = nn.Sequential()

            #create a block
            if layer_type.lower() == 'convolutional':

                filters = int(config['filters'])
                conv  = create_conv_layer(config, prev_filters)
                activation, activation_name = create_activation(config['activation'])
                use_batchnorm = config.get('batch_normalize',0)
                
                #add modules to sequential block
                module.add_module(name=f"{layer_type}_{num-1}", module=conv)
                if use_batchnorm:
                  module.add_module(name=f"batch_norm_{num-1}", module=nn.BatchNorm2d(filters))

                if activation: 
                  module.add_module(name=f"{activation_name}_{num-1}", module=activation)
                  
                prev_filters = filters               

            elif layer_type.lower() == 'shortcut':
                skip = create_skip_layer(config)
                module.add_module(f"skip_{num-1}", skip)

            elif layer_type.lower() == 'route':
                route, begin, end = create_route_layer(config)
                #positive annotation
                if begin > 0: begin -= num
                if end > 0: end -= num
                
                if end < 0:
                    prev_filters = filter_list[num + begin] + filter_list[num + end]
                else:
                    prev_filters = filter_list[num + begin]

                module.add_module(f"route_{num-1}", route)
                    
            elif layer_type.lower() == 'upsample':
                stride = int(config['stride'])
                upsample = create_upsample_layer(config)
                module.add_module(name=f"{layer_type}_{num-1}", module=upsample)
                
            elif layer_type.lower() == 'yolo':
                yolo = create_yolo_layer(config)
                module.add_module(f"yolo_{num-1}", yolo)
            
            net.append(module)  
                                    
        else:
          #set up network parameters
          network_config = config
          network_config['height'] = 416
          network_config['width'] = 416
          prev_filters = int(network_config['channels'])

        filter_list.append(prev_filters)

    print(f"Network parsing complete!")  
          
    return net, network_config
  
  def load_weights(self, weight_path=None):
    assert weight_path is not None, "Please provide the weight using param weight_path"

    weights = self._get_weights(weight_path)
    ind = 0

    blocks = list(self.layers.keys())
    
    for i in range(len(self.net)):

      _,layer_type = blocks[i+1]

      if layer_type == 'convolutional':
        block = self.net[i]

        #because some conv_blocks do not use batch_norm e.g final convolution
        batch_normalize = int(self.layers[blocks[i+1]].get("batch_normalize",0))

        conv_block = block[0]

        if batch_normalize:
          
          bn_block = block[1]
          num_biases = bn_block.bias.numel()

          #load biases, reshape, copy to block
          bn_biases = torch.from_numpy(weights[ind : ind+num_biases])
          bn_biases = bn_biases.view_as(bn_block.bias.data)
          bn_block.bias.data.copy_(bn_biases)
          ind += num_biases

          #load weights, reshape, copy to block
          bn_weights = torch.from_numpy(weights[ind : ind+num_biases])
          bn_weights = bn_weights.view_as(bn_block.weight.data)
          bn_block.weight.data.copy_(bn_weights)
          ind  += num_biases

          #load means, reshape, copy to block
          bn_mean = torch.from_numpy(weights[ind : ind+num_biases])
          bn_mean = bn_mean.view_as(bn_block.running_mean)
          bn_block.running_mean.copy_(bn_mean)
          ind  += num_biases

          #load_variances, reshape, copy to block
          bn_var = torch.from_numpy(weights[ind : ind+num_biases])
          bn_var = bn_var.view_as(bn_block.running_var)
          bn_block.running_var.copy_(bn_var)
          ind  += num_biases
        
        else:
          #Number of biases
          num_biases = conv_block.bias.numel()

          #Load biases,reshape, copy to block
          conv_biases = torch.from_numpy(weights[ind: ind + num_biases])
          conv_biases = conv_biases.view_as(conv_block.bias.data)
          conv_block.bias.data.copy_(conv_biases)
          ind += num_biases

        #Load weights, reshape, copy to block
        num_weights = conv_block.weight.numel()
        conv_weights = torch.from_numpy(weights[ind:ind+num_weights])
        conv_weights = conv_weights.view_as(conv_block.weight.data)
        conv_block.weight.data.copy_(conv_weights)
        ind += num_weights 

    print('Loading weights complete!')  


  def forward(self, x):
    self.out = {}
    blocks = list(self.layers.keys())[1:]
    detections = None
    x.to(self.device)

    for i,key in enumerate(blocks):
      _,layer_type = key
      #perform forward pass for different layer types
      if layer_type in ['convolutional', 'upsample']:      
        x = self.net[i](x)
        
      elif layer_type == 'shortcut':
        from_layer_ = int(self.layers[key]['from'])
        x = self.out[i-1] + self.out[i+from_layer_]

      elif layer_type == 'route':
        route_layers = self.layers[key]['layers'].split(',')
        route_layers = [int(route_layer) for route_layer in route_layers]
        
        if route_layers[0] > 0: 
          route_layers[0] -= i 
        if len(route_layers) == 1: 
          x = self.out[i + route_layers[0]]
        else:
          if route_layers[1] > 0:
            route_layers[1] -= i

          input1 = self.out[i + route_layers[0]]
          input2 = self.out[i + route_layers[1]]

          x = torch.cat((input1, input2), 1)        

      elif layer_type == 'yolo':        
        anchors = self.net[i][0].anchors
        #Get the input dimensions
        input_dim = int(self.network_config["height"])

        #Get the number of classes
        num_classes = int(self.layers[key]["classes"])

        #get bounding boxes 
        x = x.data
        x = get_bounding_boxes(x, num_classes, anchors, input_dim, self.device)
        if detections is None:              #if no collector has been intialised. 
            detections = x
        else:       
            detections = torch.cat((detections, x), 1)   #tack on all the predictions gotten at different strides.     

      self.out[i] = x

    return detections
