#Helper functions to create YOLO architecture 

import torch
import torch.nn as nn

def parse_config(config_path):
    """
    Use the network config file to extract the required layers.
    
    :param config_path: (str) filepath of config file
    :return layers: (dict) dictionary of neural network layers' configurations
    
    """    
    #read file and remove empty lines and comments
    with open(config_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines if len(line) > 0 and line[0] not in ('#','\n')]
        
    # parse each line to obtain layer configurations
    layer_name=None
    layers = {}
    layer_config = {}
    layer_count = 0
    
    for line in lines:
        if line.startswith('['):
            if layer_config:
                layers[layer_name] = layer_config
                layer_config = {}
                layer_count += 1
            layer_name = (layer_count, line.lstrip('[').rstrip(']'))
        else:
            prop,val = line.split('=')            
            layer_config[prop.strip()] = val.strip()

    if layer_config:
      layers[layer_name] = layer_config

    return layers

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
    
def create_conv_layer(layer_config, prev_filters=3):
    """
    Create a conv layer using configurations in layer_config
    
    :params layer_config: (list) list of layer attributes
    :params prev_filters: (int) number of filters in previous layer
    :return conv: nn.Module convolutional layer
    """

    filters = int(layer_config['filters'])
    kernel_size = int(layer_config['size'])
    stride = int(layer_config['stride'])
    padding = int(layer_config['pad'])
    bias = not int(layer_config.get('batch_normalize',0))
    
    if padding:
        padding = (kernel_size - 1) // 2
    else:
        padding = 0
    
    return nn.Conv2d(prev_filters, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def create_activation(name):
    """
    Create ativation
    :params name: (str) activation name
    :return activation, activation name or None
    """
    if name.lower() == 'leaky':
      return nn.LeakyReLU(0.1, inplace=True), name.lower()
    return None, None  

def create_upsample_layer(layer_config):
  """
  Create upsampling layer from config
  :params layer_config: (dict) dictionary of layer configurations
  :return: nn.Upsample layer
  """
  return nn.Upsample(scale_factor = 2, mode = "nearest")

def create_route_layer(layer_config):
  """
  Create route layer from config
  :params layer_config: (dict) dictionary of layer configurations
  :return: EmptyLayer, begin, end
  """
  layers = layer_config['layers'].split(',')

  begin = int(layers[0])
  end = int(layers[1]) if len(layers) > 1 else 0

  return EmptyLayer(), begin, end


def create_skip_layer(layer_config):
  """
  Create skil layer from config (think resnet)
  :params layer_config: (dict) dictionary of layer configurations
  :return: EmptyLayer layer
  """
  return EmptyLayer()

def create_yolo_layer(layer_config):
  anchors = [int(a) for a in (layer_config['anchors'].strip().split(','))]
  mask = [int(m) for m in (layer_config['mask'].strip().split(','))]  
  anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
  anchors = [anchors[i] for i in mask]

  return DetectionLayer(anchors)