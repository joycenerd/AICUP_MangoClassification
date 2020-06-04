from .resnest.restnest import get_model
from efficientnet_pytorch import EfficientNet
from options import opt

def get_net(model):
    if model[0:8]=='resnest':
        get_model(model)
    elif model=='efficientnet-b7':
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes = opt.num_classes) 
        return model
    elif model == 'efficientnet-b5':
        model = EfficientNet.from_pretrained('efficientnet-b5', num_classes = opt.num_classes) 
        return model
    elif model == 'efficientnet-b4':
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes = opt.num_classes) 
        return model