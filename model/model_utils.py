from .resnest.restnest import get_model
from efficientnet_pytorch import EfficientNet
from options import opt
from .lambdaresnet import LambdaResNet50, LambdaResNet101, LambdaResNet152, LambdaResNet200, LambdaResNet270, LambdaResNet350, LambdaResNet420

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
    elif model == 'efficientnet-b3':
        model = EfficientNet.from_pretrained('efficientnet-b3', num_classes = opt.num_classes)
        return model
    elif model == 'lambdaresnet50':
        model = LambdaResNet50(num_classes=opt.num_classes)
    elif model == 'lambdaresnet101':
        model = LambdaResNet101(num_classes=opt.num_classes)
    elif model == 'lambdaresnet152':
        model = LambdaResNet152(num_classes=opt.num_classes)
    elif model == 'lambdaresnet200':
        model = LambdaResNet200(num_classes=opt.num_classes)
    elif model == 'lambdaresnet270':
        model = LambdaResNet270(num_classes=opt.num_classes)
    elif model == 'lambdaresnet350':
        model = LambdaResNet350(num_classes=opt.num_classes)
    elif model == 'lambdaresnet420':
        model = LambdaResNet420(num_classes=opt.num_classes)