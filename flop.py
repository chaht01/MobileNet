import torch
from models import MobileNet, MobileNet2, BaseLineNet
from thop import profile, clever_format


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


x = torch.randn(1, 3, 64, 64)
models = [BaseLineNet(), MobileNet2(width_mult=1.0), MobileNet2(
    width_mult=0.75), MobileNet2(width_mult=0.5), MobileNet2(width_mult=0.25)]


for model in models:
    flops, params = profile(model, inputs=(x, ))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, count_parameters(model))

# models[0](x)
