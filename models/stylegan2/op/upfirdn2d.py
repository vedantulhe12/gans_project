import torch
import torch.nn.functional as F

def upfirdn2d(x, kernel=None, up=1, down=1, pad=(0, 0)):
    # Upsample
    if up > 1:
        x = F.interpolate(x, scale_factor=up, mode='nearest')

    # Padding
    if isinstance(pad, int):
        pad = (pad, pad)
    elif len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])
    x = F.pad(x, pad, mode='reflect')

    # Convolution (optional, can be skipped or replaced with identity kernel)
    if kernel is not None:
        kernel = kernel.to(x.device)
        b, c, h, w = x.shape
        kernel = kernel.view(1, 1, *kernel.shape).expand(c, 1, -1, -1)
        x = F.conv2d(x, kernel, groups=c, padding=0)

    # Downsample
    if down > 1:
        x = F.avg_pool2d(x, down)

    return x
