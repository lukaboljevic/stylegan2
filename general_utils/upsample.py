import torch
import torch.nn as nn
import torch.nn.functional as F

from proxy import proxy


class Upsample(nn.Module):
    """
    Upsampling to 2x size, then smooting using FIR filter
    """
    def __init__(self):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2.0, mode="bilinear")
        self.smooth = Smooth()

    def forward(self, x):
        x = self.upsample(x)
        return self.smooth(x)
    
    __call__ = proxy(forward)


class Downsample(nn.Module):
    """
    Smooting using FIR filter, then downsampling to 0.5x size
    """
    def __init__(self):
        super().__init__()

        self.smooth = Smooth()

    def forward(self, x):
        # Smoothing is performed before downsampling
        x = self.smooth(x)
        return F.interpolate(x, scale_factor=0.5, mode="bilinear")
    
    __call__ = proxy(forward)


class Smooth(nn.Module):
    """
    FIR filter smoothing using 2D FIR filter
    """
    def __init__(self):
        super().__init__()
        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]
        
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()  # divide by 1/16
        
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : input tensor of shape [batch_size, in_channels, height, width], where in_channels is the
            number of input channels/features for the next convolution block (check Generator forward function)

        Returns
        -------
        out : output tensor of shape [batch_size, in_channels, height, width]
        """
        bs, ic, h, w = x.shape
        x = x.view(-1, 1, h, w)

        x = self.pad(x)
        x = F.conv2d(x, self.kernel)
        return x.view(bs, ic, h, w)
    
    __call__ = proxy(forward)
