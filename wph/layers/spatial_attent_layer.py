import torch
import torch.nn as nn

class SpatialAttentionLayer(nn.Module):
    """
    Identify where important parts of image are located.
    Helps filter out background before correlation for images
    that are not clean textures.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        shape = x.shape
        x = x.reshape(shape[0], -1, shape[-2], shape[-1])
        # x shape: [Batch, Channels, H, W]
        # We compress channels to get 2 maps: Avg and Max
        
        # TODO: think about handling real/imag parts separately instead of using magnitude
        # Handle both real and complex inputs
        if torch.is_complex(x):
            # For complex inputs, compute attention based on magnitude
            x_mag = torch.abs(x)
            avg_out = torch.mean(x_mag, dim=1, keepdim=True)
            max_out, _ = torch.max(x_mag, dim=1, keepdim=True)
        else:
            # For real inputs, compute attention normally
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Combine maps to find the "importance" of each pixel
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(attn))
        
        out = x * attn # Only the important pixels 'survive' for correlation
        return out.reshape(shape)