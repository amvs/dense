import torch
from wph.ops.backend import masks_subsample_shift
from torchvision.transforms.functional import rotate
import torch.nn.functional as F
import pdb


def create_masks_shift(J, M, N, mask_union=False, mask_angles=4):
    """
    Creates the masks_shift tensor and computes the corresponding factr_shift.

    Args:
        J (int): Number of scales.
        M (int): Spatial dimension (height).
        N (int): Spatial dimension (width).
        mask_union (bool): Whether to use the union of spatial shift masks.
        mask_angles (int): Number of angles for the masks.

    Returns:
        masks_shift (torch.Tensor): Tensor of shape (num_masks, M, N).
        factr_shift (torch.Tensor): Sum of each mask's values across spatial dimensions.
    """
    masks_shift = masks_subsample_shift(
        J, M, N, mask_union=mask_union, alpha=mask_angles
    )
    masks_shift = torch.cat((torch.zeros(1, M, N), masks_shift), dim=0)
    masks_shift[0, 0, 0] = 1.0
    factr_shift = masks_shift.sum(dim=(-2, -1))
    return masks_shift, factr_shift


def rotation_matrix(angle_degrees, batch_size, device):
    """
    Creates a rotation matrix for grid_sample.
    Note: affine_grid expects the inverse mapping (target -> source),
    so we rotate by -angle to get a counter-clockwise rotation.
    """
    angle_rad = angle_degrees * torch.pi / 180.0
    # We use -angle_rad here because grid_sample uses backward mapping
    theta = torch.tensor(-angle_rad, device=device)
    
    cos_val = torch.cos(theta)
    sin_val = torch.sin(theta)
    
    # Construct standard 2x3 rotation matrix
    # [ cos  -sin   0 ]
    # [ sin   cos   0 ]
    rot_mat = torch.tensor([
        [cos_val, -sin_val, 0],
        [sin_val,  cos_val, 0]
    ], device=device)
    
    # Expand to batch size: (B, 2, 3)
    return rot_mat.unsqueeze(0).repeat(batch_size, 1, 1)


def periodic_rotate(spatial_filters, angle_degrees):
    """
    Rotates spatial filters with periodic boundary conditions.
    
    Args:
        spatial_filters: Complex Tensor of shape (Batch, H, W)
        angle_degrees: Float, angle in degrees
        
    Returns:
        rotated_filters: Complex Tensor of shape (Batch, H, W)
    """
    B, H, W = spatial_filters.shape
    device = spatial_filters.device
    
    # 1. FFT and Shift
    f_hat = torch.fft.fft2(spatial_filters)
    f_hat_shifted = torch.fft.fftshift(f_hat, dim=(-2, -1)) # DC at center
    
    # 2. MANUAL CIRCULAR PADDING 
    # We pad the frequency domain significantly to simulate an infinite periodic grid.
    # Padding by H//2 and W//2 is sufficient to catch any rotation.
    pad_h = H // 2
    pad_w = W // 2
    
    # Split into real/imag for padding (complex padding can be finicky in older torch versions)
    real_padded = F.pad(f_hat_shifted.real, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    imag_padded = F.pad(f_hat_shifted.imag, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    
    # Stack for grid_sample: (B, 2, H_new, W_new)
    f_padded_stack = torch.stack([real_padded, imag_padded], dim=1)
    
    # Get new dimensions
    _, _, H_new, W_new = f_padded_stack.shape

    # 3. Create Grid centered on the *Original* content
    # We want a grid of size HxW, but we need to map it to the coordinate system 
    # of the larger H_new x W_new image.
    
    # Generate grid for the output size (H, W) centered at 0
    # (Using the same center-alignment logic as before)
    y = torch.arange(H, device=device) - (H // 2)
    x = torch.arange(W, device=device) - (W // 2)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # Stack (x, y)
    grid_coords = torch.stack([grid_x, grid_y], dim=-1).float() # (H, W, 2)
    grid_coords = grid_coords.unsqueeze(0).expand(B, -1, -1, -1)

    # 4. Rotate the coordinates
    angle_rad = torch.tensor(-angle_degrees * torch.pi / 180.0)
    cos_val = torch.cos(angle_rad)
    sin_val = torch.sin(angle_rad)
    
    rot_x = grid_coords[..., 0] * cos_val - grid_coords[..., 1] * sin_val
    rot_y = grid_coords[..., 0] * sin_val + grid_coords[..., 1] * cos_val
    
    # 5. Normalize coordinates to the PADDED frame
    # grid_sample expects coords in [-1, 1] relative to the *input tensor* (the padded one).
    # Center of padded image is at (W_new // 2, H_new // 2).
    
    # Shift our rotated coordinates (which are relative to 0) to the center of the padded frame
    sample_x = rot_x + (W_new // 2)
    sample_y = rot_y + (H_new // 2)
    
    # Normalize to [-1, 1] based on PADDED size
    # (sample + 0.5) * 2 / size - 1
    norm_x = (sample_x + 0.5) * 2 / W_new - 1
    norm_y = (sample_y + 0.5) * 2 / H_new - 1
    
    sample_grid = torch.stack([norm_x, norm_y], dim=-1)
    
    # 6. Sample
    # We use bilinear interpolation. Because we padded circularly, 
    # "out of bounds" checks will land on valid wrapped data.
    output_stack = F.grid_sample(
        f_padded_stack, sample_grid, mode='nearest', padding_mode='zeros', align_corners=False
    )
    
    # 7. Reconstruct
    f_hat_rot_shifted = torch.complex(output_stack[:, 0], output_stack[:, 1])
    f_hat_rot = torch.fft.ifftshift(f_hat_rot_shifted, dim=(-2, -1))
    spatial_rot = torch.fft.ifft2(f_hat_rot)
    
    return spatial_rot


def apply_phase_shifts(filters: torch.Tensor, A: int) -> torch.Tensor:
    expanded_filters = filters.expand(-1, -1, -1, A, -1, -1).clone()  # Clone to avoid in-place operations
    for a in range(A):
        i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
        phase_shift = torch.exp(i * a * (2 * torch.pi / A))
        expanded_filters[:, :, :, a, :, :] = expanded_filters[:, :, :, a, :, :] * phase_shift
    return(expanded_filters)