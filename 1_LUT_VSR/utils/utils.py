import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import torch
import warnings
warnings.filterwarnings("ignore")

from torchvision.transforms import ToPILImage
show = ToPILImage()


def upscale_and_erode_direction_idx(direction_idx, scale=2):
    """
    Nearest-neighbor upsampling + one round of directional erosion:
    propagate non-zero direction labels into zero regions.

    Input:
        direction_idx: [B, 1, H, W], values in 0~8
    Output:
        direction_idx_eroded: [B, 1, H*scale, W*scale]
    """
    # Step 1: nearest-neighbor upsampling
    direction_idx_up = F.interpolate(direction_idx.float(), scale_factor=scale, mode='nearest').long()

    # Step 2: create a 3×3 erosion kernel
    kernel = torch.ones((1, 1, 3, 3), device=direction_idx.device)

    # Step 3: mask for non-zero direction entries
    mask = (direction_idx_up != 0).float()

    # Step 4: convolve mask to detect if any neighbor is non-zero
    nearby_mask = F.conv2d(mask, kernel, padding=1) > 0

    return nearby_mask.float()


def direction_cal(x_depth, weight=1.0, threshold=0.01):
    """
    Compute direction index for each 2×2 patch in the depth map.

    Input:
        x_depth: [N, 1, H, W] depth map
    Output:
        direction_idx: [N, 1, H-1, W-1], values in [0, 12]
    """

    def ternary_sign(diff, t=threshold):
        return torch.where(diff * weight > t, 1, torch.where(diff * weight < -t, -1, 0))

    # Extract 2×2 corners
    x_padded = x_depth
    c  = x_padded[..., :-1, :-1]  # top-left
    r  = x_padded[..., :-1, 1:]   # right
    b  = x_padded[..., 1:, :-1]   # bottom
    br = x_padded[..., 1:, 1:]    # bottom-right

    # Ternary patterns
    d0 = ternary_sign(c - r)    # →
    d1 = ternary_sign(c - b)    # ↓
    d2 = ternary_sign(b - br)   # ↘
    d3 = ternary_sign(r - br)   # ↙

    # Convert 4 ternary signs to an index
    pattern_code = (d0 + 1) * 27 + (d1 + 1) * 9 + (d2 + 1) * 3 + (d3 + 1)

    # Lookup table for mapping pattern → structural direction (0–12)
    lut = torch.tensor([
        6, 6, 6, 6, 6, 6, 6, 6, 0, 2, 2, 2, 0, 0, 9, 0, 0, 8, 5, 5,
        5, 0, 0, 1, 0, 0, 8, 3, 0, 0, 3, 0, 0, 3, 11, 8, 10, 0, 0, 0, 0,
        0, 0, 0, 8, 5, 5, 5, 0, 0, 1, 0, 0, 8, 7, 0, 0, 7, 0, 0, 7, 4, 8,
        7, 0, 0, 7, 0, 0, 7, 4, 8, 0, 5, 5, 7, 12, 1, 7, 4, 8
    ], dtype=torch.long, device=x_depth.device)

    direction_idx = lut[pattern_code]
    return direction_idx


def gather_directional_neighbors(direction_idx):
    """
    Given direction_idx [1, 1, H, W], return the direction of the pixel
    and the two neighbors in its structural direction.

    Input:
        direction_idx: [1, 1, H, W], values 0~8
    Output:
        direction_feature: [1, 3, H, W] containing:
            channel 0 = self direction
            channel 1 = neighbor 1
            channel 2 = neighbor 2
    """
    B, C, H, W = direction_idx.shape
    padded = F.pad(direction_idx, (1, 1, 1, 1), mode='replicate')

    output = torch.zeros((B, 3, H, W), device=direction_idx.device, dtype=direction_idx.dtype)

    # Direction → (dx, dy) neighbor offsets
    offset_map = {
        1: [(0, 0), (-1,  0), ( 1,  0)],  # ↑
        2: [(0, 0), ( 0, -1), ( 0,  1)],  # →
        3: [(0, 0), ( 1,  0), (-1,  0)],  # ↓
        4: [(0, 0), ( 0,  1), ( 0, -1)],  # ←

        5: [(0, 0), (-1, -1), ( 1,  1)],  # ↗
        6: [(0, 0), ( 1, -1), (-1,  1)],  # ↘
        7: [(0, 0), ( 1,  1), (-1, -1)],  # ↙
        8: [(0, 0), (-1,  1), ( 1, -1)],  # ↖

        9:  [(0, 0), (-1, -1), ( 1,  1)],  # ↗
        10: [(0, 0), ( 1, -1), (-1,  1)],  # ↘
        11: [(0, 0), ( 1,  1), (-1, -1)],  # ↙
        12: [(0, 0), (-1,  1), ( 1, -1)],  # ↖
    }

    for d in range(1, 13):
        mask = (direction_idx == d)
        dxy = offset_map[d]

        for i in range(3):  # self + 2 neighbors
            dx, dy = dxy[i]
            shifted = padded[:, :, 1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
            output[0, i][mask[0, 0]] = shifted[0, 0][mask[0, 0]]

    return output


def compute_direction_and_strength_map(lowres_img):
    """
    Compute direction angle (radians) and direction strength (0–1)
    for each 2×2 patch.

    Args:
        lowres_img: [B, 3, H, W] RGB image in float32 [0, 1]

    Returns:
        direction: [B, 1, H-1, W-1], in [-1, 1]
        strength:  [B, 1, H-1, W-1], in [0, 1]
    """
    assert lowres_img.ndim == 4 and lowres_img.shape[1] == 3

    # Step 1: luminance
    R = lowres_img[:, 0:1, :, :]
    G = lowres_img[:, 1:2, :, :]
    B = lowres_img[:, 2:3, :, :]
    luminance = 0.5 * (B + R) + G

    # Diagram (pixel naming):
    '''
       | b | c |
     --+---+---+--
     e | f | g | h
     --+---+---+--
     i | j | k | l
     --+---+---+--
       | n | o |
    '''

    luminance = F.pad(luminance, (1, 1, 1, 1), mode='reflect')

    bL = luminance[..., 0:-3, 1:-2]
    cL = luminance[..., 0:-3, 2:-1]
    eL = luminance[..., 1:-2, 0:-3]
    fL = luminance[..., 1:-2, 1:-2]
    gL = luminance[..., 1:-2, 2:-1]
    hL = luminance[..., 1:-2, 3:]
    iL = luminance[..., 2:-1, 0:-3]
    jL = luminance[..., 2:-1, 1:-2]
    kL = luminance[..., 2:-1, 2:-1]
    lL = luminance[..., 2:-1, 3:]
    nL = luminance[..., 3:, 1:-2]
    oL = luminance[..., 3:, 2:-1]

    # Step 2: gradient direction
    direction_X = (gL - eL) + (kL - iL) + (hL - fL) + (lL - jL)
    direction_X_abs = torch.abs(direction_X)

    direction_Y = (jL - bL) + (kL - cL) + (nL - fL) + (oL - gL)
    direction_Y_abs = torch.abs(direction_Y)

    direction = direction_X / (direction_X_abs + direction_Y_abs + 1e-6)

    # Step 3: strength
    direction_strength = torch.sqrt(direction_X ** 2 + direction_Y ** 2 + 1e-6)

    return direction, direction_strength


def map_direction_and_strength_to_int(direction, strength, max_bin=12):
    """
    Map direction [-1,1] and strength [0,3.2] to integer bins [0,max_bin].

    Inputs:
        direction: [B,1,H,W]
        strength: [B,1,H,W]

    Outputs:
        direction_idx: [B,1,H,W] integer 0~max_bin
        strength_idx:  [B,1,H,W] integer 0~max_bin
    """
    mapped_dir = (direction + 1) / 2 * max_bin
    direction_idx = torch.round(mapped_dir).clamp(0, max_bin).long()

    mapped_strength = strength / 2 * max_bin
    strength_idx = torch.round(mapped_strength).clamp(0, max_bin).long()

    return direction_idx, strength_idx


def combine_depth_color_feature(depth_direction_feature, direction_idx, strength_idx):
    """
    If the first channel of depth_direction_feature is zero,
    fill the second and third channels with direction_idx and strength_idx.

    Inputs:
        depth_direction_feature: [B,3,H,W], int64
        direction_idx:          [B,1,H,W]
        strength_idx:           [B,1,H,W]

    Output:
        lut_feature: [B,3,H,W]
    """
    lut_feature = depth_direction_feature.clone()
    mask = (depth_direction_feature[:, 0:1, :, :] == 0)

    lut_feature[:, 1:2, :, :][mask] = direction_idx[mask]
    lut_feature[:, 2:3, :, :][mask] = strength_idx[mask]

    return lut_feature


def apply_directional_filters(x_view, filters):
    B, C, H, W = x_view.shape
    _, D, K2, Hf, Wf = filters.shape
    K = int(K2 ** 0.5)
    assert D == 4 and K == 4, "Requires 4 directions and a 4x4 kernel each."

    # Step 1: extract 4×4 patches using unfold
    x_unfold = F.unfold(x_view, kernel_size=K, padding=1, stride=1)
    x_unfold = x_unfold.view(B, C, K2, Hf, Wf)

    # Step 2: dot product between 4×4 patches and directional filters
    y = torch.einsum('bckhw, bdkhw->bcdhw', x_unfold, filters)

    # Step 3: reshape 4-direction channels into 2×2 subpixel grid (for ×2 upsampling)
    y = y.view(B, C, 2, 2, Hf, Wf)

    # Step 4: rearrange into final upsampled output
    y_up = y.permute(0, 1, 4, 2, 5, 3).contiguous().view(B, C, Hf * 2, Wf * 2)

    return y_up


######### VSR utilities
def expand_depth_and_mv(depth_unsq, mv_unsq):
    """
    Expand depth and motion vectors by selecting the minimum-depth
    pixel in each 3×3 neighborhood and copying its motion vector.

    Input:
        depth_unsq: [B,1,H,W]
        mv_unsq:    [B,2,H,W]

    Output:
        expanded_depth: [B,1,H,W]
        expanded_mv:    [B,2,H,W]
    """
    depth_unsq = 1 - depth_unsq
    n, c, h, w = depth_unsq.shape

    depth_unsq = F.pad(depth_unsq, (1, 1, 1, 1), mode='reflect')
    mv_unsq = F.pad(mv_unsq, (1, 1, 1, 1), mode='reflect')

    # Extract 3×3 neighborhoods
    depth_patches = depth_unsq.unfold(2, 3, 1).unfold(3, 3, 1)
    mv_patches = mv_unsq.unfold(2, 3, 1).unfold(3, 3, 1)

    depth_patches = depth_patches.reshape(n, c, h, w, -1)
    mv_patches = mv_patches.reshape(n, 2, h, w, -1)

    # Find min-depth index
    min_depth_indices = torch.argmin(depth_patches, dim=-1, keepdim=True)

    expanded_depth = torch.gather(depth_patches, -1, min_depth_indices).squeeze(-1)
    expanded_mv = torch.gather(mv_patches, -1, min_depth_indices.repeat(1, 2, 1, 1, 1))

    return 1 - expanded_depth, expanded_mv.squeeze(-1)


def BackwardWarping(Image: torch.Tensor, motion_T: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Perform backward warping using grid_sample.

    Inputs:
        Image:    [B, C, H, W]
        motion_T: [B, H, W, 2]
        grid:     [B, H, W, 2]

    Output:
        warped image
    """
    return F.grid_sample(Image, grid - motion_T, padding_mode='border', mode='bicubic', align_corners=True)


def clamp_color33(current_frame, history_frame):
    """
    Clamp the previous (history) frame using the min/max of a 3×3 window
    from the current frame.
    """
    max_color = torch.max_pool2d(current_frame, kernel_size=3, stride=1, padding=1)
    min_color = -torch.max_pool2d(-current_frame, kernel_size=3, stride=1, padding=1)

    clamped_history = torch.clamp(history_frame, min=min_color, max=max_color)
    return clamped_history
