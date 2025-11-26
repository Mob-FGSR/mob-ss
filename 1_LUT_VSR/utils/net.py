import torch.nn.functional as F
import torch
import torch.nn as nn

class Gaussian4x4PerPixel(nn.Module):
    def __init__(self, normalize=True, dtype=torch.float32):
        super().__init__()
        self.normalize = normalize

        # 4×4 sampling grid coordinates: y, x ∈ {0,1,2,3}
        coords = torch.arange(4, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')  # [4,4]

        # Centers (cy, cx) of the four Gaussians in continuous 4×4 grid coordinates
        centers = torch.tensor([
            [1.25, 1.25],  # 0: top-left
            [1.25, 1.75],  # 1: top-right
            [1.75, 1.25],  # 2: bottom-left
            [1.75, 1.75],  # 3: bottom-right
        ], dtype=dtype)  # [4,2]

        # Precompute squared distances (independent of pixel values, depends only on sampling grid + centers)
        # dist2[d, y, x] = (y - cy_d)^2 + (x - cx_d)^2
        cy = centers[:, 0].view(-1, 1, 1)  # [4,1,1]
        cx = centers[:, 1].view(-1, 1, 1)  # [4,1,1]
        dist2 = (yy.unsqueeze(0) - cy) ** 2 + (xx.unsqueeze(0) - cx) ** 2  # [4,4,4]

        # Register as buffer so it automatically moves with .to(device)
        self.register_buffer("dist2", dist2)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        a: [N, 1, H, W] variance (sigma^2)
        return: [N, 64, H, W]
        """
        assert a.dim() == 4, "a must be [N, 1, H, W]"
        N, C, H, W = a.shape
        assert C == 1, "Currently only supports a single variance channel per pixel"

        # Ensure dtype/device consistency
        a = a.to(self.dist2.dtype).to(self.dist2.device)

        # Prevent non-positive variances
        if torch.any(a <= 0):
            raise ValueError("Variance a must be positive")

        # self.dist2: [4,4,4] → [1,4,4,4,1,1] for broadcasting
        dist2 = self.dist2.view(1, 4, 4, 4, 1, 1)

        # a: [N,1,H,W] → [N,1,1,1,H,W]
        a_expanded = a.view(N, 1, 1, 1, H, W)

        # g: [N,4,4,4,H,W]
        g = torch.exp(- dist2 / (2.0 * a_expanded))

        if self.normalize:
            # Normalize each Gaussian within each pixel so the 4×4 patch sums to 1
            # g_sum: [N,4,1,1,H,W]
            g_sum = g.sum(dim=(2, 3), keepdim=True) + 1e-12
            g = g / g_sum

        # [N,4,4,4,H,W] → [N,64,H,W]
        g = g.view(N, 4 * 4 * 4, H, W)

        return g


class PixelMLP(nn.Module):
    def __init__(self, Cin=3, Cout=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(Cin, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, Cout)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        x = self.mlp(x)
        return x.reshape(B, H, W, -1).permute(0, 3, 1, 2)


class DirectionalLUT(nn.Module):
    def __init__(self, LUT_dim=13, embedding_dim=64, init_center=0.0625, init_range=0.01):
        super().__init__()
        self.LUT_dim = LUT_dim

        self.MLP1 = PixelMLP(Cin=3, Cout=64)
        self.MLP2 = PixelMLP(Cin=3, Cout=1)
        self.gaussian = Gaussian4x4PerPixel(normalize=True)
        self.sigmoid = nn.Sigmoid()

        self.MLP3 = PixelMLP(Cin=3, Cout=64)

    def nE_infer(self, direction_feature, LR_input):
        B, C, H, W = direction_feature.shape

        filters = self.MLP3(direction_feature.float())
        filters = filters.view(B, 4, 16, H, W)

        # Normalize along the 16 kernel elements for each direction
        sum16 = filters.sum(dim=2, keepdim=True) + 1e-12
        filters = filters / sum16

        y_up = self.apply_directional_filters(LR_input, filters)
        return y_up, sum16

    def E_infer(self, direction_feature, LR_input):
        B, C, H, W = direction_feature.shape

        # Amplitude modulation term
        F_r = self.MLP1(direction_feature.float())
        F_r = self.sigmoid(F_r)

        # Gaussian variance term
        s = self.MLP2(direction_feature.float())
        s = self.sigmoid(s) + 0.05  # Prevent s from being zero
        F_s = self.gaussian(s)
        # Already returned in [B, 64, H, W]

        # Combine Gaussian basis × amplitude
        filters = F_s * F_r
        filters = filters.view(B, 4, 16, H, W)

        # Normalize per 4×4 kernel
        sum16 = filters.sum(dim=2, keepdim=True) + 1e-12
        filters = filters / sum16

        y_up = self.apply_directional_filters(LR_input, filters)
        return y_up, sum16

    def apply_directional_filters(self, x_view, filters):
        B, C, H, W = x_view.shape
        _, D, K2, Hf, Wf = filters.shape
        K = int(K2 ** 0.5)
        assert D == 4 and K == 4, "Requires 4 directions, each with a 4x4 kernel"

        # Unfold input into 4×4 patches
        x_unfold = F.unfold(x_view, kernel_size=K, padding=1, stride=1)  # [B, C*K2, Hf*Wf]
        x_unfold = x_unfold.view(B, C, K2, Hf, Wf).contiguous()  # [B, C, 16, Hf, Wf]

        # Apply directional filters with Einstein summation
        y = torch.einsum('bckhw, bdkhw->bcdhw', x_unfold, filters)

        # Group the 4 directions into a 2×2 spatial upsampling structure
        y = y.view(B, C, 2, 2, Hf, Wf)

        # Rearrange to form 2× upsampled output
        y_up = y.permute(0, 1, 4, 2, 5, 3).contiguous().view(B, C, Hf * 2, Wf * 2)

        return y_up

    @torch.no_grad()
    def return_nELUT(self):
        D = self.LUT_dim  # 13

        # 1. Create all combinations of (d0, d1, d2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vals = torch.arange(D, device=device)  # [0, 1, ..., 12]
        d0, d1, d2 = torch.meshgrid(vals, vals, vals, indexing='ij')  # [D, D, D]

        # 2. Pack into direction_feature with shape [1, 3, H, W]
        direction_feature = torch.stack([d0, d1, d2], dim=0)  # [3, D, D, D]
        direction_feature = direction_feature.view(1, 3, D, D * D)

        # 3. Run through MLP3 + normalization
        filters = self.MLP3(direction_feature.float())  # [1, 64, D, D*D]
        B, C, H, W = filters.shape
        filters = filters.view(B, 4, 16, H, W)

        sum16 = filters.sum(dim=2, keepdim=True) + 1e-12
        filters = filters / sum16  # [1, 4, 16, D, D*D]

        # 4. Flatten (4, 16) -> 64
        filters = filters.view(1, 64, H, W)

        # 5. Rearrange to LUT: [D, D, D, 64]
        lut = filters.squeeze(0).permute(1, 2, 0)  # [D, D*D, 64]

        # Reshape width into (D, D)
        lut3 = lut.view(D, D, D, 64)

        return lut3

    @torch.no_grad()
    def return_ELUT(self):
        D = self.LUT_dim  # 13

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vals = torch.arange(D, device=device)
        d0, d1, d2 = torch.meshgrid(vals, vals, vals, indexing='ij')
        direction_feature = torch.stack([d0, d1, d2], dim=0)
        direction_feature = direction_feature.view(1, 3, D, D * D)

        # Amplitude term
        F_r = self.MLP1(direction_feature.float())
        F_r = self.sigmoid(F_r)

        # Gaussian variance term
        s = self.MLP2(direction_feature.float())
        s = self.sigmoid(s) + 0.05  # Prevent s from being zero
        F_s = self.gaussian(s)

        # Combine Gaussian basis × amplitude
        filters = F_s * F_r
        B, C, H, W = filters.shape
        filters = filters.view(B, 4, 16, H, W)

        # Normalize 16 elements per direction
        sum16 = filters.sum(dim=2, keepdim=True) + 1e-12
        filters = filters / sum16

        filters = filters.view(1, 64, H, W)

        # Rearrange into LUT shape: [D, D, D, 64]
        lut = filters.squeeze(0).permute(1, 2, 0)  # [D, D*D, 64]
        lut3 = lut.view(D, D, D, 64)

        return lut3
