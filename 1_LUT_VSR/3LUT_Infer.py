import os
import warnings
warnings.filterwarnings("ignore")
import time
import torch.nn.functional as F
import torch
from utils.net import DirectionalLUT
from utils.ssim import ssim
from utils.utils import direction_cal, gather_directional_neighbors, upscale_and_erode_direction_idx
from utils.utils import compute_direction_and_strength_map, map_direction_and_strength_to_int
from utils.utils import combine_depth_color_feature, apply_directional_filters
from utils.utils import expand_depth_and_mv, BackwardWarping, clamp_color33
import warnings
warnings.filterwarnings("ignore")

from torchvision import utils as vutils
from tqdm import tqdm

import argparse
from torchvision.transforms import ToPILImage
show = ToPILImage()

GLOBAL_vis_debug = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ISR(img, depth, LUT):
    Depth_direction_idx = direction_cal(depth, threshold=0.02)
    Depth_direction_feature = gather_directional_neighbors(Depth_direction_idx)

    '''
    #######################################################
    # Stage 2 calculate the directional feature of Brightness
    #######################################################
    '''
    direction, direction_strength = compute_direction_and_strength_map(img)
    direction_idx, strength_idx = map_direction_and_strength_to_int(direction, direction_strength)

    '''
    #######################################################
    # Stage 3 combine the two features of Depth and Brightness
    #######################################################
    '''
    LUT_feature = combine_depth_color_feature(Depth_direction_feature, direction_idx, strength_idx)

    assert LUT_feature.max() <= 12 and LUT_feature.min() >= 0
    d0 = LUT_feature[:, 0]  # [1,H,W]
    d1 = LUT_feature[:, 1]
    d2 = LUT_feature[:, 2]

    # LUT lookup
    out = LUT[d0, d1, d2]  # [1, H, W, 64]

    # reshape to [1,64,H,W]
    out = out.permute(0, 3, 1, 2)
    N, C, H, W = out.shape
    filters = out.view(1, 4, 16, H, W)

    X_up = apply_directional_filters(img, filters)
    X_up = F.pad(X_up, (1, 1, 1, 1), mode='replicate')

    return X_up


def VSR_Infer(LUT, data_list, save_id, Scale=2, num_epochs=10):
    H, W = 1080, 1920
    h = torch.linspace(-1, 1, H)
    w = torch.linspace(-1, 1, W)
    meshx, meshy = torch.meshgrid([h, w])
    grid = torch.stack((meshy, meshx), 2)
    grid = grid.unsqueeze(0).cuda()

    tqdm_bar = tqdm(data_list, total=int(len(data_list)), desc=f"Scale {Scale}")
    for _idx, (LR_color, LR_depth, LR_mv, HR_color, image_name) in enumerate(tqdm_bar):
        img = LR_color.to(device=device, dtype=torch.float32, non_blocking=use_pin).div_(255.0)
        depth = LR_depth.to(device=device, dtype=torch.float32, non_blocking=use_pin)
        mv  = LR_mv.to(device=device, dtype=torch.float32, non_blocking=use_pin)
        GT = HR_color.to(device=device, dtype=torch.float32, non_blocking=use_pin).div_(255.0)

        ############ 0 ISR
        ISR_img = ISR(img, depth, LUT)
        depth_HR = F.interpolate(depth, scale_factor=Scale, mode='bilinear', align_corners=True)
        mv_HR = F.interpolate(mv, scale_factor=Scale, mode='bilinear', align_corners=True)

        # catch the history frame
        if _idx == 0:
            previous_RGBD = torch.cat([ISR_img, depth_HR], dim=1)
            continue


        ############ 1 Flow UP and Expanding
        D_exp, flow_exp = expand_depth_and_mv(depth_HR, mv_HR)


        ############ 2 previsous frame warping
        motion_T = flow_exp.permute([0, 2, 3, 1])
        previous_RGBD_Warped = BackwardWarping(previous_RGBD, motion_T, grid)


        # ############ 3 Depth and Color difference detection
        RGBD_up = torch.cat([ISR_img, depth_HR], dim=1)
        D_diff = torch.abs(previous_RGBD_Warped - RGBD_up)[:, 3]

        weight = (D_diff.clone().detach() > 0.03).float()
        previous_RGBD_updata = weight * RGBD_up.clone().detach() + (1 - weight) * previous_RGBD_Warped.clone().detach()


        # ############ 4 History Frame Clamping
        previous_RGBD_updata_clamp = clamp_color33(RGBD_up, previous_RGBD_updata)

        ############ 5 Blending

        Weight_Map = 0.2
        res = Weight_Map * ISR_img.clone().detach() + (1 - Weight_Map) * previous_RGBD_updata_clamp[:,:3]

        vutils.save_image(
            ISR_img.clamp(0, 1).float(),
            f'../0_DATA/{data_id}/1080p_vsr/{image_name[0]}',
            normalize=False
        )

        previous_RGBD = torch.cat([res, depth_HR], dim=1).clone().detach()

def parse_args():
    parser = argparse.ArgumentParser(description="VSR")
    args = parser.parse_args()
    return args

def load_pth_data(data_pth):
    data_list = torch.load(data_pth, map_location='cpu')
    use_pin = torch.cuda.is_available()
    if use_pin:
        for i in range(len(data_list)):
            LR_color, LR_depth, LR_mv, HR_color, image_name = data_list[i]
            data_list[i] = [
                LR_color.pin_memory(),
                LR_depth.pin_memory(),
                LR_mv.pin_memory(),
                HR_color.pin_memory(),
                image_name
            ]
    return data_list, use_pin


if __name__ == '__main__':
    data_id = "BK"  # dataset ID: "BK" or "AL"

    args = parse_args()
    args.pth_dir = 'temp_pth/'
    data_pth = args.pth_dir + f'{data_id}_VSR_images.pth'

    # Load image data
    data_list, use_pin = load_pth_data(data_pth)


    # Load directional LUT model
    model = DirectionalLUT().cuda()

    checkpoint = torch.load('nE_ckp_4999.pth')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    nE_LUT = model.return_nELUT().cuda()      # non-edge LUT

    checkpoint = torch.load('E_ckp_4999.pth')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    E_LUT = model.return_ELUT().cuda()        # edge LUT

    # Merge the two LUTs
    LUT_matrix = torch.cat([nE_LUT[:1], E_LUT[1:]], dim=0)

    save_LUT= LUT_matrix.view(-1,64)
    torch.save(save_LUT, f'00LUT.pth')

    # Run VSR inference using the combined LUT
    os.makedirs(f'../0_DATA/{data_id}/1080p_vsr', exist_ok=True)
    VSR_Infer(LUT_matrix, data_list, data_id)


