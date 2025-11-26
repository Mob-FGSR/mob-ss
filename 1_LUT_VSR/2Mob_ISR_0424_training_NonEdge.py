import warnings
warnings.filterwarnings("ignore")
import time
import torch.nn.functional as F
import torch
import numpy
import numpy as np
from utils.net import DirectionalLUT
from utils.ssim import ssim
from utils.utils import direction_cal, gather_directional_neighbors, upscale_and_erode_direction_idx
from utils.utils import compute_direction_and_strength_map, map_direction_and_strength_to_int
from utils.utils import combine_depth_color_feature, apply_directional_filters
import os
import random
import warnings
warnings.filterwarnings("ignore")
import time
from torchvision import utils as vutils
from tqdm import tqdm
import argparse
from torchvision.transforms import ToPILImage
show = ToPILImage()

GLOBAL_vis_debug = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_lut_single(model, direction_feature, x_view, GT, edge_mask, image_name, optimizer,
                     num_iters=5000, save_dir='checkpoints'):
    model.train()
    os.makedirs(save_dir, exist_ok=True)
    L2loss = torch.nn.MSELoss(reduction='none')


    direction_feature = direction_feature.cuda()
    x_view = x_view.cuda()
    GT = GT.cuda()
    edge_mask = edge_mask.clamp(0, 1).cuda()

    pbar = tqdm(range(num_iters), total=num_iters, desc=f"Training")

    for idx, step in enumerate(pbar):
        y_up, sum16 = model.nE_infer(direction_feature.detach(), x_view.detach())

        # 加权 L2 Loss
        loss_map = L2loss(y_up, GT[:, :3, 1:-1, 1:-1])
        L2_loss = torch.mean(loss_map * (1-edge_mask)) * 100

        ssim_map = 1-ssim(y_up, GT[:, :3, 1:-1, 1:-1], window_size=7, size_average=False)
        ssim_loss = torch.mean(ssim_map * (1-edge_mask))

        L_activate = torch.mean(torch.abs(sum16 - torch.ones_like(sum16)))*0.01

        loss = L2_loss + ssim_loss + L_activate

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Step {step}, Loss {loss.item()*10000:.4f}")


    id = image_name[0][:-4]
    os.makedirs('vis', exist_ok=True)
    vutils.save_image(
        y_up.clamp(0, 1).float() * (edge_mask/2 + 0.5),
        f'vis/004_y_up_{id}.png',
        normalize=False
    )
    vutils.save_image(
        F.interpolate(x_view, scale_factor=2, mode='bilinear',align_corners=False)[:, :3, 1:-1, 1:-1],
        f'vis/004_y_up_{id}_0.png',
        normalize=False
    )
    vutils.save_image(
        y_up.clamp(0, 1).float(),
        f'vis/004_y_up_{id}_1.png',
        normalize=False
    )
    vutils.save_image(
        GT[:, :3, 1:-1, 1:-1],
        f'vis/004_y_up_{id}_2.png',
        normalize=False
    )

    torch.save({
        'model_state_dict': model.state_dict(),
        'step': step
    }, os.path.join(save_dir, f'nE_ckp_{step}.pth'))



def Super_Resolution(model, data_list, save_id, Scale=2, num_epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        print(f"Epoch {epoch+1}/{num_epochs}")

        # tqdm_bar = tqdm(data_list, total=int(len(data_list)), desc=f"Scale {Scale}")
        for (LR_color, LR_depth, LR_mv, HR_color, image_name) in data_list:

            img = LR_color.to(device=device, dtype=torch.float32, non_blocking=use_pin).div_(255.0)
            depth = LR_depth.to(device=device, dtype=torch.float32, non_blocking=use_pin)
            mv  = LR_mv.to(device=device, dtype=torch.float32, non_blocking=use_pin)
            GT = HR_color.to(device=device, dtype=torch.float32, non_blocking=use_pin).div_(255.0)

            Depth_direction_idx = direction_cal(depth, threshold=0.02)
            Depth_direction_feature = gather_directional_neighbors(Depth_direction_idx)

            edge_mask = upscale_and_erode_direction_idx(Depth_direction_idx, scale=2)  # [1, 1, H*2, W*2]

            '''
            #######################################################
            # Stage 2 calculate the directional feature of Brightness
            #######################################################
            '''
            direction, direction_strength = compute_direction_and_strength_map(img)

            direction_idx, strength_idx = map_direction_and_strength_to_int(direction, direction_strength)
            # plot_direction_hist(direction_idx, (0, 14))
            # plot_direction_hist(strength_idx, (0, 14))

            '''
            #######################################################
            # Stage 3 combine the two features of Depth and Brightness
            #######################################################
            '''
            LUT_feature = combine_depth_color_feature(Depth_direction_feature, direction_idx, strength_idx)
            # vutils.save_image(LUT_feature.float()/ 13.0, 'vis2/009_LUT_feature.png', normalize=False)

            train_lut_single(model, LUT_feature, img, GT, edge_mask, image_name, optimizer)

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
    data_id = "BK"    # BK AL

    args = parse_args()
    args.pth_dir = f'temp_pth/'
    data_pth = args.pth_dir + f'{data_id}_VSR_images.pth'
    data_list, use_pin = load_pth_data(data_pth)

    model = DirectionalLUT().cuda()
    try:
        resume_checkpoint = torch.load('nE_ckp_4999.pth')
        model.load_state_dict(resume_checkpoint['model_state_dict'], strict=False)
        print("Loading pre-trained model successfully!")
    except:
        print("Training a new model")

    Super_Resolution(model, data_list, data_id, 2)