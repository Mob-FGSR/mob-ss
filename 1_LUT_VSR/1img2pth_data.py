import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import os
import warnings
warnings.filterwarnings("ignore")
from torchvision import utils as vutils

from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as tf
from PIL import Image

import argparse
from torchvision.transforms import ToPILImage
show = ToPILImage()

def unpackage(raba_x: torch.Tensor, rgba_y: torch.Tensor) -> torch.Tensor:
    h = raba_x.shape[1]
    w = raba_x.shape[2]
    bit_shift = torch.tensor([1.0, 1.0 / 255.0, 1.0 / (255.0 * 255.0), 1.0 / (255.0 * 255.0 * 255.0)]).cuda()
    bit_shift = bit_shift.repeat(h, w, 1).permute(2, 0, 1)
    raba_x = (raba_x) * bit_shift
    raba_y = (rgba_y) * bit_shift
    x = raba_x.sum(0).unsqueeze(0) * 2 - 1
    y = raba_y.sum(0).unsqueeze(0) * 2 - 1
    res = torch.cat((x, y/-1), 0)
    return res

def unpack_mv16(mv: torch.Tensor) -> torch.Tensor:
    h = mv.shape[1]
    w = mv.shape[2]
    device = mv.device  # get device of mv tensor (cpu or cuda)
    bit_shift = torch.tensor([1.0, 1.0 / 255.0], device=device)
    bit_shift = bit_shift.repeat(h, w, 1).permute(2, 0, 1)
    raba_x = (mv[0:2]) * bit_shift
    raba_y = (mv[2:4]) * bit_shift
    x = raba_x.sum(0).unsqueeze(0)
    y = 1 - raba_y.sum(0).unsqueeze(0)
    res = torch.cat((x, y), 0) * 4 - 2
    return res

def unpackage_depth(depth: torch.Tensor) -> torch.Tensor:
    h = depth.shape[1]
    w = depth.shape[2]
    bit_shift = torch.tensor([1.0, 1.0 / 255.0, 1.0 / (255.0 * 255.0), 1.0 / (255.0 * 255.0 * 255.0)]).cuda()
    bit_shift = bit_shift.repeat(h, w, 1).permute(2, 0, 1)
    raba_x = (depth) * bit_shift
    x = raba_x.sum(0).unsqueeze(0)
    return x

class IMG_VSR(Dataset):
    def __init__(self,
                 root_dir: str,
                 batch_size: int,
                 Scale:int,
                 Jitter:bool,
                 ):
        self.Scale = Scale
        self.Jitter = Jitter
        self.batch_size = batch_size
        self.root_dir = root_dir

        print("training with data of path:")
        print(self.root_dir)
        img = os.listdir(os.path.join(self.root_dir, '540p_color')) #'Inputs'
        img.sort(key=lambda x: int(x[:-4]))
        img = img[2:-2]

        self.imglist = [i for i in img for j in range(self.batch_size)]
        self.transform = tf.ToTensor()

        ulist, vlist = [], []
        if self.Scale == 2:
            x_list = [1, 0, 1, 1, 0, 1, 0, 0]
            y_list = [1, 0, 1, 0, 1, 0, 1, 0]
            for indx in range(len(self.imglist)):
                ulist.append(x_list[indx%8])
                vlist.append(y_list[indx%8])
        elif self.Scale == 3:
            x_list = [0, 1, 0, 1, 2, 2, 2, 1, 0]
            y_list = [0, 2, 1, 1, 2, 1, 0, 0, 2]
            for indx in range(len(self.imglist)):
                ulist.append(x_list[indx%9])
                vlist.append(y_list[indx%9])

        if self.Jitter:
            self.ulist = [i for i in ulist for j in range(self.batch_size)]
            self.vlist = [i for i in vlist for j in range(self.batch_size)]
        else:
            self.ulist = [0 for i in self.imglist]
            self.vlist = [0 for i in self.imglist]


    def __getitem__(self, index):
        image_name = self.imglist[index]

        view_path = os.path.join(self.root_dir, '540p_color', str(int(image_name[:-4])).zfill(4)+'.png')
        img = Image.open(view_path)
        img_view = self.transform(img).cuda()[0:3]

        GT_path = os.path.join(self.root_dir, '1080p_color', str(int(image_name[:-4])).zfill(4)+'.png')
        GT = Image.open(GT_path)
        GT = self.transform(GT).cuda()[0:3]

        depth_path = os.path.join(self.root_dir, '540p_depth', str(int(image_name[:-4])).zfill(4)+'.png')
        img_depth = self.transform(Image.open(depth_path)).cuda()
        img_depth = unpackage_depth(img_depth)

        d_mask_path = os.path.join(self.root_dir, '540p_dynamic_mask', str(int(image_name[:-4])).zfill(4)+'.png')
        img_d_mask = self.transform(Image.open(d_mask_path)).cuda()
        img_depth_ = img_depth + img_d_mask[:1]*0.01

        # Motion Vector
        mv_path = os.path.join(self.root_dir, '540p_motion_vector', str(int(image_name[:-4])).zfill(4)+'.png')
        mv_img = Image.open(mv_path)
        mv = self.transform(mv_img)  # (C,H,W)
        mv = unpack_mv16(mv)  # -> (Cm,H,W)

        return img_view, img_depth_, mv, GT, image_name

    def __len__(self) -> int:
        return len(self.imglist)



# -----------------------------------
# ISR
# -----------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="VSR")
    args = parser.parse_args()
    return args

def main(args):
    Data_buffer = []
    with torch.no_grad():
        dataset = IMG_VSR(root_dir=args.data_dir, batch_size=args.test_batch_size, Scale=args.Scale, Jitter=False) # 70, end=180   140, end=250

        data_loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False)

        tqdm_bar = tqdm(data_loader, desc=f'Test: ', total=int(len(data_loader)))
        for batch_idx, (LR_color, LR_depth, LR_mv, HR_color, image_name) in enumerate(tqdm_bar):
            LR_color = (LR_color * 255).byte().cpu()
            LR_depth = LR_depth.half().cpu()
            LR_mv = LR_mv.half().cpu()
            HR_color = (HR_color * 255).byte().cpu()
            Data_buffer.append([LR_color, LR_depth, LR_mv, HR_color, image_name])
    torch.save(Data_buffer, args.save_dir)

if __name__ == '__main__':
    data_id = "BK"    # BK   WT

    args = parse_args()
    args.data_dir = f'../0_DATA/{data_id}/'
    args.output_dir = f'temp_pth/'
    os.makedirs(args.output_dir, exist_ok=True)

    args.test_batch_size = 1
    args.Scale = 2

    args.save_dir = args.output_dir + f'{data_id}_VSR_images.pth'

    main(args)


