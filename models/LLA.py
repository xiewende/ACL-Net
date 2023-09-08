###
#local pixel refinement
###

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_kernel():
    
    weight = torch.zeros(8, 1, 3, 3)
    weight[0, 0, 0, 0] = 1
    weight[1, 0, 0, 1] = 1
    weight[2, 0, 0, 2] = 1

    weight[3, 0, 1, 0] = 1
    weight[4, 0, 1, 2] = 1

    weight[5, 0, 2, 0] = 1
    weight[6, 0, 2, 1] = 1
    weight[7, 0, 2, 2] = 1

    return weight

# Implement of low-level affinity
class LLA(nn.Module):
    
    def __init__(self, dilations,):
        super().__init__()
        self.dilations = dilations
        # self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d]*4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b*c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)
 
        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)
        
        for d in self.dilations:
            pos_xy.append(ker*d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs):
        # masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)
        b, c, h, w = imgs.shape
        _imgs = self.get_dilated_neighbors(imgs) # B c 48 H W
        _pos = self.pos.to(_imgs.device)

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1,1,_imgs.shape[self.dim],1,1) # B c 48 H W
        _pos_rep = _pos.repeat(b, 1, 1, h, w) # B 1 48 H W

        _imgs_abs = torch.abs(_imgs - _imgs_rep) # B c 48 H W
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True) # b c 1 H w
        _pos_std = torch.std(_pos_rep, dim=self.dim, keepdim=True)# b 1 1 H w

        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1)**2 # B 1 48 H W
        aff = aff.mean(dim=1, keepdim=True)

        pos_aff = -(_pos_rep / (_pos_std + 1e-8) / self.w1)**2 # B 1 48 H W
        #pos_aff = pos_aff.mean(dim=1, keepdim=True)

        aff_final = F.softmax(aff, dim=2) + self.w2 * F.softmax(pos_aff, dim=2) # B 1 48 H W
        
        # refined
        # for _ in range(self.num_iter):
        #     _masks = self.get_dilated_neighbors(masks) # B 2 48 H W
        #     masks = (_masks * aff).sum(2)     # 2 2 48 384 384 
        return aff_final


if __name__ == '__main__':
    image = torch.rand(2,3,4,4)
    # mask = torch.rand(2,2,4,4)
    # par = PAR(num_iter=10, dilations=[1,2,4,8,12,24])
    # # att = Attention_Guide()
    # masks = par(image,mask)
    # print(masks.shape)

    lla = LLA(dilations=[1,2,4,8,12,24])
    aff = lla(image)
    print(aff.shape)
    