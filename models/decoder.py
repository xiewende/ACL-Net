from os import pread
from tkinter.tix import Tree
import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import ASPP, get_syncbn
from .affinity_generated import Aff_Generated


class dec_deeplabv3(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
    ):
        super(dec_deeplabv3, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )
        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        aspp_out = self.aspp(x)
        res = self.head(aspp_out)
        return res

class dec_deeplabv3_plus(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
        rep_head=False,
        need_aff = True,
    ):
        super(dec_deeplabv3_plus, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.rep_head = rep_head
        self.need_aff = need_aff

        self.low_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        if self.rep_head:

            self.representation = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            )


        # 是否添加aff
        if self.need_aff:
            self.aff_gen = Aff_Generated(in_dim=128)
        
    def forward(self, x):

        x1, x2, x3, x4 = x

        aspp_out = self.aspp(x4)  # 4 1280 24 24 
        low_feat = self.low_conv(x1) # 4 256 96 86
        aspp_out = self.head(aspp_out) # 4 256 24 24
        h, w = low_feat.size()[-2:]
        aspp_out = F.interpolate(
            aspp_out, size=(h, w), mode="bilinear", align_corners=True
        )
        aspp_out = torch.cat((low_feat, aspp_out), dim=1)

       
        res = {"pred": self.classifier(aspp_out)} # 4 2 97 97
        # print("self.classifier(aspp_out)",self.classifier(aspp_out).shape) 57 * 57
        
        if self.rep_head:
            res["rep"] = self.representation(aspp_out) # 4, 256, 97, 97 
            res["aff"] = self.aff_gen(aspp_out)
            # print("self.representation(aspp_out)",self.representation(aspp_out).shape) 57 * 57

        return res


class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19, sync_bn=False):
        super(Aux_Module, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.aux = nn.Sequential(
            nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        res = self.aux(x)
        return res
