import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from unet_parts import *
from unet_parts_att_transformer import *
from unet_parts_att_multiscale import *

class TransAttUnet_DenseNet121(nn.Module):
    def __init__(self, n_classes, pretrained_weights_path=None, in_channels=3):
        super(TransAttUnet_DenseNet121, self).__init__()

        self.n_classes = n_classes

        # RadImageNetようにDenseNet121をロード(3チャンネル入力を想定)
        densenet = models.densenet121(pretrained=False)

        # 入力がモノクロの場合は、最初の畳み込み層をリサイズ
        if in_channels != 3:
            densenet.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 事前学習済みの重みをロード
        if pretrained_weights_path:
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')

            # ネットワーク構造を一致させるための調整
            densenet.load_state_dict(state_dict, strict=False)
            print(f" Loaded RadImageNet pretrained weights from: {pretrained_weights_path}")
        
        # DenseNet121のエンコーダー各層を抽出
        self.enc_init = nn.Sequential(
            densenet.features.conv0,
            densenet.features.norm0,
            densenet.features.relu0
        ) # 出力: 64ch (1/2サイズ)

        self.pool1 = densenet.features.pool0 # 64ch (1/4サイズ)
        self.dense_block1 = densenet.features.denseblock1 # 出力: 256ch (1/4サイズ)
        self.trans_layer1 = densenet.features.transition1

        self.dense_block2

