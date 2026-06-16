import torch
import torch.nn as nn

class SpatialTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        print("\n🎉 带时空注意力的UNet2D模型（自动特征提取版）")
        self.pool = nn.MaxPool2d(2)
        self.d1 = DoubleConv(in_ch, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)
        self.spatial_temporal_attn = SpatialTemporalAttention(dim=512, num_heads=4)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.u4 = DoubleConv(512, 256)
        self.u3 = DoubleConv(256, 128)
        self.u2 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_ch, kernel_size=1)
        self.fused_feature = None

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        d4_att = self.spatial_temporal_attn(d4)
        f1 = torch.mean(d1, dim=(2, 3))
        f2 = torch.mean(d2, dim=(2, 3))
        f3 = torch.mean(d3, dim=(2, 3))
        f4 = torch.mean(d4_att, dim=(2, 3))
        feat = torch.cat([f1, f2, f3, f4], dim=1)
        self.fused_feature = torch.nn.functional.normalize(feat, p=2, dim=1)
        u4 = self.u4(torch.cat([self.up4(d4_att), d3], dim=1))
        u3 = self.u3(torch.cat([self.up3(u4), d2], dim=1))
        u2 = self.u2(torch.cat([self.up2(u3), d1], dim=1))
        return self.out(u2)

    def extract_features(self):
        if self.fused_feature is None:
            raise ValueError("请先执行一次 model(x) 推理，再调用 extract_features()")
        return self.fused_feature