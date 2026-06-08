import torch
import torch.nn as nn

# 时空注意力
import torch
import torch.nn as nn


# 时空注意力模块（融合空间维度特征注意力 + 序列维度注意力，通用Transformer注意力架构）
class SpatialTemporalAttention(nn.Module):
	def __init__(self, dim, num_heads=4, qkv_bias=False):
		super().__init__()
		self.num_heads = num_heads  # 注意力头数
		head_dim = dim // num_heads  # 每个头的维度 = 总维度 // 头数
		self.scale = head_dim ** -0.5  # 缩放因子：1/√d，防止点积数值过大导致softmax饱和

		# 线性层：将输入特征映射为 Q、K、V 三个矩阵（3*dim）
		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

		# 输出投影层：注意力计算后映射回原始维度
		self.proj = nn.Linear(dim, dim)

	def forward(self, x):
		# 输入形状：[B, C, H, W] → 批量、通道、高、宽
		B, C, H, W = x.shape

		# 1. 展平空间维度 + 调整维度顺序
		# flatten(2)：把 H,W 展平 → [B, C, H*W]
		# transpose(1,2) → [B, H*W, C]（符合Transformer输入格式：序列长度, 特征维度）
		x = x.flatten(2).transpose(1, 2)

		# 2. 线性变换生成 Q, K, V
		# 输出 shape: [B, N, 3*C]，N=H*W 序列长度
		qkv = self.qkv(x)

		# 3. 拆分维度 + 多头分组
		# reshape: [B, N, 3, 头数, 每头维度]
		# permute: 调整为 [3, B, 头数, N, 每头维度]
		qkv = qkv.reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

		# 拆分得到 Q、K、V，shape 均为：[B, 头数, N, 每头维度]
		q, k, v = qkv[0], qkv[1], qkv[2]

		# 4. 计算注意力分数：Q @ K^T / √d_k
		# attn shape: [B, 头数, N, N]
		attn = (q @ k.transpose(-2, -1)) * self.scale

		# 5. Softmax 归一化得到注意力权重
		attn = attn.softmax(dim=-1)

		# 6. 注意力权重 @ Value
		# 结果 shape: [B, 头数, N, 每头维度]
		# transpose + reshape → 合并多头：[B, N, C]
		x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

		# 7. 最终线性投影
		x = self.proj(x)

		# 8. 恢复形状为 [B, C, H, W]，返回CNN格式特征
		x = x.transpose(1, 2).reshape(B, C, H, W)

		return x

# 双卷积
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

# UNet 主模型 + 自动特征提取
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

		# 🔥 用来存储最终融合特征的变量
		self.fused_feature = None

	def forward(self, x):
		# 编码
		d1 = self.d1(x)
		d2 = self.d2(self.pool(d1))
		d3 = self.d3(self.pool(d2))
		d4 = self.d4(self.pool(d3))
		d4_att = self.spatial_temporal_attn(d4)

		# ===================== 🔥 核心：推理时自动计算并保存特征 =====================
		f1 = torch.mean(d1, dim=(2, 3))
		f2 = torch.mean(d2, dim=(2, 3))
		f3 = torch.mean(d3, dim=(2, 3))
		f4 = torch.mean(d4_att, dim=(2, 3))
		feat = torch.cat([f1, f2, f3, f4], dim=1)
		self.fused_feature = torch.nn.functional.normalize(feat, p=2, dim=1)

		# 解码
		u4 = self.u4(torch.cat([self.up4(d4_att), d3], dim=1))
		u3 = self.u3(torch.cat([self.up3(u4), d2], dim=1))
		u2 = self.u2(torch.cat([self.up2(u3), d1], dim=1))
		out = self.out(u2)
		return out

	# ===================== ✅ 无参提取特征！不用输入 x =====================
	def extract_features(self):
		if self.fused_feature is None:
			raise ValueError("请先执行一次 model(x) 推理，再提取特征！")
		return self.fused_feature