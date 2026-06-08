import torch
import torch.nn as nn
import numpy as np


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
		print("\n🎉 普通UNet2D + 多层融合特征提取")
		self.pool = nn.MaxPool2d(2)

		# 编码器
		self.d1 = DoubleConv(in_ch, 64)
		self.d2 = DoubleConv(64, 128)
		self.d3 = DoubleConv(128, 256)
		self.d4 = DoubleConv(256, 512)

		# 解码器
		self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
		self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
		self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)

		self.u4 = DoubleConv(512, 256)
		self.u3 = DoubleConv(256, 128)
		self.u2 = DoubleConv(128, 64)
		self.out = nn.Conv2d(64, out_ch, 1)

		# 🔥 用来存储特征的变量（推理时自动写入）
		self.fused_feature = None

	def forward(self, x):
		# 编码
		d1 = self.d1(x)
		d2 = self.d2(self.pool(d1))
		d3 = self.d3(self.pool(d2))
		d4 = self.d4(self.pool(d3))

		# ===================== 🔥 关键：推理时自动提取并保存特征 =====================
		f1 = torch.mean(d1, dim=[2, 3])
		f2 = torch.mean(d2, dim=[2, 3])
		f3 = torch.mean(d3, dim=[2, 3])
		f4 = torch.mean(d4, dim=[2, 3])
		feat = torch.cat([f1, f2, f3, f4], dim=1)
		self.fused_feature = torch.nn.functional.normalize(feat, p=2, dim=1)  # 保存到模型里

		# 解码（正常不变）
		u4 = self.u4(torch.cat([self.up4(d4), d3], dim=1))
		u3 = self.u3(torch.cat([self.up3(u4), d2], dim=1))
		u2 = self.u2(torch.cat([self.up2(u3), d1], dim=1))
		out = self.out(u2)
		return out

	# ===================== 🔥 现在不需要输入 x 了！ =====================
	def extract_features(self):
		"""
		无需输入 x！直接返回刚刚推理时自动记录的融合特征
		"""
		if self.fused_feature is None:
			raise ValueError("请先执行一次 model(x) 推理，再调用 extract_features()")
		return self.fused_feature

# ===================== ✅ 推理 + 保存结果到向量（核心代码） =====================
if __name__ == '__main__':
	# 1. 初始化模型
	model = UNet2D(in_ch=1, out_ch=1)
	model.eval()  # 推理模式

	# 2. 构造输入（你换成自己的数据即可）
	x = torch.randn(1, 1, 256, 256)  # B, C, H, W

	# 3. 推理
	with torch.no_grad():
		output = model(x)  # 模型输出（分割图）

	# 4. 🔥 提取你模型自带的融合特征向量
	feature_vector = model.extract_features()

	# 5. ✅ 把 推理结果 + 特征向量 都保存成 .npy 向量
	# 保存分割输出
	np.save("unet_output.npy", output.detach().cpu().numpy())

	# 保存融合特征向量（你最想要的）
	np.save("fused_feature_vector.npy", feature_vector.detach().cpu().numpy())

	# 查看形状
	print("✅ 推理结果形状:", output.shape)
	print("✅ 融合特征向量形状:", feature_vector.shape)
	print("✅ 保存完成！")