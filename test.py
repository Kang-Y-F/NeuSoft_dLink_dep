import torch

# 检查两个权重文件的 key
print("=== nor_best.pth (UNet2D) ===")
w1 = torch.load("./Model/weights/nor_best.pth", map_location="cpu")
for k in list(w1.keys())[:5]:
    print(k)

print("\n=== atten_best.pth (AttentionUNet2D) ===")
w2 = torch.load("./Model/weights/atten_best.pth", map_location="cpu")
for k in list(w2.keys())[:10]:
    print(k)