import torch

# 检查两个权重文件的 key
print("===== nor_best.pth 完整参数列表 =====")
w1 = torch.load("./Model/weights/nor_best.pth", map_location="cpu")
for k in w1.keys():
    print(k)

print("\n===== atten_best.pth 完整参数列表 =====")
w2 = torch.load("./Model/weights/atten_best.pth", map_location="cpu")
for k in w2.keys():
    print(k)