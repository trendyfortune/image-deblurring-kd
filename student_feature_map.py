import torch
import torch.nn as nn
from models.StudentNet import StudentNet
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StudentNet().to(device)
ckpt = torch.load("s_weights/best_student.pkl", map_location=device)
model.load_state_dict(ckpt['model'])
model.eval()

conv_layers = []
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        conv_layers.append(module)
print(f"\n> Total convolutional layers: {len(conv_layers)}")
print(f"> Parameter count: {sum(p.numel() for p in model.parameters())}")

image_path = "dataset/sample_img_for_fmaps.png"
image = Image.open(image_path).convert("RGB")
transform = transforms.ToTensor()
input_tensor = transform(image).unsqueeze(0).to(device)  # [1, C, H, W]

feature_maps = []
layer_names = []

def hook_fn(module, _, output):
    feature_maps.append(output.detach().cpu())
    layer_names.append(str(module))

hooks = []
for layer in conv_layers:
    hooks.append(layer.register_forward_hook(hook_fn))

with torch.no_grad():
    _ = model(input_tensor)

for hook in hooks:
    hook.remove()

processed_feature_maps = []
titles = []

for i, fmap in enumerate(feature_maps):
    fmap = fmap.squeeze(0) 
    for c in range(0, fmap.shape[0], 16):
        channel_map = fmap[c].numpy()
        processed_feature_maps.append(channel_map)
        titles.append(f"Layer {i+1} - Channel {c}")
    if i < len(feature_maps) - 1:
        processed_feature_maps.append(fmap[fmap.shape[0]-1].numpy())
        titles.append(f"Layer {i+1} - Channel {fmap.shape[0]-1}")

cols = 5
rows = int(np.ceil(len(processed_feature_maps) / cols))

fig = plt.figure(num="Feature Maps", figsize=(cols *2, rows *1.5))

for i, fmap in enumerate(processed_feature_maps):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.imshow(fmap, cmap='gray')
    ax.axis("off")
    ax.set_title(titles[i], fontsize=10)

plt.tight_layout()
print("> feature_maps.png saved in /dataset\n")
plt.savefig("dataset/feature_maps.png")
plt.show()
