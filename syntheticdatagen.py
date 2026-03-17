
#for skin brain
from modelfunc import *
from netwroks import GlobalGenerator, get_norm_layer, get_edges, Encoder
from dataloader import *
import torchvision
import os
import torch
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
G_path = r"C:\Users\ad\Desktop\AI\heavy\gans3\G_epoch120.pth"
E_path = r"C:\Users\ad\Desktop\AI\heavy\gans3\E_epoch120.pth"
Dataset_path = r"C:\ad\adity\Desktop\AI\heavy\datasets\brain_tumor_dataset"
save_path = r"C:\Users\ad\Desktop\AI\heavy\datasets\brain_tumor_dataset_synt"

# Folder Setup
os.makedirs(save_path, exist_ok=True)
image_save = os.path.join(save_path, "images")
mask_save  = os.path.join(save_path, "masks")
os.makedirs(image_save, exist_ok=True)
os.makedirs(mask_save, exist_ok=True)

#  Style Helper
def random_style(dl):
    idx = random.randint(0, len(dl.dataset)-1)
    item = dl.dataset[idx]
    return item["input"].unsqueeze(0), item["target"].unsqueeze(0)
# Network Setup
input_nc_G = 2 + 1       # one-hot(2) + edges(1)
output_nc  = 3
style_nc   = 32
ngf        = 128
norm       = "instance"

E = Encoder(
    input_nc=output_nc,
    output_nc=style_nc,
    ngf=32,
    n_downsampling=4,
    norm_layer=get_norm_layer(norm)
).to(device)

G = GlobalGenerator(
    input_nc=input_nc_G + style_nc,   # 3 + 32 = 35 channels
    output_nc=output_nc,
    ngf=ngf,
    n_downsampling=3,
    n_blocks=9,
    norm_layer=get_norm_layer(norm)
).to(device)

G.load_state_dict(torch.load(G_path, map_location=device))
E.load_state_dict(torch.load(E_path, map_location=device))

# Dataloader
dl = get_brain_loader(dataset_path=Dataset_path, batch_size=1)

if __name__ == "__main__":
    print("started inference")
    image_num = 0

    for batch in dl:

        real_img = batch['input'].to(device)
        mask     = batch['target'].to(device)

        #real_img = (real_img - 0.5) * 2.0  

        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]

        mask = (mask > 0).long()    # binarize
        mask_4d = mask.unsqueeze(1) # (B,1,H,W)

        one_hot = one_hot_encode_label(mask, 2)
        edges = get_edges(mask_4d)
        cond = torch.cat([one_hot, edges], dim=1)


        #  STYLE MIXING

        # style_image, style_mask = random_style(dl)

        # # normalize style image
        # style_image = style_image.to(device)
        # style_image = (style_image - 0.5) * 2.0

        # # process style mask
        # style_mask = style_mask.to(device)
        # style_mask = (style_mask > 0).long()
        # if style_mask.dim() == 3:
        #     style_mask = style_mask.unsqueeze(1)

        style_feat = E(real_img, mask_4d)
        gen_input = torch.cat([cond, style_feat], dim=1)

        with torch.no_grad():
            fake_img = G(gen_input)

        out = (fake_img.cpu() * 0.5 + 0.5).clamp(0, 1)
        out_img = out.squeeze(0)

        mask_to_save = mask.cpu().float()
        if mask_to_save.dim() == 2:
            mask_to_save = mask_to_save.unsqueeze(0)

        save_image_path = os.path.join(image_save, f"{image_num}.png")
        save_mask_path  = os.path.join(mask_save,  f"{image_num}.png")

        torchvision.utils.save_image(out_img, save_image_path)
        torchvision.utils.save_image(mask_to_save, save_mask_path)

        print(f"{image_num} saved")

        image_num += 1
