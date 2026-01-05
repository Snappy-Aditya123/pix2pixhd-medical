from dataloader import *
import torch
import functools
from torch import nn
from modelfunc import (
    Pix2PixLesionModel,
    one_hot_encode_label,
    get_edges,
    VGGLoss
    )
from netwroks import (
    MultiscaleDiscriminator,
    get_norm_layer,
    weights_init,
    GANLoss,
    GlobalGenerator
)
import os
import torchvision
from pathlib import Path

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

image_dir = r"C:\Users\ad\Desktop\gans\export\export\out512\images_512p"
semantic_dir = r"C:\Users\ad\Desktop\gans\export\export\semantic512"
instance_dir = r"C:\Users\ad\Desktop\gans\export\export\bordermap512"
save_path = Path(r'C:\Users\ad\Desktop\gans\export\synthetic_images')
params = {
    # Model capacity
    "label_nc": 7,           # number of semantic labels
    "output_nc": 3,
    "ngf": 96,
    "ndf": 96,
    "norm": "instance",      # pix2pixHD uses INSTANCE NORM
    "lr": 0.0002,
    "beta1": 0.5,
    "beta2": 0.999,

    # Training
    "batch_size": 1,
    "num_epochs": 80,
    "gpu_ids": [0],

    # Loss weights
    "lambda_feat": 10.0,     # feature matching loss
    # (no L1, no VGG because pix2pixHD doesn't use them)

    # Checkpoint
    "checkpoints_dir": "./checkpoints_pix2pixHD",
}
if __name__ == "__main__":

    train_ds = SkinLesionDataset(
            img_dir=image_dir,
            semantic_dir=semantic_dir,
            instance_dir=instance_dir,
            size=(256, 256)       # recommended
        )

    train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    label_nc = params["label_nc"]
    input_nc_G = label_nc + 1   # one-hot + edges
    output_nc = params["output_nc"]

        # ---- Generator ----
    G = GlobalGenerator(
            input_nc=input_nc_G,
            output_nc=output_nc,
            ngf=params["ngf"],
            n_downsampling=3,
            n_blocks=9,
            norm_layer=get_norm_layer(params["norm"])
        )
    state_dict = torch.load(r'C:\Users\as0118d\Desktop\gans\checkpoints_pix2pixHD\G_epoch80.pth')

    # Then load the state_dict into the model
    G.load_state_dict(state_dict)
    for i, (semantic, instance, real_img) in enumerate(train_dl):
                G.eval()
                sem_onehot = one_hot_encode_label(semantic, label_nc)  # [B, label_nc, H, W]
                inst_map = instance.unsqueeze(1).float()               # [B,1,H,W]
                edges = get_edges(inst_map)                            # [B,1,H,W]

                cond = torch.cat([sem_onehot, edges], dim=1)       
                fake_img = G(cond)
                print(fake_img.shape)

                output_folder = r'C:\Users\as0118d\Desktop\generated_images'
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # Assuming fake_img is a batch of 4 images
                for j in range(fake_img.size(0)):  # Loop through the batch
                    img_path = os.path.join(output_folder, f'image_{i+1}.png')
                    fake_img = fake_img * 0.5 + 0.5  # Rescale to [0, 1]
                    fake_img = torch.clamp(fake_img, 0, 1)  # Ensure the values are within [0, 1]

                    torchvision.utils.save_image(fake_img[j], img_path)
                    
                    print(f"Image {j+1} saved to {img_path}")
                if i == 100:
                    break
