import os
import torch
from torch import nn, optim
from tqdm import tqdm
import torchvision
from torch.cuda.amp import autocast, GradScaler

from dataloader import get_brain_loader
from modelfunc import *
from netwroks import *


def train_pix2pixHD_brain(params, train_loader):

    # Device + basic setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    os.makedirs("live_preview_brain2", exist_ok=True)
    os.makedirs(params["checkpoints_dir"], exist_ok=True)

    label_nc = params["label_nc"]
    input_nc_G = label_nc + 1      # one-hot + edges
    output_nc = params["output_nc"]


    #   Create Generator + Discriminator

    G = GlobalGenerator(
        input_nc=input_nc_G,
        output_nc=output_nc,
        ngf=params["ngf"],
        n_downsampling=3,
        n_blocks=9,
        norm_layer=get_norm_layer(params["norm"])
    ).to(device)
    G.apply(weights_init)

    D = MultiscaleDiscriminator(
        input_nc=input_nc_G + output_nc,
        ndf=params["ndf"],
        n_layers=3,
        norm_layer=get_norm_layer(params["norm"]),
        num_D=3,
        getIntermFeat=True
    ).to(device)
    D.apply(weights_init)

    # Optimizers + Loss functions
    optimizer_G = optim.Adam(G.parameters(), lr=params["lr"],
                             betas=(params["beta1"], params["beta2"]))
    optimizer_D = optim.Adam(D.parameters(), lr=params["lr"],
                             betas=(params["beta1"], params["beta2"]))

    criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
    criterionFeat = nn.L1Loss()
    vgg_loss = VGGLoss(device=device)

    scaler = GradScaler()

    global_step = 0
    print(f"Training on {device}")

    #   Training Loop

    for epoch in range(params["num_epochs"]):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['num_epochs']}")
        for batch in pbar:
            global_step += 1

            # 1) Load input mask + ground truth MRI
            mask = batch[1].to(device)
            real_img = batch[0].to(device)

            # Convert mask → one-hot + edges
            one_hot = one_hot_encode_label(mask, label_nc)
            edges = get_edges((mask * 0.5 + 0.5))
            cond = torch.cat([one_hot, edges], dim=1)

            optimizer_D.zero_grad()
            with autocast():
                with torch.no_grad():
                    fake_img = G(cond)

                pred_real = D(torch.cat([cond, real_img], dim=1))
                pred_fake = D(torch.cat([cond, fake_img], dim=1))

                loss_D_real = criterionGAN(pred_real, True)
                loss_D_fake = criterionGAN(pred_fake, False)
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
         
            optimizer_G.zero_grad()

            with autocast():
                fake_img = G(cond)
                pred_fake_for_G = D(torch.cat([cond, fake_img], dim=1))

                # GAN loss
                loss_G_GAN = criterionGAN(pred_fake_for_G, True)

                # Feature matching loss
                loss_G_FM = 0
                pred_real_detached = D([cond, real_img])   # real features
                for real_f, fake_f in zip(pred_real_detached, pred_fake_for_G):
                    for rf, ff in zip(real_f[:-1], fake_f[:-1]):   # skip final logits
                        loss_G_FM += criterionFeat(ff, rf)
                loss_G_FM = loss_G_FM * 10.0 * (1.0 / 3)

                # VGG loss
                loss_vgg_val = vgg_loss(real_img, fake_img) * 10.0

                loss_G = loss_G_GAN + loss_G_FM + loss_vgg_val

            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()

            # Save visual results every fixed number of steps
            if global_step % 50 == 0:
                with torch.no_grad():
                    vis_fake = (fake_img * 0.5 + 0.5).cpu()
                    vis_real = (real_img * 0.5 + 0.5).cpu()
                    grid = torch.cat([vis_fake, vis_real], dim=3)
                    torchvision.utils.save_image(grid, f"live_preview_brain2/{global_step}.png")

            pbar.set_postfix(D=float(loss_D), G=float(loss_G))

        # Save checkpoints every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(G.state_dict(), f"{params['checkpoints_dir']}/G_epoch{epoch+1}.pth")
            torch.save(D.state_dict(), f"{params['checkpoints_dir']}/D_epoch{epoch+1}.pth")


if __name__ == "__main__":
    params = {
        "label_nc": 2,
        "output_nc": 3,
        "ngf": 128,
        "ndf": 128,
        "norm": "instance",
        "lr": 0.0002,
        "beta1": 0.5,
        "beta2": 0.999,
        "batch_size": 4,
        "num_epochs": 100,
        "checkpoints_dir": "./checkpoints2_brain",
    }
    dataset_path = r"C:\Users\ad\Desktop\gans2\brain_tumor_dataset"
    loader = get_brain_loader(dataset_path, batch_size=params["batch_size"])
    print(f"Found {len(loader.dataset)} samples.")
    train_pix2pixHD_brain(params, loader)