import os
import torch
from torch import nn, optim
import torchvision
from torch.cuda.amp import autocast, GradScaler
from dataloader import SkinLesionDataset
from modelfunc import one_hot_encode_label
from netwroks import (
    MultiscaleDiscriminator,
    get_norm_layer,
    weights_init,
    get_edges,
    GANLoss,
    GlobalGenerator,
    VGGLoss
)


def train_pix2pixHD(params, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("live_preview", exist_ok=True)
    label_nc = params["label_nc"]
    input_nc_G = label_nc + 1   
    output_nc = params["output_nc"]
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
    optimizer_G = optim.Adam(G.parameters(), lr=params["lr"],
                             betas=(params["beta1"], params["beta2"]))
    optimizer_D = optim.Adam(D.parameters(), lr=params["lr"],
                             betas=(params["beta1"], params["beta2"]))
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G,
        lr_lambda=lambda e: 1.0 if e < 80 else max(0, 1 - (e - 80) / 40)
    )
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D,
        lr_lambda=lambda e: 1.0 if e < 80 else max(0, 1 - (e - 80) / 40)
    )
    vgg_loss = VGGLoss(device=device)
    criterionGAN = GANLoss(use_lsgan=True,
                           tensor=torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor)
    criterionFeat = nn.L1Loss()
    VIS_FREQ = 50
    global_step = 0
    scaler = GradScaler()
    for epoch in range(params["num_epochs"]):
        for i, (semantic, instance, real_img) in enumerate(train_loader):
            global_step += 1

            semantic = semantic.to(device)  
            instance = instance.to(device)   
            real_img = real_img.to(device)  

            sem_onehot = one_hot_encode_label(semantic, label_nc)  
            inst_map = instance.unsqueeze(1).float()               
            edges = get_edges(inst_map)                            
            cond = torch.cat([sem_onehot, edges], dim=1)           
            optimizer_D.zero_grad()
            with autocast():
                fake_img = G(cond).detach()
                real_pair = torch.cat([cond, real_img], dim=1)
                fake_pair = torch.cat([cond, fake_img], dim=1)
                pred_real = D(real_pair)
                pred_fake = D(fake_pair)
                loss_D_real = criterionGAN(pred_real, True)
                loss_D_fake = criterionGAN(pred_fake, False)
                loss_D = (loss_D_real + loss_D_fake)

            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)

            optimizer_G.zero_grad()
    
            with autocast():
                fake_img = G(cond)
                fake_pair = torch.cat([cond, fake_img], dim=1)
                pred_fake_for_G = D(fake_pair)
                loss_G_GAN = criterionGAN(pred_fake_for_G, True)
                feat_scales = len(pred_fake_for_G)
                feat_layers = len(pred_fake_for_G[0]) - 1
                lambda_feat = params.get("lambda_feat", 10.0)
                loss_G_FM = 0
                for s in range(feat_scales):
                    for l in range(feat_layers):
                        loss_G_FM += criterionFeat(
                            pred_fake_for_G[s][l],
                            pred_real[s][l].detach())
                        
                loss_vgg = vgg_loss.forward(real_img, fake_img)
                loss_G_FM = loss_G_FM * lambda_feat / (feat_scales * feat_layers)
                vgg_weight = 0.2 * max(0.0, 1 - (epoch - 80) / 40)
                loss_G = loss_G_GAN + loss_G_FM + loss_vgg * vgg_weight
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()
            if i % VIS_FREQ == 0:
                with torch.no_grad():
                    fake_b = (fake_img.cpu() * 0.5 + 0.5).clamp(0, 1)
                    save_path = os.path.join("live_preview", f"epoch_{epoch}_step_{i}.png")
                    torchvision.utils.save_image(fake_b, save_path)
        scheduler_G.step()
        scheduler_D.step()
        if (epoch + 1) % 5 == 0:
            ckpt_dir = params["checkpoints_dir"]
            os.makedirs(ckpt_dir, exist_ok=True)   
            G_path = os.path.join(ckpt_dir, f"G_epoch{epoch+1}.pth")
            D_path = os.path.join(ckpt_dir, f"D_epoch{epoch+1}.pth")
            torch.save(G.state_dict(), G_path)
            torch.save(D.state_dict(), D_path)
params = {
    "label_nc": 7,           
    "output_nc": 3,
    "ngf": 96,
    "ndf": 96,
    "norm": "instance",      
    "lr": 0.0002,
    "beta1": 0.5,
    "beta2": 0.999,
    "batch_size": 4,
    "num_epochs": 120,
    "gpu_ids": [0],
    "lambda_feat": 10.0,     
    "checkpoints_dir": "./checkpoints_pix2pixHD",
}
if __name__ == "__main__":
    image_dir = r"C:\Users\as0118d\Desktop\gans2\export\images_512p"
    semantic_dir = r"C:\Users\as0118d\Desktop\gans2\export\semantic512"
    instance_dir = r"C:\Users\as0118d\Desktop\gans2\export\bordermap512"
    train_ds = SkinLesionDataset(
        img_dir=image_dir,
        semantic_dir=semantic_dir,
        instance_dir=instance_dir,
        size=(256, 256)       
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    train_pix2pixHD(params, train_dl)