import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.MIMOUNet import build_net
from models.StudentNet import StudentNet
from data.data_load import train_dataloader, valid_dataloader 
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import torch.nn.functional as F
#from data.padding import pad_to_multiple, crop_to_original
from piq import ssim as ssim_loss_fn, LPIPS
import time

def evaluate_model(model, dataloader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    num_images = 0

    with torch.no_grad():
        for blur, sharp in dataloader:
            blur = blur.to(device)
            sharp = sharp.to(device)

            #blur, pad_h, pad_w = pad_to_multiple(blur, multiple = 16)

            output = model(blur)

            #output = crop_to_original(output, pad_h, pad_w)

            output = torch.clamp(output, 0, 1)

            for i in range(output.size(0)):  # Loop over batch
                pred = output[i].cpu().numpy().transpose(1, 2, 0)
                label = sharp[i].cpu().numpy().transpose(1, 2, 0)

                pred_uint8 = np.clip(pred * 255, 0, 255).astype(np.uint8)
                label_uint8 = np.clip(label * 255, 0, 255).astype(np.uint8)

                psnr = peak_signal_noise_ratio(label, pred, data_range=1.0)
                ssim_score = structural_similarity(label_uint8, pred_uint8, channel_axis=2, data_range=255)

                total_psnr += psnr
                total_ssim += ssim_score
                num_images += 1

    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    return avg_psnr, avg_ssim


def train_student(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher = build_net('MIMO-UNet').to(device)
    student = StudentNet().to(device)

    state_dict = torch.load(args.teacher_weights, map_location=device)
    teacher.load_state_dict(state_dict['model'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    lpips_loss = LPIPS(replace_pooling=True).to(device).eval()

    optimizer = optim.Adam(student.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    train_loader = train_dataloader(args.data_dir, batch_size = args.batch_size, num_workers = args.num_workers, use_transform=False)

    val_loader = valid_dataloader("dataset/validation", batch_size=1, num_workers=2)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)

    best_ssim = 0
    best_model_path = os.path.join(args.save_dir, "best_student.pkl")

    try:
        for epoch in range(args.epochs):
            student.train()
            total_loss = 0

            loop = tqdm(train_loader, desc = f"Epoch [{epoch + 1}/{args.epochs}]")
            for blur, sharp in loop:
                blur = blur.to(device)
                sharp = sharp.to(device)

                #blur, pad_h, pad_w = pad_to_multiple(blur, multiple = 16)

                with torch.no_grad():
                    teacher_output = teacher(blur)[-1]

                student_output = student(blur)

                #student_output = crop_to_original(student_output, pad_h, pad_w)
                #teacher_output = crop_to_original(teacher_output, pad_h, pad_w)

                student_output_clamped = torch.clamp(student_output, 0, 1)
                sharp_clamped = torch.clamp(sharp, 0, 1)

                loss_lpips = lpips_loss(student_output_clamped, sharp_clamped).mean()
                loss_gt = l1_loss(student_output, sharp)
                loss_kd = mse_loss(student_output, teacher_output)
                loss_ssim = 1 - ssim_loss_fn(student_output_clamped, sharp_clamped, data_range=1.)

                loss = 0.5 * loss_kd + 0.2 * loss_gt + 0.2 * loss_ssim + 0.1 * loss_lpips

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current_lr = optimizer.param_groups[0]['lr']
                
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item(), lr=current_lr)

            print(f"Epoch {epoch + 1}, Avg Loss: {total_loss / len(train_loader):.4f}")
            if (epoch + 1) %  2 == 0:
                psnr, ssim = evaluate_model(student, val_loader, device)
                print(f"Validation - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
                scheduler.step(ssim)  # Let scheduler adapt LR based on SSIM
                
                if ssim > best_ssim:
                    best_ssim = ssim
                    torch.save({'model': student.state_dict()}, best_model_path)
                    print(f"\n✅ Saved new best model at Epoch {epoch + 1} (SSIM: {ssim:.4f})\n")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by the user")
        
    print("\n-- Best model saved as best_student.pkl under /s_weights --")


if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="dataset/div2k")
    parser.add_argument("--teacher_weights", type=str, default="t_weights/MIMO-UNet.pkl")
    parser.add_argument("--save_dir", type=str, default="s_weights/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=21)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--num_workers", type=int, default=7)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    start_time = time.time()

    train_student(args)

    end_time = time.time()
    elapsed = end_time - start_time
    elapsed_min = int(elapsed // 60)
    elapsed_sec = int(elapsed % 60)
    print(f"\n Total training time: {elapsed_min} min {elapsed_sec} sec")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = StudentNet().to(device).half()
    ckpt = torch.load("s_weights/best_student.pkl", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy_input = torch.randn(1, 3, 1088, 1920).to(device).half()

    # Convert using torch.jit.script
    if os.path.exists("s_weights/best_student.pkl"):
        with torch.no_grad():
            scripted = torch.jit.script(model)
        torch.jit.save(scripted, "s_weights/best_student_scripted.pt")
        print("\n-- The scripted model for the same saved as best_student_scripted.pt --\n")
