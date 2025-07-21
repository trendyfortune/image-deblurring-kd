import torch
from models.MIMOUNet import build_net
from data.data_load import valid_dataloader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from tqdm import tqdm
#from data.padding import pad_to_multiple, crop_to_original

def evaluate_model(model, dataloader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    num_images = 0

    with torch.no_grad():
        for blur, sharp in tqdm(dataloader, desc="Evaluating Teacher"):
            blur = blur.to(device)
            sharp = sharp.to(device)

            #blur, pad_h, pad_w = pad_to_multiple(blur, multiple=16)

            output = model(blur)[-1]  
            #output = crop_to_original(output, pad_h, pad_w)
            output = torch.clamp(output, 0, 1)

            for i in range(output.size(0)):
                pred = output[i].cpu().numpy().transpose(1, 2, 0)
                label = sharp[i].cpu().numpy().transpose(1, 2, 0)

                pred_uint8 = np.clip(pred * 255, 0, 255).astype(np.uint8)
                label_uint8 = np.clip(label * 255, 0, 255).astype(np.uint8)

                psnr = peak_signal_noise_ratio(label, pred, data_range=1.0)
                ssim = structural_similarity(label_uint8, pred_uint8, channel_axis=2, data_range=255)

                total_psnr += psnr
                total_ssim += ssim
                num_images += 1

    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    return avg_psnr, avg_ssim


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_weights", type=str, default="t_weights/MIMO-UNet.pkl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_net("MIMO-UNet").to(device)
    state_dict = torch.load(args.teacher_weights, map_location=device)
    model.load_state_dict(state_dict["model"])
    model.eval()

    val_loader = valid_dataloader("dataset/validation", batch_size=args.batch_size, num_workers=args.num_workers)

    psnr, ssim = evaluate_model(model, val_loader, device)
    print(f"\n Teacher Evaluation Results:\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}")
