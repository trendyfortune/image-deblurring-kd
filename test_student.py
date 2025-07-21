import os
import time
import sys
import cv2
import argparse
import numpy as np
from torch.backends import cudnn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
from data.data_load import test_dataloader
from contextlib import contextmanager
#from models.MIMOUNet import build_net
#from data.padding import pad_to_multiple, crop_to_original

@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output"""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

with suppress_stderr():
    import torch_tensorrt
    import torch

def compute_metrics(output, gt):
    output = torch.clamp(output, 0, 1)
    gt = torch.clamp(gt, 0, 1)

    pred_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    gt_np = gt.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    pred_uint8 = (pred_np * 255).astype(np.uint8)
    gt_uint8 = (gt_np * 255).astype(np.uint8)

    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
    ssim_score = ssim(gt_uint8, pred_uint8, channel_axis=2, data_range=255)
    return psnr, ssim_score

def evaluate(args):
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nloading...\n")

    student_jit = torch.jit.load(args.student_model).to(device).eval()

    #Torch-TensorRT (TorchScript API)
    student = torch_tensorrt.ts.compile(
            student_jit,
            inputs=[ torch_tensorrt.ts.Input(shape=(1, 3, 1080, 1920), dtype=torch.float16)],
            enabled_precisions={torch.float16}, 
            truncate_long_and_double=True,
        )

    #teacher = build_net("MIMO-UNet").to(device)
    #state_dict = torch.load(args.teacher_weights, map_location=device)
    #teacher.load_state_dict(state_dict["model"])
    #teacher.eval()

    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=4)

    os.makedirs("dataset/results", exist_ok=True)

    s_psnr = s_ssim = t_psnr = t_ssim = 0
    count = 0

    #warmup     
    with torch.no_grad():
        for i, (blur, _, _) in enumerate(dataloader):
            if i >= 5:
                break
            blur = blur.to(device).to(memory_format=torch.channels_last).contiguous().half()
            #blur, _, _ = pad_to_multiple(blur, multiple=16)
            _ = student(blur)
    
    with torch.no_grad():
        for blur, sharp, name in dataloader:
            if count >= args.num_test_images:
                break

            blur = blur.to(device)
            sharp = sharp.to(device)
            #blur, pad_h, pad_w = pad_to_multiple(blur, multiple=16)
            blur = blur.half()

            # Teacher
            #t_out = teacher(blur)[-1]
            #t_out = crop_to_original(t_out, pad_h, pad_w)
            
            # Student
            s_out = student(blur)
            #s_out = crop_to_original(s_out, pad_h, pad_w)

            s_out_np = s_out.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
            s_out_np = (np.clip(s_out_np, 0, 1) * 255).astype(np.uint8).copy()

            s_out_np = cv2.putText(s_out_np, 'student', (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,0), 4, cv2.LINE_AA)

            filename = os.path.basename(name[0])
            save_path = os.path.join("dataset/results", filename)
            cv2.imwrite(save_path, cv2.cvtColor(s_out_np, cv2.COLOR_RGB2BGR))

            # Metrics
            psnr_s, ssim_s = compute_metrics(s_out, sharp)
            #psnr_t, ssim_t = compute_metrics(t_out, sharp)

            s_psnr += psnr_s
            s_ssim += ssim_s
            #t_psnr += psnr_t
            #t_ssim += ssim_t
            count += 1
            if count==1:
                print("\n")
            print(f"{name[0]} | Student SSIM: {ssim_s:.4f}, PSNR: {psnr_s:.2f}")

    print("\n================== SUMMARY ===================")
    print(f"\n> Evaluated on {count} images")
    print(f"> Student Avg SSIM: {s_ssim / count:.4f}, PSNR: {s_psnr / count:.2f}")
    print("> Check /results for imgs")
    #print(f"> Teacher Avg SSIM: {t_ssim / count:.4f}, PSNR: {t_psnr / count:.2f}")
    print("\n==============================================\n")

    count = total_time = 0
    torch.cuda.synchronize()
    print("calculating fps...\n")
    # FPS Benchmark
    with torch.no_grad():
        for blur, _, _ in dataloader:
            if count >= args.num_test_images:
                break

            blur = blur.to(device).to(memory_format=torch.channels_last).contiguous().half()
            #blur, pad_h, pad_w = pad_to_multiple(blur, multiple=16)

            start = time.time()
            _ = student(blur)
            torch.cuda.synchronize()
            end = time.time()

            total_time += (end - start)
            count += 1
    
    print("\n========== RAW INFERENCE BENCHMARK ===========\n")
    print(f"⚡ Pure Inference FPS: {count / total_time:.2f} frames/sec")
    print("\n==============================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--student_model', type=str, default='s_weights/best_student_scripted.pt')
    #parser.add_argument('--teacher_weights', type=str, default='t_weights/MIMO-UNet.pkl')
    parser.add_argument('--data_dir', type=str, default='dataset/div2k')
    parser.add_argument('--num_test_images', type=int, default=100)

    args = parser.parse_args()
    evaluate(args)
