# image-deblurring-kd
Lightweight image deblurring using knowledge distillation
# Student-Teacher Image Deblurring with MIMO-UNet

This project implements a lightweight **StudentNet** model trained via **knowledge distillation** from a pretrained **MIMO-UNet** teacher model to perform real-time image deblurring.  

The goal is to achieve fast and efficient image restoration using a compact student network suitable for deployment.  

Report: [Intel_Unnati_Project_report_image_kd.pdf](https://github.com/user-attachments/files/21189142/Intel_Unnati_Project_report_image_kd.pdf)  
Demo video: https://www.youtube.com/watch?v=AV7A3DCs9LQ  



## Dependencies
- Python (3.12.x)
- Pytorch (2.6)
- CUDA (12.6)
- TensorRT (10.7), Torch-TensorRT (2.6)
  
(Nvidia RTX 4060 was used for this project)  

---
## Dataset

- Download DIV2K dataset.

- Arrange the ```dataset``` folder as follows:

```
├─ dataset
   └─ DIV2K_HR               % 900 images
      ├─ train               % 700 images   
      |  ├─ xxxx.png
      |  ├─ ......
      │
      ├─ test                % 100 images (taken from train)
      │  ├─ xxxx.png
      │  ├─ ......
      |
      └─ valid              % 100 images
         ├─ xxxx.png
         ├─ ......

```

- Prepare the dataset by running the command below:

   ```python image_prep.py ```

- After preparingdata set, the data folder should be like the format below:

```
├─ dataset
   ├─ div2k
   |   ├─ train
   |   |
   |   └─ test
   |
   ├─ DIV2K_HR
   |   └─ ......
   |
   └─ validation
       └─ valid              
    
```
---
## Train

To train the StudentNet model, run the command below:

   ```python train_student.py ```

Student weights saved in "s_weights/" (pretrained MIMO-UNet is used as teacher model)  
<br>

---

## Test

To test the StudentNet model, run the command below:

``` python test_student.py ```

Output images are stored in "results/"  
(optionally, run ```student_feature_map.py``` to get the feature maps)
<br>

---
<br>

## Performance  

StudentNet FPS range : 55-65 fps    
(Processes 1920x1080 images)  
<br>  

| <img width='100%' alt="Blurred" src="https://github.com/user-attachments/assets/861eeb8b-9719-4b04-8fdc-944870851207" /> |<img width="95%" alt="Student_op" src="https://github.com/user-attachments/assets/e799aba0-88c7-4634-99b2-779819b8c66e" /> | 
| :--: | :--: | 
| <img width='99%' alt="Blurred_zoomed" src="https://github.com/user-attachments/assets/463dd50c-5ae9-4a77-8916-da2677bb1201" /> | <img width='95%' alt="Student_op_zoomed" src="https://github.com/user-attachments/assets/819c28be-4469-40dd-bf00-47ae8c96b7fe" /> | 
| Blurred | StudentNet Output | 

<br>

---

|   Method    | MIMO-UNet | StudentNet | 
| :---------: | :-------: | :--------: | 
|  PSNR (dB)  |   34.78   |   32.00    | 
|    SSIM     |   0.949   |   0.933    |  
