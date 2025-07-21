import os
import cv2
import random
import argparse
from tqdm import tqdm
import cv2

DOWNSCALE_FACTORS = [1.5, 1.6, 1.8, 2.0]

def center_crop(img, crop_height=1080, crop_width=1920):
    h, w, _ = img.shape

    # Rotate cond
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]

    start_y = h // 2 - crop_height // 2 if h > crop_height else 0
    end_y = start_y + crop_height if h > crop_height else h

    start_x = w // 2 - crop_width // 2 if w > crop_width else 0
    end_x = start_x + crop_width if w > crop_width else w

    return img[start_y:end_y, start_x:end_x]


def gen_train_images(src, dst):

    PATCH_SIZE = 256
    PATCHES_PER_IMAGE = 6
    factor_counts = {factor: 0 for factor in DOWNSCALE_FACTORS}

    os.makedirs( os.path.join(dst, 'train/blur'), exist_ok=True)
    os.makedirs( os.path.join(dst, 'train/sharp'), exist_ok=True)

    image_files = [f for f in os.listdir(os.path.join(src, 'train'))]

    patch_id = 0
    print("\n")
    for file in tqdm(image_files, desc="Generating images for training"):
        path = os.path.join(os.path.join(src, 'train'), file)
        img = cv2.imread(path)

        scale = random.choice(DOWNSCALE_FACTORS)
        factor_counts[scale] += 1

        img = center_crop(img, 1080, 1920)
        h, w = img.shape[:2]

        new_w = int(w / scale)
        new_h = int(h / scale)

        img_small = cv2.resize(img, (new_w, new_h), cv2.INTER_CUBIC)
        img_blurry = cv2.resize(img_small, (w, h), cv2.INTER_CUBIC)

        for _ in range(PATCHES_PER_IMAGE):
            top = random.randint(0, h - PATCH_SIZE)
            left = random.randint(0, w - PATCH_SIZE)

            patch_sharp = img[top:top + PATCH_SIZE, left:left + PATCH_SIZE]
            patch_blur = img_blurry[top:top + PATCH_SIZE, left:left + PATCH_SIZE]

            cv2.imwrite(f"{os.path.join(dst, 'train/blur')}/{patch_id:05d}.png", patch_blur)
            cv2.imwrite(f"{os.path.join(dst, 'train/sharp')}/{patch_id:05d}.png", patch_sharp)

            patch_id += 1

    print(f" Created {patch_id} patch pairs ")
    print("\n Downscale factor usage summary:")
    for factor in sorted(factor_counts):
        print(f"  Factor {factor}: {factor_counts[factor]} images")
    print("\n")


def gen_test_images(src, dst):

    os.makedirs( os.path.join(dst, 'test/blur'), exist_ok=True)
    os.makedirs( os.path.join(dst, 'test/sharp'), exist_ok=True)

    image_files = [f for f in os.listdir(os.path.join(src, 'test'))]
    count = 0
    for file in tqdm(image_files, desc="Generating images for testing"):
        path = os.path.join(os.path.join(src, 'test'), file)
        img = cv2.imread(path)

        if img.shape[0] > img.shape[1]:  # height > width
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        img = center_crop(img, 1080, 1920)

        h, w = img.shape[:2]
        scale = random.choice(DOWNSCALE_FACTORS)
        new_w = int(w / scale)
        new_h = int(h / scale)

        img_small = cv2.resize(img, (new_w, new_h), cv2.INTER_CUBIC)
        img_blurry = cv2.resize(img_small, (w, h), cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join(dst, 'test/blur', file), img_blurry)
        cv2.imwrite(os.path.join(dst, 'test/sharp', file), img)
        count += 1

    print(f" Created {count} images for testing")
    print("\n")


def gen_valid_images(src):

    os.makedirs( os.path.join('dataset/validation/valid/blur'), exist_ok=True)
    os.makedirs( os.path.join('dataset/validation/valid/sharp'), exist_ok=True)

    image_files = [f for f in os.listdir(os.path.join(src, 'valid'))]
    count = 0
    for file in tqdm(image_files, desc="Generating images for validation"):
        path = os.path.join(os.path.join(src, 'valid'), file)
        img = cv2.imread(path)

        if img.shape[0] > img.shape[1]:  # height > width
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        img = center_crop(img, 1080, 1920)

        h, w = img.shape[:2]
        scale = random.choice(DOWNSCALE_FACTORS)
        new_w = int(w / scale)
        new_h = int(h / scale)

        img_small = cv2.resize(img, (new_w, new_h), cv2.INTER_CUBIC)
        img_blurry = cv2.resize(img_small, (w, h), cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join('dataset/validation/valid/blur', file), img_blurry)
        cv2.imwrite(os.path.join('dataset/validation/valid/sharp', file), img)
        count += 1

    print(f" Created {count} images for validation")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--root_src', default='dataset/DIV2K_HR', type=str)
    parser.add_argument('--root_dst', default='dataset/div2k', type=str)
    

    args = parser.parse_args()

    if not os.path.exists(args.root_dst):
        os.mkdir(args.root_dst)

    gen_train_images(args.root_src, args.root_dst)
    gen_test_images(args.root_src, args.root_dst)
    gen_valid_images(args.root_src)
    print("\n")