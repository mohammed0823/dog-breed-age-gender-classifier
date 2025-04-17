import os
import numpy as np
import kagglehub
from PIL import Image, ImageOps, ImageFilter, ImageStat
from PIL.ImageFilter import EDGE_ENHANCE, CONTOUR, SMOOTH, SMOOTH_MORE, BoxBlur, EDGE_ENHANCE_MORE, DETAIL, MaxFilter, \
    MinFilter, MedianFilter, GaussianBlur

# Download latest version
path = kagglehub.dataset_download("user164919/the-dogage-dataset")
print("Path to dataset files:", path)

# Get the folder paths from user input
folder1_path = input("Enter the path for the raw data folder: ").strip()
expert_adult_path = input("Enter the path for the expert adult data folder: ").strip()
save_path = input("Enter the path where you want to save processed images: ").strip()
alt_save_path = input("Enter the path where you want to save background removed images: ").strip()
aug_save_path = input("Enter the path where you want to save augmented images: ").strip()

# Iterate through folder
for filename in os.listdir(folder1_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Get image
        img_path = os.path.join(folder1_path, filename)
        imgOG = Image.open(img_path)
        imgOG = imgOG.convert('RGB')

        ##PROCESSING
        imgOG = imgOG.resize((256, 256))
        img = imgOG.convert('L')
        img = img.filter(ImageFilter.EDGE_ENHANCE)
        img = ImageOps.equalize(img, mask=None)

        ##AUGMENTING
        augImgs = []
        augImgs.append(img.transpose(Image.Transpose.FLIP_TOP_BOTTOM))
        augImgs.append(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
        augImgs.append(img.rotate(45))
        augImgs.append(img.rotate(135))
        augImgs.append(img.rotate(225))
        augImgs.append(img.rotate(315))

        # ADDING RANDOM NOISE
        imgNoisy = np.array(img)
        noise = np.random.randint(100, size=(256, 256))
        for i in range(len(imgNoisy)):
            for j in range(len(imgNoisy[0])):
                if imgNoisy[i][j] + noise[i][j] <= 255:
                    imgNoisy[i][j] = (imgNoisy[i][j] + noise[i][j])
        augImgs.append(Image.fromarray(imgNoisy))

        ##MASKING
        # Find color band with the highest contrast, measured by standard deviation of pixel intensity
        imgBands = list(imgOG.split())
        maxStdDev = 0
        maxContrast = 0
        stats = ImageStat.Stat(imgOG)
        for band, _ in enumerate(imgOG.getbands()):
            if stats.stddev[band] > maxStdDev:
                maxStdDev = stats.stddev[band]
                maxContrast = band

        # Create mask of dog
        threshold = 127
        mask = imgBands[maxContrast].point(lambda x: 255 if x > threshold else 0)

        mask = mask.filter(ImageFilter.GaussianBlur(5))
        mask = mask.point(lambda x: 255 if x > threshold else 0)

        # Make a box in center of mask, count all black
        mask = np.array(mask)
        b = 0
        for i in range(64, 192, 1):
            for j in range(64, 192, 1):
                if mask[i][j] == 0:
                    b += 1
        mask = Image.fromarray(mask)

        # If black is majority in the box
        if b >= 8192:
            mask = ImageOps.invert(mask)

        mask = mask.filter(ImageFilter.BoxBlur(20))

        # Add dog back to blurred background
        background = img.filter(ImageFilter.BoxBlur(20))
        background.paste(img, (0, 0), mask)

        ##SAVING
        # Convert all image types to jpg and save to respective folder
        filename = filename.rsplit('.', 1)[0]
        # img.save(os.path.join(save_path, filename+".jpg"))
        background.save(os.path.join(alt_save_path, filename + ".jpg"))

        for i in range(len(augImgs)):
            tempname = filename + " " + str(i)
            # augImgs[i].save(os.path.join(aug_save_path, tempname + ".jpg"))