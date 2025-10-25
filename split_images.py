import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import glob

def split_image_matplotlib(image_path):
    img = mpimg.imread(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    height, width = img.shape[0], img.shape[1]
    print(height, width)
    mid_y, mid_x = height // 2, width // 2
    print(mid_y, mid_x)
    # Define the four quadrants
    quadrants = [
        img[:mid_y, :mid_x],    # Top-left
        img[:mid_y, mid_x:],    # Top-right
        img[mid_y:, :mid_x],    # Bottom-left
        img[mid_y:, mid_x:],    # Bottom-right
    ]

    for i, quadrant in enumerate(quadrants, start=1):
        save_path = f"/content/drive/MyDrive/Nevagim/samples/split_images/images/{basename}_{i}.png"
        plt.imsave(save_path, quadrant)
        print(f"Saved: {save_path}")


images_to_split = glob.glob('/content/drive/MyDrive/Nevagim/samples/new samples/*')
for image in images_to_split:
  split_image_matplotlib(image)
