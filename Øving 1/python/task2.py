import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

grass = 'data/grass.jpg'

# 2.1)


def widthAndHeightImage(image):
    img = Image.open(image)

    width, height = img.size

    return width, height


print(widthAndHeightImage(grass))


# 2.2)
grass_img = Image.open(grass)

grass_array = np.array(grass_img)

ga0 = grass_array[:, :, 0]
ga1 = grass_array[:, :, 1]
ga2 = grass_array[:, :, 2]


def task2_2():

    plt.subplot(1, 3, 1)
    plt.imshow(ga0, cmap='gray')

    # Assume this is the green channel since the green trees are brightest in this image
    plt.subplot(1, 3, 2)
    plt.imshow(ga1, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.imshow(ga2, cmap='gray')
    plt.show()


# 2.3)
threshold_value = 120
thresholded = (ga1 > threshold_value).astype(np.uint8) * 255


def task2_3():
    plt.imshow(thresholded, cmap="gray")
    plt.axis("off")
    plt.show()

# Seems like having a value of 120 makes it so that the trees are standing out and having a black outline


# 2.4)
rgb = grass_array.astype(np.float32)

R = rgb[:, :, 0]
G = rgb[:, :, 1]
B = rgb[:, :, 2]

S = R + G + B
S[S == 0] = 1.0

r = R / S
g = G / S
b = B / S


def task2_4():
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(r, cmap="gray", vmin=0, vmax=1)
    plt.title("r")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(g, cmap="gray", vmin=0, vmax=1)
    plt.title("g")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(b, cmap="gray", vmin=0, vmax=1)
    plt.title("b")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


task2_4()


# 2.5)
threshold_value = 0.4
thresholded_5 = (g > threshold_value).astype(np.uint8) * 255


def task2_5():
    plt.imshow(thresholded_5, cmap="gray")
    plt.axis("off")
    plt.show()


task2_5()
