import numpy as np


def rgb_to_gray(I):
    """
    Converts a HxWx3 RGB image to a HxW grayscale image as
    described in the text.
    """
    R = I[:, :, 0]
    G = I[:, :, 1]
    B = I[:, :, 2]
    return (G+R+B)/3


def central_difference(I):
    """
    Computes the gradient in the x and y direction using
    a central difference filter, and returns the resulting
    gradient images (Ix, Iy) and the gradient magnitude Im.
    """
    # I = I.astype(np.float32)

    kernel = np.array([0.5, 0.0, -0.5])

    H, W = I.shape

    Ix = np.zeros_like(I)
    Iy = np.zeros_like(I)

    for i in range(H):
        Ix[i, :] = np.convolve(I[i, :], kernel, mode='same')

    for i in range(W):
        Iy[:, i] = np.convolve(I[:, i], kernel.T, mode='same')
    Im = np.sqrt(Ix**2 + Iy**2)
    return Ix, Iy, Im


def gaussian(I, sigma):
    """
    Applies a 2-D Gaussian blur with standard deviation sigma to
    a grayscale image I.
    """

    # Hint: The size of the kernel should depend on sigma. A common
    # choice is to make the half-width be 3 standard deviations. The
    # total kernel width is then 2*np.ceil(3*sigma) + 1.

    result = np.zeros_like(I)
    H, W = I.shape

    kernel_size = 2 * np.ceil(3 * sigma) + 1
    kernel = np.zeros(int(kernel_size))
    for i in range(-kernel_size//2, kernel_size//2+1):
        kernel[i] = np.random.normal(0, sigma)

    for i in range(H):
        result[i, :] = np.convolve(I[i, :], kernel, mode='same')

    for i in range(W):
        result[:, i] = np.convolve(result[:, i], kernel.T, mode='same')

    return result


def extract_edges(Ix, Iy, Im, threshold):
    """
    Returns the x, y coordinates of pixels whose gradient
    magnitude is greater than the threshold. Also, returns
    the angle of the image gradient at each extracted edge.
    """
    edges = Im > threshold
    y, x = np.nonzero(edges)
    theta = np.arctan2(Iy[edges], Ix[edges])

    return [x, y, theta]  # Placeholder
