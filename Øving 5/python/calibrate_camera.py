#
# The code in this script is largely copied from the official tutorial
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
#
# Please read the tutorial for explanations of the OpenCV functions.

import numpy as np
import cv2 as cv
import glob
from os.path import join, basename, realpath, dirname, exists, splitext

# This string should point to the folder containing the images
# used for calibration. The same folder will hold the output.
# "*.jpg" means that any file with a .jpg extension is used
image_path_pattern = 'data/calibration/*.jpg'
output_folder = dirname(image_path_pattern)

#
# TASK: Specify these
#
board_size = (4, 7)  # Number of internal corners of the checkerboard (see tutorial)
square_size = 30 # Real world length of the sides of the squares

#
# Tip:
# You can pass flags to calibrateCameraExtended to specify what distortion
# coefficients to estimate. See https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
# Below are two examples of flags that may be interesting to try.
#
calibrate_flags = 0 # Use default settings (three radial and two tangential)
# calibrate_flags = cv.CALIB_ZERO_TANGENT_DIST|cv.CALIB_FIX_K3 # Disable tangential distortion and third radial distortion coefficient

# Flags to findChessboardCorners that improve performance
detect_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Termination criteria for cornerSubPix routine
subpix_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#
# Detect checkerboard points
#
# Note: This first tries to use existing checkerboard detections,
# if present in the output folder. You can delete the 'u_all.npy'
# file to force the script to re-detect all the corners.
#
if exists(join(output_folder, 'u_all.npy')):
    u_all = np.load(join(output_folder, 'u_all.npy'))
    X_all = np.load(join(output_folder, 'X_all.npy'))
    image_size = np.loadtxt(join(output_folder, 'image_size.txt')).astype(np.int32)
    print('Using previous checkerboard detection results.')
else:
    X_board = np.zeros((board_size[0]*board_size[1], 3), np.float32)
    X_board[:,:2] = square_size*np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    X_all = []
    u_all = []
    image_size = None
    image_paths = glob.glob(image_path_pattern)
    for image_path in sorted(image_paths):
        print('%s...' % basename(image_path), end='')

        I = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if not image_size:
            image_size = I.shape
        elif I.shape != image_size:
            print('Image size is not identical for all images.')
            print('Check image "%s" against the other images.' % basename(image_path))
            quit()

        ok, u = cv.findChessboardCorners(I, (board_size[0],board_size[1]), detect_flags)
        if ok:
            print('detected all %d checkerboard corners.' % len(u))
            X_all.append(X_board)
            u = cv.cornerSubPix(I, u, (11,11), (-1,-1), subpix_criteria)
            u_all.append(u)
        else:
            print('failed to detect checkerboard corners, skipping.')

    np.savetxt(join(output_folder, 'image_size.txt'), image_size)
    np.save(join(output_folder, 'u_all.npy'), u_all) # Detected checkerboard corner locations
    np.save(join(output_folder, 'X_all.npy'), X_all) # Corresponding 3D pattern coordinates

print('Calibrating. This may take a minute or two...', end='')
results = cv.calibrateCameraExtended(X_all, u_all, image_size, None, None, flags=calibrate_flags)
print('Done!')
# ret, mtx, dist, lok, bok=results

ok, K, dc, rvecs, tvecs, std_int, std_ext, per_view_errors = results

mean_errors = []
for i in range(len(X_all)):
    u_hat, _ = cv.projectPoints(X_all[i], rvecs[i], tvecs[i], K, dc)
    vector_errors = (u_hat - u_all[i])[:,0,:] # the indexing here is because OpenCV likes to add extra dimensions.
    scalar_errors = np.linalg.norm(vector_errors, axis=1)
    mean_errors.append(np.mean(scalar_errors))

np.savetxt(join(output_folder, 'K.txt'), K) # Intrinsic matrix (3x3)
np.savetxt(join(output_folder, 'dc.txt'), dc) # Distortion coefficients
np.savetxt(join(output_folder, 'mean_errors.txt'), mean_errors)
np.savetxt(join(output_folder, 'std_int.txt'), std_int) # Standard deviations of intrinsics (entries in K and distortion coefficients)
print('Calibration data is saved in the folder "%s"' % realpath(output_folder))

# folder = "data/calibration"
# img_path = join(folder, "image035.jpg")  # change

# img = cv.imread(img_path)
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# ok, corners = cv.findChessboardCorners(gray, board_size)
# print("found corners:", ok, "count:", 0 if corners is None else len(corners))

# vis = img.copy()
# if ok:
#     cv.drawChessboardCorners(vis, board_size, corners, ok)

# cv.imshow("corners", vis)
# cv.waitKey(0)
# cv.destroyAllWindows()


def sampleCoeff(dc_mean_1x5, std_dc_5):
    dc_mean = np.asarray(dc_mean_1x5, dtype=float).reshape(1, 5)
    std_dc  = np.asarray(3*std_dc_5, dtype=float).reshape(1, 5)
    return np.random.normal(loc=dc_mean, scale=std_dc)


folder = "data"
img_path = join(folder, f"sample_image.jpg")  # change
img = cv.imread(img_path)
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(K, dc, (w,h), 0, (w,h))

for i in range(10):
    dc1=sampleCoeff(dc, std_int[4:9])

    # undistort
    dst = cv.undistort(img, K, dc1, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(f'calibration_result_img0{53}_r_{i}.png', dst)