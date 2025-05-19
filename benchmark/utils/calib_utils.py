import os.path

import numpy as np


def rescale_calib_params(calib_path, rescaled_calib_dir):
    """
    Rescale the calibration parameters in the given file using the original resolution vs new resolution.
    The original resolution is gotten by:
    1. Reading the intrinsic matrix K4 from the calibration file.
    2. The original width and height are calculated from the cx and cy values in K4.
    3. The new resolution is set to 1728x972.
    4. The scale factor is calculated as the ratio of the new width to the original width.
    5. The intrinsic matrix K4 is rescaled using the scale factor.
    6. The rescaled intrinsic matrix, distortion coefficients, translation vectors, and rotation vectors are saved
    in a new file with the prefix "Rescaled_" added to the original filename.

    Parameters:
    ----------
    calib_path : str
        Path to the calibration file.
    scale_factor : float
        Scale factor to rescale the calibration parameters.

    Returns:
    -------
    None
    """
    data = np.load(calib_path)
    K4 = data['mtx']
    dist = data['dist']
    tvecs = data['tvecs']
    rvecs = data['rvecs']

    cx, cy = K4[0, 2], K4[1, 2]
    orig_w = int(round(2 * cx))
    orig_h = int(round(2 * cy))

    resized_w, resized_h = 1728.0, 972.0

    s = resized_w / orig_w

    K_res = K4.copy()
    K_res[0, 0] *= s  # fx
    K_res[1, 1] *= s  # fy
    K_res[0, 2] *= s  # cx
    K_res[1, 2] *= s  # cy

    np.savez(f"{os.path.join(rescaled_calib_dir, 'Rescaled_' + os.path.basename(calib_path))}",
             mtx=K_res,
             dist=dist,
             tvecs=tvecs,
             rvecs=rvecs)


if __name__ == '__main__':
    calib_path = '/media/samuel/Expansion/DataFromRFE/PhD/QUBPHEO/calibrationparameters/p15-CAM_AV-intrinsic.npz'
    rescale_calib_params(calib_path, 'calib_data')
