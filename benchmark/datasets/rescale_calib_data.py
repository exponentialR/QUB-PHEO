import os
from tqdm import tqdm
from utils.calib_utils import rescale_calib_params


if __name__ == '__main__':
    original_calib_dir = '/media/samuel/Expansion/DataFromRFE/PhD/QUBPHEO/calibrationparameters'
    rescaled_calib_dir = 'calib_data'
    os.makedirs(rescaled_calib_dir, exist_ok=True)

    calib_files_list = [file for file in os.listdir(original_calib_dir) if file.endswith('.npz')]
    print(f'FOUND {len(calib_files_list)} CALIBRATION FILES')

    success_count = 0
    for calib_file in tqdm(calib_files_list, desc='Rescaling Calibration Files'):
        calib_path = os.path.join(original_calib_dir, calib_file)
        rescale_calib_params(calib_path, rescaled_calib_dir)
        print(f'✅ Rescaled: {calib_file} into {rescaled_calib_dir}')
        success_count += 1
    print(f'✅ Rescaled {success_count} calibration files.')
    print(f'❌ Failed to rescale {len(calib_files_list) - success_count} calibration files.')

