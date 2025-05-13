'''
Script to copy aerial-view .mp4 videos (containing a specific key) from the QUB-PHEO dataset's Annotated directory
and organize them into a destination folder for further preprocessing, preserving folder structure and displaying
progress bars using tqdm.

Usage:
    python copy_videos_with_tqdm.py

Adjust the FOLDER_PATH (QUB-PHEO Annotated directory), FOLDER_TO_COPYTO (destination for aerial view videos),
and key_for_extract variables in the __main__ section as needed.

Author: Samuel Adebayo
'''

__author__ = "Samuel Adebayo"

import os
import shutil
from tqdm import tqdm


def copy_videos(src_dir: str, dst_dir: str, key_for_extract: str = 'CAM_AV') -> int:
    """
    Copy .mp4 files containing a specified key from each subdirectory of src_dir
    to corresponding subdirectories in dst_dir, with progress bars.

    Args:
        src_dir (str): Path to the source directory containing subfolders.
        dst_dir (str): Path to the destination directory where files will be copied.
        key_for_extract (str, optional): Substring to filter filenames. Defaults to 'CAM_AV'.

    Returns:
        int: Total number of files copied.
    """

    file_count = 0
    subfolders = sorted(os.listdir(src_dir))

    for folder in tqdm(subfolders, desc="Processing folders", unit="folder"):
        folder_path = os.path.join(src_dir, folder)
        target_folder = os.path.join(dst_dir, folder)

        if not os.path.isdir(folder_path):
            continue

        os.makedirs(target_folder, exist_ok=True)

        video_files = [f for f in os.listdir(folder_path)
                       if f.endswith('.mp4') and key_for_extract in f]

        for file_name in tqdm(video_files, desc=f"Copying in {folder}", leave=False, unit="file"):
            src_file = os.path.join(folder_path, file_name)
            dst_file = os.path.join(target_folder, file_name)

            if os.path.exists(dst_file):
                continue

            shutil.copy2(src_file, dst_file)
            file_count += 1

    return file_count


if __name__ == '__main__':
    FOLDER_PATH = '/home/samuel/extended_storage/Datasets/QUB-PHEO/segmented'
    FOLDER_TO_COPYTO = '/home/samuel/extended_storage/Datasets/QUB-PHEO/process_wait'
    KEY_FOR_EXTRACT = 'CAM_AV'

    total_copied = copy_videos(FOLDER_PATH, FOLDER_TO_COPYTO, KEY_FOR_EXTRACT)
    print(f'Overall number of files copied: {total_copied}')
