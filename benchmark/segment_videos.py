"""
This script segments videos based on filenames in a specific directory.
"""
__author__ = "Samuel Adebayo"

import os
from utils import extract_segment

def main(existingSegment_directory, baseVid_directory, output_directory, view_name):
    os.makedirs(output_directory, exist_ok=True)
    print(view_name)
    control_count = 0
    video_list = []
    print(f'{os.listdir(existingSegment_directory)}')
    subtask_folders = sorted([os.path.join(existing_segment_directory, subtask) for subtask in os.listdir(existingSegment_directory)])

    for subtask in subtask_folders:
        subtask_videos = sorted([os.path.join(subtask, video) for video in os.listdir(subtask) if video.lower().endswith('.mp4') and view_name.lower() in video.lower()])
        video_list.extend(subtask_videos)
        control_count += len(subtask_videos)
    print(f'Total videos to process: {control_count}')
    extract_segment(baseVid_directory, video_list, output_directory, view_name)

if __name__ == '__main__':
    existing_segment_directory = '/home/samuel/extended_storage/Datasets/QUB-PHEO/segmented'
    base_videos_directory = '/home/samuel/extended_storage/Datasets/QUB-PHEO/corrected'
    output_directory = '/home/samuel/ml_projects/QUBPHEO/benchmark'
    view_name = 'CAM_AV'

    main(existing_segment_directory, base_videos_directory, output_directory, view_name)

