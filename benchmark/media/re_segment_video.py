"""
This script re-segments videos that were previously segmented by QUB-HRI/segment_video.py.
But were segmented incorrectly. Most of these videos (120 across all 5 views, 24 videos per view) were segmented incorrectly.
"""
__author__ = "Samuel Adebayo"

import os
import subprocess

from tqdm import tqdm

file_name = 'p34-CAM_AV-STAIRWAY_MS-CS-72.58102160072437_74.04535045742935.mp4'
file_name_split = file_name.split('-')
print(file_name_split)
start_time, end_time = file_name_split[-1].split('_')[0], file_name_split[-1].split('_')[1][0:-4]
print(start_time, end_time)
# print(file_name_split[-1]

def main(bad_segment_directory, base_videos_directory, output_directory, view_name='cam_av'):
    """
    Re-segment videos that were previously segmented by QUB-HRI/segment_video.py.
    But were segmented incorrectly. Most of these videos (120 across all 5 views, 24 videos per view) were segmented incorrectly.
    """
    os.makedirs(output_directory, exist_ok=True)
    for video_name in tqdm(os.listdir(bad_segment_directory), desc='Bad Segments'):
        if view_name.lower() in video_name and video_name.lower().endswith('.mp4'):
            video_name_split = video_name.split('-')
            part_id, cam_view, task_name, subtask_name, timestamp = video_name_split[0], video_name_split[1], video_name_split[2], video_name_split[3], video_name_split[4]
            main_base_video = os.path.join(base_videos_directory, part_id, cam_view, task_name + '.mp4')
            start_time, end_time = video_name_split[-1].split('_')[0], video_name_split[-1].split('_')[1][0:-4]

            output_segment_filepath = os.path.join(output_directory, video_name_split, video_name)

            if os.path.exists(main_base_video):
                command = [
                    'ffmpeg',
                    '-i', main_base_video,
                    '-ss', str(start_time),
                    '-to', str(end_time),
                    '-c:v', 'copy',
                    '-c:a', 'copy',
                    output_segment_filepath
                ]
                subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Re-segmented {video_name} from {main_base_video} to {output_segment_filepath}")
            else:
                print(f"Base video {main_base_video} does not exist. Skipping {video_name}.")
                continue




