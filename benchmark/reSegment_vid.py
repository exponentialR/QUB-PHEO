"""
This script re-segments videos previously segmented by QUB-HRI/segment_video.py.
But were segmented incorrectly. Most of these videos (120 across all 5 views, 24 videos per view) were segmented incorrectly.
"""
__author__ = "Samuel Adebayo"

import os
import subprocess
from tqdm import tqdm

def main(badSegment_txt, baseVid_directory, output_directory, view_name):
    os.makedirs(output_directory, exist_ok=True)
    bad_counter = 0
    subtask_dict = {}
    with open(badSegment_txt, 'r') as f:
        bad_segment_files = [line.strip() for line in f]

    for video_name in tqdm(bad_segment_files, desc='Bad Segments'):
        if view_name.lower() not in video_name.lower() or not video_name.lower().endswith('.mp4'):
            continue


        part_id, cam_view, task_name, subtask_name, ts = video_name.split('-')
        start_str, end_str = ts.split('_')
        end_str = end_str[:-4]  # strip “.mp4”

        try:
            start = float(start_str)
            end   = float(end_str)
        except ValueError:
            print(f"⚠️  Couldn't parse times in {video_name}, skipping.")
            continue

        duration = end - start
        if duration <= 0:
            print(f"⚠️  Non-positive duration ({duration:.3f}s) for {video_name}, skipping.")
            continue

        # build paths
        main_base_video = os.path.join(baseVid_directory,
                                       part_id,
                                       f"{cam_view}_P",
                                       f"{task_name}.mp4")
        # put subtask count in a dictionary
        if subtask_name not in subtask_dict:
            subtask_dict[subtask_name] = 1
        else:
            subtask_dict[subtask_name] += 1
        output_seg_folder = os.path.join(output_directory, subtask_name)
        os.makedirs(output_seg_folder, exist_ok=True)
        output_path = os.path.join(output_seg_folder, video_name)

        if not os.path.isfile(main_base_video):
            print(f"❌  Base missing: {main_base_video}, skipping {video_name}")
            continue

        # cmd = [
        #     "ffmpeg",
        #     "-y",                     # overwrite if exists
        #     "-ss", f"{start:.6f}",    # seek *into* the input
        #     "-i", main_base_video,
        #     "-t", f"{duration:.6f}",  # copy exactly this many seconds
        #     "-c", "copy",             # stream copy both audio/video
        #     output_path
        # ]
        cmd = [
            "ffmpeg",
            "-y",  # overwrite if exists
            "-ss", f"{start:.6f}",  # seek *into* the input
            "-i", main_base_video,  # source file
            "-t", f"{duration:.6f}",  # duration to extract
            "-c:v", "copy",  # copy *only* video stream
            "-an",  # drop any audio stream
            output_path
        ]

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print(f"❌  ffmpeg failed on {video_name}:\n{res.stderr}")
            continue

        print(f"✅  Re-segmented {video_name}")
        bad_counter += 1

    print(f"\nTotal bad segments treated: {bad_counter}")
    s_dict = {key: subtask_dict[key] for key in sorted(subtask_dict)}
    print(f'Syntax of subtask names: {s_dict}')



if __name__ == '__main__':

    badSegment_txt = 'bad_segments.txt'
    base_videos_directory = '/home/samuel/extended_storage/Datasets/QUB-PHEO/corrected'
    output_directory = '/home/samuel/ml_projects/QUBPHEO/benchmark'
    view_name = 'CAM_AV'

    main(badSegment_txt, base_videos_directory, output_directory, view_name)


