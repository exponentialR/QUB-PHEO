import subprocess
from tqdm import tqdm
import os

def get_bad_segment(folder_directory):
    """
    This function gets the filenames of all the bad segments in the given directory,
    Then saves them into a text file called bad_segments.txt.
    :param folder_directory:
    :return:
    """
    bad_segments = []
    for bad_files in os.listdir(folder_directory):
        if bad_files.lower().endswith('mp4'):
            bad_segments.append(bad_files)
    with open('../bad_segments.txt', 'w') as f:
        for bad_file in bad_segments:
            f.write(bad_file + '\n')
    print(f"Bad segments saved to bad_segments.txt")


def extract_segment(base_video_directory, control_video_list, output_directory, view_name, retain_audio=False):
    """
    This function extracts the segments from the base video directory and saves them to the output directory.
    :param retain_audio:
    :param base_video_directory:
    :param control_video_list:
    :param output_directory:
    :param view_name:
    :return:
    """
    video_counter = 0
    for vid_n in tqdm(control_video_list, desc='Video Segments'):
        video_name = os.path.basename(vid_n)
        if view_name.lower() not in video_name.lower() or not video_name.lower().endswith('.mp4'):
            continue

        part_id, cam_view, task_name, subtask_name, ts = video_name.split('-')
        start_str, end_str = ts.split('_')
        end_str = end_str[:-4]  # strip “.mp4”

        try:
            start = float(start_str)
            end = float(end_str)
        except ValueError:
            print(f"⚠️  Couldn't parse times in {video_name}, skipping.")
            continue

        duration = end - start
        if duration <= 0:
            print(f"⚠️  Non-positive duration ({duration:.3f}s) for {video_name}, skipping.")
            continue

        # build paths
        main_base_video = os.path.join(base_video_directory,
                                       part_id,
                                       f"{cam_view}_P",
                                       f"{task_name}.mp4")
        output_seg_folder = os.path.join(output_directory, subtask_name)
        os.makedirs(output_seg_folder, exist_ok=True)
        output_path = os.path.join(output_seg_folder, video_name)
        if not os.path.isfile(main_base_video):
            print(f"❌  Base missing: {main_base_video}, skipping {video_name}")
            continue

        if not retain_audio:
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
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", f"{start:.6f}",
                "-i", main_base_video,
                "-t", f"{duration:.6f}",
                "-c", "copy",
                output_path
            ]

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print(f"❌  ffmpeg failed on {video_name}:\n{res.stderr}")
            continue

        print(f"✅  Segmented {video_name}")
        video_counter += 1

    print(f"\nTotal video segments treated: {video_counter}")# strip

if __name__ == '__main__':
    """=======================TESTING UTILS OF GETTING BAD SEGMENTS ========================================"""
    folder_directory = '/home/samueladebayo/Documents/PhD/QUBPHEO/corrupted-segment'
    get_bad_segment(folder_directory)
