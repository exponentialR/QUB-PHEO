
__author__ = "Samuel Adebayo"

import os
from utils import extract_bbox_from_video
from ultralytics import YOLO
from tabulate import tabulate
from pathlib import Path
from tqdm import tqdm
import shutil

def extract_bbox(video_directory:Path, h5_directory:Path, camera_view='CAM_AV', model_path=None):
    """

    :param video_directory:
    :param h5_directory:
    :return:
    """
    video_list = sorted([video for video in video_directory.rglob('*.mp4') if camera_view.lower() in str(video).lower()])
    # h5_list = sorted([h5 for h5 in h5_directory.rglob('*.h5') if camera_view.lower() in str(h5).lower()])
    model_path = 'models/Lego_YOLO.pt'
    yolo_model = YOLO(model_path)
    summary_results = []
    for video_file in tqdm(video_list, desc=f'Extracting BBoxes from Videos', unit='video'):
        h5_path = video_file.with_suffix('.h5')
        h5_path_ = h5_directory / h5_path.relative_to(video_directory)
        if os.path.exists(h5_path_):
            video_name, total_rec, total_hands, surrogate_hands_track = extract_bbox_from_video(video_file, yolo_model,
                                                                                                h5_path_)
            summary_results.append({
                "Video": video_name,
                "Total Rectangles": total_rec,
                "Total Hands": total_hands,
                "Extra Hands": len(surrogate_hands_track)
            })

    markdown_table = tabulate(summary_results, headers="keys", tablefmt="github")

    print("\n📊 Summary of Extracted Bounding Boxes:")
    print(markdown_table)
    total_files = len(summary_results)

    # Save to .md file
    output_md_path = "bounding_box_summary.md"
    with open(output_md_path, 'w') as f:
        f.write("# Bounding Box Summary\n\n")
        f.write(markdown_table)
        f.write(f"\n\n**Total Processed Videos:** {total_files}\n")

    print(f"\n✅ Markdown summary saved to {output_md_path}")

if __name__ == '__main__':
    video_direc = Path('/home/samuel/ml_projects/QUBPHEO/benchmark/videos/segmented')
    h5_direc = Path('/home/samuel/ml_projects/QUBPHEO/benchmark/landmarks')
    extract_bbox(video_direc, h5_direc, camera_view='CAM_AV')




