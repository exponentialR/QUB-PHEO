"""
extract_bbox.py

This script performs batch extraction of 2D bounding boxes from video datasets using a
YOLOv8 model and writes results into pre-existing HDF5 files. It serves as a preprocessing
utility for preparing large-scale visual data—especially for applications in human-robot
interaction, object tracking, or perception-driven modelling.

Videos are filtered by a camera view keyword (e.g., 'CAM_AV') and matched with HDF5 files
using consistent naming. For each matched pair, bounding boxes are extracted frame-by-frame
via `extract_bbox_from_video()` and inserted into the corresponding datasets:
    - 'rec_bboxes'        : shape (N, 30, 4)
    - 'surrogate_hands'   : shape (N, 2, 4)

A summary of all processed videos—including counts of detected rectangles and surrogate hands—
is rendered to the console and exported as a Markdown file for reproducibility and audit.

Key Features:
-------------
- Selective processing by camera view substring (e.g., CAM_AV)
- Integration with pre-labelled HDF5 timestamp-aligned data
- Per-video extraction statistics
- Markdown-formatted summary report for reproducibility

Author  : Samuel Adebayo
Updated : 18 May 2025
"""

__author__ = "Samuel Adebayo"

import os
from utils.bbox_utils import extract_bbox_from_video
from ultralytics import YOLO
from tabulate import tabulate
from pathlib import Path
from tqdm import tqdm
import shutil

def extract_bbox(video_directory:Path, h5_directory:Path, camera_view='CAM_AV', model_path=None):
    """
    Extracts bounding boxes from all videos under a specified directory using a YOLOv8 model,
    and writes the results into their corresponding HDF5 files.

    For each video that matches the specified `camera_view` filter, the function:
      - Loads the video and finds the corresponding HDF5 file.
      - Applies a YOLOv8 model to detect bounding boxes frame-by-frame.
      - Populates the 'rec_bboxes' and 'surrogate_hands' datasets within the HDF5 file.
      - Logs total rectangles and hands detected.
      - Tracks and logs excess hands beyond the per-frame limit (2 hands max).

    At the end of processing, a Markdown-formatted summary report is printed to the console
    and saved to `sample_data/bbox_preprocess/bbox_summary.md`.

    Parameters:
    ----------
    video_directory : Path
        Path to the root directory containing input `.mp4` videos. All subdirectories are searched recursively.
    h5_directory : Path
        Path to the root directory containing corresponding `.h5` files with matching filenames.
    camera_view : str, optional
        Keyword filter for selecting videos by camera view name (e.g., 'CAM_AV'). Case-insensitive.
    model_path : str, optional
        Path to the YOLO model weights file. If not specified, defaults to 'weights/Lego_YOLO.pt'.

    Returns:
    -------
    None
        This function produces side effects: bounding box datasets are updated in-place in each
        HDF5 file, and a Markdown summary is saved to disk.
    """
    video_list = sorted([video for video in video_directory.rglob('*.mp4') if camera_view.lower() in str(video).lower()])
    # h5_list = sorted([h5 for h5 in h5_directory.rglob('*.h5') if camera_view.lower() in str(h5).lower()])
    model_path = 'weights/Lego_YOLO.pt'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Downloading...")

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
    output_md_path = "sample_data/bbox_preprocess/bbox_summary.md"
    with open(output_md_path, 'w') as f:
        f.write("# Bounding Box Summary\n\n")
        f.write(markdown_table)
        f.write(f"\n\n**Total Processed Videos:** {total_files}\n")

    print(f"\n✅ Markdown summary saved to {output_md_path}")

if __name__ == '__main__':
    video_direc = Path('sample_data/bbox_preprocess/videos')
    h5_direc = Path('sample_data/bbox_preprocess/h5')
    extract_bbox(video_direc, h5_direc, camera_view='CAM_AV')




