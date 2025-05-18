"""
bbox_utils.py

This module provides utilities for extracting 2D bounding boxes from video frames
using a YOLOv8 model. It is designed for structured preprocessing of video data
in human-robot interaction and visual perception pipelines, particularly when
bounding box annotations are to be written directly into HDF5 datasets.

The core function, `extract_bbox_from_video`, reads a video frame-by-frame,
applies the YOLO model to detect rectangular objects and surrogate hands,
and populates two datasets in the associated HDF5 file:
    - 'rec_bboxes'        : shape (N, 30, 4)
    - 'surrogate_hands'   : shape (N, 2, 4)

Each detection is filtered by class index (e.g., class 10 for hands) and capped
at a predefined number of objects per frame to ensure consistent dataset size.

Warnings are logged for any anomalies such as:
    - Frame count mismatches between video and HDF5 timestamps
    - More than 2 hands or 30 rectangles detected per frame

Usage of this module assumes consistent naming between videos and HDF5 files,
as well as availability of timestamp metadata within the HDF5 structure.

Author  : Samuel Adebayo
Updated : 18 May 2025
"""


import os
import h5py
import numpy as np
import cv2
from ultralytics import YOLO

def extract_bbox_from_video(video_path, model, h5_path):
    """
    Extracts bounding boxes from a video using YOLOv8 and stores them in an HDF5 file.

    Parameters:
    ----------
    video_path : str or Path
        Path to the input video file (.mp4 format).
    model : YOLO
        A preloaded YOLOv8 model from the `ultralytics` library.
    h5_path : str or Path
        Path to the corresponding HDF5 file where extracted bounding boxes will be saved.

    Returns:
    -------
    video_name : str
        Name of the processed video file.
    total_rec : int
        Total number of rectangular bounding boxes detected and saved.
    total_hands : int
        Total number of surrogate hands detected and saved.
    surrogate_hands_track : dict
        Dictionary tracking extra surrogate hands detected beyond the limit (2 per frame).

    Notes:
    ------
    - YOLO predictions are assumed to use `xywhn` (normalised centre coordinates).
    - Detected class 10 objects are considered 'surrogate hands'.
    - All other classes are considered rectangular objects.
    - The HDF5 structure must include a 'timestamps' dataset to validate frame counts.
    """

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    total_rec = 0
    total_hands = 0
    with h5py.File(h5_path, 'a') as h5f:
        h5_frame_count = h5f['timestamps'].shape[0] if 'timestamps' in h5f else 0
        if frame_count != h5_frame_count:
            print(f"Warning: Frame count mismatch for {video_path}. Expected {frame_count}, found {h5_frame_count}.")

        if 'rec_bboxes' not in h5f.keys():
            print(f"Creating Rectangle BBoxes dataset for {video_path}...")
            rec_bboxes_dset = h5f.create_dataset('rec_bboxes', (frame_count, 30, 4), dtype=np.float32)
        else:
            rec_bboxes_dset = h5f['rec_bboxes']

        if 'surrogate_hands' not in h5f.keys():
            print(f"Creating Surrogate Hands dataset for {video_path}...")
            surrogate_hands_dset = h5f.create_dataset('surrogate_hands', (frame_count, 2, 4), dtype=np.float32)
        else:
            surrogate_hands_dset = h5f['surrogate_hands']

        for frame_idx in range(frame_count):
            hand_count, rec_count = 0, 0
            surrogate_hands_track = {}
            success, frame = cap.read()
            if not success:
                print(f"Error reading frame {frame_idx} from {video_path}.")
                break
            results = model(frame, verbose=False)
            rectangle_bboxes, surrogate_hands = np.zeros((30, 4), dtype=np.float32), np.zeros((2, 4), dtype=np.float32)

            num_boxes = len(results[0].boxes)
            if num_boxes > 36:
                print(f"Warning: More than 30 boxes detected in frame {frame_idx}. Only the first 30 will be saved.")
            for idx, box in enumerate(results[0].boxes[:36]):
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # convert to seconds
                x, y, w, h = box.xywhn[0].tolist()
                class_idx = int(box.cls)

                if class_idx == 10:
                    if hand_count < 2:
                        surrogate_hands[hand_count] = [x, y, w, h]
                        hand_count += 1
                    else:
                        print(f'{video_path} has more than 2 surrogate hands in frame {frame_idx}.')
                        surrogate_hands_track[f'{os.path.basename(video_path)}_{frame_idx}_{timestamp}'] = [x, y, w, h]
                        continue
                else:
                    if rec_count < 30:
                        rectangle_bboxes[rec_count] = [x, y, w, h]
                        rec_count += 1
                    else:
                        continue
            total_rec += rec_count
            total_hands += hand_count


            surrogate_hands_dset[frame_idx] = surrogate_hands
            rec_bboxes_dset[frame_idx] = rectangle_bboxes
    cap.release()
    print(f"✅  Extracted bounding boxes from {video_path} to {h5_path}.")
    return os.path.basename(video_path), total_rec, total_hands, surrogate_hands_track

if __name__ == "__main__":
    """=======================TESTING UTILS OF EXTRACTING BBOXES ========================================"""
    h5_path = '../sample_data/bbox_preprocess/p01-CAM_AV-BIAH_RB-BHO-0.0007253558395636801_1.4222339664428545.h5'
    video_path = '../sample_data/bbox_preprocess/p01-CAM_AV-BIAH_RB-BHO-0.0007253558395636801_1.4222339664428545.mp4'
    model_path = '../weights/Lego_YOLO.pt'
    yolo_model = YOLO(model_path)
    extract_bbox_from_video(video_path, yolo_model, h5_path)