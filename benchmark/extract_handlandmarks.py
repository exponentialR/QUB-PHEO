import re

import cv2
import mediapipe as mp
import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm
import pandas as pd
from utils import handLandMarkExtractor

# Suppress GLOG INFO and WARNING (0=INFO, 1=WARNING, 2=ERROR, 3=FATAL)
os.environ['GLOG_minloglevel']   = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging as _logging
_logging._warn_preinit_stderr = False
_logging.set_verbosity(_logging.ERROR)


def argParser():
    parser = argparse.ArgumentParser(description="Extract hand landmarks from a video.")
    parser.add_argument("--videoDataset_dir", type=str, default="/home/samuel/ml_projects/QUBPHEO/benchmark",
                        help="Path to the input video file.")
    parser.add_argument("--outputH5_dir", type=str, default="/home/samuel/ml_projects/QUBPHEO/av_landmark",
                        help="Path to the output HDF5 file.")
    parser.add_argument("--camera_view", type=str, default="CAM_AV",)
    parser.add_argument("--detection_confidence", type=float, default=0.3,
                        help="Detection confidence threshold.")
    parser.add_argument("--tracking_confidence", type=float, default=0.3,
                        help="Tracking confidence threshold.")
    return parser.parse_args()


def main():
    args = argParser()
    videoDataset_dir = args.videoDataset_dir
    outputH5_dir = args.outputH5_dir
    cam_view = args.camera_view
    detection_confidence = args.detection_confidence
    tracking_confidence = args.tracking_confidence
    subtaskPaths = sorted([os.path.join(videoDataset_dir, subtask) for subtask in os.listdir(videoDataset_dir) if os.path.isdir(os.path.join(videoDataset_dir, subtask))])

    subtask_list = []
    for subtask_path in tqdm(subtaskPaths, desc='Compiling Subtask Files'):
        subtask_dirPath = os.path.join(videoDataset_dir, subtask_path)
        subtask_files = sorted([os.path.join(subtask_dirPath, file) for file in os.listdir(subtask_dirPath) if file.lower().endswith('.mp4') and cam_view.lower() in file.lower()])
        subtask_list.extend(subtask_files)

    print(f"Total number of videos: {len(subtask_list)}")
    subtasksList = sorted(subtask_list)
    videoNameList = [os.path.basename(subtask) for subtask in subtasksList]
    data = []
    for line in videoNameList:
        match = re.match(r"(p\d+)-(.+)-(.+)-(.+)-(\d*\.?\d+)_([\d\.]+)\.mp4", line)
        if match:
            participant, camera, task, subtask, start, end = match.groups()
            data.append({
                "participant": participant,
                "camera": camera,
                "task": task,
                "subtask": subtask,
                "start_time": float(start),
                "end_time": float(end),
                "filename": line
            })
    df = pd.DataFrame(data)

    # Sort by participant, task, and start_time
    df_sorted = df.sort_values(by=["participant", "task", "start_time"])
    df_sorted.to_csv('subtasksList_byTask_time.csv', index=False, sep='\t')

    for subtask_video in tqdm(subtasksList, desc='Processing Videos'):
        video_name = os.path.basename(subtask_video)
        participant, camera, task, subtask, timestamp = video_name.split('-')
        H5_dir = os.path.join(outputH5_dir, subtask)
        os.makedirs(H5_dir, exist_ok=True)
        h5_fileName = os.path.join(H5_dir, f"{video_name[:-4]}.h5")
        print(f'H5 file name: {h5_fileName}')
        detector = handLandMarkExtractor(
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence
        )
        detector.extract(subtask_video, h5_fileName)


if __name__ == "__main__":
    main()
