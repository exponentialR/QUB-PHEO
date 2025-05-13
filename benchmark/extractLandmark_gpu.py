"""
extractLandmark_gpu.py

Extracts per-frame left/right hand landmarks from videos using MediaPipe’s GPU-accelerated HandLandmarker API.
Suppresses verbose C++ and TensorFlow logs, compiles a list of subtasks, processes each MP4 in a directory
(with timestamps and handedness), and writes results to HDF5 (and a summary CSV).

Author: Samuel Adebayo
Date:   13 May 2025
"""
__author__ = "Samuel Adebayo"

import os
os.environ['GLOG_minloglevel']  = '2'    # 0=INFO,1=WARNING,2=ERROR,3=FATAL
os.environ['GLOG_stderrthreshold'] = '2' # send only ERROR+ to stderr
os.environ['GLOG_logtostderr'] = '1'     # force logs to stderr (so stderrthreshold applies)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=INFO,1=WARNING,2=ERROR,3=FATAL

import warnings
warnings.filterwarnings(
    "ignore",
    message="All log messages before absl::InitializeLog\\(\\) is called are written to STDERR"
)
warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype\\(\\) is deprecated"
)
import absl.logging as _logging
_logging._warn_preinit_stderr = False
_logging.set_verbosity(_logging.ERROR)
import cv2
import time
import h5py
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import argparse
import re
from utils import _MuteStderr

os.environ['GLOG_minloglevel']   = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class handLandMarkExtractor:
    """
    GPU-accelerated hand landmark extractor using MediaPipe Tasks API.

    Coordinates:
        - x, y: Normalized [0,1] relative to image width/height.
        - z: Orthographic normalized depth relative to wrist origin.
    """
    def __init__(self,
                 model_path: str = 'models/hand_landmarker.task',
                 num_hands: int = 4,
                 tracking_confidence: float = 0.5):
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.GPU
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_tracking_confidence=tracking_confidence
        )
        with _MuteStderr():
            self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray, timestamp: float):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        with _MuteStderr():
            result = self.detector.detect_for_video(mp_image, timestamp_ms=int(timestamp * 1e3))

        left_frame = np.zeros((21, 3), dtype=np.float32)
        right_frame = np.zeros((21, 3), dtype=np.float32)

        for lmks, hand_cls in zip(result.hand_landmarks or [], result.handedness or []):
            coords = np.array([[lmk.x, lmk.y, lmk.z] for lmk in lmks], dtype=np.float32)

            label = hand_cls[0].category_name.lower()  # or .label if you prefer
            if label == 'left':
                left_frame = coords
            else:
                right_frame = coords

        return (left_frame, right_frame)


    def extract(self, input_vid: str, output_h5: str):
        """
        Extracts left/right hand landmarks per frame to HDF5.
        Saves 'left_landmarks' and 'right_landmarks' of shape [N,21,3]
        and 'timestamps' of shape [N].
        """
        if not os.path.isfile(input_vid):
            raise FileNotFoundError(f"Input video not found: {input_vid}")

        cap = cv2.VideoCapture(input_vid)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_vid}")

        timestamps, left_all, right_all = [], [], []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            left_lms, right_lms = self.process(frame, timestamp)
            left_all.append(left_lms)
            right_all.append(right_lms)
            timestamps.append(timestamp)

        cap.release()

        os.makedirs(os.path.dirname(output_h5) or '.', exist_ok=True)
        with h5py.File(output_h5, 'w') as hf:
            hf.create_dataset('left_landmarks',  data=np.stack(left_all,  axis=0))
            hf.create_dataset('right_landmarks', data=np.stack(right_all, axis=0))
            hf.create_dataset('timestamps',      data=np.array(timestamps, dtype=np.float32))

        print(f"Extracted {len(timestamps)} frames of hand landmarks to '{output_h5}'.")



def argParser():
    parser = argparse.ArgumentParser(description="Extract hand landmarks from a video.")
    parser.add_argument("--videoDataset_dir", type=str, default="/home/samuel/extended_storage/Datasets/QUB-PHEO/segmented",
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
    csv_path = 'subtasks_byTask_time.csv'

    if os.path.exists(csv_path):
        df_sorted.to_csv(csv_path,
                         index=False,
                         sep='\t',
                         mode='a',
                         header=False)
    else:
        df_sorted.to_csv(csv_path,
                         index=False,
                         sep='\t',
                         mode='w',
                         header=True)

    for subtask_video in tqdm(subtasksList, desc='Processing Videos'):
        video_name = os.path.basename(subtask_video)
        participant, camera, task, subtask, timestamp = video_name.split('-')
        H5_dir = os.path.join(outputH5_dir, subtask)
        os.makedirs(H5_dir, exist_ok=True)
        h5_fileName = os.path.join(H5_dir, f"{video_name[:-4]}.h5")
        print(f'H5 file name: {h5_fileName}')
        detector = handLandMarkExtractor(
            tracking_confidence=tracking_confidence
        )
        detector.extract(subtask_video, h5_fileName)


if __name__ == "__main__":
    main()