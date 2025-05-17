import os
import subprocess
from tqdm import tqdm
import cv2
import mediapipe as mp
import argparse
import os
import h5py
import numpy as np
import csv
import datetime
import cv2
from ultralytics import YOLO

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
    with open('bad_segments.txt', 'w') as f:
        for bad_file in bad_segments:
            f.write(bad_file + '\n')
    print(f"Bad segments saved to bad_segments.txt")


def delete_dataset_hf5(hdf5_file, dataset_name):
    """
    Deletes a dataset from an HDF5 file if it exists.
    :param hdf5_file: Path to the HDF5 file.
    :param dataset_name: Name of the dataset to delete.
    """
    with h5py.File(hdf5_file, 'a') as f:
        if dataset_name in f:
            del f[dataset_name]
            print(f"✅  Deleted dataset '{dataset_name}' from '{hdf5_file}'.")
            return True
        else:
            print(f"❌ Dataset '{dataset_name}' not found in '{hdf5_file}'.")
            return False


def extract_segment(baseVideoDirectory, controlVidList, outputDirectory, view_name, retain_audio=False):
    """
    This function extracts the segments from the base video directory and saves them to the output directory.
    :param baseVideoDirectory:
    :param controlVidList:
    :param outputDirectory:
    :param view_name:
    :return:
    """
    video_counter = 0
    for vid_n in tqdm(controlVidList, desc='Video Segments'):
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
        main_base_video = os.path.join(baseVideoDirectory,
                                       part_id,
                                       f"{cam_view}_P",
                                       f"{task_name}.mp4")
        output_seg_folder = os.path.join(outputDirectory, subtask_name)
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


def extract_bbox_from_video(video_path, model, h5_path):
    """
    Extracts bounding boxes from a video using YOLOv8.
    :param video_path: Path to the input video file.
    :return: List of bounding boxes for each frame.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    """
    TODO: Handle the paths for h5 correctly
    """
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


class _MuteStderr:
    def __enter__(self):
        self.orig_stderr_fd = os.dup(2)
        os.close(2)
        os.open(os.devnull, os.O_RDWR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.close(2)
        os.dup2(self.orig_stderr_fd, 2)
        os.close(self.orig_stderr_fd)


class handLandMarkExtractor:
    """
    Wrapper around MediaPipe Hands for detection and drawing of hand landmarks.
    coordinates:
        - x, y: normalized coordinates of the landmark in the range [0, 1] relative to the image width and height
        - z: depth of the landmark in the range [-1, 1] relative to the palm. It is the Orthographic approaximation to wrist
            wrist origin; smaller z values
           indicate landmarks closer to the camera plane. Not true metric depth without calibration.
    """
    def __init__ (self, detection_confidence=0.5, tracking_confidence=0.5, max_num_hands=4, model_complexity=1):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process(self, frame):
        """
        Detect hands in the given BGR frame. Returns list of hand landmarks or empty list.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def extract(self, input_vid, output_h5):
        """
           Extracts left/right hand landmarks per frame into HDF5.
           Saves datasets 'left_landmarks' and 'right_landmarks' of shape [N,21,3]
           and 'timestamps' of shape [N].
       """
        if not os.path.exists(input_vid) or not os.path.isfile(input_vid):
            print(f'Error - input video {input_vid} does not exist or is not a file.')

        cap = cv2.VideoCapture(input_vid)
        if not cap.isOpened():
            print(f'Error - could not open video {input_vid}.')
            return

        timestamps = []
        left_all, right_all = [], []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # convert to seconds
            results = self.process(frame)
            landmarks_list = results.multi_hand_landmarks if results.multi_hand_landmarks else []

            handedness_list = [h.classification[0].label for h in (results.multi_handedness or [])]
            left_frame = np.zeros((21, 3), dtype=np.float32)
            right_frame = np.zeros((21, 3), dtype=np.float32)

            for idx, hand_landmarks in enumerate(landmarks_list):
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                label = handedness_list[idx].lower()
                if label == 'left':
                    left_frame = coords
                else:
                    right_frame = coords

            left_all.append(left_frame)
            right_all.append(right_frame)
            timestamps.append(timestamp)

        cap.release()


        with h5py.File(output_h5, 'w') as h5f:
            h5f.create_dataset('left_landmarks', data=np.array(left_all, dtype=np.float32))
            h5f.create_dataset('right_landmarks', data=np.array(right_all, dtype=np.float32))
            h5f.create_dataset('timestamps', data=np.array(timestamps, dtype=np.float32))
        print(f"Extracted {len(timestamps)} frames of hand landmarks to '{output_h5}'.")


def mergeHdf5Datasets(hdf5_1, hdf5_2,
                      datasets_to_merge=None,
                      rename_map=None,
                      log_path='hdf5_merge_log.csv'):
    """
    Merge selected datasets from one HDF5 file into another, with optional renaming,
    log any shape mismatches, and return a success flag.

    Parameters
    ----------
    hdf5_1 : str
        Path to the source HDF5 file (read-only).
    hdf5_2 : str
        Path to the destination HDF5 file (append mode).
    datasets_to_merge : list of str, optional
        List of dataset names (or full paths) in `hdf5_1` to copy.
        Defaults to ['bounding_boxes', 'normalized_gaze'].
    rename_map : dict, optional
        Mapping from original dataset names to new names in `hdf5_2`.
        E.g. {'bounding_boxes': 'bboxes', 'normalized_gaze': 'norm_gaze'}.
    log_path : str, optional
        CSV file path where shape‐mismatch events are appended.
        Defaults to 'hdf5_merge_log.csv'.

    Returns
    -------
    bool
        True if *all* datasets in `datasets_to_merge` were found and merged
        (with no shape mismatches); False if *any* were missing or mismatched.
    """
    # defaults
    if datasets_to_merge is None:
        datasets_to_merge = ['bounding_boxes', 'normalized_gaze']
    if rename_map is None:
        rename_map = {
            'bounding_boxes': 'bboxes',
            'normalized_gaze': 'norm_gaze'
        }

    # ensure log header exists once
    header = ['timestamp', 'source_file', 'src_dataset', 'src_shape',
              'dest_file', 'dest_dataset', 'dest_shape']
    if not os.path.isfile(log_path):
        with open(log_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

    all_success = True

    with h5py.File(hdf5_1, 'r') as f1, h5py.File(hdf5_2, 'a') as f2:
        for src_name in datasets_to_merge:
            if src_name not in f1:
                print(f"Warning: '{src_name}' not found in {hdf5_1}. Skipping.")
                all_success = False
                continue

            dest_name = rename_map.get(src_name, src_name)

            # if target exists, compare shapes
            if dest_name in f2:
                src_shape = f1[src_name].shape
                dest_shape = f2[dest_name].shape

                if src_shape != dest_shape:
                    print(f"Shape mismatch for {src_name} → {dest_name}: "
                          f"source {src_shape} vs destination {dest_shape}. Skipping.")
                    # log mismatch
                    with open(log_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([
                            datetime.datetime.now().isoformat(),
                            hdf5_1, src_name, src_shape,
                            hdf5_2, dest_name, dest_shape
                        ])
                    all_success = False
                    continue
                else:
                    # safe to overwrite
                    del f2[dest_name]

            # perform the copy
            f1.copy(src_name, f2, name=dest_name)

    # if all_success:
    #     print(f"✅ All datasets merged successfully into '{hdf5_2}'.")
    # else:
    #     print(f"❌ Some datasets were skipped. Check logs for details.")

    return all_success


if __name__ == "__main__":
    """=======================TESTING UTILS OF EXTRACTING BBOXES ========================================"""
    h5_path = 'sample_data/bbox_preprocess/p01-CAM_AV-BIAH_RB-BHO-0.0007253558395636801_1.4222339664428545.h5'
    video_path = 'sample_data/bbox_preprocess/p01-CAM_AV-BIAH_RB-BHO-0.0007253558395636801_1.4222339664428545.mp4'
    model_path = 'weights/Lego_YOLO.pt'
    yolo_model = YOLO(model_path)
    extract_bbox_from_video(video_path, yolo_model, h5_path)

    """=======================TESTING UTILS OF DELETING DATASET IN HDF5 ========================================"""
    # hd5_file = 'sample_data/delete_dataset/p01-CAM_AV-BIAH_RB-BHO-25.420644039568998_26.674916343042494.h5'
    # dataset_name = 'bboxes'
    # delete_dataset_hf5(hd5_file, dataset_name)


    """=======================TESTING UTILS OF MERGING DATASET ========================================"""
    # h5_1 = '/home/samuel/ml_projects/QUBPHEO/benchmark/test_run/p1/BHO/p01-CAM_AV-BIAH_RB-BHO-0.0007253558395636801_1.4222339664428545.h5'
    # h5_2 = '/home/samuel/ml_projects/QUBPHEO/benchmark/test_run/p2/BHO/p01-CAM_AV-BIAH_RB-BHO-0.0007253558395636801_1.4222339664428545.h5'
    #
    # success = mergeHdf5Datasets(
    #     h5_1,
    #     h5_2,
    #     datasets_to_merge=['bounding_boxes', 'normalized_gaze'],
    #     rename_map={'bounding_boxes': 'bboxes', 'normalized_gaze': 'norm_gaze'}
    # )
    # if success:
    #     print("Merge completed without issues.")
    # else:
    #     print("Merge completed with issues. Check logs for details.")

    """=======================TESTING UTILS OF EXTRACTING HAND LANDMARKS ========================================"""
    # parser = argparse.ArgumentParser(description="Extract hand landmarks from a video.")
    # parser.add_argument("--input", type=str, default="media/p04-CAM_AV-STAIRWAY_MS-CS-13.140981640849496_16.403350468435704.mp4",
    #                     help="Path to the input video file.")
    # parser.add_argument("--output", type=str, default="p04-CAM_AV-STAIRWAY_MS-CS-13.140981640849496_16.403350468435704.h5",
    #                     help="Path to the output HDF5 file.")
    # parser.add_argument("--detection_confidence", type=float, default=0.5,
    #                     help="Detection confidence threshold.")
    # parser.add_argument("--tracking_confidence", type=float, default=0.5,
    #                     help="Tracking confidence threshold.")
    # args = parser.parse_args()
    #
    #
    # detector = handLandMarkExtractor(
    #     detection_confidence=args.detection_confidence,
    #     tracking_confidence=args.tracking_confidence
    # )
    # detector.extract(args.input, args.output)

"""=======================TESTING UTILS OF GETTING BAD SEGMENTS ========================================"""

# if __name__ == '__main__':
#     folder_directory = '/home/samueladebayo/Documents/PhD/QUBPHEO/corrupted-segment'
#     get_badSegment(folder_directory)


