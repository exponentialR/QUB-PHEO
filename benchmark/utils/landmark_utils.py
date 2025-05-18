import mediapipe as mp
import argparse
import os
import h5py
import numpy as np
import cv2


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

if __name__ == '__main__':
    """=======================TESTING UTILS OF EXTRACTING HAND LANDMARKS ========================================"""
    parser = argparse.ArgumentParser(description="Extract hand landmarks from a video.")
    parser.add_argument("--input", type=str, default="media/p04-CAM_AV-STAIRWAY_MS-CS-13.140981640849496_16.403350468435704.mp4",
                        help="Path to the input video file.")
    parser.add_argument("--output", type=str, default="p04-CAM_AV-STAIRWAY_MS-CS-13.140981640849496_16.403350468435704.h5",
                        help="Path to the output HDF5 file.")
    parser.add_argument("--detection_confidence", type=float, default=0.5,
                        help="Detection confidence threshold.")
    parser.add_argument("--tracking_confidence", type=float, default=0.5,
                        help="Tracking confidence threshold.")
    args = parser.parse_args()


    detector = handLandMarkExtractor(
        detection_confidence=args.detection_confidence,
        tracking_confidence=args.tracking_confidence
    )
    detector.extract(args.input, args.output)