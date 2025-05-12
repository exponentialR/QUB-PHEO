'''
Script to detect and annotate hand landmarks in a video using MediaPipe, with optional CSV logging of detection metadata.

This tool reads an input video file, processes each frame to detect hand landmarks, draws the detected landmarks,
and outputs an annotated video. Optionally, a CSV file can be generated logging the timestamp and number of hands detected per frame.

Usage:
    python hand_landmark_detector.py \
        --input /path/to/input.mp4 \
        --output /path/to/output.mp4 \
        [--csv /path/to/log.csv] \
        [--detection_confidence 0.7] \
        [--tracking_confidence 0.7] \
        [--show] \
        [--no_fps]

Author: Samuel Adebayo
'''
__author__ = "Samuel Adebayo"


import cv2
import mediapipe as mp
import time
import csv
import argparse
import os

class HandLandmarkDetector:
    """
    Wrapper around MediaPipe Hands for detection and drawing of hand landmarks.
    """
    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5, max_num_hands=4, model_complexity=1):
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

    def detect(self, frame):
        """
        Detect hands in the given BGR frame. Returns list of hand landmarks or empty list.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results.multi_hand_landmarks if results.multi_hand_landmarks else []

    def draw_landmarks(self, frame, landmarks_list):
        """
        Draw detected hand landmarks on the frame in-place.
        """
        for hand_landmarks in landmarks_list:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Detect and plot hand landmarks in a video.")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file (.mp4, .avi, etc.)")
    parser.add_argument("--output", type=str, required=True, help="Path to save annotated output video.")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to save detection metadata CSV.")
    parser.add_argument("--detection_confidence", type=float, default=0.5, help="Min confidence for hand detection.")
    parser.add_argument("--tracking_confidence", type=float, default=0.5, help="Min confidence for hand tracking.")
    parser.add_argument("--show", action="store_true", help="Show annotated video in real time.")
    parser.add_argument("--no_fps", action="store_true", help="Disable FPS overlay.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Prepare input and output
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open input video {args.input}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # CSV logging
    csv_file = None
    csv_writer = None
    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        csv_file = open(args.csv, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp_s", "num_hands"])

    detector = HandLandmarkDetector(
        detection_confidence=args.detection_confidence,
        tracking_confidence=args.tracking_confidence
    )

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # seconds
        landmarks = detector.detect(frame)
        num_hands = len(landmarks)

        if num_hands > 0:
            detector.draw_landmarks(frame, landmarks)

        # Overlay FPS and hand count if enabled
        if not args.no_fps:
            curr_time = time.time()
            fps_text = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
            prev_time = curr_time
            status_text = f"FPS: {fps_text:.1f} | Hands: {num_hands}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame)

        if csv_writer:
            csv_writer.writerow([f"{timestamp:.3f}", num_hands])

        if args.show:
            cv2.imshow("Hand Landmarks", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()
    if csv_file:
        csv_file.close()

    print(f"Processing complete. Output saved to {args.output}")
    if args.csv:
        print(f"Metadata logged to {args.csv}")


if __name__ == "__main__":
    main()
