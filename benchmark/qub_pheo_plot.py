import os
import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

# ——— Setup ———
trained_model = 'weights/Lego_YOLO.pt'
video_path = '/home/samuel/ml_projects/QUBPHEO/benchmark/videos/segmented/BHO/p08-CAM_AV-BIAH_RB-BHO-20.414318230243815_22.799820194594822.mp4'
# video_path = "sample_data/videos/p31-CAM_AV-BRIDGE_BV-RSA-13.310799666859792_15.203448450173813.mp4"
output_dir = 'output_frames'
os.makedirs(output_dir, exist_ok=True)

# Load YOLOv8
model = YOLO(trained_model)

# Initialise MediaPipe Hands
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

frame_number = 0
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    centre  = (w//2, h//2)
    std_dev = (w*0.1, h*0.1)

    # 1) YOLO inference
    # flip frame for correct orientation
    # frame = cv2.flip(frame, 1)
    yolo_res = model(frame, verbose=False)[0]

    # 2) Start from a clean copy
    annotated = frame.copy()

    # 3) Draw your own boxes + small text
    for box in yolo_res.boxes:
        # get absolute coords
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls            = int(box.cls)
        conf           = box.conf[0]

        # rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # label = “name xx%”
        label = f"{model.names[cls]} {conf:.2f}"

        # smaller fontScale and thinner thickness
        font      = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.6
        thickness = 1

        # compute text size so we can draw a filled bg if you like
        (w_text, h_text), baseline = cv2.getTextSize(label, font, fontScale, thickness)
        # optional: draw background rectangle
        cv2.rectangle(
            annotated,
            (x1, y1 - h_text - baseline),
            (x1 + w_text, y1),
            (255, 0, 0),
            cv2.FILLED
        )
        # now put the text in white
        cv2.putText(
            annotated,
            label,
            (x1, y1 - baseline),
            font,
            fontScale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA
        )

    # 4) MediaPipe hands (unchanged)
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_out = hands.process(rgb)
    if hands_out.multi_hand_landmarks:
        for lm in hands_out.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated, lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
            )

    # 5) Random gaze dot
    gx = int(np.random.normal(centre[0], std_dev[0])); gx = np.clip(gx, 0, w-1)
    gy = int(np.random.normal(centre[1], std_dev[1])); gy = np.clip(gy, 0, h-1)

    cv2.circle(annotated, (gx, gy), 25, (0, 255, 0), -1)
    cv2.putText(annotated, 'gaze ray',
                (gx + 12, gy - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (120, 0, 255), 2)

    # 6) Show & save
    cv2.imshow('Annotations', annotated)
    cv2.imwrite(f"{output_dir}/frame_{frame_number:04d}.jpg", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_number += 1

cap.release()
cv2.destroyAllWindows()
hands.close()
