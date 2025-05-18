from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
# results = model.train(data='config.yaml', epochs=100, imgsz=640)
trained_model = 'weights/Lego_YOLO.pt'



import cv2
from ultralytics import YOLO

model = YOLO(trained_model)

# video_path = "sample_data/videos/p31-CAM_AV-BRIDGE_BV-RSA-13.310799666859792_15.203448450173813.mp4"
video_path = '/home/samuel/ml_projects/QUBPHEO/benchmark/videos/segmented/RBNS/p01-CAM_AV-BIAH_BV-RBNS-14.68566331487069_21.838886652801722.mp4'
cap = cv2.VideoCapture(video_path)


frame_number = 0
while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame, verbose=False)

        annotated_frame = results[0].plot()
        for box in results[0].boxes:
            x, y, w, h = box.xywhn[0].tolist()
            class_idx = int(box.cls)
            # if class_idx == 10:
                # print the class name
                # print(f" Frame Number: {frame_number} | Bounding Box: {x, y, w, h} | class ID: {class_idx} | Class Name: {model.names[class_idx]}")
            print(f" Frame Number: {frame_number} | Bounding Box: {x, y, w, h} | class ID: {class_idx} | Class Name: {model.names[class_idx]}")

        frame_number += 1

        cv2.imshow("YOLOv8 Inference", annotated_frame)


        if cv2.waitKey(200) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()