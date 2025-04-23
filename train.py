"""
import torch
from ultralytics import YOLO
import cv2
import os
from multiprocessing import freeze_support

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the pretrained YOLO model
    model = YOLO("yolo11n-pose.pt").to(device)  # Ensure the model is moved to the GPU

    # Train the model with valid parameters
    result = model.train(
        data="hand-keypoints.yaml",  # Dataset configuration
        epochs=200,                 # Number of epochs
        imgsz=640,                  # Image size
        batch=16,                   # Increased batch size
        workers=4                   # Number of data loader workers
    )

    # Open a video stream (0 for the default webcam)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform prediction on the current frame
        results = model.predict(source=frame, save=False, show=False, device=device)

        # Draw bounding boxes and labels on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Hand Gesture Detection", annotated_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    freeze_support()  # Added this line for Windows support
    main()
    """
