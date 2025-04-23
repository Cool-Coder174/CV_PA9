"""
import cv2
from ultralytics import YOLO
import torch
from multiprocessing import freeze_support

def run_detection():
    # Load the specific model from train7
    model = YOLO('runs/pose/train8/weights/best.pt')
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Run inference with pose estimation
        results = model.predict(
            source=frame,
            conf=0.5,    # Confidence threshold
            show=False,
            device=device
        )
        
        # Draw results on frame with keypoints
        annotated_frame = results[0].plot()
        
        # Display the frame
        cv2.imshow("YOLOv8 Pose Detection", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    freeze_support()
    run_detection()
"""