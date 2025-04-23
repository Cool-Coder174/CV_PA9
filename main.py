import time
import cv2
import numpy as np
import torch
from pynput.keyboard import Controller, Key
from ultralytics import YOLO
from multiprocessing import freeze_support

class GestureDetector:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)
        self.cap = cv2.VideoCapture(0)
        self.keyboard = Controller()
        self.aim_frozen = False
        self.shoot_triggered = False
        self.reload_timer_running = False
        self.reload_start_time = 0
        self.reload_triggered = False

    def detect_gesture(self, keypoints):
        if keypoints is None or len(keypoints) == 0:
            return "none"
            
        try:
            palm_base = keypoints[0]
            fingertip_points = keypoints[1:5]

            if len(keypoints) < 5:
                return "none"

            spread = np.linalg.norm(fingertip_points[0][:2] - fingertip_points[-1][:2])
            palm_width = np.linalg.norm(keypoints[1][:2] - keypoints[2][:2])
            
            denominator = np.max(keypoints[:,0]) - np.min(keypoints[:,0])
            if denominator < 1e-5:
                aspect_ratio = 0
            else:
                aspect_ratio = (np.max(keypoints[:,1]) - np.min(keypoints[:,1])) / denominator

            if spread < palm_width * 0.7:
                return "fist"
            elif spread > palm_width * 1.4 and 1.2 < aspect_ratio < 1.8:
                return "shoot"
            elif np.var(keypoints[1:5,0]) < 10:
                return "reload"

            return "none"
        except (IndexError, ValueError):
            return "none"

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            results = self.model.predict(frame, device=self.device, conf=0.5, show=False)
            gesture = "none"

            if results and len(results) > 0 and results[0].keypoints is not None:
                keypoints_data = results[0].keypoints.data
                if keypoints_data is not None and len(keypoints_data) > 0:
                    keypoints = keypoints_data.cpu().numpy()[0]
                    gesture = self.detect_gesture(keypoints)

                    if gesture == "fist":
                        self.aim_frozen = True
                    elif gesture == "shoot":
                        self.aim_frozen = False
                        if not self.shoot_triggered:
                            self.keyboard.press(Key.space)
                            self.keyboard.release(Key.space)
                            self.shoot_triggered = True
                    else:
                        self.shoot_triggered = False

                    if gesture == "reload":
                        if not self.reload_timer_running:
                            self.reload_start_time = time.time()
                            self.reload_timer_running = True
                        elif time.time() - self.reload_start_time > 1.0 and not self.reload_triggered:
                            self.keyboard.press('r')
                            self.keyboard.release('r')
                            self.reload_triggered = True
                    else:
                        self.reload_timer_running = False
                        self.reload_triggered = False

            # Draw the results on the frame
            annotated_frame = results[0].plot()
            
            # Add gesture text to the frame
            cv2.putText(annotated_frame, f"Gesture: {gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Hand Gesture Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    freeze_support()
    model_path = 'runs/pose/train8/weights/best.pt'
    detector = GestureDetector(model_path)
    detector.run()