# CV_PA9

# Hand Gesture Duck Hunt Controller

This project implements a computer vision-based hand gesture controller for a Duck Hunt-style game using YOLO pose detection. Players can control the game using specific hand gestures instead of traditional input methods.

## Features

- Real-time hand gesture detection using YOLOv8 pose estimation
- Multiple gesture controls:
  - **Fist**: Freeze aim
  - **Open Hand**: Shoot
  - **Vertical Hand**: Reload
- Webcam integration for gesture tracking
- CUDA support for GPU acceleration (if available)

## Requirements
```
bash
ultralytics
opencv-python
torch
torchvision
torchaudio
pynput
numpy
```
## Installation

1. Clone the repository:
```
bash
git clone https://github.com/Cool-Coder174/CV_PA9.git
cd CV_PA9
```
2. Install the required packages:
```
bash
pip install -r requirements.txt
```
3. Ensure you have a webcam connected to your system

## Usage

1. Run the main script:
```
bash
python main.py
```
2. Position yourself in front of the webcam
3. Use the following gestures to control the game:
   - Make a fist to freeze aim
   - Open hand to shoot
   - Vertical hand (fingers aligned vertically) to reload
4. Press 'q' to quit the application

## How It Works

The system uses YOLOv8's pose estimation to detect hand keypoints in real-time through your webcam. These keypoints are analyzed to recognize specific gestures:

- **Fist Detection**: Measures the spread between fingertips
- **Shoot Detection**: Analyzes hand aspect ratio and finger spread
- **Reload Detection**: Checks vertical alignment of fingers

When gestures are recognized, the system simulates corresponding keyboard inputs:
- Shoot: Spacebar
- Reload: 'R' key

## Technical Details

- Model: YOLOv11n pose estimation
- Input: Webcam feed (480x640 resolution)
- Processing: Real-time frame analysis (30+ FPS with GPU)
- Output: Keyboard simulation for game control

## Troubleshooting

- Ensure good lighting conditions for better hand detection
- Keep your hand within the camera's field of view
- Maintain a reasonable distance from the camera (approximately arm's length)
- If using GPU, ensure CUDA is properly installed

## Contributing

Feel free to fork the project and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT License](LICENSE)

## Acknowledgments

- YOLOv8 for the pose estimation model
- Ultralytics for the YOLO implementation
- OpenCV for image processing capabilities


This README provides:
1. A clear project description
2. Installation instructions
3. Usage guidelines
4. Technical details
5. Troubleshooting tips
6. Contributing guidelines

Would you like me to modify any section or add additional information to the README?

