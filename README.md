# MediaPipe-and-Leap_Motion-integration-Marker-TEST-

# Leap Motion + MediaPipe Hand Tracking with Stickers

This project combines Google MediaPipe and Leap Motion for precise fingertip tracking in a virtual keyboard application, enhanced by sticker markers as suggested by Dr. M. It’s designed for an ITRC research task in a VR context.

## Files
- **`leap_hand_tracker.py`**: Main script for MediaPipe + Leap Motion with sticker fusion.
- **`virtual_keyboard.py`**: Virtual QWERTY keyboard layout and detection.
- **`utils.py`**: Sticker detection and Kalman filter utilities.
- **`train_sticker_model.py`**: Deep learning model for sticker detection.
- **`requirements.txt`**: Dependencies.

## Features
- Hand tracking with MediaPipe (RGB) and Leap Motion (depth).
- Sticker-based fingertip enhancement per Dr. M.’s advice.
- Optional deep learning model for robust sticker detection.
- Virtual keyboard with key press detection.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/leap-hand-tracking-stickers.git
   cd leap-hand-tracking-stickers

   2. Set Up Conda Environment:
```bash
conda create -n mp python=3.9
conda activate mp
pip install -r requirements.txt
```
3. Leap Motion Setup:
- Install Leap Motion SDK from Ultraleap.

- Start the Leap service (leapd).

## Usage
1. Attach Stickers: Place bright-colored stickers (e.g., green) on fingertips.

2. Run Tracker:
```bash
python leap_hand_tracker.py
```
- Requires a camera and Leap Motion connected.

- Displays tracking with keyboard overlay.

3. Train Model (Optional):
- Collect dataset in sticker_dataset/ (images + .txt labels).

- Run:
```bash
python train_sticker_model.py
```
## Customization
- Sticker Color: Adjust HSV in detect_stickers (utils.py).

- Keyboard Layout: Edit virtual_keyboard.py.

- Fusion Weights: Tweak fuse_data weights for MediaPipe/Leap/stickers.

## Troubleshooting
- Leap Motion: Ensure leapd is running (ps aux | grep leapd).

- Camera: Check permissions (sudo chmod 666 /dev/video0).

- Accuracy: Train the deep learning model or adjust Kalman params.

## License
- MIT License. See LICENSE for details.

```
### Implementation Steps
#### 1. Setup Hardware
- Attach Leap Motion to your VR headset (e.g., Quest 2 via mount).
- Place stickers on fingertips (test visibility with your camera).
#### 2. Test Tracking
- Run `leap_hand_tracker.py`:
  ```bash
  python leap_hand_tracker.py
```
- Verify Leap Motion, MediaPipe, and sticker detection work together.

3. Deep Learning Enhancement
- Collect a dataset with Leap Motion-aligned camera images.

- Train with train_sticker_model.py and integrate into utils.py.

4. VR Integration (Optional)
- Replace get_camera_feed with your VR camera API if using in VR.

- Add VR rendering as needed.

### How It Addresses The Task
- Leap Motion: Provides accurate depth (z) for fingertips, overcoming MediaPipe’s RGB-only limitations.

- Stickers: Boosts x and y precision per your supervisor’s suggestion.

