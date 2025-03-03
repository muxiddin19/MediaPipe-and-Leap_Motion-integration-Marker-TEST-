# leap_hand_tracker.py
import cv2
import mediapipe as mp
import leap
import numpy as np
from virtual_keyboard import draw_keyboard, detect_key_press, keyboard_layout
from utils import detect_stickers, initialize_kalman

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Leap Motion setup
leap_controller = leap.Controller()

def get_camera_feed():
    """Initialize camera feed (webcam or VR passthrough for MediaPipe)."""
    cap = cv2.VideoCapture(0)  # Replace with VR camera if needed
    if not cap.isOpened():
        print("Failed to open camera.")
        exit(1)
    return cap

def get_leap_fingertips():
    """Extract fingertip positions from Leap Motion."""
    frame = leap_controller.frame()
    if not frame.hands:
        return None
    fingertips = {}
    for hand in frame.hands:
        hand_type = "Left" if hand.is_left else "Right"
        tips = []
        for finger in hand.fingers:
            tip = finger.bone(leap.Bone.TYPE_DISTAL).next_joint
            tips.append((tip.x, tip.y, tip.z))  # In mm
        fingertips[hand_type] = tips
    return fingertips

def fuse_data(mp_landmarks, leap_tips, sticker_positions, image_width, image_height, kfs):
    """Fuse MediaPipe, Leap Motion, and sticker data with Kalman filtering."""
    fused = mp_landmarks
    if not leap_tips and not sticker_positions:
        return fused
    
    leap_fingers = list(leap_tips.values())[0] if leap_tips else [None] * 5
    for i, kf in enumerate(kfs):
        tip_idx = mp_hands.HandLandmark.INDEX_FINGER_TIP + i * 4
        mp_tip = mp_landmarks.landmark[tip_idx]
        
        # Default to MediaPipe
        x, y, z = mp_tip.x, mp_tip.y, mp_tip.z
        
        # Sticker enhancement (x, y)
        if i < len(sticker_positions):
            sx, sy = sticker_positions[i]
            x = (sx / image_width) * 0.8 + mp_tip.x * 0.2
            y = (sy / image_height) * 0.8 + mp_tip.y * 0.2
        
        # Leap Motion depth (z)
        if leap_fingers[i]:
            leap_z = leap_fingers[i][2] / 1000  # Convert mm to normalized
            z = leap_z * 0.7 + mp_tip.z * 0.3  # Favor Leap for depth
        
        # Kalman update
        kf.predict()
        kf.update(np.array([x, y, z]))
        fused.landmark[tip_idx].x, fused.landmark[tip_idx].y, fused.landmark[tip_idx].z = kf.x
    
    return fused

def main():
    """Main loop for Leap Motion + MediaPipe hand tracking."""
    cap = get_camera_feed()
    kfs = [initialize_kalman() for _ in range(5)]  # Kalman filters for fingertips

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read camera frame.")
                break

            # Process with MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw virtual keyboard
            image = draw_keyboard(image)

            # Get Leap Motion and sticker data
            leap_data = get_leap_fingertips()
            sticker_positions = detect_stickers(image)

            # Process hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    fused_landmarks = fuse_data(
                        hand_landmarks, leap_data, sticker_positions, 
                        image.shape[1], image.shape[0], kfs
                    )
                    mp_drawing.draw_landmarks(
                        image, fused_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    pressed_keys = detect_key_press(fused_landmarks, image, keyboard_layout)
                    if pressed_keys:
                        print(f"Pressed: {pressed_keys}")

            # Display (replace with VR rendering if needed)
            cv2.imshow('Leap Hand Tracking', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
