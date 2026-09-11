# ==============================================================================
# PIP INSTALLATION COMMAND:
# pip install opencv-python mediapipe pyautogui numpy
# ==============================================================================

import sys
import os
import time
import urllib.request
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Force UTF-8 stdout if possible on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ------------------------------------------------------------------------------
# 1. SYSTEM & SAFETY CONFIGURATION
# ------------------------------------------------------------------------------
# Disable PyAutoGUI failsafe so cursor moving to corners does not raise FailSafeException.
# NOTE: This disables the automatic safety stop (moving mouse to screen corner).
pyautogui.FAILSAFE = False

# Set PyAutoGUI internal pause to 0 for instantaneous, unthrottled cursor movement per frame.
pyautogui.PAUSE = 0

# Retrieve primary screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# ------------------------------------------------------------------------------
# 2. TRACKING & GESTURE TUNING PARAMETERS
# ------------------------------------------------------------------------------
# Exponential Moving Average (EMA) smoothing factor (0.0 < alpha <= 1.0)
# Higher values (e.g. 0.6 - 0.8) -> Faster response, less smoothing
# Lower values (e.g. 0.2 - 0.4) -> Ultra smooth, slight inertia
SMOOTHING_ALPHA = 0.35

# Active Region / Sensitivity Margin (Normalized [0, 1] range)
# Defines an active bounding box inside the camera frame to prevent neck strain.
ACTIVE_X_MIN, ACTIVE_X_MAX = 0.25, 0.75
ACTIVE_Y_MIN, ACTIVE_Y_MAX = 0.25, 0.75

# --- MOUTH-OPEN CLICK GESTURE CONFIGURATION ---
# Mouth-open threshold: Ratio of (vertical inner lip distance) / (inter-ocular eye distance).
# When this ratio exceeds the threshold, a mouse click is triggered.
# Typical values: Closed mouth ~ 0.05 - 0.15; Open mouth ~ 0.30 - 0.50+.
MOUTH_OPEN_THRESHOLD = 0.35

# Minimum cooldown time (in seconds) between successive clicks.
# Prevents repeated rapid clicks while the mouth remains continuously open.
CLICK_COOLDOWN_SECONDS = 1.0

# Visual click flash duration in seconds on the debug screen
CLICK_FLASH_DURATION = 0.35

# ------------------------------------------------------------------------------
# 3. STATE MEMORY VARIABLES
# ------------------------------------------------------------------------------
smoothed_x = SCREEN_WIDTH / 2
smoothed_y = SCREEN_HEIGHT / 2
last_known_x = smoothed_x
last_known_y = smoothed_y

# Click gesture state
last_click_time = 0.0
click_flash_until = 0.0

# ------------------------------------------------------------------------------
# 4. INITIALIZE MEDIAPIPE FACE MESH DETECTOR
# ------------------------------------------------------------------------------
print("=" * 60)
print("  [*] NOSE-CONTROLLED CURSOR TRACKER + MOUTH-OPEN CLICK")
print("=" * 60)
print(f"[*] Screen Resolution Detected: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
print(f"[*] Mouth-Open Click Threshold: {MOUTH_OPEN_THRESHOLD} (Cooldown: {CLICK_COOLDOWN_SECONDS}s)")
print("[*] Initializing MediaPipe Face Mesh...")

class FaceMeshTracker:
    def __init__(self):
        self.use_tasks_api = not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'face_mesh')
        
        if self.use_tasks_api:
            model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
            if not os.path.exists(model_path):
                print("[*] Downloading MediaPipe Face Landmarker model...")
                model_url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
                urllib.request.urlretrieve(model_url, model_path)
                print("[*] Model downloaded successfully.")

            from mediapipe.tasks.python import vision, BaseOptions
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
        else:
            self.detector = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        print("[+] MediaPipe Face Mesh initialized successfully.")

    def process(self, rgb_frame):
        """
        Extracts:
        - Nose Tip (Landmark 1)
        - Upper Lip Inner Center (Landmark 13)
        - Lower Lip Inner Center (Landmark 14)
        - Left Eye Outer Corner (Landmark 33)
        - Right Eye Outer Corner (Landmark 263)
        """
        if self.use_tasks_api:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.detector.detect(mp_img)
            if result.face_landmarks and len(result.face_landmarks) > 0:
                lms = result.face_landmarks[0]
                return True, {
                    'nose': (lms[1].x, lms[1].y),
                    'upper_lip': (lms[13].x, lms[13].y),
                    'lower_lip': (lms[14].x, lms[14].y),
                    'left_eye': (lms[33].x, lms[33].y),
                    'right_eye': (lms[263].x, lms[263].y),
                }
            return False, None
        else:
            result = self.detector.process(rgb_frame)
            if result.multi_face_landmarks and len(result.multi_face_landmarks) > 0:
                lms = result.multi_face_landmarks[0].landmark
                return True, {
                    'nose': (lms[1].x, lms[1].y),
                    'upper_lip': (lms[13].x, lms[13].y),
                    'lower_lip': (lms[14].x, lms[14].y),
                    'left_eye': (lms[33].x, lms[33].y),
                    'right_eye': (lms[263].x, lms[263].y),
                }
            return False, None

    def close(self):
        if hasattr(self.detector, 'close'):
            self.detector.close()

tracker = FaceMeshTracker()

# ------------------------------------------------------------------------------
# 5. INITIALIZE WEBCAM
# ------------------------------------------------------------------------------
print("[*] Connecting to webcam...")

def open_camera():
    backends = [
        (0, cv2.CAP_DSHOW),
        (0, cv2.CAP_ANY),
        (1, cv2.CAP_DSHOW),
        (1, cv2.CAP_ANY)
    ]
    for index, backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"[+] Webcam detected and opened successfully on camera index {index}!")
                return cap
            cap.release()
    return None

cap = open_camera()

if cap is None:
    print("[!] ERROR: Could not access any webcam.")
    print("    Please ensure your camera is connected and not blocked by another app/permissions.")
    sys.exit(1)

# Set up named window
window_name = "Nose Cursor - Tracking Feed"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 480)

print("[*] Tracking Target: Nose Tip (Landmark 1)")
print("[*] Gesture Action: Open mouth to trigger Left Click")
print("[*] Live cursor tracking active.")
print("[*] Press 'q' inside the webcam window to exit.")
print("=" * 60)

# ------------------------------------------------------------------------------
# 6. HELPER MATH FUNCTIONS
# ------------------------------------------------------------------------------
def euclidean_distance(pt1, pt2):
    """Calculates 2D Euclidean distance between two normalized coordinate points."""
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# ------------------------------------------------------------------------------
# 7. MAIN CONTINUOUS REAL-TIME TRACKING & GESTURE LOOP
# ------------------------------------------------------------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("[!] Warning: Empty or unreadable frame from webcam.")
        continue

    current_time = time.time()

    # STEP A: Flip frame horizontally for intuitive mirror-like navigation
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    # STEP B: Convert BGR to RGB for MediaPipe inference
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_found, landmarks = tracker.process(rgb_frame)

    # STEP C: Face Detection Gate
    if face_found:
        nose_norm = landmarks['nose']
        norm_x, norm_y = nose_norm

        # STEP D: Coordinate Mapping using np.interp
        # Map normalized coordinates from active region to screen pixels
        raw_screen_x = np.interp(norm_x, [ACTIVE_X_MIN, ACTIVE_X_MAX], [0, SCREEN_WIDTH])
        raw_screen_y = np.interp(norm_y, [ACTIVE_Y_MIN, ACTIVE_Y_MAX], [0, SCREEN_HEIGHT])

        # Clamp screen coordinates within screen dimensions
        raw_screen_x = np.clip(raw_screen_x, 0, SCREEN_WIDTH - 1)
        raw_screen_y = np.clip(raw_screen_y, 0, SCREEN_HEIGHT - 1)

        # STEP E: Exponential Moving Average (EMA) Smoothing
        smoothed_x = (smoothed_x * (1.0 - SMOOTHING_ALPHA)) + (raw_screen_x * SMOOTHING_ALPHA)
        smoothed_y = (smoothed_y * (1.0 - SMOOTHING_ALPHA)) + (raw_screen_y * SMOOTHING_ALPHA)

        # Save last known valid position
        last_known_x = smoothed_x
        last_known_y = smoothed_y

        # STEP F: Real-Time System Cursor Movement
        pyautogui.moveTo(int(smoothed_x), int(smoothed_y))

        # ----------------------------------------------------------------------
        # STEP G: MOUTH-OPEN CLICK GESTURE LOGIC
        # ----------------------------------------------------------------------
        # 1. Calculate vertical distance between upper inner lip (13) & lower inner lip (14)
        lip_vertical_dist = euclidean_distance(landmarks['upper_lip'], landmarks['lower_lip'])

        # 2. Calculate stable face scale reference: inter-ocular distance between outer eye corners (33 & 263)
        # This makes the mouth-open ratio scale-invariant (works whether you are close or far from webcam).
        eye_reference_dist = euclidean_distance(landmarks['left_eye'], landmarks['right_eye'])

        # 3. Normalized Mouth-Open Ratio:
        mouth_open_ratio = lip_vertical_dist / max(eye_reference_dist, 1e-6)

        # 4. Check gesture trigger condition:
        # Mouth open exceeds threshold AND cooldown period has elapsed
        time_since_last_click = current_time - last_click_time
        can_click = time_since_last_click >= CLICK_COOLDOWN_SECONDS

        is_clicking_now = False
        if mouth_open_ratio >= MOUTH_OPEN_THRESHOLD and can_click:
            # Trigger a single click at the current cursor position
            pyautogui.click(int(smoothed_x), int(smoothed_y))
            last_click_time = current_time
            click_flash_until = current_time + CLICK_FLASH_DURATION
            is_clicking_now = True

        # Check if click visual flash is currently active
        is_flash_active = current_time < click_flash_until

        # ----------------------------------------------------------------------
        # STEP H: VISUAL DEBUG FEEDBACK & OVERLAYS
        # ----------------------------------------------------------------------
        nose_pixel_x = int(norm_x * frame_w)
        nose_pixel_y = int(norm_y * frame_h)

        # Lip landmarks in pixel coordinates
        upper_lip_px = (int(landmarks['upper_lip'][0] * frame_w), int(landmarks['upper_lip'][1] * frame_h))
        lower_lip_px = (int(landmarks['lower_lip'][0] * frame_w), int(landmarks['lower_lip'][1] * frame_h))

        # Draw lip tracking points and aperture line
        cv2.circle(frame, upper_lip_px, 3, (0, 255, 255), -1)
        cv2.circle(frame, lower_lip_px, 3, (0, 255, 255), -1)
        cv2.line(frame, upper_lip_px, lower_lip_px, (0, 255, 255), 1)

        # Draw nose tracking dot:
        # Flashes to bright magenta/cyan when a click is triggered, otherwise green
        if is_flash_active:
            dot_color = (0, 0, 255)       # Bright Red/Flash on Click
            halo_color = (0, 255, 255)    # Yellow outer glow
            halo_radius = 18
        else:
            dot_color = (0, 255, 0)       # Normal Green
            halo_color = (0, 255, 255)    # Subtle Cyan/Yellow glow
            halo_radius = 12

        cv2.circle(frame, (nose_pixel_x, nose_pixel_y), 8, dot_color, -1)
        cv2.circle(frame, (nose_pixel_x, nose_pixel_y), halo_radius, halo_color, 2)

        # Draw active interaction bounding box
        cv2.rectangle(
            frame,
            (int(ACTIVE_X_MIN * frame_w), int(ACTIVE_Y_MIN * frame_h)),
            (int(ACTIVE_X_MAX * frame_w), int(ACTIVE_Y_MAX * frame_h)),
            (255, 255, 0),
            1
        )

        # Overlay text: Tracking status
        cv2.putText(
            frame,
            "Face Detected - Tracking Live",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Overlay text: Cursor position
        cv2.putText(
            frame,
            f"Cursor: ({int(smoothed_x)}, {int(smoothed_y)})",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 255, 200),
            1,
            cv2.LINE_AA
        )

        # Overlay text: Live Mouth-Open Ratio & Threshold readout
        # Color turns bright yellow/cyan when open ratio passes threshold
        ratio_color = (0, 255, 255) if mouth_open_ratio >= MOUTH_OPEN_THRESHOLD else (220, 220, 220)
        cv2.putText(
            frame,
            f"Mouth Ratio: {mouth_open_ratio:.2f} (Threshold: {MOUTH_OPEN_THRESHOLD:.2f})",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            ratio_color,
            2 if mouth_open_ratio >= MOUTH_OPEN_THRESHOLD else 1,
            cv2.LINE_AA
        )

        # Overlay text: Cooldown status indicator
        if not can_click:
            remaining_cd = max(0.0, CLICK_COOLDOWN_SECONDS - time_since_last_click)
            cv2.putText(
                frame,
                f"Click Cooldown: {remaining_cd:.1f}s",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),  # Orange
                1,
                cv2.LINE_AA
            )
        else:
            cv2.putText(
                frame,
                "Click Ready: Open Mouth",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (150, 255, 150),  # Light Green
                1,
                cv2.LINE_AA
            )

        # Visual CLICK Flash Banner
        if is_flash_active:
            # Draw prominent click notification bar
            cv2.rectangle(frame, (frame_w // 2 - 120, 20), (frame_w // 2 + 120, 70), (0, 0, 255), -1)
            cv2.putText(
                frame,
                "*** CLICK! ***",
                (frame_w // 2 - 95, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    else:
        # Face NOT detected -> Freeze cursor at last known position
        cv2.putText(
            frame,
            "No Face Detected - Cursor Paused",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    # Display Quit Prompt overlay
    cv2.putText(
        frame,
        "Press 'q' in this window to quit",
        (20, frame_h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    # Show live webcam debug feed window
    cv2.imshow(window_name, frame)

    # Check for 'q' key press to break out of loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[*] Exit requested by user. Terminating tracking...")
        break

# ------------------------------------------------------------------------------
# 8. CLEANUP & RELEASE RESOURCES
# ------------------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
tracker.close()
print("[*] Cleanup complete. Nose Cursor stopped successfully.")
