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
pyautogui.FAILSAFE = False

# Set PyAutoGUI internal pause to 0 for instantaneous, unthrottled cursor movement and scrolling.
pyautogui.PAUSE = 0

# Retrieve primary screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# ------------------------------------------------------------------------------
# 2. TRACKING & GESTURE TUNING PARAMETERS
# ------------------------------------------------------------------------------
# --- CURSOR MOVEMENT (NOSE TRACKING) ---
SMOOTHING_ALPHA = 0.35          # Exponential Moving Average (EMA) smoothing factor
ACTIVE_X_MIN, ACTIVE_X_MAX = 0.25, 0.75  # Active bounding box (prevents neck strain)
ACTIVE_Y_MIN, ACTIVE_Y_MAX = 0.25, 0.75

# --- MOUTH-OPEN CLICK GESTURE ---
MOUTH_OPEN_THRESHOLD = 0.35     # Normalized vertical lip opening threshold
CLICK_COOLDOWN_SECONDS = 1.0    # Cooldown between successive clicks
CLICK_FLASH_DURATION = 0.35     # Duration in seconds for on-screen click flash

# --- DUAL-SPEED VIRTUAL BUTTON SCROLL ZONES ---
# Button column span on the right edge of the webcam view (normalized [0, 1] space)
BUTTON_X_MIN, BUTTON_X_MAX = 0.68, 0.97

# Vertical bounds for the 2-speed zones:
FAST_UP_Y_MIN, FAST_UP_Y_MAX     = 0.03, 0.22   # "FAST UP (Turbo)"
SLOW_UP_Y_MIN, SLOW_UP_Y_MAX     = 0.23, 0.42   # "SLOW UP (Smooth)"
# Neutral Gap: 0.42 <= y <= 0.58 (Safe resting zone, no action)
SLOW_DOWN_Y_MIN, SLOW_DOWN_Y_MAX = 0.58, 0.77   # "SLOW DOWN (Smooth)"
FAST_DOWN_Y_MIN, FAST_DOWN_Y_MAX = 0.78, 0.97   # "FAST DOWN (Turbo)"

# Scroll Speed Tiers (Windows mouse wheel delta units)
SLOW_SCROLL_SPEED = 80          # Normal/Smooth reading speed
FAST_SCROLL_SPEED = 240         # Turbo/Quick navigation speed
SCROLL_INTERVAL = 0.04          # Minimum interval between scroll steps for smooth continuous action

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

# Scroll rate-limiter state
last_scroll_time = 0.0

# ------------------------------------------------------------------------------
# 4. INITIALIZE MEDIAPIPE FACE MESH & HAND DETECTORS
# ------------------------------------------------------------------------------
print("=" * 60)
print("  [*] NOSE CURSOR + MOUTH CLICK + 2-SPEED VIRTUAL SCROLLER")
print("=" * 60)
print(f"[*] Screen Resolution Detected: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
print(f"[*] Mouth Click Threshold: {MOUTH_OPEN_THRESHOLD} (Cooldown: {CLICK_COOLDOWN_SECONDS}s)")
print(f"[*] Scroll Speeds: Slow={SLOW_SCROLL_SPEED}, Fast={FAST_SCROLL_SPEED}")
print("[*] Initializing MediaPipe Face & Hand Landmarkers...")

class VisionPipeline:
    def __init__(self):
        self.use_tasks_api = not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'face_mesh')
        script_dir = os.path.dirname(__file__)

        if self.use_tasks_api:
            from mediapipe.tasks.python import vision, BaseOptions

            # 1. Face Landmarker Model Setup
            face_model_path = os.path.join(script_dir, 'face_landmarker.task')
            if not os.path.exists(face_model_path):
                print("[*] Downloading Face Landmarker model...")
                urllib.request.urlretrieve(
                    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                    face_model_path
                )

            face_options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=face_model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.4,
                min_face_presence_confidence=0.4,
                min_tracking_confidence=0.4
            )
            self.face_detector = vision.FaceLandmarker.create_from_options(face_options)

            # 2. Hand Landmarker Model Setup
            hand_model_path = os.path.join(script_dir, 'hand_landmarker.task')
            if not os.path.exists(hand_model_path):
                print("[*] Downloading Hand Landmarker model...")
                urllib.request.urlretrieve(
                    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                    hand_model_path
                )

            hand_options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3
            )
            self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)

        else:
            self.face_detector = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4
            )
            self.hand_detector = mp.solutions.hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )

        print("[+] Vision models initialized successfully.")

    def process_face(self, rgb_frame):
        """Extracts facial landmarks for cursor tracking and click gesture."""
        if self.use_tasks_api:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.face_detector.detect(mp_img)
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
            result = self.face_detector.process(rgb_frame)
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

    def process_hand(self, rgb_frame):
        """
        Extracts index fingertip (Landmark 8) and wrist (Landmark 0).
        Checks if the finger is extended relative to the wrist/palm.
        """
        if self.use_tasks_api:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.hand_detector.detect(mp_img)
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                lms = result.hand_landmarks[0]
                dist_tip_wrist = (lms[8].x - lms[0].x)**2 + (lms[8].y - lms[0].y)**2
                dist_pip_wrist = (lms[6].x - lms[0].x)**2 + (lms[6].y - lms[0].y)**2
                is_extended = dist_tip_wrist > dist_pip_wrist
                return True, is_extended, (lms[8].x, lms[8].y), [(lm.x, lm.y) for lm in lms]
            return False, False, None, []
        else:
            result = self.hand_detector.process(rgb_frame)
            if result.multi_hand_landmarks and len(result.multi_hand_landmarks) > 0:
                lms = result.multi_hand_landmarks[0].landmark
                dist_tip_wrist = (lms[8].x - lms[0].x)**2 + (lms[8].y - lms[0].y)**2
                dist_pip_wrist = (lms[6].x - lms[0].x)**2 + (lms[6].y - lms[0].y)**2
                is_extended = dist_tip_wrist > dist_pip_wrist
                return True, is_extended, (lms[8].x, lms[8].y), [(lm.x, lm.y) for lm in lms]
            return False, False, None, []

    def close(self):
        if hasattr(self.face_detector, 'close'):
            self.face_detector.close()
        if hasattr(self.hand_detector, 'close'):
            self.hand_detector.close()

pipeline = VisionPipeline()

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
print("[*] Click Gesture: Open mouth to trigger Left Click")
print("[*] 2-Speed Scroll: FAST UP / SLOW UP / NEUTRAL / SLOW DOWN / FAST DOWN")
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

    # Run Face & Hand detections concurrently
    face_found, face_landmarks = pipeline.process_face(rgb_frame)
    hand_found, is_extended, index_tip, all_hand_lms = pipeline.process_hand(rgb_frame)

    # --------------------------------------------------------------------------
    # STEP C: NOSE TRACKING & CURSOR MOVEMENT
    # --------------------------------------------------------------------------
    if face_found:
        nose_norm = face_landmarks['nose']
        norm_x, norm_y = nose_norm

        # Coordinate Mapping using np.interp
        raw_screen_x = np.interp(norm_x, [ACTIVE_X_MIN, ACTIVE_X_MAX], [0, SCREEN_WIDTH])
        raw_screen_y = np.interp(norm_y, [ACTIVE_Y_MIN, ACTIVE_Y_MAX], [0, SCREEN_HEIGHT])

        # Clamp screen coordinates within screen dimensions
        raw_screen_x = np.clip(raw_screen_x, 0, SCREEN_WIDTH - 1)
        raw_screen_y = np.clip(raw_screen_y, 0, SCREEN_HEIGHT - 1)

        # Exponential Moving Average (EMA) Smoothing
        smoothed_x = (smoothed_x * (1.0 - SMOOTHING_ALPHA)) + (raw_screen_x * SMOOTHING_ALPHA)
        smoothed_y = (smoothed_y * (1.0 - SMOOTHING_ALPHA)) + (raw_screen_y * SMOOTHING_ALPHA)

        last_known_x = smoothed_x
        last_known_y = smoothed_y

        # Real-time system cursor movement
        pyautogui.moveTo(int(smoothed_x), int(smoothed_y))

        # ----------------------------------------------------------------------
        # STEP D: MOUTH-OPEN CLICK GESTURE
        # ----------------------------------------------------------------------
        lip_vertical_dist = euclidean_distance(face_landmarks['upper_lip'], face_landmarks['lower_lip'])
        eye_reference_dist = euclidean_distance(face_landmarks['left_eye'], face_landmarks['right_eye'])
        mouth_open_ratio = lip_vertical_dist / max(eye_reference_dist, 1e-6)

        time_since_last_click = current_time - last_click_time
        can_click = time_since_last_click >= CLICK_COOLDOWN_SECONDS

        if mouth_open_ratio >= MOUTH_OPEN_THRESHOLD and can_click:
            pyautogui.click(int(smoothed_x), int(smoothed_y))
            last_click_time = current_time
            click_flash_until = current_time + CLICK_FLASH_DURATION

        is_flash_active = current_time < click_flash_until

        # Draw Face & Lip tracking visuals
        nose_pixel_x = int(norm_x * frame_w)
        nose_pixel_y = int(norm_y * frame_h)

        upper_lip_px = (int(face_landmarks['upper_lip'][0] * frame_w), int(face_landmarks['upper_lip'][1] * frame_h))
        lower_lip_px = (int(face_landmarks['lower_lip'][0] * frame_w), int(face_landmarks['lower_lip'][1] * frame_h))

        cv2.circle(frame, upper_lip_px, 3, (0, 255, 255), -1)
        cv2.circle(frame, lower_lip_px, 3, (0, 255, 255), -1)
        cv2.line(frame, upper_lip_px, lower_lip_px, (0, 255, 255), 1)

        # Nose dot indicator
        if is_flash_active:
            dot_color = (0, 0, 255)       # Red flash on click
            halo_color = (0, 255, 255)
            halo_radius = 18
        else:
            dot_color = (0, 255, 0)
            halo_color = (0, 255, 255)
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

        # Tracking status text
        cv2.putText(frame, "Face: Tracking Live", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Cursor: ({int(smoothed_x)}, {int(smoothed_y)})", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)

        ratio_color = (0, 255, 255) if mouth_open_ratio >= MOUTH_OPEN_THRESHOLD else (220, 220, 220)
        cv2.putText(frame, f"Mouth: {mouth_open_ratio:.2f} (Thresh: {MOUTH_OPEN_THRESHOLD:.2f})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ratio_color, 1, cv2.LINE_AA)

        # Click flash banner
        if is_flash_active:
            cv2.rectangle(frame, (frame_w // 2 - 110, 15), (frame_w // 2 + 110, 60), (0, 0, 255), -1)
            cv2.putText(frame, "*** CLICK! ***", (frame_w // 2 - 85, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    else:
        cv2.putText(frame, "Face: Not Detected (Paused)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # STEP E: 2-SPEED VIRTUAL BUTTON SCROLL ZONE SYSTEM
    # --------------------------------------------------------------------------
    btn_x1 = int(BUTTON_X_MIN * frame_w)
    btn_x2 = int(BUTTON_X_MAX * frame_w)

    fup_y1, fup_y2   = int(FAST_UP_Y_MIN * frame_h), int(FAST_UP_Y_MAX * frame_h)
    sup_y1, sup_y2   = int(SLOW_UP_Y_MIN * frame_h), int(SLOW_UP_Y_MAX * frame_h)
    sdown_y1, sdown_y2 = int(SLOW_DOWN_Y_MIN * frame_h), int(SLOW_DOWN_Y_MAX * frame_h)
    fdown_y1, fdown_y2 = int(FAST_DOWN_Y_MIN * frame_h), int(FAST_DOWN_Y_MAX * frame_h)

    # Check button collisions with index fingertip
    is_fast_up   = False
    is_slow_up   = False
    is_slow_down = False
    is_fast_down = False

    if hand_found and index_tip is not None:
        tip_x, tip_y = index_tip
        if BUTTON_X_MIN <= tip_x <= BUTTON_X_MAX:
            if FAST_UP_Y_MIN <= tip_y <= FAST_UP_Y_MAX:
                is_fast_up = True
            elif SLOW_UP_Y_MIN <= tip_y <= SLOW_UP_Y_MAX:
                is_slow_up = True
            elif SLOW_DOWN_Y_MIN <= tip_y <= SLOW_DOWN_Y_MAX:
                is_slow_down = True
            elif FAST_DOWN_Y_MIN <= tip_y <= FAST_DOWN_Y_MAX:
                is_fast_down = True

    # Continuous scroll execution with 2 speeds
    if is_fast_up:
        if current_time - last_scroll_time >= SCROLL_INTERVAL:
            pyautogui.scroll(FAST_SCROLL_SPEED)
            last_scroll_time = current_time
        scroll_status_text = "FAST UP (TURBO)"
        scroll_status_color = (0, 255, 255)
    elif is_slow_up:
        if current_time - last_scroll_time >= SCROLL_INTERVAL:
            pyautogui.scroll(SLOW_SCROLL_SPEED)
            last_scroll_time = current_time
        scroll_status_text = "SLOW UP (SMOOTH)"
        scroll_status_color = (0, 255, 0)
    elif is_slow_down:
        if current_time - last_scroll_time >= SCROLL_INTERVAL:
            pyautogui.scroll(-SLOW_SCROLL_SPEED)
            last_scroll_time = current_time
        scroll_status_text = "SLOW DOWN (SMOOTH)"
        scroll_status_color = (0, 255, 0)
    elif is_fast_down:
        if current_time - last_scroll_time >= SCROLL_INTERVAL:
            pyautogui.scroll(-FAST_SCROLL_SPEED)
            last_scroll_time = current_time
        scroll_status_text = "FAST DOWN (TURBO)"
        scroll_status_color = (0, 165, 255)
    else:
        scroll_status_text = "IDLE"
        scroll_status_color = (180, 180, 180)

    # --------------------------------------------------------------------------
    # STEP F: RENDER 2-SPEED VIRTUAL BUTTONS & HAND VISUALS
    # --------------------------------------------------------------------------
    btn_overlay = frame.copy()

    # Helper renderer for each button tier
    def render_button(y1, y2, is_active, label_line1, label_line2, active_bg=(0, 220, 0)):
        if is_active:
            cv2.rectangle(btn_overlay, (btn_x1, y1), (btn_x2, y2), active_bg, -1)
            border_c = (0, 255, 0) if active_bg == (0, 220, 0) else (0, 255, 255)
            thick = 3
            text_c = (0, 0, 0)
        else:
            cv2.rectangle(btn_overlay, (btn_x1, y1), (btn_x2, y2), (100, 50, 0), -1)
            border_c = (255, 200, 0)
            thick = 2
            text_c = (255, 255, 255)
        return border_c, thick, text_c

    # 1. FAST UP Button
    fup_bc, fup_th, fup_tc = render_button(fup_y1, fup_y2, is_fast_up, "FAST", "UP", active_bg=(0, 230, 255))
    # 2. SLOW UP Button
    sup_bc, sup_th, sup_tc = render_button(sup_y1, sup_y2, is_slow_up, "SLOW", "UP", active_bg=(0, 220, 0))
    # 3. SLOW DOWN Button
    sdown_bc, sdown_th, sdown_tc = render_button(sdown_y1, sdown_y2, is_slow_down, "SLOW", "DOWN", active_bg=(0, 220, 0))
    # 4. FAST DOWN Button
    fdown_bc, fdown_th, fdown_tc = render_button(fdown_y1, fdown_y2, is_fast_down, "FAST", "DOWN", active_bg=(0, 165, 255))

    # Blend button backgrounds
    cv2.addWeighted(btn_overlay, 0.45, frame, 0.55, 0, frame)

    # Draw solid border outlines
    cv2.rectangle(frame, (btn_x1, fup_y1), (btn_x2, fup_y2), fup_bc, fup_th)
    cv2.rectangle(frame, (btn_x1, sup_y1), (btn_x2, sup_y2), sup_bc, sup_th)
    cv2.rectangle(frame, (btn_x1, sdown_y1), (btn_x2, sdown_y2), sdown_bc, sdown_th)
    cv2.rectangle(frame, (btn_x1, fdown_y1), (btn_x2, fdown_y2), fdown_bc, fdown_th)

    # Render Button Labels
    # FAST UP
    fup_mid = (fup_y1 + fup_y2) // 2
    cv2.putText(frame, "▲▲ FAST UP", (btn_x1 + 10, fup_mid + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, fup_tc, 2, cv2.LINE_AA)

    # SLOW UP
    sup_mid = (sup_y1 + sup_y2) // 2
    cv2.putText(frame, "▲ SLOW UP", (btn_x1 + 10, sup_mid + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, sup_tc, 2, cv2.LINE_AA)

    # NEUTRAL GAP
    neutral_mid = (sup_y2 + sdown_y1) // 2
    cv2.putText(frame, "[ NEUTRAL ]", (btn_x1 + 10, neutral_mid + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)

    # SLOW DOWN
    sdown_mid = (sdown_y1 + sdown_y2) // 2
    cv2.putText(frame, "▼ SLOW DOWN", (btn_x1 + 8, sdown_mid + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sdown_tc, 2, cv2.LINE_AA)

    # FAST DOWN
    fdown_mid = (fdown_y1 + fdown_y2) // 2
    cv2.putText(frame, "▼▼ FAST DOWN", (btn_x1 + 8, fdown_mid + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, fdown_tc, 2, cv2.LINE_AA)

    # Draw Hand Landmarks / Pointer Marker
    if hand_found and index_tip is not None:
        tip_px_x = int(index_tip[0] * frame_w)
        tip_px_y = int(index_tip[1] * frame_h)
        is_any_active = is_fast_up or is_slow_up or is_slow_down or is_fast_down
        pointer_color = (0, 255, 0) if is_any_active else (0, 255, 255)
        
        # Draw all hand landmarks
        for lm_x, lm_y in all_hand_lms:
            cv2.circle(frame, (int(lm_x * frame_w), int(lm_y * frame_h)), 2, (0, 180, 255), -1)

        # Highlight Fingertip Marker
        cv2.circle(frame, (tip_px_x, tip_px_y), 9, pointer_color, -1)
        cv2.circle(frame, (tip_px_x, tip_px_y), 15, (255, 255, 255), 2)
        cv2.putText(frame, "POINTER", (tip_px_x + 15, tip_px_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # Render Scroll Status on HUD
    cv2.putText(
        frame,
        f"Scroll: {scroll_status_text}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        scroll_status_color,
        2 if scroll_status_text != "IDLE" else 1,
        cv2.LINE_AA
    )

    # Display Quit Prompt overlay
    cv2.putText(
        frame,
        "Press 'q' in this window to quit",
        (20, frame_h - 15),
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
pipeline.close()
print("[*] Cleanup complete. Nose Cursor stopped successfully.")
