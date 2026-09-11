# ==============================================================================
# PIP INSTALLATION COMMAND:
# pip install opencv-python mediapipe pyautogui numpy pygame
# ==============================================================================

import sys
import os
import time
import threading
import urllib.request
import ctypes
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize Pygame Mixer for non-blocking asynchronous audio playback
try:
    import pygame
    pygame.mixer.init()
except Exception as e:
    print(f"[!] Pygame mixer notice: {e}")

# Force UTF-8 stdout if possible on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ------------------------------------------------------------------------------
# 1. SYSTEM & SAFETY CONFIGURATION
# ------------------------------------------------------------------------------
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# ------------------------------------------------------------------------------
# FULLSCREEN INVERSION SUPPORT
# ------------------------------------------------------------------------------
fullscreen_inversion = False
original_display_orientation = None


def _get_foreground_window_details():
    """Returns the active window title, class, and screen rectangle on Windows."""
    if sys.platform != "win32":
        return "", "", None

    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return "", "", None

        title_buffer = ctypes.create_unicode_buffer(512)
        class_buffer = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, title_buffer, len(title_buffer))
        user32.GetClassNameW(hwnd, class_buffer, len(class_buffer))

        class Rect(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        rect = Rect()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return title_buffer.value, class_buffer.value, None
        return title_buffer.value, class_buffer.value, rect
    except Exception:
        return "", "", None


def is_youtube_fullscreen():
    """Detects a fullscreen YouTube browser window without requiring browser hooks."""
    title, window_class, rect = _get_foreground_window_details()
    if rect is None or title.lower().startswith(window_name.lower()):
        return False

    fills_screen = (
        rect.left <= 0 and rect.top <= 0 and
        rect.right >= SCREEN_WIDTH and rect.bottom >= SCREEN_HEIGHT
    )
    browser_window = window_class in {
        "Chrome_WidgetWin_1",
        "MozillaWindowClass",
        "ApplicationFrameWindow",
    }
    return fills_screen and browser_window and "youtube" in title.lower()


class _DisplayMode(ctypes.Structure):
    _fields_ = [
        ("dmDeviceName", ctypes.c_wchar * 32),
        ("dmSpecVersion", ctypes.c_ushort),
        ("dmDriverVersion", ctypes.c_ushort),
        ("dmSize", ctypes.c_ushort),
        ("dmDriverExtra", ctypes.c_ushort),
        ("dmFields", ctypes.c_ulong),
        ("dmPositionX", ctypes.c_long),
        ("dmPositionY", ctypes.c_long),
        ("dmDisplayOrientation", ctypes.c_ulong),
        ("dmDisplayFixedOutput", ctypes.c_ulong),
        ("dmColor", ctypes.c_short),
        ("dmDuplex", ctypes.c_short),
        ("dmYResolution", ctypes.c_short),
        ("dmTTOption", ctypes.c_short),
        ("dmCollate", ctypes.c_short),
        ("dmFormName", ctypes.c_wchar * 32),
        ("dmLogPixels", ctypes.c_ushort),
        ("dmBitsPerPel", ctypes.c_ulong),
        ("dmPelsWidth", ctypes.c_ulong),
        ("dmPelsHeight", ctypes.c_ulong),
        ("dmDisplayFlags", ctypes.c_ulong),
        ("dmDisplayFrequency", ctypes.c_ulong),
        ("dmICMMethod", ctypes.c_ulong),
        ("dmICMIntent", ctypes.c_ulong),
        ("dmMediaType", ctypes.c_ulong),
        ("dmDitherType", ctypes.c_ulong),
        ("dmReserved1", ctypes.c_ulong),
        ("dmReserved2", ctypes.c_ulong),
        ("dmPanningWidth", ctypes.c_ulong),
        ("dmPanningHeight", ctypes.c_ulong),
    ]


def set_display_inverted(inverted):
    """Attempts a 180-degree primary-display rotation and never raises."""
    global original_display_orientation
    if sys.platform != "win32":
        return False

    try:
        user32 = ctypes.windll.user32
        mode = _DisplayMode()
        mode.dmSize = ctypes.sizeof(_DisplayMode)
        if not user32.EnumDisplaySettingsW(None, 0xFFFFFFFF, ctypes.byref(mode)):
            return False

        if original_display_orientation is None:
            original_display_orientation = mode.dmDisplayOrientation

        mode.dmDisplayOrientation = (
            (original_display_orientation + 2) % 4
            if inverted else original_display_orientation
        )
        mode.dmFields = 0x00000080  # DM_DISPLAYORIENTATION
        result = user32.ChangeDisplaySettingsW(ctypes.byref(mode), 0)
        return result == 0
    except Exception as error:
        print(f"[!] Display rotation unavailable; using software inversion only: {error}")
        return False


def update_fullscreen_inversion(active):
    """Applies inversion once per state change; display rotation is best effort."""
    global fullscreen_inversion
    if active == fullscreen_inversion:
        return

    fullscreen_inversion = active
    rotated = set_display_inverted(active)
    if active:
        print(f"[!] FULLSCREEN INVERSION ACTIVATED (display rotation: {'yes' if rotated else 'software only'})")
    else:
        print("[+] Fullscreen inversion cleared.")

# ------------------------------------------------------------------------------
# 2. TUNING PARAMETERS & THRESHOLDS
# ------------------------------------------------------------------------------
# --- NOSE CURSOR TRACKING ---
SMOOTHING_ALPHA = 0.22          # Smooth, calm cursor motion
ACTIVE_X_MIN, ACTIVE_X_MAX = 0.20, 0.80  # Comfortable head navigation bounds
ACTIVE_Y_MIN, ACTIVE_Y_MAX = 0.20, 0.80

# --- MOUTH-OPEN CLICK GESTURE ---
MOUTH_OPEN_THRESHOLD = 0.38     # Normalized vertical lip ratio
CLICK_COOLDOWN_SECONDS = 1.2    # Cooldown between clicks
CLICK_FLASH_DURATION = 0.35     # Visual flash duration

# --- 2-SPEED VIRTUAL BUTTON SCROLLER ---
BUTTON_X_MIN, BUTTON_X_MAX = 0.65, 0.98
FAST_UP_Y_MIN, FAST_UP_Y_MAX     = 0.05, 0.22   # "FAST UP"
SLOW_UP_Y_MIN, SLOW_UP_Y_MAX     = 0.24, 0.42   # "SLOW UP"
SLOW_DOWN_Y_MIN, SLOW_DOWN_Y_MAX = 0.58, 0.76   # "SLOW DOWN"
FAST_DOWN_Y_MIN, FAST_DOWN_Y_MAX = 0.78, 0.95   # "FAST DOWN"

SLOW_SCROLL_SPEED = 40          # Gentle reading scroll
FAST_SCROLL_SPEED = 120         # Turbo scroll
SCROLL_INTERVAL = 0.08          # Timed interval between scroll ticks

# --- YOUTUBE PRANK MODE CONFIGURATION ---
PRANK_MODE_DEFAULT = True       # Starts Enabled (Toggle with 'r')
LOOKING_FRAME_BUFFER = 4        # Debounce buffer frames (~0.12s) for clean transition
JERK_COOLDOWN_SECONDS = 3.0     # 3-second cooldown on sneeze/head jerk restart
JERK_DOWNWARD_VELOCITY_THRESH = 0.65  # Normalized downward speed threshold
JERK_DISPLACEMENT_THRESH = 0.06       # Sudden delta y threshold

# ------------------------------------------------------------------------------
# 3. GLOBAL SINGLE-EVENT MEDIA & VIDEO CONTROLLER
# ------------------------------------------------------------------------------
def toggle_media_play_pause():
    """
    Sends a SINGLE clean hardware Media Play/Pause toggle event (VK_MEDIA_PLAY_PAUSE = 0xB3).
    Works globally on Windows across Chrome, Edge, Brave, and Firefox even when
    the OpenCV webcam window is currently focused.
    """
    try:
        if sys.platform == "win32":
            VK_MEDIA_PLAY_PAUSE = 0xB3
            ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 0, 0)
            time.sleep(0.01)
            ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 2, 0)
        else:
            pyautogui.press('k')
    except Exception:
        pyautogui.press('k')

def restart_youtube_video():
    """
    Restarts the YouTube video to 0:00 by sending shortcut '0' / 'home'.
    """
    try:
        pyautogui.press('0')
        pyautogui.press('home')
    except Exception:
        pass

# ------------------------------------------------------------------------------
# 4. ASSET MANAGEMENT & AUDIO PIPELINE (WITH CRASH-PROOF FALLBACKS)
# ------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(SCRIPT_DIR, 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

def load_png(filename):
    """Loads a 4-channel PNG with alpha transparency from assets/."""
    try:
        path = os.path.join(ASSETS_DIR, filename)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img
    except Exception:
        pass
    return None

judge_cat_img = load_png('judge_cat.png')
happy_cat_img = load_png('happy_cat.png')
shocked_cat_img = load_png('shocked_cat.png')

def play_sound_async(sound_name):
    """Plays audio on a non-blocking daemon thread so webcam frames never drop."""
    def _play():
        try:
            path = os.path.join(ASSETS_DIR, sound_name)
            if os.path.exists(path):
                if 'pygame' in sys.modules and pygame.mixer.get_init():
                    sound = pygame.mixer.Sound(path)
                    sound.play()
                    return
            if sys.platform == "win32":
                import winsound
                winsound.Beep(880, 150)
        except Exception:
            pass
    threading.Thread(target=_play, daemon=True).start()

def overlay_png(frame, overlay_img, x, y, target_w=None, target_h=None, fallback_label="MEME CAT"):
    """
    Cleanly alpha-blends a 4-channel PNG over the OpenCV BGR frame.
    If image is missing, draws a crash-proof stylish cartoon fallback box.
    """
    h_frame, w_frame = frame.shape[:2]
    tw = int(target_w) if target_w is not None else 180
    th = int(target_h) if target_h is not None else 180

    if overlay_img is None:
        bx1, by1 = max(0, x), max(0, y)
        bx2, by2 = min(w_frame, x + tw), min(h_frame, y + th)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (40, 40, 40), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        cv2.putText(frame, fallback_label, (bx1 + 10, by1 + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return frame

    if target_w is not None and target_h is not None:
        overlay = cv2.resize(overlay_img, (tw, th), interpolation=cv2.INTER_AREA)
    else:
        overlay = overlay_img

    h_ov, w_ov = overlay.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_frame, x + w_ov), min(h_frame, y + h_ov)

    if x1 >= x2 or y1 >= y2:
        return frame

    ov_x1, ov_y1 = max(0, -x), max(0, -y)
    ov_x2, ov_y2 = ov_x1 + (x2 - x1), ov_y1 + (y2 - y1)

    overlay_crop = overlay[ov_y1:ov_y2, ov_x1:ov_x2]
    frame_crop = frame[y1:y2, x1:x2]

    if overlay_crop.shape[2] == 4:
        alpha = (overlay_crop[:, :, 3] / 255.0)[:, :, np.newaxis]
        bgr = overlay_crop[:, :, :3]
        blended = (bgr * alpha + frame_crop * (1.0 - alpha)).astype(np.uint8)
        frame[y1:y2, x1:x2] = blended
    else:
        frame[y1:y2, x1:x2] = overlay_crop[:, :, :3]

    return frame

# ------------------------------------------------------------------------------
# 5. STATE MEMORY VARIABLES
# ------------------------------------------------------------------------------
smoothed_x = SCREEN_WIDTH / 2
smoothed_y = SCREEN_HEIGHT / 2
last_known_x = smoothed_x
last_known_y = smoothed_y

# Click & Scroll states
last_click_time = 0.0
click_flash_until = 0.0
last_scroll_time = 0.0

# Prank Mode states
prank_active = PRANK_MODE_DEFAULT
video_state = "INITIAL"
looking_counter = 0
not_looking_counter = 0
happy_cat_until = 0.0

# Head Jerk / Sneeze detection state
prev_nose_y = None
prev_pose_time = 0.0
last_jerk_time = 0.0
jerk_alert_until = 0.0

# ------------------------------------------------------------------------------
# 6. INITIALIZE MEDIAPIPE FACE MESH & HAND DETECTORS
# ------------------------------------------------------------------------------
print("=" * 65)
print("  [*] NOSE CURSOR + 2-SPEED SCROLLER + YOUTUBE PRANK MODE")
print("=" * 65)
print(f"[*] Screen Resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
print(f"[*] YouTube Prank Mode: {'ACTIVE' if prank_active else 'OFF'} (Press 'r' to toggle)")
print("[*] RULE 1: Watch Screen -> Look away pauses video ('LOOK AT THE SCREEN!')")
print("[*] RULE 2: No Sneeze/Jerks -> Sudden jerk restarts video to 0:00")
print("[*] Hotkeys: Press 'r' to toggle prank mode | 'q' to exit.")
print("=" * 65)

class VisionPipeline:
    def __init__(self):
        self.use_tasks_api = not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'face_mesh')

        if self.use_tasks_api:
            from mediapipe.tasks.python import vision, BaseOptions

            face_model_path = os.path.join(SCRIPT_DIR, 'face_landmarker.task')
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

            hand_model_path = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')
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
                min_hand_detection_confidence=0.25,
                min_hand_presence_confidence=0.25,
                min_tracking_confidence=0.25
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
                min_detection_confidence=0.25,
                min_tracking_confidence=0.25
            )

        print("[+] Vision models initialized successfully.")

    def process_face(self, rgb_frame):
        """Extracts facial landmarks for cursor tracking, click, and head orientation."""
        if self.use_tasks_api:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.face_detector.detect(mp_img)
            if result.face_landmarks and len(result.face_landmarks) > 0:
                lms = result.face_landmarks[0]
                return True, {
                    'nose': (lms[1].x, lms[1].y),
                    'forehead': (lms[10].x, lms[10].y),
                    'chin': (lms[152].x, lms[152].y),
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
                    'forehead': (lms[10].x, lms[10].y),
                    'chin': (lms[152].x, lms[152].y),
                    'upper_lip': (lms[13].x, lms[13].y),
                    'lower_lip': (lms[14].x, lms[14].y),
                    'left_eye': (lms[33].x, lms[33].y),
                    'right_eye': (lms[263].x, lms[263].y),
                }
            return False, None

    def process_hand(self, rgb_frame):
        """Extracts index fingertip (Landmark 8) for virtual button scrolling."""
        if self.use_tasks_api:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.hand_detector.detect(mp_img)
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                lms = result.hand_landmarks[0]
                return True, (lms[8].x, lms[8].y), [(lm.x, lm.y) for lm in lms]
            return False, None, []
        else:
            result = self.hand_detector.process(rgb_frame)
            if result.multi_hand_landmarks and len(result.multi_hand_landmarks) > 0:
                lms = result.multi_hand_landmarks[0].landmark
                return True, (lms[8].x, lms[8].y), [(lm.x, lm.y) for lm in lms]
            return False, None, []

    def close(self):
        if hasattr(self.face_detector, 'close'):
            self.face_detector.close()
        if hasattr(self.hand_detector, 'close'):
            self.hand_detector.close()

pipeline = VisionPipeline()

# ------------------------------------------------------------------------------
# 7. INITIALIZE WEBCAM
# ------------------------------------------------------------------------------
print("[*] Connecting to webcam...")

def open_camera():
    backends = [(0, cv2.CAP_DSHOW), (0, cv2.CAP_ANY), (1, cv2.CAP_DSHOW), (1, cv2.CAP_ANY)]
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
    sys.exit(1)

window_name = "Nose Cursor - YouTube Meme Prank Suite"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 720, 540)

print("[*] Tracking Target: Nose Tip (Landmark 1)")
print("[*] Controls: Move nose to steer, open mouth to click, hover right boxes to scroll.")
print("[*] Mode: Look at screen to play | Look away to pause | Sneeze/jerk restarts video")
print("[*] Press 'r' to toggle prank mode ON/OFF | Press 'q' to exit.")
print("=" * 65)

# ------------------------------------------------------------------------------
# 8. HELPER MATH FUNCTIONS
# ------------------------------------------------------------------------------
def euclidean_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# ------------------------------------------------------------------------------
# 9. MAIN CONTINUOUS REAL-TIME TRACKING & PRANK LOOP
# ------------------------------------------------------------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("[!] Warning: Empty or unreadable frame from webcam.")
        continue

    current_time = time.time()
    update_fullscreen_inversion(is_youtube_fullscreen())

    # STEP A: Flip frame horizontally for intuitive mirror-like navigation
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    # STEP B: Convert BGR to RGB for MediaPipe inference
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run Face & Hand detections concurrently
    face_found, face_landmarks = pipeline.process_face(rgb_frame)
    hand_found, index_tip, all_hand_lms = pipeline.process_hand(rgb_frame)

    raw_looking_at_screen = False

    # --------------------------------------------------------------------------
    # STEP C: NOSE TRACKING, GAZE POSE & SNEEZE/JERK DETECTION
    # --------------------------------------------------------------------------
    if face_found:
        nose_norm = face_landmarks['nose']
        norm_x, norm_y = nose_norm

        # 1. Coordinate Mapping using np.interp
        cursor_x_targets = [SCREEN_WIDTH, 0] if fullscreen_inversion else [0, SCREEN_WIDTH]
        cursor_y_targets = [SCREEN_HEIGHT, 0] if fullscreen_inversion else [0, SCREEN_HEIGHT]
        raw_screen_x = np.interp(norm_x, [ACTIVE_X_MIN, ACTIVE_X_MAX], cursor_x_targets)
        raw_screen_y = np.interp(norm_y, [ACTIVE_Y_MIN, ACTIVE_Y_MAX], cursor_y_targets)
        raw_screen_x = np.clip(raw_screen_x, 0, SCREEN_WIDTH - 1)
        raw_screen_y = np.clip(raw_screen_y, 0, SCREEN_HEIGHT - 1)

        # 2. Exponential Moving Average (EMA) Smoothing
        smoothed_x = (smoothed_x * (1.0 - SMOOTHING_ALPHA)) + (raw_screen_x * SMOOTHING_ALPHA)
        smoothed_y = (smoothed_y * (1.0 - SMOOTHING_ALPHA)) + (raw_screen_y * SMOOTHING_ALPHA)
        last_known_x, last_known_y = smoothed_x, smoothed_y

        # Move system mouse cursor
        pyautogui.moveTo(int(smoothed_x), int(smoothed_y))

        # 3. Head Orientation & Gaze Pose Detection
        forehead_to_nose = euclidean_distance(face_landmarks['forehead'], face_landmarks['nose'])
        nose_to_chin = euclidean_distance(face_landmarks['nose'], face_landmarks['chin'])
        nose_to_left_eye = euclidean_distance(face_landmarks['nose'], face_landmarks['left_eye'])
        nose_to_right_eye = euclidean_distance(face_landmarks['nose'], face_landmarks['right_eye'])

        pitch_ratio = forehead_to_nose / max(nose_to_chin, 1e-6)
        yaw_symmetry = min(nose_to_left_eye, nose_to_right_eye) / max(max(nose_to_left_eye, nose_to_right_eye), 1e-6)

        # Looking forward at the screen:
        if 0.60 <= pitch_ratio <= 1.80 and yaw_symmetry >= 0.40:
            raw_looking_at_screen = True
        else:
            raw_looking_at_screen = False

        # 4. Sneeze / Sudden Downward Head Jerk Detection
        if prev_nose_y is not None and prev_pose_time > 0:
            dt = max(current_time - prev_pose_time, 1e-4)
            dy = norm_y - prev_nose_y
            downward_speed = dy / dt

            # If sudden downward movement / sneeze / jerk detected
            if (downward_speed >= JERK_DOWNWARD_VELOCITY_THRESH or dy >= JERK_DISPLACEMENT_THRESH):
                if prank_active and (current_time - last_jerk_time) >= JERK_COOLDOWN_SECONDS:
                    last_jerk_time = current_time
                    jerk_alert_until = current_time + 2.5
                    restart_youtube_video()
                    play_sound_async('banana_cat_cry.mp3')
                    print("[!] SNEEZE / HEAD JERK DETECTED! Restarting YouTube video to 0:00...")

        prev_nose_y = norm_y
        prev_pose_time = current_time

        # 5. Mouth-Open Click Gesture
        lip_vertical_dist = euclidean_distance(face_landmarks['upper_lip'], face_landmarks['lower_lip'])
        eye_dist = euclidean_distance(face_landmarks['left_eye'], face_landmarks['right_eye'])
        mouth_open_ratio = lip_vertical_dist / max(eye_dist, 1e-6)

        can_click = (current_time - last_click_time) >= CLICK_COOLDOWN_SECONDS
        if mouth_open_ratio >= MOUTH_OPEN_THRESHOLD and can_click:
            pyautogui.click(int(smoothed_x), int(smoothed_y))
            last_click_time = current_time
            click_flash_until = current_time + CLICK_FLASH_DURATION

        # Visual feedback: Nose dot
        nose_pixel_x, nose_pixel_y = int(norm_x * frame_w), int(norm_y * frame_h)
        is_flash_active = current_time < click_flash_until
        dot_color = (0, 0, 255) if is_flash_active else (0, 255, 0)
        cv2.circle(frame, (nose_pixel_x, nose_pixel_y), 8, dot_color, -1)
        cv2.circle(frame, (nose_pixel_x, nose_pixel_y), 14, (0, 255, 255), 2)

    else:
        # Face not detected -> user is definitely looking away / stepped away
        raw_looking_at_screen = False
        prev_nose_y = None

    # --------------------------------------------------------------------------
    # STEP D: "WATCH THE SCREEN" PRANK RULE STATE MACHINE
    # --------------------------------------------------------------------------
    if prank_active:
        if raw_looking_at_screen:
            looking_counter += 1
            not_looking_counter = 0
        else:
            not_looking_counter += 1
            looking_counter = 0

        # Looking Away -> Automatically PAUSE YouTube video
        if not_looking_counter >= LOOKING_FRAME_BUFFER and video_state != "PAUSED":
            toggle_media_play_pause()
            play_sound_async('huh_cat.mp3')
            video_state = "PAUSED"
            print("[!] PRANK RULE: LOOKED AWAY! YouTube PAUSED.")

        # Looking Back at Screen -> Automatically UNPAUSE YouTube video & show happy cat
        elif looking_counter >= LOOKING_FRAME_BUFFER and video_state != "PLAYING":
            toggle_media_play_pause()
            play_sound_async('happy_cat.mp3')
            happy_cat_until = current_time + 1.5
            video_state = "PLAYING"
            print("[+] PRANK RULE: LOOKED BACK! YouTube RESUMED.")

    # --------------------------------------------------------------------------
    # STEP E: 2-SPEED VIRTUAL BUTTON SCROLL ZONE SYSTEM
    # --------------------------------------------------------------------------
    btn_x1, btn_x2 = int(BUTTON_X_MIN * frame_w), int(BUTTON_X_MAX * frame_w)
    fup_y1, fup_y2     = int(FAST_UP_Y_MIN * frame_h), int(FAST_UP_Y_MAX * frame_h)
    sup_y1, sup_y2     = int(SLOW_UP_Y_MIN * frame_h), int(SLOW_UP_Y_MAX * frame_h)
    sdown_y1, sdown_y2 = int(SLOW_DOWN_Y_MIN * frame_h), int(SLOW_DOWN_Y_MAX * frame_h)
    fdown_y1, fdown_y2 = int(FAST_DOWN_Y_MIN * frame_h), int(FAST_DOWN_Y_MAX * frame_h)

    is_fast_up = is_slow_up = is_slow_down = is_fast_down = False

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

    # Continuous scroll execution
    if is_fast_up:
        if current_time - last_scroll_time >= SCROLL_INTERVAL:
            pyautogui.scroll(FAST_SCROLL_SPEED)
            last_scroll_time = current_time
        scroll_status_text, scroll_status_color = "FAST UP", (0, 255, 255)
    elif is_slow_up:
        if current_time - last_scroll_time >= SCROLL_INTERVAL:
            pyautogui.scroll(SLOW_SCROLL_SPEED)
            last_scroll_time = current_time
        scroll_status_text, scroll_status_color = "SLOW UP", (0, 255, 0)
    elif is_slow_down:
        if current_time - last_scroll_time >= SCROLL_INTERVAL:
            pyautogui.scroll(-SLOW_SCROLL_SPEED)
            last_scroll_time = current_time
        scroll_status_text, scroll_status_color = "SLOW DOWN", (0, 255, 0)
    elif is_fast_down:
        if current_time - last_scroll_time >= SCROLL_INTERVAL:
            pyautogui.scroll(-FAST_SCROLL_SPEED)
            last_scroll_time = current_time
        scroll_status_text, scroll_status_color = "FAST DOWN", (0, 165, 255)
    else:
        scroll_status_text, scroll_status_color = "IDLE", (180, 180, 180)

    # --------------------------------------------------------------------------
    # STEP F: RENDER BUTTONS & HAND HUD
    # --------------------------------------------------------------------------
    btn_overlay = frame.copy()

    def render_button(y1, y2, is_active, active_bg=(0, 220, 0)):
        if is_active:
            cv2.rectangle(btn_overlay, (btn_x1, y1), (btn_x2, y2), active_bg, -1)
            border_c = (0, 255, 0) if active_bg == (0, 220, 0) else (0, 255, 255)
            thick = 3
            text_c = (0, 0, 0)
        else:
            cv2.rectangle(btn_overlay, (btn_x1, y1), (btn_x2, y2), (80, 40, 0), -1)
            border_c = (255, 180, 0)
            thick = 2
            text_c = (255, 255, 255)
        return border_c, thick, text_c

    fup_bc, fup_th, fup_tc     = render_button(fup_y1, fup_y2, is_fast_up, active_bg=(0, 230, 255))
    sup_bc, sup_th, sup_tc     = render_button(sup_y1, sup_y2, is_slow_up, active_bg=(0, 220, 0))
    sdown_bc, sdown_th, sdown_tc = render_button(sdown_y1, sdown_y2, is_slow_down, active_bg=(0, 220, 0))
    fdown_bc, fdown_th, fdown_tc = render_button(fdown_y1, fdown_y2, is_fast_down, active_bg=(0, 165, 255))

    cv2.addWeighted(btn_overlay, 0.45, frame, 0.55, 0, frame)

    cv2.rectangle(frame, (btn_x1, fup_y1), (btn_x2, fup_y2), fup_bc, fup_th)
    cv2.rectangle(frame, (btn_x1, sup_y1), (btn_x2, sup_y2), sup_bc, sup_th)
    cv2.rectangle(frame, (btn_x1, sdown_y1), (btn_x2, sdown_y2), sdown_bc, sdown_th)
    cv2.rectangle(frame, (btn_x1, fdown_y1), (btn_x2, fdown_y2), fdown_bc, fdown_th)

    cv2.putText(frame, "FAST UP", (btn_x1 + 16, (fup_y1 + fup_y2) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, fup_tc, 2, cv2.LINE_AA)
    cv2.putText(frame, "SLOW UP", (btn_x1 + 16, (sup_y1 + sup_y2) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, sup_tc, 2, cv2.LINE_AA)
    cv2.putText(frame, "[ NEUTRAL ]", (btn_x1 + 10, (sup_y2 + sdown_y1) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.putText(frame, "SLOW DOWN", (btn_x1 + 10, (sdown_y1 + sdown_y2) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sdown_tc, 2, cv2.LINE_AA)
    cv2.putText(frame, "FAST DOWN", (btn_x1 + 10, (fdown_y1 + fdown_y2) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, fdown_tc, 2, cv2.LINE_AA)

    # Hand Skeleton & Pointer Marker
    if hand_found and index_tip is not None:
        tip_px_x, tip_px_y = int(index_tip[0] * frame_w), int(index_tip[1] * frame_h)
        for lm_x, lm_y in all_hand_lms:
            cv2.circle(frame, (int(lm_x * frame_w), int(lm_y * frame_h)), 2, (0, 180, 255), -1)
        pointer_color = (0, 255, 0) if (is_fast_up or is_slow_up or is_slow_down or is_fast_down) else (0, 255, 255)
        cv2.circle(frame, (tip_px_x, tip_px_y), 9, pointer_color, -1)
        cv2.circle(frame, (tip_px_x, tip_px_y), 15, (255, 255, 255), 2)
        cv2.putText(frame, "POINTER", (tip_px_x + 15, tip_px_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # STEP G: PRANK MODE MEME CAT OVERLAYS & ALERTS
    # --------------------------------------------------------------------------
    if prank_active:
        # 1. SNEEZE / JERK RESTART OVERLAY (Highest priority)
        if current_time < jerk_alert_until:
            cv2.rectangle(frame, (15, 15), (frame_w - 15, 80), (0, 0, 200), -1)
            cv2.putText(frame, "EMOTION DETECTED! RESTARTING VIDEO!", (25, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "VIDEO RESTARTED TO 0:00 (NO JERKS / SNEEZING)", (25, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 255, 255), 1, cv2.LINE_AA)

            shock_sz = int(min(frame_w, frame_h) * 0.45)
            frame = overlay_png(frame, shocked_cat_img, frame_w // 2 - shock_sz // 2, frame_h // 2 - shock_sz // 2, shock_sz, shock_sz, fallback_label="SCREAMING CAT")

        # 2. LOOKING AWAY: WARNING & JUDGE MEME CAT
        elif video_state == "PAUSED":
            # Red Header Warning Banner: LOOK AT THE SCREEN!
            cv2.rectangle(frame, (15, 15), (frame_w - 15, 80), (0, 0, 230), -1)
            cv2.putText(frame, "LOOK AT THE SCREEN!", (30, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "VIDEO PAUSED! LOOK BACK TO RESUME PLAYBACK", (30, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 255, 255), 1, cv2.LINE_AA)

            cat_sz = int(min(frame_w, frame_h) * 0.40)
            frame = overlay_png(frame, judge_cat_img, 30, frame_h - cat_sz - 30, cat_sz, cat_sz, fallback_label="LOOK AT SCREEN CAT")

        # 3. LOOKED BACK: HAPPY MEME CAT POPUP
        elif current_time < happy_cat_until:
            cv2.rectangle(frame, (15, 15), (frame_w - 15, 65), (0, 180, 0), -1)
            cv2.putText(frame, "GOOD! KEEP WATCHING!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

            happy_sz = int(min(frame_w, frame_h) * 0.35)
            frame = overlay_png(frame, happy_cat_img, 30, frame_h - happy_sz - 30, happy_sz, happy_sz, fallback_label="HAPPY CAT")

    # --------------------------------------------------------------------------
    # STEP H: HUD SYSTEM READOUTS
    # --------------------------------------------------------------------------
    prank_status_text = "PRANK MODE: ACTIVE" if prank_active else "PRANK MODE: OFF"
    prank_status_color = (0, 255, 0) if prank_active else (140, 140, 140)
    cv2.putText(frame, f"{prank_status_text} (Press 'r' to toggle)", (20, frame_h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, prank_status_color, 1, cv2.LINE_AA)

    gaze_label = "LOOKING AT SCREEN (PLAYING)" if raw_looking_at_screen else "LOOKING AWAY (PAUSED)"
    gaze_color = (0, 255, 0) if raw_looking_at_screen else (0, 0, 255)
    cv2.putText(frame, f"Gaze: {gaze_label}", (20, frame_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.46, gaze_color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"Scroll: {scroll_status_text}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.50, scroll_status_color, 1, cv2.LINE_AA)

    # Rotate the complete preview while keeping the punishment banner readable.
    display_frame = cv2.rotate(frame, cv2.ROTATE_180) if fullscreen_inversion else frame
    if fullscreen_inversion:
        banner_text = "FULLSCREEN PRIVILEGES REVOKED. ENJOY UPSIDE DOWN."
        banner_scale = 0.62
        banner_thickness = 2
        text_size = cv2.getTextSize(
            banner_text, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thickness
        )[0]
        banner_x = max(10, (frame_w - text_size[0]) // 2)
        cv2.rectangle(display_frame, (10, 12), (frame_w - 10, 68), (20, 20, 190), -1)
        cv2.putText(
            display_frame, banner_text, (banner_x, 50), cv2.FONT_HERSHEY_SIMPLEX,
            banner_scale, (255, 255, 255), banner_thickness, cv2.LINE_AA
        )

    # Show live webcam debug feed window
    cv2.imshow(window_name, display_frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("[*] Exit requested by user. Terminating tracking...")
        break
    elif key == 27:
        update_fullscreen_inversion(False)
    elif key == ord('r') or key == ord('R'):
        prank_active = not prank_active
        print(f"[*] YOUTUBE PRANK MODE TOGGLED: {'ACTIVE' if prank_active else 'OFF'}")

# ------------------------------------------------------------------------------
# 10. CLEANUP & RELEASE RESOURCES
# ------------------------------------------------------------------------------
update_fullscreen_inversion(False)
cap.release()
cv2.destroyAllWindows()
pipeline.close()
print("[*] Cleanup complete. Nose Cursor stopped successfully.")
