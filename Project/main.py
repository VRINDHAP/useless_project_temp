# ==============================================================================
# PIP INSTALLATION COMMAND:
# pip install opencv-python mediapipe pyautogui numpy
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

# Fullscreen & Inversion Trap Global States
fullscreen_inversion = False
original_display_orientation = None
_display_lock = threading.Lock()

# ------------------------------------------------------------------------------
# WINDOW, YOUTUBE & DISPLAY ROTATION HELPERS
# ------------------------------------------------------------------------------
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


def is_youtube_active():
    """Detects if YouTube is open/active in the foreground browser automatically."""
    title, window_class, rect = _get_foreground_window_details()
    title_lower = title.lower()
    return "youtube" in title_lower or "youtu.be" in title_lower


def is_youtube_fullscreen(current_window_name="Nose Cursor"):
    """Detects if YouTube has entered a fullscreen window state."""
    title, window_class, rect = _get_foreground_window_details()
    if rect is None:
        return False
    if current_window_name and title.lower().startswith(current_window_name.lower()):
        return False

    fills_screen = (
        rect.left <= 0 and rect.top <= 0 and
        rect.right >= SCREEN_WIDTH and rect.bottom >= SCREEN_HEIGHT
    )
    browser_window = window_class in {
        "Chrome_WidgetWin_1",
        "MozillaWindowClass",
        "ApplicationFrameWindow",
    } or "chrome" in window_class.lower() or "firefox" in window_class.lower() or "edge" in window_class.lower()
    
    return fills_screen and (browser_window or "youtube" in title.lower()) and ("youtube" in title.lower() or "youtu.be" in title.lower())


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
    """Attempts a 180-degree primary-display rotation on Windows safely."""
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

        # Rotate 180 degrees (add 2 mod 4)
        target_orientation = ((original_display_orientation + 2) % 4) if inverted else original_display_orientation
        mode.dmDisplayOrientation = target_orientation
        mode.dmFields = 0x00000080  # DM_DISPLAYORIENTATION
        result = user32.ChangeDisplaySettingsW(ctypes.byref(mode), 0)
        return result == 0
    except Exception as error:
        print(f"[!] Display rotation error: {error}")
        return False


def update_fullscreen_inversion(active):
    """
    Applies display rotation in a background daemon thread so the OpenCV
    camera loop never freezes or drops below 30 FPS.
    """
    global fullscreen_inversion
    if active == fullscreen_inversion:
        return

    fullscreen_inversion = active

    def _async_rotate():
        with _display_lock:
            rotated = set_display_inverted(active)
            if active:
                print(f"[!] FULLSCREEN INVERSION ACTIVATED: Display flipped 180° ({'Hardware OK' if rotated else 'Software active'})")
            else:
                print("[+] FULLSCREEN EXITED: Display restored to normal right-side up.")

    threading.Thread(target=_async_rotate, daemon=True).start()

# ------------------------------------------------------------------------------
# 2. TUNING PARAMETERS & THRESHOLDS
# ------------------------------------------------------------------------------
# --- ADAPTIVE NOSE CURSOR TRACKING (Full Screen, All 4 Corners & Bottommost Region) ---
ACTIVE_X_MIN, ACTIVE_X_MAX = 0.35, 0.65  # Natural horizontal tilt reaches left and right edges easily
ACTIVE_Y_MIN, ACTIVE_Y_MAX = 0.36, 0.56  # Natural vertical tilt reaches top and bottommost taskbar easily

# --- MOUTH-OPEN CLICK GESTURE (Normalized by Face Height for Yaw Invariance) ---
mouth_click_enabled = True      # Enabled by default (Toggle with 'm')
MOUTH_OPEN_THRESHOLD = 0.14     # Deliberate open-mouth threshold normalized by face height
CLICK_COOLDOWN_SECONDS = 1.0    # Cooldown between clicks
CLICK_FLASH_DURATION = 0.30     # Visual flash duration

# --- 2-SPEED VIRTUAL BUTTON SCROLLER (RIGHT SIDE OF CAMERA) ---
# (UP Buttons Above / Top, DOWN Buttons Below / Bottom)
BUTTON_X_MIN, BUTTON_X_MAX = 0.68, 0.98
FAST_UP_Y_MIN, FAST_UP_Y_MAX     = 0.08, 0.24   # "FAST UP" (Top)
SLOW_UP_Y_MIN, SLOW_UP_Y_MAX     = 0.26, 0.42   # "SLOW UP" (Upper)
SLOW_DOWN_Y_MIN, SLOW_DOWN_Y_MAX = 0.58, 0.74   # "SLOW DOWN" (Lower)
FAST_DOWN_Y_MIN, FAST_DOWN_Y_MAX = 0.76, 0.92   # "FAST DOWN" (Bottom)

SLOW_SCROLL_SPEED = 50          # Smooth reading scroll delta
FAST_SCROLL_SPEED = 140         # Turbo scroll delta
SCROLL_INTERVAL = 0.04          # Fluid 25Hz scroll tick interval

# --- INVERSE ATTENTION TRAP & GAZE BUFFER SETTINGS ---
inverse_attention_trap = True   # Enabled by default (Toggle with 'i' / 'I')
GAZE_BUFFER_SECONDS = 0.50      # 0.5-second buffer so normal eye blinks do not spam 'k'
GAZE_TOGGLE_COOLDOWN = 0.40     # 400ms cooldown between play/pause toggles

# --- YOUTUBE AD REWIND TAX CONFIGURATION ---
ad_mode_active = False          # Toggle with 'a' / 'A'
ad_strikes = 0
ad_not_looking_start = 0.0
last_ad_penalty_time = 0.0
ad_penalty_banner_text = ""
ad_penalty_banner_until = 0.0
AD_LOOKAWAY_THRESHOLD_SEC = 1.0
AD_PENALTY_COOLDOWN_SEC = 2.0

# ------------------------------------------------------------------------------
# 3. GLOBAL CLEAN MEDIA & VIDEO CONTROLLER
# ------------------------------------------------------------------------------
def send_hardware_scroll(clicks):
    """
    Dispatches ultra-smooth direct hardware mouse wheel scroll events.
    Positive value scrolls UP (page content moves down, viewport up).
    Negative value scrolls DOWN (page content moves up, viewport down).
    """
    if sys.platform == "win32":
        try:
            # Win32 MOUSEEVENTF_WHEEL = 0x0800
            ctypes.windll.user32.mouse_event(0x0800, 0, 0, int(clicks), 0)
            return
        except Exception:
            pass
    try:
        pyautogui.scroll(int(clicks))
    except Exception:
        pass


def send_youtube_play_pause():
    """
    Sends clean play/pause events directly to YouTube without clicking anything.
    Uses 'k' (standard universal YouTube shortcut) in a background thread to maintain >30 FPS.
    """
    def _send():
        try:
            pyautogui.press('k')
        except Exception:
            try:
                if sys.platform == "win32":
                    VK_K = 0x4B
                    scan = ctypes.windll.user32.MapVirtualKeyW(VK_K, 0)
                    ctypes.windll.user32.keybd_event(VK_K, scan, 0, 0)
                    time.sleep(0.02)
                    ctypes.windll.user32.keybd_event(VK_K, scan, 2, 0)
            except Exception:
                pass
    threading.Thread(target=_send, daemon=True).start()

# ------------------------------------------------------------------------------
# 4. PATHS & INITIALIZATION
# ------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(__file__)

# ------------------------------------------------------------------------------
# 5. AD REWIND TAX EXECUTOR
# ------------------------------------------------------------------------------
def trigger_ad_penalty(current_time):
    """
    The Ad Rewind Tax:
    - Strike 1 & 2: Inattention > 1s rewinds ad backward 5-10s ('left' twice).
    - Strike 3: Total reset to 0:00 ('0' / 'home') and resets strike count.
    """
    global ad_strikes, last_ad_penalty_time, ad_penalty_banner_text, ad_penalty_banner_until
    last_ad_penalty_time = current_time
    ad_strikes += 1

    if ad_strikes < 3:
        ad_penalty_banner_text = f"ATTENTION BREACH: REWINDING SPONSOR MESSAGE (+5s) [STRIKES: {ad_strikes}/3]"
        ad_penalty_banner_until = current_time + 3.0

        def _rewind_ad():
            try:
                pyautogui.press('left', presses=2, interval=0.04)
            except Exception as e:
                print(f"[!] Ad rewind error: {e}")

        threading.Thread(target=_rewind_ad, daemon=True).start()
        print(f"[!] AD PENALTY (Strike {ad_strikes}/3): Inattention detected! Rewound ad backward by 5-10s.")
    else:
        ad_penalty_banner_text = "STRIKE 3! AD FULLY RESTARTED FOR DISRESPECT."
        ad_penalty_banner_until = current_time + 3.5
        ad_strikes = 0

        def _restart_ad():
            try:
                pyautogui.press('0')
                pyautogui.press('home')
            except Exception as e:
                print(f"[!] Ad restart error: {e}")

        threading.Thread(target=_restart_ad, daemon=True).start()
        print("[!] AD PENALTY (STRIKE 3 TOTAL RESET): Ad fully restarted to 0:00!")

# ------------------------------------------------------------------------------
# 6. STATE MEMORY VARIABLES
# ------------------------------------------------------------------------------
smoothed_x = SCREEN_WIDTH / 2
smoothed_y = SCREEN_HEIGHT / 2
last_known_x = smoothed_x
last_known_y = smoothed_y

# Click & Scroll states
last_click_time = 0.0
click_flash_until = 0.0
last_scroll_time = 0.0
smoothed_tip_x = None
smoothed_tip_y = None

# Automatic Gaze & Buffer states
video_state = "INITIAL"
last_gaze_toggle_time = 0.0
was_youtube_active = False

# 0.5-second Gaze Buffer states
gaze_buffered_state = "AWAY"    # "LOOKING" or "AWAY"
gaze_candidate_state = None     # candidate state undergoing 0.5s debounce
gaze_candidate_start = 0.0      # start timestamp of candidate state

# ------------------------------------------------------------------------------
# 7. INITIALIZE MEDIAPIPE FACE MESH & HAND DETECTORS
# ------------------------------------------------------------------------------
print("=" * 65)
print("  [*] NOSE CURSOR + 2-SPEED SCROLLER + INVERSE ATTENTION + INVERSION")
print("=" * 65)
print(f"[*] Screen Resolution:       {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
print(f"[*] Click Using Mouth:        {'ENABLED' if mouth_click_enabled else 'DISABLED'} (Press 'm' to toggle)")
print(f"[*] YouTube Ad Mode:          {'ACTIVE' if ad_mode_active else 'OFF'} (Press 'a' to toggle)")
print(f"[*] Inverse Attention Trap:   {'ON' if inverse_attention_trap else 'OFF'} (Press 'i' to toggle)")
print(f"[*] Fullscreen Inversion Trap: ACTIVE ON FULLSCREEN (Press 'f' / 'Esc' to exit)")
print("[*] Controls: Move nose to steer, open mouth to click, hover right boxes to scroll.")
print("[*] Hotkeys: [M: Mouth Click | A: Ad Mode | I: Inverse Attention | F/Esc: Fullscreen | R: Resync | Q: Exit]")
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
                    'outer_upper_lip': (lms[0].x, lms[0].y),
                    'outer_lower_lip': (lms[17].x, lms[17].y),
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
                    'outer_upper_lip': (lms[0].x, lms[0].y),
                    'outer_lower_lip': (lms[17].x, lms[17].y),
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
# 8. INITIALIZE WEBCAM
# ------------------------------------------------------------------------------
print("[*] Connecting to webcam...")

def open_camera():
    backends = [(0, cv2.CAP_DSHOW), (0, cv2.CAP_ANY), (1, cv2.CAP_DSHOW), (1, cv2.CAP_ANY)]
    for index, backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"[+] Webcam detected and opened successfully on camera index {index} (640x480 @ 30FPS)!")
                return cap
            cap.release()
    return None

cap = open_camera()

if cap is None:
    print("[!] ERROR: Could not access any webcam.")
    sys.exit(1)

window_name = "Nose Cursor - YouTube Smart Assistant"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 720, 540)

# ------------------------------------------------------------------------------
# 9. HELPER MATH FUNCTIONS
# ------------------------------------------------------------------------------
def euclidean_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# ------------------------------------------------------------------------------
# 10. MAIN CONTINUOUS REAL-TIME TRACKING LOOP
# ------------------------------------------------------------------------------
last_yt_check_time = 0.0
youtube_active_now = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("[!] Warning: Empty or unreadable frame from webcam.")
        continue

    current_time = time.time()

    # Throttled YouTube Active & Fullscreen Check (Runs every 0.10s for instant response)
    if current_time - last_yt_check_time >= 0.10:
        youtube_active_now = is_youtube_active()
        if youtube_active_now and not was_youtube_active:
            video_state = "INITIAL"
            gaze_candidate_state = None
        was_youtube_active = youtube_active_now

        # Auto-detect fullscreen vs normal/closed/back state
        yt_is_fullscreen = youtube_active_now and is_youtube_fullscreen(window_name)
        if yt_is_fullscreen:
            if not fullscreen_inversion:
                update_fullscreen_inversion(True)
        else:
            # When back is clicked, video closed, or fullscreen exited -> return to straight normal
            if fullscreen_inversion:
                update_fullscreen_inversion(False)

        last_yt_check_time = current_time

    # Instant Global Escape Key Check (Reverts to normal immediately if pressed anywhere)
    if sys.platform == "win32" and fullscreen_inversion:
        if (ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000) != 0:
            update_fullscreen_inversion(False)

    # STEP A: Flip frame horizontally for intuitive mirror-like navigation
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    # STEP B: Convert BGR to RGB for MediaPipe inference
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run Face & Hand detections concurrently
    face_found, face_landmarks = pipeline.process_face(rgb_frame)
    hand_found, index_tip, all_hand_lms = pipeline.process_hand(rgb_frame)

    raw_looking_at_screen = False
    mouth_open_ratio = 0.0

    # Normalized camera coordinates of the nose dot
    norm_cam_x, norm_cam_y = 0.5, 0.5

    # --------------------------------------------------------------------------
    # STEP C: GAZE ORIENTATION & POSE CHECK
    # --------------------------------------------------------------------------
    if face_found:
        nose_norm = face_landmarks['nose']
        norm_cam_x, norm_cam_y = nose_norm

        # Distances for Orientation & Scale
        forehead_to_chin = euclidean_distance(face_landmarks['forehead'], face_landmarks['chin'])
        forehead_to_nose = euclidean_distance(face_landmarks['forehead'], face_landmarks['nose'])
        nose_to_chin = euclidean_distance(face_landmarks['nose'], face_landmarks['chin'])
        nose_to_left_eye = euclidean_distance(face_landmarks['nose'], face_landmarks['left_eye'])
        nose_to_right_eye = euclidean_distance(face_landmarks['nose'], face_landmarks['right_eye'])

        pitch_ratio = forehead_to_nose / max(nose_to_chin, 1e-6)
        yaw_symmetry = min(nose_to_left_eye, nose_to_right_eye) / max(max(nose_to_left_eye, nose_to_right_eye), 1e-6)

        # STRICT LOOKING FORWARD AT SCREEN CHECK:
        # Looking straight ahead / down at taskbar: yaw_symmetry >= 0.68, pitch 0.55 to 1.90
        # If user looks away/right/outside: yaw_symmetry drops significantly
        if (0.55 <= pitch_ratio <= 1.90) and (yaw_symmetry >= 0.68):
            raw_looking_at_screen = True
        else:
            raw_looking_at_screen = False

        # Calculate mouth opening normalized by face height (invariant to head yaw rotation)
        lip_inner_dist = euclidean_distance(face_landmarks['upper_lip'], face_landmarks['lower_lip'])
        mouth_open_ratio = lip_inner_dist / max(forehead_to_chin, 1e-6)

        # ----------------------------------------------------------------------
        # CURSOR STEERING (NORMAL VS INVERTED DIRECTION TRAP)
        # ----------------------------------------------------------------------
        if raw_looking_at_screen:
            if fullscreen_inversion:
                # FEATURE 1 PUNISHMENT: Inverted nose-cursor controls
                # Moving head UP moves cursor DOWN, moving LEFT moves it RIGHT
                raw_screen_x = np.interp(norm_cam_x, [ACTIVE_X_MIN, ACTIVE_X_MAX], [SCREEN_WIDTH - 1, 0])
                raw_screen_y = np.interp(norm_cam_y, [ACTIVE_Y_MIN, ACTIVE_Y_MAX], [SCREEN_HEIGHT - 1, 0])
            else:
                # Standard intuitive navigation: All 4 corners
                raw_screen_x = np.interp(norm_cam_x, [ACTIVE_X_MIN, ACTIVE_X_MAX], [0, SCREEN_WIDTH - 1])
                raw_screen_y = np.interp(norm_cam_y, [ACTIVE_Y_MIN, ACTIVE_Y_MAX], [0, SCREEN_HEIGHT - 1])

            raw_screen_x = np.clip(raw_screen_x, 0, SCREEN_WIDTH - 1)
            raw_screen_y = np.clip(raw_screen_y, 0, SCREEN_HEIGHT - 1)

            # Continuous Exponential Velocity Filter (Smooth & Jitter-Free)
            delta_dist = np.sqrt((raw_screen_x - smoothed_x) ** 2 + (raw_screen_y - smoothed_y) ** 2)
            adaptive_alpha = 0.18 + 0.47 * (1.0 - np.exp(-delta_dist / 45.0))
            
            smoothed_x = (smoothed_x * (1.0 - adaptive_alpha)) + (raw_screen_x * adaptive_alpha)
            smoothed_y = (smoothed_y * (1.0 - adaptive_alpha)) + (raw_screen_y * adaptive_alpha)
            last_known_x, last_known_y = smoothed_x, smoothed_y

            # Move system mouse cursor with 0ms Win32 hardware direct call
            if sys.platform == "win32":
                ctypes.windll.user32.SetCursorPos(int(smoothed_x), int(smoothed_y))
            else:
                pyautogui.moveTo(int(smoothed_x), int(smoothed_y))

            # Mouth-Open Click Gesture
            can_click = (current_time - last_click_time) >= CLICK_COOLDOWN_SECONDS
            if mouth_click_enabled and (mouth_open_ratio >= MOUTH_OPEN_THRESHOLD) and can_click and (yaw_symmetry >= 0.78):
                pyautogui.click(int(smoothed_x), int(smoothed_y))
                last_click_time = current_time
                click_flash_until = current_time + CLICK_FLASH_DURATION
                print(f"[+] MOUTH CLICK TRIGGERED! (Ratio: {mouth_open_ratio:.2f})")

        # Visual feedback: Nose dot
        nose_pixel_x, nose_pixel_y = int(norm_cam_x * frame_w), int(norm_cam_y * frame_h)
        is_flash_active = current_time < click_flash_until
        dot_color = (0, 0, 255) if is_flash_active else (0, 255, 0)
        cv2.circle(frame, (nose_pixel_x, nose_pixel_y), 8, dot_color, -1)
        cv2.circle(frame, (nose_pixel_x, nose_pixel_y), 14, (0, 255, 255), 2)

    else:
        # Face not detected -> user is looking away / stepped away
        raw_looking_at_screen = False

    # --------------------------------------------------------------------------
    # STEP D: 0.5-SECOND BUFFERED GAZE STATE TRACKER & PLAYBACK CONTROLLER
    # --------------------------------------------------------------------------
    current_instant_gaze = "LOOKING" if raw_looking_at_screen else "AWAY"

    # Debounce 0.5s buffer to ignore normal eye blinks
    if current_instant_gaze != gaze_buffered_state:
        if gaze_candidate_state != current_instant_gaze:
            gaze_candidate_state = current_instant_gaze
            gaze_candidate_start = current_time
        elif (current_time - gaze_candidate_start) >= GAZE_BUFFER_SECONDS:
            gaze_buffered_state = current_instant_gaze
            gaze_candidate_state = None
    else:
        gaze_candidate_state = None

    # FEATURE 2: INVERSE ATTENTION TRAP (Look Away to Play)
    if ad_mode_active:
        # Penalty Trigger: Look away > 1.0s continuously during ad
        if not raw_looking_at_screen:
            if ad_not_looking_start == 0.0:
                ad_not_looking_start = current_time
            else:
                inattention_duration = current_time - ad_not_looking_start
                if inattention_duration >= AD_LOOKAWAY_THRESHOLD_SEC:
                    if (current_time - last_ad_penalty_time) >= AD_PENALTY_COOLDOWN_SEC:
                        trigger_ad_penalty(current_time)
                    ad_not_looking_start = 0.0
        else:
            ad_not_looking_start = 0.0

    elif inverse_attention_trap and (youtube_active_now or True):
        can_toggle_gaze = (current_time - last_gaze_toggle_time) >= GAZE_TOGGLE_COOLDOWN

        # Rule 1: Looking AWAY from screen -> The video MUST PLAY
        if gaze_buffered_state == "AWAY":
            if (video_state == "PAUSED" or video_state == "INITIAL") and can_toggle_gaze:
                send_youtube_play_pause()
                video_state = "PLAYING"
                last_gaze_toggle_time = current_time
                print("[+] INVERSE ATTENTION: Looking Away -> RESUMING PLAYBACK (GOOD)")

        # Rule 2: Looking DIRECTLY AT screen -> The video MUST PAUSE
        elif gaze_buffered_state == "LOOKING":
            if (video_state == "PLAYING" or video_state == "INITIAL") and can_toggle_gaze:
                send_youtube_play_pause()
                video_state = "PAUSED"
                last_gaze_toggle_time = current_time
                print("[!] INVERSE ATTENTION: Looking At Screen FORBIDDEN -> PAUSED!")

    # --------------------------------------------------------------------------
    # STEP E: 2-SPEED VIRTUAL BUTTON SCROLL ZONE SYSTEM (RIGHT SIDE)
    # --------------------------------------------------------------------------
    btn_x1, btn_x2 = int(BUTTON_X_MIN * frame_w), int(BUTTON_X_MAX * frame_w)
    fup_y1, fup_y2     = int(FAST_UP_Y_MIN * frame_h), int(FAST_UP_Y_MAX * frame_h)
    sup_y1, sup_y2     = int(SLOW_UP_Y_MIN * frame_h), int(SLOW_UP_Y_MAX * frame_h)
    sdown_y1, sdown_y2 = int(SLOW_DOWN_Y_MIN * frame_h), int(SLOW_DOWN_Y_MAX * frame_h)
    fdown_y1, fdown_y2 = int(FAST_DOWN_Y_MIN * frame_h), int(FAST_DOWN_Y_MAX * frame_h)

    is_fast_up = is_slow_up = is_slow_down = is_fast_down = False

    # Smooth fingertip tracking to eliminate landmark jitter
    if hand_found and index_tip is not None:
        raw_tx, raw_ty = index_tip
        if smoothed_tip_x is None or smoothed_tip_y is None:
            smoothed_tip_x, smoothed_tip_y = raw_tx, raw_ty
        else:
            tip_dist = np.hypot(raw_tx - smoothed_tip_x, raw_ty - smoothed_tip_y)
            tip_alpha = 0.35 + 0.45 * (1.0 - np.exp(-tip_dist * 20.0))
            smoothed_tip_x = (1.0 - tip_alpha) * smoothed_tip_x + tip_alpha * raw_tx
            smoothed_tip_y = (1.0 - tip_alpha) * smoothed_tip_y + tip_alpha * raw_ty

        tip_x, tip_y = smoothed_tip_x, smoothed_tip_y

        if BUTTON_X_MIN <= tip_x <= BUTTON_X_MAX:
            if FAST_UP_Y_MIN <= tip_y <= FAST_UP_Y_MAX:
                is_fast_up = True
            elif SLOW_UP_Y_MIN <= tip_y <= SLOW_UP_Y_MAX:
                is_slow_up = True
            elif SLOW_DOWN_Y_MIN <= tip_y <= SLOW_DOWN_Y_MAX:
                is_slow_down = True
            elif FAST_DOWN_Y_MIN <= tip_y <= FAST_DOWN_Y_MAX:
                is_fast_down = True
    else:
        smoothed_tip_x, smoothed_tip_y = None, None
        tip_x, tip_y = None, None

    # Continuous fluid scroll execution (UP at top, DOWN at bottom)
    if is_fast_up:
        if current_time - last_scroll_time >= SCROLL_INTERVAL:
            send_hardware_scroll(FAST_SCROLL_SPEED)
            last_scroll_time = current_time
        scroll_status_text, scroll_status_color = "FAST UP", (0, 255, 255)
    elif is_slow_up:
        if current_time - last_scroll_time >= SCROLL_INTERVAL:
            send_hardware_scroll(SLOW_SCROLL_SPEED)
            last_scroll_time = current_time
        scroll_status_text, scroll_status_color = "SLOW UP", (0, 255, 0)
    elif is_slow_down:
        if current_time - last_scroll_time >= SCROLL_INTERVAL:
            send_hardware_scroll(-SLOW_SCROLL_SPEED)
            last_scroll_time = current_time
        scroll_status_text, scroll_status_color = "SLOW DOWN", (0, 165, 255)
    elif is_fast_down:
        if current_time - last_scroll_time >= SCROLL_INTERVAL:
            send_hardware_scroll(-FAST_SCROLL_SPEED)
            last_scroll_time = current_time
        scroll_status_text, scroll_status_color = "FAST DOWN", (0, 80, 255)
    else:
        scroll_status_text, scroll_status_color = "IDLE", (180, 180, 180)

    # --------------------------------------------------------------------------
    # STEP F: RENDER MOUTH-CLICK & SCROLL BUTTON HUD
    # --------------------------------------------------------------------------
    hud_overlay = frame.copy()

    # 1. "CLICK USING YOUR MOUTH" Status Box (Top-Left)
    mc_bx1, mc_by1 = int(0.02 * frame_w), int(0.08 * frame_h)
    mc_bx2, mc_by2 = int(0.48 * frame_w), int(0.18 * frame_h)

    if mouth_click_enabled:
        cv2.rectangle(hud_overlay, (mc_bx1, mc_by1), (mc_bx2, mc_by2), (0, 150, 0), -1)
        cv2.rectangle(frame, (mc_bx1, mc_by1), (mc_bx2, mc_by2), (0, 255, 0), 2)
        cv2.putText(frame, "CLICK USING MOUTH: ON", (mc_bx1 + 10, mc_by1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 2, cv2.LINE_AA)

        # Live Real-time Mouth Meter
        meter_w = mc_bx2 - mc_bx1 - 20
        fill_w = int(np.clip((mouth_open_ratio / (MOUTH_OPEN_THRESHOLD * 1.5)), 0.0, 1.0) * meter_w)
        cv2.rectangle(frame, (mc_bx1 + 10, mc_by1 + 30), (mc_bx1 + 10 + meter_w, mc_by1 + 42), (50, 50, 50), -1)
        bar_color = (0, 255, 0) if mouth_open_ratio >= MOUTH_OPEN_THRESHOLD else (0, 200, 255)
        cv2.rectangle(frame, (mc_bx1 + 10, mc_by1 + 30), (mc_bx1 + 10 + fill_w, mc_by1 + 42), bar_color, -1)
        thresh_x = mc_bx1 + 10 + int((MOUTH_OPEN_THRESHOLD / (MOUTH_OPEN_THRESHOLD * 1.5)) * meter_w)
        cv2.line(frame, (thresh_x, mc_by1 + 28), (thresh_x, mc_by1 + 44), (0, 0, 255), 2)
    else:
        cv2.rectangle(hud_overlay, (mc_bx1, mc_by1), (mc_bx2, mc_by2), (35, 35, 35), -1)
        cv2.rectangle(frame, (mc_bx1, mc_by1), (mc_bx2, mc_by2), (90, 90, 90), 2)
        cv2.putText(frame, "CLICK USING MOUTH: OFF", (mc_bx1 + 10, (mc_by1 + mc_by2) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA)

    # 2. 4 Scroll Buttons (Right Side - Top-to-Bottom: FAST UP -> SLOW UP -> NEUTRAL -> SLOW DOWN -> FAST DOWN)
    def render_scroll_button(y1, y2, is_active, active_bg, idle_border=(255, 180, 0), idle_bg=(40, 25, 15)):
        if is_active:
            cv2.rectangle(hud_overlay, (btn_x1, y1), (btn_x2, y2), active_bg, -1)
            border_c = (255, 255, 255)
            thick = 3
            text_c = (0, 0, 0)
        else:
            cv2.rectangle(hud_overlay, (btn_x1, y1), (btn_x2, y2), idle_bg, -1)
            border_c = idle_border
            thick = 2
            text_c = (240, 240, 240)
        return border_c, thick, text_c

    fup_bc, fup_th, fup_tc       = render_scroll_button(fup_y1, fup_y2, is_fast_up, active_bg=(0, 230, 255), idle_border=(0, 200, 220), idle_bg=(35, 35, 15))
    sup_bc, sup_th, sup_tc       = render_scroll_button(sup_y1, sup_y2, is_slow_up, active_bg=(0, 230, 0), idle_border=(0, 180, 0), idle_bg=(15, 35, 15))
    sdown_bc, sdown_th, sdown_tc = render_scroll_button(sdown_y1, sdown_y2, is_slow_down, active_bg=(0, 165, 255), idle_border=(0, 140, 220), idle_bg=(15, 30, 45))
    fdown_bc, fdown_th, fdown_tc = render_scroll_button(fdown_y1, fdown_y2, is_fast_down, active_bg=(0, 80, 255), idle_border=(0, 80, 220), idle_bg=(20, 15, 45))

    cv2.addWeighted(hud_overlay, 0.45, frame, 0.55, 0, frame)

    # Render Button Borders and Clear Labels (UP at Top, DOWN at Bottom)
    cv2.rectangle(frame, (btn_x1, fup_y1), (btn_x2, fup_y2), fup_bc, fup_th)
    cv2.rectangle(frame, (btn_x1, sup_y1), (btn_x2, sup_y2), sup_bc, sup_th)
    cv2.rectangle(frame, (btn_x1, sdown_y1), (btn_x2, sdown_y2), sdown_bc, sdown_th)
    cv2.rectangle(frame, (btn_x1, fdown_y1), (btn_x2, fdown_y2), fdown_bc, fdown_th)

    cv2.putText(frame, "FAST UP", (btn_x1 + 12, (fup_y1 + fup_y2) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.44, fup_tc, 2, cv2.LINE_AA)
    cv2.putText(frame, "SLOW UP", (btn_x1 + 12, (sup_y1 + sup_y2) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.44, sup_tc, 2, cv2.LINE_AA)
    cv2.putText(frame, "[ NEUTRAL ]", (btn_x1 + 10, (sup_y2 + sdown_y1) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.putText(frame, "SLOW DOWN", (btn_x1 + 8, (sdown_y1 + sdown_y2) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, sdown_tc, 2, cv2.LINE_AA)
    cv2.putText(frame, "FAST DOWN", (btn_x1 + 8, (fdown_y1 + fdown_y2) // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, fdown_tc, 2, cv2.LINE_AA)

    # Hand Skeleton & Smoothed Pointer Marker
    if hand_found and tip_x is not None and tip_y is not None:
        tip_px_x, tip_px_y = int(tip_x * frame_w), int(tip_y * frame_h)
        for lm_x, lm_y in all_hand_lms:
            cv2.circle(frame, (int(lm_x * frame_w), int(lm_y * frame_h)), 2, (0, 180, 255), -1)
        pointer_active = (is_fast_up or is_slow_up or is_slow_down or is_fast_down)
        pointer_color = (0, 255, 0) if pointer_active else (0, 255, 255)
        cv2.circle(frame, (tip_px_x, tip_px_y), 9, pointer_color, -1)
        cv2.circle(frame, (tip_px_x, tip_px_y), 15, (255, 255, 255), 2)
        cv2.putText(frame, "POINTER", (tip_px_x + 15, tip_px_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # STEP G: RENDER WARNING BANNERS & INVERSE ATTENTION HUD
    # --------------------------------------------------------------------------
    # 1. AD REWIND TAX PENALTY BANNER (During Ad Mode)
    if ad_mode_active and current_time < ad_penalty_banner_until and ad_penalty_banner_text:
        banner_h = 68
        banner_y1 = 44
        banner_y2 = banner_y1 + banner_h

        overlay_banner = frame.copy()
        cv2.rectangle(overlay_banner, (10, banner_y1), (frame_w - 10, banner_y2), (0, 0, 220), -1)
        cv2.addWeighted(overlay_banner, 0.90, frame, 0.10, 0, frame)
        cv2.rectangle(frame, (10, banner_y1), (frame_w - 10, banner_y2), (0, 255, 255), 2)

        font_scale = 0.46 if len(ad_penalty_banner_text) > 42 else 0.54
        text_size = cv2.getTextSize(ad_penalty_banner_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = max(16, (frame_w - text_size[0]) // 2)
        cv2.putText(frame, ad_penalty_banner_text, (text_x, banner_y1 + 42), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    # 2. INVERSE ATTENTION TRAP HUD STATUS (GREEN / RED)
    if inverse_attention_trap and not ad_mode_active:
        if gaze_buffered_state == "AWAY":
            hud_gaze_text = "LOOKING AWAY: PLAYING (GOOD)"
            hud_gaze_bg = (0, 140, 0)
            hud_gaze_border = (0, 255, 0)
            hud_text_color = (255, 255, 255)
        else:
            hud_gaze_text = "LOOKING AT SCREEN FORBIDDEN! PAUSED"
            hud_gaze_bg = (0, 0, 180)
            hud_gaze_border = (0, 0, 255)
            hud_text_color = (255, 255, 255)

        banner_y1 = 40
        banner_y2 = 72
        overlay_gaze = frame.copy()
        cv2.rectangle(overlay_gaze, (10, banner_y1), (frame_w - 10, banner_y2), hud_gaze_bg, -1)
        cv2.addWeighted(overlay_gaze, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (10, banner_y1), (frame_w - 10, banner_y2), hud_gaze_border, 2)

        ts = cv2.getTextSize(hud_gaze_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)[0]
        tx = max(15, (frame_w - ts[0]) // 2)
        cv2.putText(frame, hud_gaze_text, (tx, banner_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, hud_text_color, 2, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # STEP H: TOP STATUS BAR & HUD READOUTS
    # --------------------------------------------------------------------------
    top_bar_h = 36
    overlay_top = frame.copy()
    if fullscreen_inversion:
        cv2.rectangle(overlay_top, (0, 0), (frame_w, top_bar_h), (0, 0, 180), -1)
        status_title = "FULLSCREEN TRAP: UPSIDE DOWN (Press 'F'/'Esc' to reset)"
        status_color = (0, 255, 255)
    elif ad_mode_active:
        cv2.rectangle(overlay_top, (0, 0), (frame_w, top_bar_h), (0, 80, 180), -1)
        status_title = f"AD MODE: ACTIVE | STRIKES: {ad_strikes}/3"
        status_color = (0, 255, 255)
    elif inverse_attention_trap:
        cv2.rectangle(overlay_top, (0, 0), (frame_w, top_bar_h), (0, 100, 150), -1)
        status_title = f"INVERSE ATTENTION: ON ({video_state})"
        status_color = (0, 255, 255)
    elif is_youtube_active():
        cv2.rectangle(overlay_top, (0, 0), (frame_w, top_bar_h), (0, 0, 150), -1)
        status_title = f"YOUTUBE: ACTIVE ({video_state})"
        status_color = (0, 255, 255)
    else:
        cv2.rectangle(overlay_top, (0, 0), (frame_w, top_bar_h), (35, 35, 35), -1)
        status_title = "SYSTEM: READY"
        status_color = (180, 180, 180)

    cv2.addWeighted(overlay_top, 0.70, frame, 0.30, 0, frame)
    cv2.putText(frame, status_title, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "[M: Mouth | I: Inverse | A: Ad | F: Flip]", (frame_w - 320, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

    # Bottom status readouts
    gaze_label = "LOOKING AT SCREEN (PAUSED)" if gaze_buffered_state == "LOOKING" else "LOOKING AWAY: (PLAYING)"
    gaze_color = (0, 0, 255) if gaze_buffered_state == "LOOKING" else (0, 255, 0)
    cv2.putText(frame, f"Gaze: {gaze_label}", (20, frame_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, gaze_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Scroll: {scroll_status_text}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, scroll_status_color, 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # STEP I: WEBCAM PREVIEW INVERSION & WARNING BANNER
    # --------------------------------------------------------------------------
    # FEATURE 1: Flip preview upside down during fullscreen punishment
    display_frame = cv2.rotate(frame, cv2.ROTATE_180) if fullscreen_inversion else frame

    if fullscreen_inversion:
        banner_text = "FULLSCREEN PRIVILEGES REVOKED. ENJOY UPSIDE DOWN."
        banner_scale = 0.52
        banner_thickness = 2
        text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thickness)[0]
        banner_x = max(10, (frame_w - text_size[0]) // 2)
        cv2.rectangle(display_frame, (10, 8), (frame_w - 10, 58), (0, 0, 220), -1)
        cv2.rectangle(display_frame, (10, 8), (frame_w - 10, 58), (0, 255, 255), 2)
        cv2.putText(
            display_frame, banner_text, (banner_x, 40), cv2.FONT_HERSHEY_SIMPLEX,
            banner_scale, (255, 255, 255), banner_thickness, cv2.LINE_AA
        )

    # Show live webcam feed window
    cv2.imshow(window_name, display_frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        print("[*] Exit requested by user. Terminating tracking...")
        break
    elif key == ord('m') or key == ord('M'):
        mouth_click_enabled = not mouth_click_enabled
        print(f"[*] MOUTH-OPEN CLICK GESTURE: {'ENABLED' if mouth_click_enabled else 'DISABLED'}")
    elif key == ord('i') or key == ord('I'):
        inverse_attention_trap = not inverse_attention_trap
        print(f"[*] INVERSE ATTENTION TRAP (Look Away to Play): {'ON' if inverse_attention_trap else 'OFF'}")
    elif key == ord('f') or key == ord('F'):
        # Toggle Fullscreen Inversion Trap
        update_fullscreen_inversion(not fullscreen_inversion)
    elif key == 27:  # Escape key
        if fullscreen_inversion:
            update_fullscreen_inversion(False)
    elif key == ord('a') or key == ord('A'):
        ad_mode_active = not ad_mode_active
        if not ad_mode_active:
            ad_strikes = 0
            ad_not_looking_start = 0.0
            ad_penalty_banner_text = ""
        print(f"[*] YOUTUBE AD REWIND TAX MODE TOGGLED: {'ACTIVE' if ad_mode_active else 'OFF'}")
    elif key == ord('r') or key == ord('R'):
        send_youtube_play_pause()
        video_state = "PLAYING" if video_state == "PAUSED" else "PAUSED"
        last_gaze_toggle_time = current_time
        print(f"[*] MANUAL RESYNC: Toggled playback state to {video_state}")

# ------------------------------------------------------------------------------
# 11. CLEANUP & RELEASE RESOURCES
# ------------------------------------------------------------------------------
update_fullscreen_inversion(False)
cap.release()
cv2.destroyAllWindows()
pipeline.close()
print("[*] Cleanup complete. Nose Cursor stopped successfully.")
