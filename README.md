<img width="1280" height="640" alt="git (1)" src="https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd" />

# LookAway

## Basic Details
### Team Name:Vrindha P's Team

### Team Members
- **Team Lead:** Vrindha P - Government Engineering College Thrissur
---

## Project Description
**LookAway** is an interactive computer-vision system built with MediaPipe and OpenCV that replaces traditional mouse navigation with nose tracking and mouth-open clicking, while doubling as a chaotic "ragebait" attention enforcer for YouTube.
---

## The Problem (that doesn't exist)
It is actually creating the problem

---


## Technical Details

### Technologies/Components Used

#### For Software:
- **Languages used:** Python 3.10+
- **Frameworks used:** Google MediaPipe (Face Landmarker & Hand Landmarker Solutions)
- **Libraries used:**
  - `opencv-python` (`cv2`) — Real-time camera feed capture, HUD rendering, and meme overlays
  - `pyautogui` — Programmatic OS cursor positioning and hotkey dispatching
  - `numpy` — Dynamic landmark normalization and smoothing filters
  - `ctypes` (`user32.dll`) — Native Windows display rotation and active window detection
  - `threading` — Asynchronous non-blocking background workers ensuring smooth 30+ FPS
- **Tools used:**
  - VS Code — Development and debugging
  - Git & GitHub — Version control and repository hosting
  - PyInstaller — Standalone executable packaging

---

## Implementation

### For Software:

#### Installation
```bash
# Clone the repository
git clone https://github.com/VRINDHAP/useless_project_temp.git
cd useless_project_temp
cd Project

# Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install opencv-python mediapipe pyautogui numpy
```

#### Run
```bash
# Start the LookAway system
python main.py
```

#### Key Controls
| Key | Action |
|---|---|
| `M` | Toggle Mouth-Click Control |
| `I` | Toggle Inverse-Attention Trap (Pause when Looking, Play when Looking Away) |
| `A` | Toggle YouTube Ad Mode (Resume when Looking, Pause/Rewind when Looking Away) |
| `F` / `Esc` | Toggle or Exit Fullscreen Display Inversion |
| `R` | Resync / Recenter Cursor Tracking |
| `Q` | Quit LookAway |

---

## Project Documentation

### For Software:

#### Screenshots
![Screenshot 1](Project/assets/Screenshot%202026-09-12%20070315.png)  

**Image Path:** [Project/assets/Screenshot 2026-09-12 070315.png`](Project/assets/Screenshot%202026-09-12%20070315.png)

![Screenshot 2](Project/assets/Screenshot%202026-09-12%20070333.png)  

**Image Path:** [Project/assets/Screenshot 2026-09-12 070333.png`](Project/assets/Screenshot%202026-09-12%20070333.png)

![Screenshot 3](Project/assets/Screenshot%202026-09-12%20070352.png)  

**Image Path:** [Project/assets/Screenshot 2026-09-12 070352.png`](Project/assets/Screenshot%202026-09-12%20070352.png)

#### Diagrams
```mermaid
flowchart TD
    A[Webcam Feed (30+ FPS)] --> B[MediaPipe Face Landmarker]
    B --> C[Nose Tip Tracking]
    B --> D[Mouth Open Ratio Calculation]
    B --> E[Gaze & Attention Direction]
    
    C --> F[PyAutoGUI Desktop Cursor Movement]
    D --> G{Mouth > Threshold?}
    G -- Yes --> H[Simulate Left Click]
    G -- No --> I[Idle / Re-arm]
    
    E --> J{Active Window == YouTube?}
    J -- Fullscreen Detected --> K[ctypes Win32 180° Display Rotation]
    J -- Looking Away > 1.0s --> L[Ad Rewind Tax: Rewind 10s or Reset to 0:00]
    J -- Look Away in Video --> M[Auto Pause Playback 'k']
```
*Workflow diagram illustrating landmark extraction, gesture handling, display inversion, and YouTube attention enforcement.*

---

## Project Demo

### Video
### 🎥 Project Demo

[▶️ Watch Demo Video](https://youtu.be/gZrAWg5hlnY?si=7VovAM4bwYkVu-zM) 


Made with ❤️ at TinkerHub Useless Projects

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--26-26?link=https%3A%2F%2Ftinkerhub.org%2Fevents%2F1M8ORET9A1%2Fuseless-projects-3.0)
