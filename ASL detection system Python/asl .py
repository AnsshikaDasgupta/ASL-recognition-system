"""
ASL Fingerspelling Recognizer  (webcam -> letters -> words -> sentence -> speech)

Setup:
    pip install opencv-python mediapipe numpy pyttsx3
    Put 'hand_landmarker.task' in the same folder as this script
    (https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

Controls:
    Hold a sign 0.8s  -> types the letter        Hold to 2.0s -> doubles it
    Hide your hand 1.2s -> space (ends the word)
    J and Z           -> draw them in the air (typed instantly)
    Enter = read sentence aloud    Space = space    Backspace = delete
    C = clear     T = training mode     D = debug values     Q / Esc = quit

Training mode (improves accuracy a lot):
    Press T, show a sign, HOLD its letter key to record samples (20-30 is plenty).
    Backspace wipes the last letter you recorded. Esc leaves training mode.
    J and Z are motion letters and are detected automatically.
"""

import os
import sys
import json
import time
import threading
import traceback
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "hand_landmarker.task")
SAMPLES_PATH = os.path.join(BASE, "asl_samples.json")

CAM_INDEX = 0
CAM_W, CAM_H = 640, 480
SIDEBAR_W, BOTTOM_H = 320, 160
W, H = CAM_W + SIDEBAR_W, CAM_H + BOTTOM_H

TIME_TO_CONFIRM = 0.8     # hold to type
TIME_TO_DOUBLE = 2.0      # keep holding to double
SPACE_TIME = 1.2          # hand hidden this long -> space
MOTION_COOLDOWN = 1.0     # pause after J / Z

# Finger "extended" thresholds: (tip-to-wrist) / (knuckle-to-wrist) for index, middle, ring, pinky
EXT_T = [1.70, 1.70, 1.65, 1.55]

KNN_MAX_DIST = 1.5        # max distance for a trained sample to count as a match
KNN_MAX_PER_LETTER = 300

# Colours (BGR)
BG = (28, 24, 22)
PANEL = (40, 35, 33)
CARD = (58, 52, 49)
TEXT = (240, 240, 240)
MUTED = (160, 155, 150)
ACCENT = (120, 210, 80)
ORANGE = (60, 160, 255)
BLUE = (255, 160, 60)
RED = (90, 90, 230)
FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = "ASL Fingerspelling"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
    (18, 19), (19, 20), (0, 17),
]
TIPS = [8, 12, 16, 20]
MCPS = [5, 9, 13, 17]


# ----------------------------------------------------------------------------
# Camera thread (always keeps only the newest frame -> no lag build-up)
# ----------------------------------------------------------------------------
class Camera:
    def __init__(self, index):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.failures = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                self.failures = 0
                with self.lock:
                    self.frame = f
            else:
                self.failures += 1
                time.sleep(0.02)

    def read(self):
        with self.lock:
            f, self.frame = self.frame, None
        return f

    def release(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


# ----------------------------------------------------------------------------
# Text to speech (own thread, fresh engine each time -> never blocks video)
# ----------------------------------------------------------------------------
_tts_lock = threading.Lock()


def speak(text):
    if pyttsx3 is None:
        print("pyttsx3 is not installed - run: pip install pyttsx3")
        return

    def run():
        with _tts_lock:
            try:
                try:
                    import pythoncom  # Windows only
                    pythoncom.CoInitialize()
                except Exception:
                    pass
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print("Speech error:", e)

    threading.Thread(target=run, daemon=True).start()


# ----------------------------------------------------------------------------
# Hand geometry
# ----------------------------------------------------------------------------
def to_points(landmarks, w, h, label):
    """Pixel-space landmarks. Left hands are mirrored so one rule set fits both."""
    pts = np.array([[l.x * w, l.y * h, l.z * w] for l in landmarks], dtype=np.float32)
    if label == "Left":
        pts[:, 0] = w - pts[:, 0]
    return pts


def palm_size(pts):
    return float(np.linalg.norm(pts[9, :2] - pts[0, :2])) + 1e-6


def finger_ratios(pts):
    wrist = pts[0, :2]
    out = []
    for tip, mcp in zip(TIPS, MCPS):
        out.append(float(np.linalg.norm(pts[tip, :2] - wrist) /
                         (np.linalg.norm(pts[mcp, :2] - wrist) + 1e-6)))
    return out


def pointing(pts, a, b):
    v = pts[b, :2] - pts[a, :2]
    if abs(v[0]) > abs(v[1]) * 1.2:
        return "side"
    return "up" if v[1] < 0 else "down"


def feature_vector(pts):
    p = pts - pts[0]
    p = p / palm_size(pts)
    p[:, 2] *= 0.5
    return p.flatten().astype(np.float32)


def rule_letter(pts, ratios, ext):
    """Geometric rules. Returns a static letter or None."""
    P = palm_size(pts)

    def d(a, b):
        return float(np.linalg.norm(pts[a, :2] - pts[b, :2])) / P

    idx, mid, ring, pnk = ext
    thumb_idx = d(4, 8)
    thumb_mcp = d(4, 5)
    direction = pointing(pts, 5, 8)
    mean_r = sum(ratios) / 4.0

    # --- only index extended: D, G, Q, L
    if idx and not (mid or ring or pnk):
        if direction == "down":
            return "Q"
        if direction == "side":
            return "G"
        if thumb_mcp > 0.9:
            return "L"
        return "D"

    # --- index + middle: H, P, R, K, V, U
    if idx and mid and not (ring or pnk):
        if direction == "side":
            return "H"
        if direction == "down":
            return "P"
        i_side = pts[5, 0] - pts[9, 0]
        t_side = pts[8, 0] - pts[12, 0]
        if i_side * t_side < 0 and abs(t_side) / P > 0.08 and abs(i_side) / P > 0.08:
            return "R"
        gap = d(8, 12)
        if d(4, 10) < 0.35 and gap > 0.3:
            return "K"
        return "V" if gap > 0.45 else "U"

    if idx and mid and ring and not pnk:
        return "W"
    if idx and mid and ring and pnk:
        return "B"

    # --- pinky only: I / Y  (J is detected by motion)
    if pnk and not (idx or mid or ring):
        return "Y" if thumb_mcp > 0.9 else "I"

    # --- F: index touches thumb, other three up
    if mid and ring and pnk and not idx:
        return "F" if thumb_idx < 0.4 else None

    # --- nothing extended: X, O, C, A, E, T, N, M, S
    if not any(ext):
        if ratios[0] > 1.3 and max(ratios[1:]) < 1.25:
            return "X"
        if mean_r >= 1.35:
            return "O" if thumb_idx < 0.35 else "C"
        if thumb_mcp > 0.55 and pts[4, 1] < pts[6, 1] and d(4, 10) > 0.45:
            return "A"
        if mean_r > 1.22 and d(4, 12) < 0.55:
            return "E"
        a, b = pts[5, :2], pts[17, :2]
        ab = b - a
        ap = pts[4, :2] - a
        t = float(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-6))
        line_d = abs(ab[0] * ap[1] - ab[1] * ap[0]) / (np.linalg.norm(ab) + 1e-6) / P
        if line_d < 0.55 and -0.1 <= t <= 1.3:
            if t < 0.30:
                return "T"
            if t < 0.52:
                return "N"
            return "M"
        return "S"

    return None


# ----------------------------------------------------------------------------
# Optional trained classifier (k-nearest neighbours on your own samples)
# ----------------------------------------------------------------------------
class KNN:
    def __init__(self, path):
        self.path = path
        self.data = {}
        self.X = None
        self.labels = np.array([])
        self.stale = True
        self.dirty = False
        self.load()

    def load(self):
        try:
            with open(self.path, "r") as f:
                raw = json.load(f)
            self.data = {k: [np.array(v, dtype=np.float32) for v in vs] for k, vs in raw.items()}
        except FileNotFoundError:
            pass
        except Exception as e:
            print("Could not read samples file:", e)
        self.stale = True

    def save(self):
        if not self.dirty:
            return
        try:
            with open(self.path, "w") as f:
                json.dump({k: [v.tolist() for v in vs] for k, vs in self.data.items()}, f)
            self.dirty = False
        except Exception as e:
            print("Could not save samples:", e)

    def add(self, letter, vec):
        lst = self.data.setdefault(letter, [])
        lst.append(vec)
        if len(lst) > KNN_MAX_PER_LETTER:
            lst.pop(0)
        self.stale = True
        self.dirty = True

    def wipe(self, letter):
        if letter in self.data:
            del self.data[letter]
            self.stale = True
            self.dirty = True

    def count(self, letter):
        return len(self.data.get(letter, []))

    def _rebuild(self):
        rows, labels = [], []
        for k, vs in self.data.items():
            for v in vs:
                rows.append(v)
                labels.append(k)
        self.X = np.array(rows, dtype=np.float32) if rows else None
        self.labels = np.array(labels)
        self.stale = False

    def predict(self, vec, k=5):
        if self.stale:
            self._rebuild()
        if self.X is None:
            return None
        dist = np.linalg.norm(self.X - vec, axis=1)
        order = np.argsort(dist)[:k]
        if dist[order[0]] > KNN_MAX_DIST:
            return None
        votes = Counter()
        for i in order:
            if dist[i] <= KNN_MAX_DIST:
                votes[str(self.labels[i])] += 1.0 / (dist[i] + 0.05)
        return votes.most_common(1)[0][0]


# ----------------------------------------------------------------------------
# Motion letters (J and Z)
# ----------------------------------------------------------------------------
class Motion:
    def __init__(self):
        self.buf = deque()

    def reset(self):
        self.buf.clear()

    def push(self, t, x, y):
        self.buf.append((t, x, y))
        while self.buf and t - self.buf[0][0] > 2.2:
            self.buf.popleft()

    def energy(self, window=0.4):
        if len(self.buf) < 2:
            return 0.0
        now = self.buf[-1][0]
        pts = [p for p in self.buf if now - p[0] <= window]
        total = 0.0
        for a, b in zip(pts, pts[1:]):
            total += float(np.hypot(b[1] - a[1], b[2] - a[2]))
        return total

    def detect_j(self):
        pts = list(self.buf)
        if len(pts) < 5:
            return False
        now = pts[-1][0]
        for win in (0.6, 1.0, 1.4):
            seg = [p for p in pts if now - p[0] <= win]
            if len(seg) < 5 or seg[-1][0] - seg[0][0] < 0.25:
                continue
            ys = [p[2] for p in seg]
            drop = max(ys) - ys[0]
            path = sum(float(np.hypot(b[1] - a[1], b[2] - a[2])) for a, b in zip(seg, seg[1:]))
            if drop > 0.5 and path > 0.7:
                return True
        return False

    def detect_z(self):
        """Corner-based: right -> diagonal down-left -> right (works for either hand direction)."""
        pts = list(self.buf)
        if len(pts) < 8:
            return False
        now = pts[-1][0]
        for win in (0.8, 1.2, 1.6, 2.0):
            seg = [p for p in pts if now - p[0] <= win]
            n = len(seg)
            if n < 8:
                continue
            ys = [p[2] for p in seg]
            for s in (1, -1):
                xs = [s * p[1] for p in seg]
                iA = int(np.argmax(xs[:n - 3]))
                iB = iA + int(np.argmin(xs[iA:]))
                if iB <= iA or iB >= n - 1:
                    continue
                if (xs[iA] - xs[0] > 0.25 and xs[iA] - xs[iB] > 0.4 and
                        ys[iB] - ys[iA] > 0.15 and xs[-1] - xs[iB] > 0.25):
                    return True
        return False


# ----------------------------------------------------------------------------
# Smoothing + typing logic
# ----------------------------------------------------------------------------
class Stabilizer:
    def __init__(self):
        self.buf = deque(maxlen=10)
        self.cur = None

    def clear(self):
        self.buf.clear()
        self.cur = None

    def push(self, letter):
        self.buf.append(letter)
        c = Counter(x for x in self.buf if x)
        if not c:
            self.cur = None
            return None
        best, n = c.most_common(1)[0]
        if self.cur and c.get(self.cur, 0) >= 4 and (best == self.cur or n < 7):
            return self.cur
        self.cur = best if n >= 6 else None
        return self.cur


class Typer:
    def __init__(self):
        self.word = ""
        self.sentence = ""
        self.cur = None
        self.since = 0.0
        self.committed = False
        self.doubled = False
        self.lost_at = None
        self.space_done = False

    def reset_hold(self):
        self.cur, self.committed, self.doubled = None, False, False

    def commit_word(self):
        if self.word:
            self.sentence += self.word + " "
            self.word = ""

    def backspace(self):
        if self.word:
            self.word = self.word[:-1]
        elif self.sentence:
            self.sentence = self.sentence[:-1]

    def clear(self):
        self.word = self.sentence = ""
        self.reset_hold()

    def update(self, letter, hand, now, moving=False):
        if not hand:
            if self.lost_at is None:
                self.lost_at = now
            lost_for = now - self.lost_at
            if lost_for > 0.3:
                self.reset_hold()
            if lost_for >= SPACE_TIME and not self.space_done:
                self.commit_word()
                self.space_done = True
            return
        self.lost_at = None
        self.space_done = False

        if letter != self.cur:
            self.cur, self.since = letter, now
            self.committed = self.doubled = False
            return
        if letter is None:
            return
        if moving and not self.committed:
            self.since = now
            return
        el = now - self.since
        if el >= TIME_TO_CONFIRM and not self.committed:
            self.word += letter
            self.committed = True
        if el >= TIME_TO_DOUBLE and not self.doubled:
            self.word += letter
            self.doubled = True

    def progress(self, now):
        if self.cur is None:
            return 0.0, False
        el = now - self.since
        if not self.committed:
            return min(el / TIME_TO_CONFIRM, 1.0), False
        if not self.doubled:
            return min((el - TIME_TO_CONFIRM) / (TIME_TO_DOUBLE - TIME_TO_CONFIRM), 1.0), True
        return 1.0, True


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
def put(img, text, org, scale=0.6, color=TEXT, thick=1):
    cv2.putText(img, text, (int(org[0]), int(org[1])), FONT, scale, color, thick, cv2.LINE_AA)


def put_center(img, text, cx, cy, scale, color, thick):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, (int(cx - tw / 2), int(cy + th / 2)), FONT, scale, color, thick, cv2.LINE_AA)


def rrect(img, x1, y1, x2, y2, color, r=10):
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def wrap(text, max_w, scale, thick):
    lines, cur = [], ""
    for word in text.split():
        test = (cur + " " + word).strip()
        if cv2.getTextSize(test, FONT, scale, thick)[0][0] > max_w and cur:
            lines.append(cur)
            cur = word
        else:
            cur = test
    if cur:
        lines.append(cur)
    return lines


def draw_hand(frame, landmarks, w, h):
    px = [(int(l.x * w), int(l.y * h)) for l in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px[a], px[b], ACCENT, 2, cv2.LINE_AA)
    for i, p in enumerate(px):
        cv2.circle(frame, p, 6 if i in TIPS or i == 4 else 4, ORANGE if i in TIPS or i == 4 else TEXT, -1, cv2.LINE_AA)


BUTTONS = []


def build_buttons():
    labels = [("speak", "Speak (Enter)"), ("space", "Space"), ("back", "Backspace"),
              ("clear", "Clear (C)"), ("train", "Train (T)")]
    bw = (W - 32 - 4 * 10) // 5
    for i, (k, l) in enumerate(labels):
        x1 = 16 + i * (bw + 10)
        BUTTONS.append((k, l, (x1, CAM_H + 98, x1 + bw, CAM_H + 146)))


def draw_ui(canvas, S):
    now = S["now"]
    canvas[:] = BG
    canvas[0:CAM_H, 0:CAM_W] = S["frame"]

    # ---------------- sidebar ----------------
    sx = CAM_W
    cv2.rectangle(canvas, (sx, 0), (W, CAM_H), PANEL, -1)
    put(canvas, "ASL Fingerspelling", (sx + 16, 32), 0.75, TEXT, 2)
    if S["training"]:
        rrect(canvas, sx + 228, 14, sx + 306, 40, ORANGE, 8)
        put_center(canvas, "TRAIN", sx + 267, 27, 0.5, BG, 2)

    rrect(canvas, sx + 16, 50, W - 16, 252, CARD, 14)
    cx, cy, r = sx + SIDEBAR_W // 2, 132, 62
    cv2.circle(canvas, (cx, cy), r, PANEL, -1, cv2.LINE_AA)
    prog, doubling = S["progress"]
    cv2.circle(canvas, (cx, cy), r, (80, 74, 70), 6, cv2.LINE_AA)
    if prog > 0 and not S["training"]:
        cv2.ellipse(canvas, (cx, cy), (r, r), -90, 0, 360 * prog,
                    ORANGE if doubling else ACCENT, 6, cv2.LINE_AA)
    shown = S["letter"] or "-"
    put_center(canvas, shown, cx, cy, 3.0, TEXT if S["letter"] else MUTED, 6)
    if not S["hand"]:
        status = "Show your hand"
    elif S["training"]:
        status = "Hold a letter key to record"
    elif S["letter"] is None:
        status = "Sign not recognised"
    elif doubling:
        status = "Keep holding to double"
    else:
        status = "Hold steady to type"
    put_center(canvas, status, cx, 230, 0.5, MUTED, 1)

    rrect(canvas, sx + 16, 264, W - 16, 330, CARD, 12)
    put(canvas, "WORD", (sx + 30, 284), 0.4, MUTED, 1)
    word = S["typer"].word
    put(canvas, (word[-13:] if word else "...") + ("_" if int(now * 2) % 2 == 0 else ""),
        (sx + 30, 316), 0.9, ACCENT if word else MUTED, 2)

    y = 352
    if S["training"]:
        put(canvas, "Hold a letter key to record the sign", (sx + 16, y), 0.45, TEXT, 1)
        put(canvas, "Bksp = wipe last letter   Esc = done", (sx + 16, y + 20), 0.42, MUTED, 1)
        letters = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"]
        for i, c in enumerate(letters):
            n = S["knn"].count(c)
            put(canvas, f"{c}:{n}", (sx + 16 + (i % 6) * 49, y + 50 + (i // 6) * 22), 0.45,
                ACCENT if n >= 20 else MUTED, 1)
    elif S["debug"]:
        rr = " ".join(f"{v:.2f}" for v in S["ratios"])
        put(canvas, "ratios I M R P: " + rr, (sx + 16, y), 0.42, TEXT, 1)
        put(canvas, "extended: " + "".join("1" if e else "0" for e in S["ext"]), (sx + 16, y + 22), 0.45, TEXT, 1)
        put(canvas, f"source: {S['source']}   fps: {S['fps']:.0f}", (sx + 16, y + 44), 0.45, TEXT, 1)
        put(canvas, f"samples: {sum(len(v) for v in S['knn'].data.values())}", (sx + 16, y + 66), 0.45, MUTED, 1)
    else:
        for i, line in enumerate(["Hold 0.8s : type letter", "Hold 2.0s : double letter",
                                  "Hide hand 1.2s : space", "J / Z : draw in the air",
                                  "D debug  |  Q quit"]):
            put(canvas, line, (sx + 16, y + i * 22), 0.45, MUTED, 1)

    # toast on camera
    if S["toast"][0] and now < S["toast"][1]:
        msg = S["toast"][0]
        tw = cv2.getTextSize(msg, FONT, 0.6, 2)[0][0]
        rrect(canvas, 16, CAM_H - 50, 40 + tw, CAM_H - 14, (20, 20, 20), 10)
        put(canvas, msg, (28, CAM_H - 26), 0.6, TEXT, 2)

    # ---------------- bottom ----------------
    cv2.rectangle(canvas, (0, CAM_H), (W, H), PANEL, -1)
    rrect(canvas, 16, CAM_H + 8, W - 16, CAM_H + 88, CARD, 12)
    put(canvas, "SENTENCE", (30, CAM_H + 26), 0.4, MUTED, 1)
    sentence = S["typer"].sentence
    lines = wrap(sentence, W - 80, 0.75, 2)[-2:]
    if not lines:
        put(canvas, "Your sentence will appear here...", (30, CAM_H + 58), 0.7, MUTED, 1)
    for i, ln in enumerate(lines):
        put(canvas, ln, (30, CAM_H + 56 + i * 26 - (0 if len(lines) == 2 else 0)), 0.75, TEXT, 2)

    mx, my = S["mouse"]
    for key, label, (x1, y1, x2, y2) in BUTTONS:
        hover = x1 <= mx <= x2 and y1 <= my <= y2
        base = {"speak": ACCENT, "clear": RED}.get(key, CARD)
        if key == "train" and S["training"]:
            base = ORANGE
        col = tuple(min(c + 25, 255) for c in base) if hover else base
        rrect(canvas, x1, y1, x2, y2, col, 10)
        dark = key in ("speak",) or (key == "train" and S["training"])
        put_center(canvas, label, (x1 + x2) / 2, (y1 + y2) / 2, 0.55, BG if dark else TEXT, 2)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Missing model file: {MODEL_PATH}")
        print("Download: https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/1/hand_landmarker.task")
        return

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    cam = Camera(CAM_INDEX)
    knn = KNN(SAMPLES_PATH)
    typer, stab = Typer(), Stabilizer()
    z_motion, j_motion = Motion(), Motion()
    build_buttons()

    ui = {"mouse": (0, 0), "click": None}

    def on_mouse(event, x, y, flags, param):
        ui["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONUP:
            ui["click"] = (x, y)

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    state = {"training": False, "debug": False, "toast": ("", 0.0)}
    last_trained = None
    cooldown_until = 0.0
    z_last = j_last = 0.0
    trail = deque()
    last_ts = 0
    t0 = time.perf_counter()
    fps, last_t = 0.0, time.perf_counter()
    frames_shown = 0
    err_seen = set()

    def toast(msg, secs=1.8):
        state["toast"] = (msg, time.time() + secs)

    def do_action(name):
        if name == "speak":
            text = (typer.sentence + " " + typer.word).strip().lower()
            if text:
                speak(text)
                toast("Reading aloud...")
            else:
                toast("Nothing to read yet")
        elif name == "space":
            typer.commit_word()
        elif name == "back":
            typer.backspace()
        elif name == "clear":
            typer.clear()
        elif name == "train":
            state["training"] = not state["training"]
            typer.reset_hold()
            stab.clear()
            if not state["training"]:
                knn.save()
                toast("Samples saved")

    print("--- ASL Fingerspelling active (press Q to quit) ---")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                if cam.failures > 150:
                    print("Camera is not delivering frames. Check CAM_INDEX / other apps using the camera.")
                    break
                key = cv2.waitKey(5) & 0xFF
                if key in (ord("q"), 27) and not state["training"]:
                    break
                continue

            now = time.time()
            if frame.shape[1] != CAM_W or frame.shape[0] != CAM_H:
                frame = cv2.resize(frame, (CAM_W, CAM_H))
            frame = cv2.flip(frame, 1)

            hand = False
            letter_raw = None
            stable = None
            source = "-"
            ratios, ext = [0, 0, 0, 0], [False] * 4
            vec = None
            moving = False

            try:
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ts = int((time.perf_counter() - t0) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

                if result.hand_landmarks:
                    hand = True
                    lms = result.hand_landmarks[0]
                    label = result.handedness[0][0].category_name
                    draw_hand(frame, lms, CAM_W, CAM_H)
                    pts = to_points(lms, CAM_W, CAM_H, label)
                    P = palm_size(pts)
                    ratios = finger_ratios(pts)
                    ext = [r > t for r, t in zip(ratios, EXT_T)]
                    vec = feature_vector(pts)

                    letter_raw = knn.predict(vec)
                    source = "trained"
                    if letter_raw is None:
                        letter_raw = rule_letter(pts, ratios, ext)
                        source = "rules"

                    # ---- motion letters J / Z
                    t = time.perf_counter()
                    index_only = ext[0] and not any(ext[1:])
                    pinky_only = ext[3] and not any(ext[:3])
                    if index_only:
                        z_motion.push(t, pts[8, 0] / P, pts[8, 1] / P)
                        z_last = t
                    elif t - z_last > 0.25:
                        z_motion.reset()
                    if pinky_only:
                        j_motion.push(t, pts[20, 0] / P, pts[20, 1] / P)
                        j_last = t
                    elif t - j_last > 0.25:
                        j_motion.reset()
                    if index_only or pinky_only:
                        tip = lms[8 if index_only else 20]
                        trail.append((t, int(tip.x * CAM_W), int(tip.y * CAM_H)))
                        while trail and t - trail[0][0] > 1.5:
                            trail.popleft()
                    elif t - max(z_last, j_last) > 0.25:
                        trail.clear()
                    moving = (index_only and z_motion.energy() > 0.6) or \
                             (pinky_only and j_motion.energy() > 0.6)

                    if not state["training"] and now > cooldown_until:
                        motion_letter = None
                        if index_only and z_motion.detect_z():
                            motion_letter = "Z"
                        elif pinky_only and j_motion.detect_j():
                            motion_letter = "J"
                        if motion_letter:
                            typer.word += motion_letter
                            cooldown_until = now + MOTION_COOLDOWN
                            z_motion.reset()
                            j_motion.reset()
                            trail.clear()
                            stab.clear()
                            typer.reset_hold()
                            letter_raw = motion_letter
                            toast(f"Typed {motion_letter}")
                else:
                    z_motion.reset()
                    j_motion.reset()
                    trail.clear()
            except Exception:
                msg = traceback.format_exc()
                if msg not in err_seen:
                    err_seen.add(msg)
                    print(msg)

            if len(trail) > 1:
                cv2.polylines(frame, [np.array([(x, y) for _, x, y in trail], np.int32)],
                              False, ORANGE, 3, cv2.LINE_AA)

            stable = stab.push(letter_raw if hand else None)
            if now < cooldown_until:
                stable_for_typing = None
            else:
                stable_for_typing = stable

            if not state["training"]:
                typer.update(stable_for_typing, hand, now, moving)
            progress = typer.progress(now) if not state["training"] else (0.0, False)

            # ---- fps
            tn = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(tn - last_t, 1e-3))
            last_t = tn

            S = {
                "now": now, "frame": frame, "hand": hand, "letter": stable or (letter_raw if now < cooldown_until else None),
                "progress": progress, "typer": typer, "training": state["training"],
                "debug": state["debug"], "ratios": ratios, "ext": ext, "source": source,
                "fps": fps, "knn": knn, "mouse": ui["mouse"], "toast": state["toast"],
            }
            draw_ui(canvas, S)
            cv2.imshow(WIN, canvas)
            frames_shown += 1

            # ---- input
            key = cv2.waitKey(1) & 0xFF
            if ui["click"]:
                cx, cy = ui["click"]
                ui["click"] = None
                for name, _, (x1, y1, x2, y2) in BUTTONS:
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        do_action(name)

            if key != 255:
                if state["training"]:
                    if key == 27:
                        do_action("train")
                    elif key == 8:
                        if last_trained:
                            knn.wipe(last_trained)
                            toast(f"Cleared samples for {last_trained}")
                    elif 97 <= key <= 122 or 65 <= key <= 90:
                        ch = chr(key).upper()
                        if ch in "JZ":
                            toast("J and Z are motion letters (auto)")
                        elif hand and vec is not None:
                            knn.add(ch, vec)
                            last_trained = ch
                else:
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    elif key == 13:
                        do_action("speak")
                    elif key == 32:
                        do_action("space")
                    elif key == 8:
                        do_action("back")
                    elif key in (ord("c"), ord("C")):
                        do_action("clear")
                    elif key in (ord("t"), ord("T")):
                        do_action("train")
                    elif key in (ord("d"), ord("D")):
                        state["debug"] = not state["debug"]

            # window closed with the X button
            if frames_shown > 5 and cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        knn.save()
        cam.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
