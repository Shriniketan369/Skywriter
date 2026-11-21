"""
Skywriter - draw in the air with your hand and see it live on your camera feed.

GESTURES
  PINCH       thumb + index fingertips touching ... DRAW (pen down)
  OPEN HAND   any open / relaxed hand ............. pen up, move freely
  FIST        close your whole hand ............... GRAB the nearest stroke and drag it
  TWO FINGERS index + middle up, others curled .... ERASE (circle at fingertips)

KEYS (click the preview window first)
  c  clear everything      u  undo last stroke
  k  recalibrate gestures  d  show / hide debug info
  e  backlight enhancement on/off (helps tracking when light is behind you)
  q  quit

FIRST RUN: Skywriter walks you through a ~20 second calibration so it learns
what YOUR pinch, fist, two-fingers and open hand look like. Redo with k anytime.

Install:  pip install opencv-python mediapipe numpy
Run:      python skywriter.py          (python skywriter.py 1  to use camera #1)
"""
import json
import math
import os
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np

# ======================= settings =======================
MIRROR_OUTPUT = True        # output == what you see, so your writing reads correctly to viewers
CAM_W, CAM_H = 1280, 720
PROC_W = 640                # hand tracking runs on a downscaled copy (big speed-up)
PEN_COLOR = (0, 255, 255)   # BGR
PEN_SIZE = 5
ERASER_RADIUS = 40
GRAB_RADIUS = 100           # fist must be this close (px) to a stroke to grab it

DEB_IN = {"draw": 3, "grab": 3, "erase": 3, "hover": 2}   # frames a gesture must persist
DEB_OUT = {"draw": 4, "grab": 4, "erase": 3, "hover": 1}  # extra patience when leaving a mode
GRAB_COOLDOWN = 8           # frames after releasing a fist before drawing is allowed
LOST_FRAMES = 10            # tracking dropouts shorter than this do not break a stroke

PREP_S, CAPTURE_S, MIN_SAMPLES, SIGMA_FLOOR = 2.5, 2.5, 12, 0.08
HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(HERE, "skywriter_calibration.json")
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

WIN = "Skywriter preview"
COLORS = {"draw": (0, 255, 0), "hover": (220, 220, 220),
          "grab": (255, 0, 255), "erase": (0, 0, 255)}
FINGER_JOINTS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]  # index..pinky
TIPS = (8, 12, 16, 20)
N_FEATURES = 10

STEPS = [
    ("draw", "PINCH: touch thumb and index tips like holding a pen (tilt a little)"),
    ("grab", "FIST: close your whole hand"),
    ("erase", "TWO FINGERS: index + middle up, other fingers curled"),
    ("hover", "OPEN HAND: fingers spread wide"),
    ("hover", "RELAXED HAND: loose and half-curled, like moving between strokes"),
]


# ======================= camera =======================
class Camera:
    """Reads in a background thread so you always get the newest frame (no buffer lag)."""

    def __init__(self, index=0):
        self.cap = None
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
        for b in backends:
            cap = cv2.VideoCapture(index, b)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()
        self.opened = self.cap is not None
        self.lock = threading.Lock()
        self.frame, self.fresh, self.running = None, False, True
        if self.opened:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.fresh = f, True
            else:
                time.sleep(0.005)

    def read(self, timeout=0.5):
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            with self.lock:
                if self.fresh:
                    self.fresh = False
                    return self.frame
            time.sleep(0.001)
        return None

    def release(self):
        self.running = False
        time.sleep(0.05)
        if self.cap is not None:
            self.cap.release()


# ======================= math =======================
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def to_int(p):
    return (int(p[0]), int(p[1]))


def angle3(a, b, c):
    v1, v2 = a - b, c - b
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


class OneEuro:
    """Smooth when slow (no jitter), responsive when fast (no lag)."""

    def __init__(self, min_cutoff=1.5, beta=0.04, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.t_prev = self.x_prev = self.dx_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.t_prev is None:
            self.t_prev, self.x_prev, self.dx_prev = t, x, np.zeros_like(x)
            return x
        dt = max(t - self.t_prev, 1e-3)
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * ((x - self.x_prev) / dt) + (1 - a_d) * self.dx_prev
        a = self._alpha(self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat)), dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.t_prev, self.x_prev, self.dx_prev = t, x_hat, dx_hat
        return x_hat


def make_filters():
    return {"pen": OneEuro(), "palm": OneEuro(), "erase": OneEuro()}


# ======================= hand tracking backends =======================
class Hand:
    def __init__(self, img, world):
        self.img = img      # (21, 2) pixel coordinates in the full-size frame
        self.world = world  # (21, 3) metric 3D coordinates (rotation friendly)


class LegacyTracker:
    """mediapipe < 0.10.3x : mp.solutions.hands"""

    def __init__(self, mp):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1, model_complexity=0,
            min_detection_confidence=0.5, min_tracking_confidence=0.3)

    def process(self, rgb, w, h):
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None
        lm = res.multi_hand_landmarks[0].landmark
        img = np.array([[p.x * w, p.y * h] for p in lm], dtype=float)
        wl = getattr(res, "multi_hand_world_landmarks", None)
        if wl:
            world = np.array([[p.x, p.y, p.z] for p in wl[0].landmark], dtype=float)
        else:
            world = np.array([[p.x * w, p.y * w, p.z * w] for p in lm], dtype=float)
        return Hand(img, world)


def ensure_model_file():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    print("Downloading hand tracking model (one time, ~8 MB)...")
    tmp = MODEL_PATH + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, MODEL_PATH)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "Could not download the hand model (%s).\nDownload it manually from:\n  %s\n"
            "and save it as:\n  %s" % (e, MODEL_URL, MODEL_PATH))
    return MODEL_PATH


class TasksTracker:
    """mediapipe >= 0.10.3x : Tasks API HandLandmarker (needs hand_landmarker.task)"""

    def __init__(self, mp):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model_file()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.3)
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_ts = 0

    def process(self, rgb, w, h):
        ts = max(int(time.perf_counter() * 1000), self.last_ts + 1)
        self.last_ts = ts
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(rgb))
        res = self.landmarker.detect_for_video(image, ts)
        if not res.hand_landmarks:
            return None
        lm = res.hand_landmarks[0]
        img = np.array([[p.x * w, p.y * h] for p in lm], dtype=float)
        if res.hand_world_landmarks:
            world = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]], dtype=float)
        else:
            world = np.array([[p.x * w, p.y * w, p.z * w] for p in lm], dtype=float)
        return Hand(img, world)


def make_tracker():
    import mediapipe as mp
    try:
        has_legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
    except Exception:  # noqa: BLE001
        has_legacy = False
    return LegacyTracker(mp) if has_legacy else TasksTracker(mp)


# ======================= gesture learning =======================
def features(w):
    """10 rotation/scale-invariant numbers describing the hand shape."""
    palm = float(np.linalg.norm(w[0] - w[9])) + 1e-9
    c = w[[0, 5, 9, 13, 17]].mean(axis=0)
    f = [float(np.linalg.norm(w[4] - w[8])) / palm]                 # thumb-index gap
    f += [float(np.linalg.norm(w[t] - c)) / palm for t in TIPS]     # fingertips to palm centre
    f.append(float(np.linalg.norm(w[4] - c)) / palm)                # thumb tip to palm centre
    f += [angle3(w[a], w[b], w[k]) / 180.0 for a, b, k in FINGER_JOINTS]  # finger straightness
    return np.array(f, dtype=float)


class Model:
    """Nearest-prototype classifier built from the user's own calibration."""

    def __init__(self, protos, sigma):
        self.protos = protos                  # list of (label, vector)
        self.sigma = np.asarray(sigma, dtype=float)
        self.min_sep = self._separation()

    def distances(self, f):
        out = {}
        for label, mu in self.protos:
            d = float(np.linalg.norm((f - mu) / self.sigma))
            if label not in out or d < out[label]:
                out[label] = d
        return out

    def classify(self, f):
        d = self.distances(f)
        label = min(d, key=d.get)
        return label, d

    def _separation(self):
        best = 1e9
        for i, (la, a) in enumerate(self.protos):
            for lb, b in self.protos[i + 1:]:
                if la != lb:
                    best = min(best, float(np.linalg.norm((a - b) / self.sigma)))
        return best

    @staticmethod
    def build(results):
        protos, variances = [], []
        for label, samples in results:
            protos.append((label, samples.mean(axis=0)))
            variances.append(samples.var(axis=0))
        sigma = np.maximum(np.sqrt(np.mean(variances, axis=0)), SIGMA_FLOOR)
        return Model(protos, sigma)

    def save(self, path=None):
        path = path or CALIB_PATH
        data = {"protos": [[l, v.tolist()] for l, v in self.protos],
                "sigma": self.sigma.tolist()}
        try:
            with open(path, "w") as fh:
                json.dump(data, fh)
            return True
        except OSError:
            return False

    @staticmethod
    def load(path=None):
        path = path or CALIB_PATH
        try:
            with open(path) as fh:
                data = json.load(fh)
            protos = [(str(l), np.array(v, dtype=float)) for l, v in data["protos"]]
            sigma = np.array(data["sigma"], dtype=float)
            labels = {l for l, _ in protos}
            if (not protos or len(sigma) != N_FEATURES or
                    any(len(v) != N_FEATURES for _, v in protos) or
                    not {"draw", "grab", "erase", "hover"} <= labels):
                return None
            return Model(protos, sigma)
        except Exception:  # noqa: BLE001
            return None


class Calibrator:
    def __init__(self):
        self.i, self.phase, self.t0 = 0, "prep", None
        self.samples, self.results, self.note = [], [], ""

    def update(self, now, f):
        """Feed one frame. Returns a Model when calibration is complete."""
        if self.t0 is None:
            self.t0 = now
        el = now - self.t0
        if self.phase == "prep":
            if el >= PREP_S:
                self.phase, self.t0, self.samples = "capture", now, []
            return None
        if f is not None:
            self.samples.append(f)
        if el < CAPTURE_S:
            return None
        if len(self.samples) >= MIN_SAMPLES:
            self.results.append((STEPS[self.i][0], np.array(self.samples)))
            self.i += 1
            self.note = ""
            if self.i >= len(STEPS):
                return Model.build(self.results)
        else:
            self.note = "Hand not seen - repeating this step"
        self.phase, self.t0 = "prep", now
        return None

    def lines(self, now):
        el = now - (self.t0 if self.t0 is not None else now)
        if self.phase == "prep":
            status = "Get ready... %.0f" % max(0.0, PREP_S - el + 0.49)
        else:
            status = "HOLD IT... %.1fs" % max(0.0, CAPTURE_S - el)
        out = ["CALIBRATION %d/%d   %s" % (min(self.i + 1, len(STEPS)), len(STEPS), status),
               STEPS[min(self.i, len(STEPS) - 1)][1]]
        if self.note:
            out.append(self.note)
        return out


class Gestures:
    """Turns noisy per-frame labels into stable modes, with safety rules."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mode, self.cand, self.cand_n, self.no_draw = "hover", None, 0, 0

    def update(self, raw):
        cur = self.mode
        if self.no_draw > 0:
            self.no_draw -= 1
        if raw == "draw" and (cur == "grab" or self.no_draw > 0):
            raw = "hover"          # releasing a fist can never start a stroke
        if raw == cur:
            self.cand, self.cand_n = None, 0
            return cur
        self.cand_n = self.cand_n + 1 if raw == self.cand else 1
        self.cand = raw
        if self.cand_n >= max(DEB_IN[raw], DEB_OUT.get(cur, 1)):
            if cur == "grab":
                self.no_draw = GRAB_COOLDOWN
            self.mode, self.cand, self.cand_n = raw, None, 0
        return self.mode


# ======================= strokes =======================
class Stroke:
    def __init__(self, color, size):
        self.pts, self.color, self.size, self.t = [], color, size, time.perf_counter()


def path_len(s):
    return sum(dist(s.pts[i], s.pts[i + 1]) for i in range(len(s.pts) - 1))


def add_point(s, p):
    p = (float(p[0]), float(p[1]))
    if s.pts:
        last = s.pts[-1]
        d = dist(last, p)
        if d < 1.5:
            return
        n = int(d // 6)
        for i in range(1, n + 1):
            t = i / (n + 1.0)
            s.pts.append((last[0] + (p[0] - last[0]) * t, last[1] + (p[1] - last[1]) * t))
    s.pts.append(p)


def erase_at(strokes, c, r):
    out, changed = [], False
    for s in strokes:
        pieces, seg = [], []
        for p in s.pts:
            if dist(p, c) <= r:
                if seg:
                    pieces.append(seg)
                    seg = []
            else:
                seg.append(p)
        if seg:
            pieces.append(seg)
        if len(pieces) == 1 and len(pieces[0]) == len(s.pts):
            out.append(s)
            continue
        changed = True
        for pc in pieces:
            if len(pc) >= 2:
                n = Stroke(s.color, s.size)
                n.pts = pc
                out.append(n)
    return out, changed


def nearest_stroke(strokes, c):
    best, bd = None, GRAB_RADIUS
    for s in strokes:
        for p in s.pts:
            d = dist(p, c)
            if d < bd:
                bd, best = d, s
    return best


def paint(canvas, mask, s, color, size):
    if not s.pts:
        return
    arr = np.round(np.array(s.pts)).astype(np.int32).reshape(-1, 1, 2)
    if len(arr) == 1:
        c = (int(arr[0, 0, 0]), int(arr[0, 0, 1]))
        cv2.circle(canvas, c, max(1, size // 2), color, -1)
        cv2.circle(mask, c, max(1, size // 2), 255, -1)
    else:
        cv2.polylines(canvas, [arr], False, color, size, cv2.LINE_AA)
        cv2.polylines(mask, [arr], False, 255, size, cv2.LINE_AA)


def paint_tail(canvas, mask, s, start):
    """Paint only the newly added part of a stroke (fast path while writing)."""
    seg = s.pts[max(start - 1, 0):]
    if len(seg) < 2:
        paint(canvas, mask, s, s.color, s.size)
        return
    arr = np.round(np.array(seg)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [arr], False, s.color, s.size, cv2.LINE_AA)
    cv2.polylines(mask, [arr], False, 255, s.size, cv2.LINE_AA)


def render_into(canvas, mask, strokes, selected):
    canvas[:] = 0
    mask[:] = 0
    for s in strokes:
        if s is selected:
            paint(canvas, mask, s, (255, 255, 255), s.size + 6)   # highlight while held
        paint(canvas, mask, s, s.color, s.size)


# ======================= engine (all logic, no camera) =======================
class Engine:
    def __init__(self, model=None, load=True):
        self.model = model if model is not None else (Model.load() if load else None)
        self.strokes, self.current, self.grabbed, self.last_palm = [], None, None, None
        self.canvas = self.mask = None
        self.dirty = False
        self.mode, self.lost = "none", 0
        self.pen_pt = self.palm_pt = self.erase_pt = None
        self.filters, self.gest = make_filters(), Gestures()
        self.calib, self.note, self.debug = None, "", None
        self.raw = None
        if self.model is None:
            self.start_calibration()

    # ---- user commands ----
    def start_calibration(self):
        self.calib, self.mode = Calibrator(), "calib"
        self.current = self.grabbed = None
        self.dirty = True

    def clear(self):
        self.strokes, self.current, self.grabbed, self.dirty = [], None, None, True

    def undo(self):
        if self.strokes:
            self.strokes.pop()
            self.current = self.grabbed = None
            self.dirty = True

    # ---- internals ----
    def ensure_canvas(self, shape):
        if self.canvas is None or self.canvas.shape != shape:
            self.canvas = np.zeros(shape, np.uint8)
            self.mask = np.zeros(shape[:2], np.uint8)
            self.dirty = True

    def _reset_tracking(self):
        self.mode = "none"
        self.gest.reset()
        self.filters = make_filters()
        if self.grabbed is not None:
            self.dirty = True
        self.current = self.grabbed = None
        self.pen_pt = self.palm_pt = self.erase_pt = self.debug = self.raw = None

    def step(self, now, hand, shape):
        """Advance one frame. hand is a Hand or None; shape is the frame shape."""
        self.ensure_canvas(shape)
        prev_mode = self.mode
        f = None
        if hand is not None:
            self.lost = 0
            f = features(hand.world)
            P = hand.img
            self.pen_pt = tuple(self.filters["pen"](mid(P[4], P[8]), now))
            self.palm_pt = tuple(self.filters["palm"](P[9], now))
            self.erase_pt = tuple(self.filters["erase"](mid(P[8], P[12]), now))
        else:
            self.lost += 1
            if self.lost > LOST_FRAMES:
                self._reset_tracking()

        if self.calib is not None:                       # calibration screen
            done = self.calib.update(now, f)
            self.mode = "calib"
            if done is not None:
                self.model, self.calib = done, None
                saved = done.save()
                self.note = "Calibration done" + ("" if saved else " (could not save file)")
                if done.min_sep < 2.0:
                    self.note += " - gestures look similar, press k to redo if misfiring"
                self.gest.reset()
                self.mode = "none"
            self._finish()
            return

        if hand is not None:
            label, dists = self.model.classify(f)
            self.raw = label
            self.mode = self.gest.update(label)
            self.debug = (label, dists)
        elif self.mode == "calib":
            self.mode = "none"

        mode = self.mode
        if mode != "draw":
            self.current = None
        if prev_mode == "grab" and mode != "grab":
            self.grabbed, self.dirty = None, True
        if mode == "grab" and prev_mode != "grab" and self.palm_pt is not None:
            # a stray dot made while closing the fist is discarded
            if self.strokes and now - self.strokes[-1].t < 0.6 and path_len(self.strokes[-1]) < 30:
                self.strokes.pop()
            self.grabbed = nearest_stroke(self.strokes, self.palm_pt)
            self.last_palm, self.dirty = self.palm_pt, True

        if mode == "draw" and self.raw == "draw" and self.pen_pt is not None:
            if self.current is None:
                self.current = Stroke(PEN_COLOR, PEN_SIZE)
                self.current.t = now
                self.strokes.append(self.current)
            n0 = len(self.current.pts)
            add_point(self.current, self.pen_pt)
            if len(self.current.pts) > n0:
                self.current.t = now
                paint_tail(self.canvas, self.mask, self.current, n0)
        elif mode == "erase" and self.erase_pt is not None:
            self.strokes, changed = erase_at(self.strokes, self.erase_pt, ERASER_RADIUS)
            self.dirty = self.dirty or changed
        elif (mode == "grab" and self.grabbed is not None and self.palm_pt is not None
              and self.last_palm is not None):
            dx, dy = self.palm_pt[0] - self.last_palm[0], self.palm_pt[1] - self.last_palm[1]
            if dx or dy:
                self.grabbed.pts = [(x + dx, y + dy) for x, y in self.grabbed.pts]
                self.dirty = True
            self.last_palm = self.palm_pt
        self._finish()

    def _finish(self):
        if self.dirty:
            render_into(self.canvas, self.mask, self.strokes, self.grabbed)
            self.dirty = False

    def compose(self, frame):
        out = frame.copy()
        if self.strokes:
            sel = self.mask > 0
            out[sel] = self.canvas[sel]
        return out


# ======================= drawing the preview =======================
def draw_cursor(img, eng):
    col = COLORS.get(eng.mode)
    if col is None:
        return
    if eng.mode == "erase" and eng.erase_pt is not None:
        cv2.circle(img, to_int(eng.erase_pt), ERASER_RADIUS, col, 2)
    elif eng.mode == "grab" and eng.palm_pt is not None:
        cv2.circle(img, to_int(eng.palm_pt), 12, col, -1)
    elif eng.pen_pt is not None:
        cv2.circle(img, to_int(eng.pen_pt), 8, col, -1 if eng.mode == "draw" else 2)


def draw_hand(img, hand):
    for x, y in hand.img:
        cv2.circle(img, (int(x), int(y)), 3, (255, 200, 0), -1)


def put(img, text, y, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def overlay(preview, eng, fps, debug, now):
    h = preview.shape[0]
    if eng.calib is not None:
        for i, line in enumerate(eng.calib.lines(now)):
            put(preview, line, 36 + i * 34, 0.85 if i == 0 else 0.75,
                (0, 255, 255) if i == 0 else (255, 255, 255))
    else:
        put(preview, "mode: %s    fps: %.0f" % (eng.mode, fps), 32)
        if eng.note:
            put(preview, eng.note, 64, 0.6, (0, 255, 255), 1)
        if debug and eng.debug:
            label, d = eng.debug
            put(preview, "closest: %s | %s" % (label, "  ".join("%s %.1f" % (k, v) for k, v in sorted(d.items()))),
                92, 0.55, (200, 255, 200), 1)
    put(preview, "c clear | u undo | k calibrate | d debug | e enhance | q quit",
        h - 14, 0.55, (200, 200, 200), 1)


def enhance(img, clahe):
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)


# ======================= main =======================
def main():
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    cam = Camera(cam_index)
    if not cam.opened:
        raise SystemExit("Could not open camera %d. Close other apps using it, or try: python skywriter.py 1" % cam_index)
    tracker = make_tracker()
    eng = Engine()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    use_clahe, debug = False, True
    fps, t_last, last_frame_t = 0.0, time.perf_counter(), time.perf_counter()

    try:
        while True:
            frame = cam.read()
            now = time.perf_counter()
            if frame is None:
                if now - last_frame_t > 5:
                    raise SystemExit("The camera stopped delivering frames.")
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            last_frame_t = now
            h, w = frame.shape[:2]
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_last, 1e-3))
            t_last = now

            small = cv2.resize(frame, (PROC_W, int(h * PROC_W / w))) if w > PROC_W else frame
            rgb = enhance(small, clahe) if use_clahe else cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            hand = tracker.process(rgb, w, h)
            eng.step(now, hand, frame.shape)

            composite = eng.compose(frame)
            output = cv2.flip(composite, 1) if MIRROR_OUTPUT else composite  # noqa: F841  (virtual camera, next step)

            ui = composite.copy()
            if eng.calib is None:
                draw_cursor(ui, eng)
            if hand is not None and (debug or eng.calib is not None):
                draw_hand(ui, hand)
            preview = cv2.flip(ui, 1)
            overlay(preview, eng, fps, debug, now)
            cv2.imshow(WIN, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                eng.clear()
            elif key == ord("u"):
                eng.undo()
            elif key == ord("k"):
                eng.start_calibration()
            elif key == ord("e"):
                use_clahe = not use_clahe
                eng.note = "Backlight enhancement " + ("ON" if use_clahe else "OFF")
            elif key == ord("d"):
                debug = not debug
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
