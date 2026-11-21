"""
Skywriter prototype (step 1)
- Tracks index fingertip with MediaPipe Hands
- Draws smoothed strokes on a transparent canvas
- Gestures: index only = DRAW, index+middle = PAUSE, open palm = ERASE
- Preview is mirrored; the 'output' frame (what a virtual camera would
  send) is NOT mirrored, and strokes are composited BEFORE any flip.

Install:  pip install opencv-python mediapipe numpy
Keys:     q = quit, c = clear all
"""
import cv2
import numpy as np
import mediapipe as mp

PEN_COLOR = (0, 255, 255)   # BGR (yellow) - adaptive ink comes in step 3
PEN_SIZE = 6
ERASER_SIZE = 40
SMOOTHING = 0.35            # 0 = no smoothing, closer to 1 = heavy smoothing

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)


def fingers_up(lm):
    """Return [index, middle, ring, pinky] booleans (assumes upright hand)."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return [lm[t].y < lm[p].y for t, p in zip(tips, pips)]


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam")

    canvas = None          # BGR strokes
    mask = None            # where strokes exist
    prev = None            # last smoothed fingertip point
    smooth = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if canvas is None:
            canvas = np.zeros_like(frame)
            mask = np.zeros((h, w), dtype=np.uint8)

        # Hand tracking runs on the UNMIRRORED frame so coordinates match output.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        mode = "none"
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            idx, mid, ring, pinky = fingers_up(lm)
            raw = np.array([lm[8].x * w, lm[8].y * h])

            # Exponential moving average to remove jitter
            smooth = raw if smooth is None else smooth + (raw - smooth) * (1 - SMOOTHING)
            pt = (int(smooth[0]), int(smooth[1]))

            if idx and not mid and not ring and not pinky:
                mode = "draw"
                if prev is not None:
                    cv2.line(canvas, prev, pt, PEN_COLOR, PEN_SIZE, cv2.LINE_AA)
                    cv2.line(mask, prev, pt, 255, PEN_SIZE, cv2.LINE_AA)
                prev = pt
            elif idx and mid and not ring and not pinky:
                mode = "pause"
                prev = None
            elif idx and mid and ring and pinky:
                mode = "erase"
                palm = (int(lm[9].x * w), int(lm[9].y * h))
                cv2.circle(canvas, palm, ERASER_SIZE, (0, 0, 0), -1)
                cv2.circle(mask, palm, ERASER_SIZE, 0, -1)
                prev = None
            else:
                prev = None
        else:
            prev = None
            smooth = None

        # Composite strokes onto the video (unmirrored = what meeting sees)
        output = frame.copy()
        output[mask > 0] = canvas[mask > 0]

        # Mirrored preview for the user only
        preview = cv2.flip(output, 1)
        cv2.putText(preview, f"mode: {mode}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Skywriter preview (mirrored)", preview)
        # In step 4, `output` is what gets sent to the virtual camera.

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            canvas[:] = 0
            mask[:] = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()