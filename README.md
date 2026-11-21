<h1 align="center">Skywriter</h1>

<p align="center">
  Draw in the air with your hand.<br>
  Your webcam tracks your fingertip, and three simple gestures control the pen.
</p>

<!-- Add a 5 to 10 second GIF of you drawing, then remove this line and the closing arrow below:
<p align="center">
  <img src="assets/demo.gif" alt="Skywriter demo" width="640">
</p>
-->

> **Status:** early prototype (step 1 of 4). Drawing, pausing and erasing work. Adaptive ink and virtual camera output are planned.

## What it does

Skywriter turns your webcam into a canvas. It follows your index fingertip and draws smooth strokes on top of the live video. You switch between drawing, pausing and erasing by changing how many fingers you hold up.

It runs on your own computer.

## Gestures

| Gesture | Action |
| --- | --- |
| Index finger only | **Draw** |
| Index and middle fingers | **Pause** (move without drawing) |
| Open palm (four fingers up) | **Erase** (wipes a circle around your palm) |

| Key | Action |
| --- | --- |
| `q` | Quit |
| `c` | Clear everything |

Keep your hand upright. The gesture rules assume your fingers point up.

## How it works

1. **Capture:** OpenCV reads frames from your webcam.
2. **Track:** MediaPipe Hands finds 21 landmarks on your hand.
3. **Decide:** simple rules compare each fingertip with the joint below it to see which fingers are up. That picks the gesture. There is no trained gesture model, so the rules are easy to read and change.
4. **Smooth:** an exponential moving average removes jitter from the fingertip position, so lines look clean.
5. **Draw:** strokes go on a separate canvas and are placed over the video.

### Mirror-safe output

The preview you see is mirrored, so it feels like a mirror. The frame that a future virtual camera would send is **not** mirrored, and strokes are added to it **before** any flip. That way, writing looks correct to the people on the other end.

## Getting started

You need Python 3 and a webcam. MediaPipe works best with Python 3.9 to 3.12.

```bash
git clone https://github.com/Shriniketan369/skywriter.git
cd skywriter
pip install -r requirements.txt
python skywriter.py
```

Use a well-lit room and keep your hand inside the frame.

## Settings

At the top of `skywriter.py`:

| Setting | What it changes |
| --- | --- |
| `PEN_COLOR` | Ink color, in BGR (default yellow) |
| `PEN_SIZE` | Line thickness |
| `ERASER_SIZE` | Size of the erase circle |
| `SMOOTHING` | `0` is no smoothing. Values closer to `1` are smoother but lag more |

## Roadmap

- [x] Step 1: fingertip tracking, smooth strokes, draw / pause / erase
- [ ] Step 3: adaptive ink color
- [ ] Step 4: send the output to a virtual camera, so you can draw in video calls

## Built with

- [OpenCV](https://opencv.org) for video capture and drawing
- [MediaPipe](https://developers.google.com/mediapipe) for hand tracking
- [NumPy](https://numpy.org) for the canvas

## License

MIT
