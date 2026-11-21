<h1 align="center">Skywriter</h1>

<p align="center">
  Draw in the air with your hand.<br>
  Your webcam tracks your hand, and simple gestures control the drawing.
</p>

<!-- Add a 5 to 10 second GIF of you drawing, then remove this line and the closing arrow below:
<p align="center">
  <img src="assets/demo.gif" alt="Skywriter demo" width="640">
</p>
-->

## What it does

Skywriter turns your webcam into a canvas. It finds your hand in each video frame, follows your fingertip, and draws wherever you point. Gestures decide what the app does, such as drawing or stopping.

It runs on your own computer.

## How it works

1. **Capture:** OpenCV reads frames from your webcam.
2. **Track:** MediaPipe finds 21 landmarks on your hand in each frame.
3. **Decide:** simple rules on those landmarks (for example, which fingers are up) pick the gesture.
4. **Draw:** the app draws on a canvas and shows it over the live video.

The gestures come from plain rules, not a trained model, so you can read and change them in the code.

## Gestures

<!-- Fill this in with your real gestures: -->

| Gesture | Action |
| --- | --- |
| *(your gesture)* | *(what it does)* |
| *(your gesture)* | *(what it does)* |

## Getting started

You need Python 3 and a webcam. MediaPipe works best with Python 3.9 to 3.12.

```bash
git clone https://github.com/Shriniketan369/skywriter.git
cd skywriter
pip install -r requirements.txt
python skywriter.py
```

Use a well-lit room and keep your hand clear of the edges of the frame.

## Built with

- [OpenCV](https://opencv.org) for video capture and drawing
- [MediaPipe](https://developers.google.com/mediapipe) for hand tracking
- [NumPy](https://numpy.org) for the canvas

## Ideas for next

- Tune the gestures so they work in more lighting
- Add colors and an eraser
- Save the drawing as an image

## License

MIT
