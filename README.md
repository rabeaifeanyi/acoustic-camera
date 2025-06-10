# Acoustic Camera

This project aims to create an acoustic camera that can visualize sound sources in a room.

## Overview

Here is a brief overview of the project structure:

![alt text](overview.png "Title")

## Requirements

### Hardware

- Microphone Array
- USB Camera (optional)

### Python packages

- Python == 3.11
- Acoular
- TensorFlow (optional)
- OpenCV (optional)
- Flask (optional)

## Environment
run `conda create -n acoustic_camera python=3.11`.
run `conda activate acoustic_camera`.
run `pip install -r requirements.txt`.

## Usage

To run the acoustic camera, clone this repository. Navigate into folder `acoustic-camera`.

To run in default, run `python start.py`.

To run without the flask underlay, use flag `--no-flask`.

To run with a specified model, use flag `--model path-to-folder-with-model`.
