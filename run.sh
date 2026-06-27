#!/bin/bash
xhost +local:docker
docker run --rm \
  --device=/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/snd \
  -v /run/user/$(id -u)/pulse:/run/user/$(id -u)/pulse \
  -e PULSE_SERVER=unix:/run/user/$(id -u)/pulse/native \
  --device /dev/dri \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  driver-fatigue-ai
