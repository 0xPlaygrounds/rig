# Local rig-candle pose example

This workspace example estimates human poses with a YOLOv8 pose checkpoint
through the root `rig` facade. It reads the checkpoint and decodes the images
itself; `rig-candle` only receives the checkpoint bytes and RGB frames.

From the repository root:

```bash
./examples/candle_pose/download_model.sh
cargo run --release -p candle_pose
cargo run --release -p candle_pose -- frame-001.jpg frame-002.jpg frame-003.jpg
```

The downloader fetches Candle's safetensors copy of `yolov8n-pose` and a photo
of a cycling race, both pinned to immutable revisions and checked by SHA-256.
Every image named on the command line is one frame of a video; each frame's
people are printed as soon as it is estimated.

The YOLOv8 pose weights are Ultralytics' and are licensed under AGPL-3.0.
Check that license before you ship them.

Set `MODEL_DIR` for both commands to store and load the files elsewhere.
