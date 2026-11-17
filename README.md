# gst-vignettecorrect

Lens falloff (vignetting) correction for DeepStream pipelines on NVIDIA Jetson, with a numpy reference that defines the maths.

Each pixel is multiplied by `1 / falloff` at its radius from the lens centre, using the lens manufacturer's relative-illumination chart. Only lens falloff is corrected; uneven scene lighting is left alone.

## Status

| Part | State |
|---|---|
| Python reference (`vignettecorrect/reference.py`) | Done, tested |
| Per-camera config format (`vignettecorrect/config.py`) | Done, tested |
| GStreamer element `vignettecorrect` (C++/CUDA) | Built and verified on one Jetson; see "Tested on" |

## Python reference

```python
import cv2
from vignettecorrect import read_config, apply_gain

config = read_config("configs/kowa_lm3ncm_wp__bfs_pge_63s4c.txt")
image = cv2.imread("frame.png", cv2.IMREAD_UNCHANGED)  # (H, W, 3) uint8
gain = config.gain_map(*image.shape[:2])               # (H, W) float32, build once per size
corrected = apply_gain(image, gain, knee=config.knee, highlight_mode=config.highlight_mode)
```

Channel order does not matter: the maths uses each pixel's brightest channel and scales all three alike. Gain is applied to the stored 8-bit values, with no linearisation.

Training images and the device must use the same config. Never correct an image twice.

## Config file

One file per camera and lens setup, in GLib key-file syntax. Lists are separated by `;`.

| Key | Meaning | Default |
|---|---|---|
| `image-height-mm` | Chart radii in millimetres, starting at 0 | required |
| `relative-illumination` | Chart brightness at each radius, centre = 1, never rising | required |
| `pixel-size-um` | Pixel size at the resolution of the frames being corrected | required |
| `centre-x`, `centre-y` | Lens centre as a fraction of width and height | 0.5, 0.5 |
| `highlight-mode` | `knee` or `clip` | `knee` |
| `knee` | Highlight protection point, used in `knee` mode | 200 |
| `reference-width`, `reference-height` | Full frame size, when frames are a centred crop | the frame's own size |

If frames are binned or scaled, `pixel-size-um` is the effective pixel size after that, not the sensor's native one.

### Highlight modes

- **`knee`:** full gain until a pixel's brightest channel would reach the knee; brighter originals are compressed into knee–255. Adds no clipping and keeps colour ratios.
- **`clip`:** `clip(round(pixel × gain))` per channel, the pure lens map. Bright pixels near the corners can clip.

A correction with values above 255 is not possible on 8-bit frames.

## GStreamer element

`vignettecorrect` corrects NVMM RGBA frames in place on the GPU. It is Jetson only for now.

| Property | Type | Default | Meaning |
|---|---|---|---|
| `config-file` | string | none, required | Path to the camera's config file |
| `enable` | boolean | true | False passes frames through untouched |
| `cache-mappings` | boolean | true | Keep each pooled buffer mapped into CUDA between frames. False maps and unmaps every frame, which is slower |

```
... ! nvvideoconvert ! "video/x-raw(memory:NVMM),format=RGBA" \
    ! vignettecorrect config-file=/path/to/camera.txt ! ...
```

- It corrects every frame in a buffer, so it works on one stream or on a batch after `nvstreammux`. After the muxer, all streams share one config.
- A missing or invalid config, a non-RGBA frame or a CUDA failure stops the pipeline with an error. It never runs uncorrected silently.
- Frames are modified without a writable copy, so do not put it on one branch of a `tee`.
- Run with `GST_DEBUG=vignettecorrect:4` to log per-frame timings and the number of mappings made when the stream stops.

### Build

On the Jetson, with DeepStream and the CUDA toolkit installed:

```bash
cd plugin
make
GST_PLUGIN_PATH=$PWD gst-inspect-1.0 vignettecorrect
```

`GST_PLUGIN_PATH` loads the element without installing anything system-wide.

### Verify on the device

```bash
python3 tools/verify_on_device.py --video recording.mp4 \
    --config configs/kowa_lm3ncm_wp__bfs_pge_63s4c.txt --plugin-dir plugin
```

This decodes a few frames with and without the element and checks the element's output against the numpy reference. It uses the hardware decoder and the GPU for a few seconds.

## Tests

```bash
python3 -m pytest
```

No GPU, camera or Jetson needed. `tests/test_cpp_host.py` compiles `plugin/host_test`, which runs the same per-pixel C++ code as the CUDA kernel on the CPU, and compares it with numpy. It needs `g++` and the GLib development files, and is skipped without them.

## Limits

- **The chart is a design value:** manufacturer charts are usually simulations in air at one wavelength and one aperture. Stopping the lens down flattens falloff, so a wide-open chart over-corrects the corners.
- **Water and housings are not modelled.**
- **Noise:** corner gain of about 2.5× also amplifies sensor noise and compression artefacts.
- **Lens centre:** assumed at the frame centre unless set in the config.

## Tested on

Jetson Orin NX, L4T R36.4.7, DeepStream 7.1, CUDA 12.6, GStreamer 1.20.3, with 3072×2048 HEVC video from FLIR BFS-PGE-63S4C cameras:

- **Matches the reference:** the element's output equals the numpy reference to within one grey level, with fewer than 0.0001% of values differing. Checked on consecutive frames in knee mode and clip mode, and with an off-centre lens and a reference size.
- **In an inference pipeline:** two streams into `nvstreammux` (batch 2) and `nvinfer` (TensorRT FP16 detector), then `nvtracker` and `nvdsanalytics`. Frames taken downstream of `nvinfer` match the reference in both placements: one element per stream before the muxer, and one element on the batch after it.
- **Cost on its own:** about 0.3 ms of pipeline time per frame when only decoding. With `cache-mappings=false` it is about 3.8 ms; with `enable=false` the cost is nil.
- **Cost with inference:** about 1.2 to 1.4 ms per frame in either placement. The RGBA conversion adds a little on top, and peak memory rises by 50 to 100 MB.
- **Mappings:** each stream recycles 4 buffers, so 4 mappings are made per stream (8 on a batch of two) and all are released at teardown.
- **Not yet tested:** hundreds of pipelines created and destroyed in one long-running process, and live camera sources.

## Licence

MIT; see `LICENSE`. Building the element needs NVIDIA's DeepStream SDK and CUDA toolkit, which are not included and have their own licence terms.
