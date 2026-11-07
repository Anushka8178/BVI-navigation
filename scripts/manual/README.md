# Manual checks (need hardware — not run by pytest)

These scripts play sound, query audio devices, or open a video file, so they are
**not** automated tests. `python -m pytest -q` never collects them (see `pytest.ini`).

Run them from the repo root with `-m` so `navigation` is importable, e.g.:

```powershell
python -m scripts.manual.test_hrtf
```

| Script | Needs | What it checks |
|---|---|---|
| `check_phone_pose.py` | phone running SensorServer | Prints live heading, x, y, steps (`--seconds 20`) |
| `test_audio_device.py` | audio device | Lists/finds the named output device via `sounddevice` |
| `test_hrtf.py` | earphones | Plays HRTF warnings at left / ahead / right |
| `test_memory_cue.py` | earphones | Memory early-warning cue (two soft low pips) at left / ahead / right, then path and urgent cues for comparison (`--audio-device`) |
| `test_pipeline_audio.py` | earphones | Same as above with console labels |
| `test_navigation_audio.py` | earphones | Fake obstacle through `NavigationPipeline`, prints warnings |
| `test_video_hrtf.py` | earphones | Same idea, positions swept for a video-sync demo |
| `test_synthetic_warning.py` | earphones | Synthetic detections → hazard classifier → HRTF warning |
| `test_video_timeline.py` | `videos/hrtf_navigation_test_video_compatible.mp4` + earphones | Plays warnings in sync with a test video |
