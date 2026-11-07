"""Manual check (needs earphones): memory early-warning cue left / ahead / right.

The memory cue is two short, soft, low pips: "a remembered hazard is
ahead that the camera does not see yet". It should sound clearly
different from (and quieter than) the urgent and path-guidance cues,
which are played once at the end for comparison.

    python -m scripts.manual.test_memory_cue
    python -m scripts.manual.test_memory_cue --audio-device 3
"""

import argparse
import time

from navigation.hrtf_renderer import HRTFRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-device", default=None, help="output device index or name")
    args = parser.parse_args()

    renderer = HRTFRenderer(output_device=args.audio_device)
    try:
        for name, azimuth, distance in (
            ("LEFT, 3 m", -60.0, 3.0),
            ("AHEAD, 5 m", 0.0, 5.0),
            ("RIGHT, 8 m (quieter)", 60.0, 8.0),
        ):
            print(f"Memory cue {name}...")
            renderer.play_memory_cue(azimuth, distance, force=True)
            time.sleep(1.5)

        print("For comparison: path guidance AHEAD...")
        renderer.play_path_guidance(0.0, force=True)
        time.sleep(1.5)
        print("For comparison: URGENT AHEAD...")
        renderer.play_warning(0.0, force=True)
        time.sleep(1.5)
    finally:
        renderer.close()
    print("Done.")


if __name__ == "__main__":
    main()
