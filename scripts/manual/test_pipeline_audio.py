import time

from navigation.hrtf_renderer import HRTFRenderer


renderer = HRTFRenderer()

print("\n=== HRTF PIPELINE AUDIO TEST ===\n")

print("Obstacle LEFT")
renderer.play_warning(-60, force=True)
time.sleep(0.7)  # playback is async

print("Obstacle AHEAD")
renderer.play_warning(0, force=True)
time.sleep(0.7)  # playback is async

print("Obstacle RIGHT")
renderer.play_warning(60, force=True)
time.sleep(0.7)  # playback is async

print("\n=== TEST COMPLETE ===")