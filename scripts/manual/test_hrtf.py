import time

from navigation.hrtf_renderer import HRTFRenderer


renderer = HRTFRenderer()

print("Playing LEFT...")
renderer.play_warning(-60, force=True)
time.sleep(1.2)

print("Playing AHEAD...")
renderer.play_warning(0, force=True)
time.sleep(1.2)

print("Playing RIGHT...")
renderer.play_warning(60, force=True)
time.sleep(1.2)

print("Done.")