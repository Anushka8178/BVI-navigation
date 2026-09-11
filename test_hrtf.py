from navigation.hrtf_renderer import HRTFRenderer


renderer = HRTFRenderer()

print("Playing LEFT...")
renderer.play_warning(-60)

print("Playing AHEAD...")
renderer.play_warning(0)

print("Playing RIGHT...")
renderer.play_warning(60)

print("Done.")