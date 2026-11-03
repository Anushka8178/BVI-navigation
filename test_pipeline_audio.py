from navigation.hrtf_renderer import HRTFRenderer


renderer = HRTFRenderer()

print("\n=== HRTF PIPELINE AUDIO TEST ===\n")

print("Obstacle LEFT")
renderer.play_warning(-60)

print("Obstacle AHEAD")
renderer.play_warning(0)

print("Obstacle RIGHT")
renderer.play_warning(60)

print("\n=== TEST COMPLETE ===")