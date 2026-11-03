import sounddevice as sd


def find_output_device(name="OPPO Enco Buds2"):
    devices = sd.query_devices()

    for index, device in enumerate(devices):
        if device["max_output_channels"] > 0:
            if name.lower() in device["name"].lower():
                return index

    return None


device = find_output_device()

print("Detected output device:", device)

if device is not None:
    print(sd.query_devices(device))
else:
    print("Earbuds not found")