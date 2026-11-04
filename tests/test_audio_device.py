"""Output-device choice: wired preferred, Bluetooth last, CLI override.

Pure logic on fake device lists, so no audio hardware is needed.
"""

import pytest

from navigation.hrtf_renderer import choose_output_device, is_bluetooth_name


def dev(name, outputs=2, inputs=0):
    return {"name": name, "max_output_channels": outputs, "max_input_channels": inputs}


# Typical Windows list: microphone, Bluetooth earbuds (listed as
# "Headphones"), BT hands-free, wired Realtek headphones, speakers.
WINDOWS = [
    dev("Microphone Array (Realtek(R) Audio)", outputs=0, inputs=2),
    dev("Headphones (OPPO Enco Buds2)"),
    dev("Headset (OPPO Enco Buds2 Hands-Free AG Audio)", outputs=1, inputs=1),
    dev("Headphones (Realtek(R) Audio)"),
    dev("Speakers (Realtek(R) Audio)"),
]


def test_prefers_wired_headphones_over_bluetooth():
    index, reason = choose_output_device(WINDOWS, default_output=1)
    assert index == 3
    assert reason == "wired headphones"


def test_bluetooth_names_detected():
    assert is_bluetooth_name("Headphones (OPPO Enco Buds2)")
    assert is_bluetooth_name("Headset (X Hands-Free AG Audio)")
    assert not is_bluetooth_name("Headphones (Realtek(R) Audio)")


def test_non_bluetooth_default_used_when_no_headphones():
    devices = [dev("Headphones (OPPO Enco Buds2)"), dev("Speakers (Realtek)"), dev("HDMI Out")]
    assert choose_output_device(devices, default_output=2) == (2, "system default output")


def test_bluetooth_default_skipped_for_wired_output():
    devices = [dev("Headphones (OPPO Enco Buds2)"), dev("Speakers (Realtek)")]
    assert choose_output_device(devices, default_output=0)[0] == 1


def test_bluetooth_only_as_last_resort():
    devices = [dev("Mic", outputs=0, inputs=1), dev("Headphones (OPPO Enco Buds2)")]
    index, reason = choose_output_device(devices, default_output=1)
    assert index == 1
    assert "Bluetooth" in reason


def test_requested_index_int_and_string():
    assert choose_output_device(WINDOWS, 3, requested=4)[0] == 4
    assert choose_output_device(WINDOWS, 3, requested=" 1 ")[0] == 1  # even Bluetooth, if asked


def test_requested_name_case_insensitive():
    assert choose_output_device(WINDOWS, 3, requested="speakers")[0] == 4
    # First output match; skips the microphone (no outputs).
    assert choose_output_device(WINDOWS, 3, requested="realtek")[0] == 3


@pytest.mark.parametrize("requested", [99, -1, "0", "no-such-device"])
def test_bad_request_raises(requested):
    # 99/-1 out of range, "0" is input-only, last matches nothing.
    with pytest.raises(ValueError):
        choose_output_device(WINDOWS, 3, requested=requested)


def test_no_outputs_raises():
    with pytest.raises(RuntimeError):
        choose_output_device([dev("Mic", outputs=0, inputs=1)], None)


def test_auto_choice_skips_wdm_ks_headphones():
    # This laptop: the only "Headphones" endpoint is raw WDM-KS.
    devices = [
        dev("Speakers (Realtek(R) Audio)") | {"hostapi_name": "MME"},
        dev("Headphones (Realtek HD Audio 2nd output)") | {"hostapi_name": "Windows WDM-KS"},
    ]
    assert choose_output_device(devices, default_output=0) == (0, "system default output")
    # Still selectable explicitly.
    assert choose_output_device(devices, 0, requested=1)[0] == 1


def test_wdm_ks_used_if_nothing_else():
    devices = [dev("Headphones (Realtek)") | {"hostapi_name": "Windows WDM-KS"}]
    assert choose_output_device(devices, None)[0] == 0


def test_keywords_from_config_are_used():
    from navigation.config import load_config

    audio = load_config()["audio"]
    devices = [dev("Speakers (Realtek)"), dev("Earphones (Jabra Wired)"), dev("Jabra Elite (BT)")]
    index, reason = choose_output_device(
        devices,
        default_output=0,
        bluetooth_keywords=audio["bluetooth_keywords"],
        headphone_keywords=audio["headphone_keywords"],
    )
    assert (index, reason) == (1, "wired headphones")

    # A custom Bluetooth keyword ("jabra") now rules out both Jabra devices.
    index, _ = choose_output_device(
        devices,
        default_output=None,
        bluetooth_keywords=["jabra"],
        headphone_keywords=audio["headphone_keywords"],
    )
    assert index == 0
    assert is_bluetooth_name("Jabra Elite (BT)", keywords=["JABRA"])
