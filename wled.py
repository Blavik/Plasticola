import socket
import time
import json
import os
import colorsys
import math

# ================================
# CONFIGURATION
# ================================

WLED_IP = "192.168.1.101"
WLED_PORT = 4048
FPS = 30
LEDS_PER_STRIP = 240
STRIP_ORDER = ["left", "back", "right", "front"]
STRIP_INDEX = {k: i for i, k in enumerate(STRIP_ORDER)}
SCROLL_SMOOTHING = 0.1

SOFT_BLUE = [30, 60, 180]
DEEP_VIOLET = [102, 0, 165]

NOISE_THRESHOLD = 0.1
BLEND_START = 0.1
BLEND_END = 0.15
VIOLET_START = 0.85
VIOLET_END = 0.9

BASE_BRIGHTNESS = 0.5
BREATH_AMPLITUDE = 1.0
GAMMA = 2.2  # perceptual correction

DOMINANCE_RELATIONS = {
    "front": {"adjacent": "back", "neighbors": ["left", "right"]},
    "back": {"adjacent": "front", "neighbors": ["left", "right"]},
    "left": {"adjacent": "right", "neighbors": ["front", "back"]},
    "right": {"adjacent": "left", "neighbors": ["front", "back"]},
}

# ================================
# COLORWHEEL
# ================================

def generate_color_wheel(steps=720):
    wheel = []
    for h in range(steps):
        base_hue = h / steps
        if base_hue < 0.25:
            sat = 0.9
            val = 1.0
        elif base_hue < 0.5:
            sat = 1.0
            val = 1.0
        elif base_hue < 0.75:
            sat = 1.0
            val = 0.85
        else:
            sat = 0.95
            val = 0.65
        r, g, b = colorsys.hsv_to_rgb(base_hue, sat, val)
        wheel.append([int(g * 255), int(r * 255), int(b * 255)])  # GRB
    return wheel

# ================================
# HELPERS
# ================================

def get_gradient(wheel, offset):
    start = int(offset)
    end = min(start + LEDS_PER_STRIP, len(wheel))
    gradient = wheel[start:end]
    if len(gradient) < LEDS_PER_STRIP:
        gradient += [gradient[-1]] * (LEDS_PER_STRIP - len(gradient))
    return gradient

def blend_gradients(prev, curr, factor):
    return [
        [
            int(p[0] + (c[0] - p[0]) * factor),
            int(p[1] + (c[1] - p[1]) * factor),
            int(p[2] + (c[2] - p[2]) * factor),
        ]
        for p, c in zip(prev, curr)
    ]

def get_coverage():
    try:
        with open("coverage.json", "r") as f:
            raw = json.load(f)
            return {k: float(raw.get(k, 0.0)) for k in STRIP_ORDER}
    except Exception:
        return {k: 0.0 for k in STRIP_ORDER}

def send_ddp_packet(strip_index, led_data):
    offset = strip_index * LEDS_PER_STRIP * 3
    payload_len = len(led_data)
    header = bytearray([
        0x41, 0x01, 0x00, 0x00,
        (offset >> 24) & 0xFF, (offset >> 16) & 0xFF,
        (offset >> 8) & 0xFF, offset & 0xFF,
        (payload_len >> 8) & 0xFF, payload_len & 0xFF
    ])
    sock.sendto(header + bytearray(led_data), (WLED_IP, WLED_PORT))

def apply_gamma(value):
    return pow(value, GAMMA)

def get_pulse_brightness(cov, t):
    if not hasattr(get_pulse_brightness, "phase"):
        get_pulse_brightness.phase = 0.0
        get_pulse_brightness.last_time = t

    dt = t - get_pulse_brightness.last_time
    get_pulse_brightness.last_time = t

    freq = 0.125 + 1.875 * cov  # 0.25–1.0 Hz
    get_pulse_brightness.phase += freq * dt
    get_pulse_brightness.phase %= 1.0

    breath = (math.sin(2 * math.pi * get_pulse_brightness.phase) + 1) / 2
    brightness = BASE_BRIGHTNESS + BREATH_AMPLITUDE * breath
    return min(apply_gamma(brightness), 1.0)

# ================================
# MAIN LOOP
# ================================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
wheel = generate_color_wheel()
max_offset = len(wheel) - LEDS_PER_STRIP
scroll_offsets = {k: 0.0 for k in STRIP_ORDER}
previous_gradients = {k: [[0, 0, 0]] * LEDS_PER_STRIP for k in STRIP_ORDER}
coverage = {k: 0.0 for k in STRIP_ORDER}
frame = 0

while True:
    t = time.time()
    raw_coverage = get_coverage()
    visual_coverage = raw_coverage.copy()

    # === DOMINANCE RULE ===
    dominant_strip = max(raw_coverage, key=raw_coverage.get)
    dominant_value = raw_coverage[dominant_strip]

    for strip in STRIP_ORDER:
        if strip == dominant_strip:
            continue
        relation = DOMINANCE_RELATIONS[dominant_strip]
        influence = None
        if strip == relation["adjacent"]:
            influence = max(0.0, dominant_value - 0.2)
        elif strip in relation["neighbors"]:
            influence = max(0.0, dominant_value - 0.1)
        if influence is not None and raw_coverage[strip] < influence:
            visual_coverage[strip] = influence

    for strip in STRIP_ORDER:
        prev_cov = coverage.get(strip, 0.0)
        target_cov = visual_coverage.get(strip, prev_cov)
        cov = prev_cov + (target_cov - prev_cov) * SCROLL_SMOOTHING
        coverage[strip] = cov

        if cov <= NOISE_THRESHOLD:
            target_gradient = [SOFT_BLUE] * LEDS_PER_STRIP

        elif BLEND_START <= cov < BLEND_END:
            alpha = (cov - BLEND_START) / (BLEND_END - BLEND_START)
            scroll_offset = 0
            gradient = get_gradient(wheel, scroll_offset)
            target_gradient = [
                [
                    int(SOFT_BLUE[0] * (1 - alpha) + color[0] * alpha),
                    int(SOFT_BLUE[1] * (1 - alpha) + color[1] * alpha),
                    int(SOFT_BLUE[2] * (1 - alpha) + color[2] * alpha),
                ]
                for color in gradient
            ]

        elif VIOLET_START <= cov < VIOLET_END:
            beta = (cov - VIOLET_START) / (VIOLET_END - VIOLET_START)
            norm_cov = (VIOLET_START - BLEND_END) / (1.0 - BLEND_END)
            scroll_offset = norm_cov * max_offset * 0.5
            gradient = get_gradient(wheel, scroll_offset)
            target_gradient = [
                [
                    int(color[0] * (1 - beta) + DEEP_VIOLET[0] * beta),
                    int(color[1] * (1 - beta) + DEEP_VIOLET[1] * beta),
                    int(color[2] * (1 - beta) + DEEP_VIOLET[2] * beta),
                ]
                for color in gradient
            ]

        elif cov >= VIOLET_END:
            target_gradient = [DEEP_VIOLET] * LEDS_PER_STRIP

        else:
            norm_cov = (cov - BLEND_END) / (1.0 - BLEND_END)
            scroll_offset = norm_cov * max_offset * 0.5
            scroll_offsets[strip] += (scroll_offset - scroll_offsets[strip]) * SCROLL_SMOOTHING
            scroll_offsets[strip] = max(0, min(scroll_offsets[strip], max_offset))
            target_gradient = get_gradient(wheel, scroll_offsets[strip])

        blended = blend_gradients(previous_gradients[strip], target_gradient, SCROLL_SMOOTHING)
        previous_gradients[strip] = blended

        brightness = get_pulse_brightness(cov, t)

        led_data = []
        for color in blended:
            led_data.extend([
                min(255, max(0, int(c * brightness)))
                for c in color
            ])

        send_ddp_packet(STRIP_INDEX[strip], led_data)

    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"🌀 Frame {frame}")
    print(f"🔥 Dominant: {dominant_strip} ({dominant_value:.2f})")
    for k in STRIP_ORDER:
        print(f"{k:<6} → Raw: {raw_coverage[k]:.2f} | Infl: {visual_coverage[k]:.2f} | Smoothed: {coverage[k]:.2f} | Offset: {int(scroll_offsets[k])}")
    print("=" * 60)

    frame += 1
    time.sleep(max(0, (1.0 / FPS) - (time.time() - t)))
