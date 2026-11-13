# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10"]
# ///
"""gen.py: Generate animated pixel-art hero headers for Rig's README.

Usage:
    uv run img/gen.py
    # or: python3 img/gen.py

Generates:
    img/header-dark.webp        (lossless animated WebP for dark backgrounds)
    img/header-light.webp       (lossless animated WebP for light backgrounds)
    img/header-dark-still.webp  (first frame static fallback)
    img/header-light-still.webp (first frame static fallback)

Features:
- An 8-tooth gear rotating seamlessly (16 frames across one tooth phase = infinite loop).
- Geometric pixel typography for 'RIG' with uniform 8-cell height and aligned baseline.
- Directional lighting highlight on the gear + 1-cell drop shadow.
- Animated terminal prompt cursor.
- Ultra-compact: ~6 KiB per animated WebP file.
"""

import math
import os
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

GLYPHS = {
    "R": [
        "XXXXXXX.",
        "XX....XX",
        "XX....XX",
        "XXXXXXX.",
        "XXXXXX..",
        "XX...XX.",
        "XX....XX",
        "XX....XX",
    ],
    "I": [
        "XXX",
        "XXX",
        "XXX",
        "XXX",
        "XXX",
        "XXX",
        "XXX",
        "XXX",
    ],
    "G": [
        "XXXXXXXX",
        "XX......",
        "XX......",
        "XX......",
        "XX..XXXX",
        "XX....XX",
        "XX....XX",
        "XXXXXXXX",
    ],
}

FRAMES = 16
FPS = 12
SCALE = 8

W_CELLS, H_CELLS = 78, 26
CX, CY = 15, 13
TEETH = 8
R_OUT = 11.5
R_IN = 8.5
R_HUB = 3.5

THEMES = {
    "dark": {
        "ink": (240, 246, 252, 255),  # #f0f6fc (GitHub dark foreground)
        "shade": (50, 25, 20, 255),  # deep warm shadow
        "gear": (220, 162, 130, 255),  # #dca282 (Rust copper)
        "gear_hi": (245, 195, 170, 255),  # bevel highlight
        "accent": (222, 91, 43, 255),  # #de5b2b (terminal cursor orange)
    },
    "light": {
        "ink": (31, 35, 40, 255),  # #1f2328 (GitHub light foreground)
        "shade": (220, 200, 190, 255),  # soft warm shadow
        "gear": (180, 110, 80, 255),  # deeper copper for light mode
        "gear_hi": (210, 140, 110, 255),
        "accent": (200, 70, 30, 255),
    },
}


def make_gear(angle):
    """Generate gear pixel coordinates with bevel highlights."""
    cells = {}
    for y in range(H_CELLS):
        for x in range(W_CELLS):
            dx, dy = x - CX, y - CY
            dist = math.hypot(dx, dy)
            if dist < R_HUB or dist > R_OUT:
                continue
            a = (math.atan2(dy, dx) - angle) % (2 * math.pi)
            tooth_phase = (a / (2 * math.pi) * TEETH) % 1.0
            r_max = R_OUT if (tooth_phase < 0.5) else R_IN
            if dist <= r_max:
                dot = (-dx - dy) / (dist * 1.414)
                cells[(x, y)] = "hi" if dot > 0.35 and dist > R_HUB + 1 else "norm"
    return cells


def get_text_cells(ox=32, oy=9):
    """Generate cell coordinates for the word RIG."""
    cells = set()
    x = ox
    for ch in ["R", "I", "G"]:
        rows = GLYPHS[ch]
        for dy, row in enumerate(rows):
            for dx, c in enumerate(row):
                if c == "X":
                    cells.add((x + dx, oy + dy))
        x += len(rows[0]) + 4
    return cells


def render(theme_name):
    """Render all frames for a given theme."""
    pal = THEMES[theme_name]
    text_cells = get_text_cells()
    frames = []

    for f in range(FRAMES):
        # Rotate by exactly one tooth pitch over FRAMES for a seamless loop
        angle = 2 * math.pi * f / (FRAMES * TEETH)
        gear = make_gear(angle)

        # Blinking cursor matching letter height
        cursor = set()
        if (f // 4) % 2 == 0:
            for cy_offset in range(8):
                cursor.add((68, 9 + cy_offset))
                cursor.add((69, 9 + cy_offset))

        solid = set(gear.keys()) | text_cells | cursor
        shadow = {(x + 1, y + 1) for x, y in solid} - solid

        img = Image.new("RGBA", (W_CELLS, H_CELLS), (0, 0, 0, 0))
        px = img.load()
        assert px is not None

        for x, y in shadow:
            if 0 <= x < W_CELLS and 0 <= y < H_CELLS:
                px[x, y] = pal["shade"]
        for (x, y), kind in gear.items():
            px[x, y] = pal["gear_hi"] if kind == "hi" else pal["gear"]
        for x, y in text_cells:
            px[x, y] = pal["ink"]
        for x, y in cursor:
            px[x, y] = pal["accent"]

        resample = getattr(getattr(Image, "Resampling", Image), "NEAREST")
        big = img.resize((W_CELLS * SCALE, H_CELLS * SCALE), resample)
        frames.append(big)

    return frames


def main():
    for theme_name in THEMES:
        frames = render(theme_name)
        out_anim = os.path.join(HERE, f"header-{theme_name}.webp")
        frames[0].save(
            out_anim,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // FPS,
            loop=0,
            lossless=True,
            background=(0, 0, 0, 0),
        )
        out_still = os.path.join(HERE, f"header-{theme_name}-still.webp")
        frames[0].save(out_still, lossless=True, background=(0, 0, 0, 0))

        w, h = frames[0].size
        print(
            f"Generated {os.path.relpath(out_anim)}: {w}x{h} (shown at {w//2}x{h//2}), "
            f"{len(frames)} frames, {os.path.getsize(out_anim) / 1024:.1f} KiB"
        )


if __name__ == "__main__":
    main()
