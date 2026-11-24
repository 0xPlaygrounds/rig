# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10"]
# ///
"""gen_blade_zoom.py: Generate the animated pixel-art Rig logo used as the README hero.

Usage:
    uv run img/pixel-logo/gen_blade_zoom.py
    # or: python3 img/pixel-logo/gen_blade_zoom.py

Generates:
    img/pixel-logo/blade-zoom-dark.webp        (lossless animated WebP for dark backgrounds)
    img/pixel-logo/blade-zoom-light.webp       (lossless animated WebP for light backgrounds)
    img/pixel-logo/blade-zoom-dark-still.webp  (static mark, for prefers-reduced-motion)
    img/pixel-logo/blade-zoom-light-still.webp (static mark, for prefers-reduced-motion)

The Rig blade stack as a continuous zoom:
- Six blades (three mirrored pairs) drift outward. A new pair opens as a slit
  at the centre and widens into place; the outer pair fades out past the edge.
- Every blade rides a continuous "slot" coordinate: s = 0, 1, 2 are the inner,
  middle and outer rest positions of the static mark. One loop moves every
  blade exactly one slot, so the loop is seamless and frame 0 is the static mark.
- Art is on a cell grid (8 px per cell), but blades move in 1/8-cell steps
  (one output pixel) so the motion reads as continuous. Edges stay hard.
- Pixel 'RIG' wordmark, 1-cell drop shadow, and a terminal cursor that blinks
  once per slot (on while the stack sits on the exact mark).
"""

import math
import os
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

GLYPHS = {
    "R": ["XXXXXXX.", "XX....XX", "XX....XX", "XXXXXXX.", "XXXXXX..", "XX...XX.", "XX....XX", "XX....XX"],
    "I": ["XXX"] * 8,
    "G": ["XXXXXXXX", "XX......", "XX......", "XX......", "XX..XXXX", "XX....XX", "XX....XX", "XXXXXXXX"],
}

SCALE = 8                      # output pixels per cell
SUB = 8                        # motion steps per cell (1/8 cell = one output pixel)
SLOT_SECONDS = 3               # time for every blade to advance one slot
W_CELLS, H_CELLS = 72, 30
ICON_X, ICON_Y, ICON_H = 2, 2, 26
TEXT_X, TEXT_Y, CURSOR_X = 37, 11, 66
TEXT_GUARD = TEXT_X - 2        # a leaving right-hand blade never draws closer to the R than this

# Blade body width per row: chamfered top and bottom, three cells wide in between.
ROW_WIDTHS = [0, 1, 2] + [3] * 20 + [2, 1, 0]
INNER_D, SPACING = 4.5, 5      # rest distances from the centre: 4.5, 9.5, 14.5 cells
LEFT_INNER_X = ICON_X + 10     # edge cell of the inner left blade at rest
RIGHT_INNER_X = ICON_X + 19    # edge cell of the inner right blade at rest
ALPHA_STEPS = 4                # fades are quantised so they stay palette-like

THEMES = {
    "dark": {
        "ink": (240, 246, 252),  # #f0f6fc (GitHub dark foreground)
        "shade": (18, 36, 22),
        "ramp": [(42, 107, 56), (28, 138, 28), (0, 181, 18)],  # outer -> inner
        "edge": (34, 209, 84),  # #22d154
        "accent": (34, 209, 84),
    },
    "light": {
        "ink": (1, 34, 18),
        "shade": (206, 222, 208),
        "ramp": [(24, 67, 31), (20, 99, 20), (0, 139, 14)],
        "edge": (34, 209, 84),
        "accent": (0, 139, 14),
    },
}


def quantise(a):
    if a <= 0:
        return 0.0
    return min(1.0, math.ceil(a * ALPHA_STEPS - 1e-9) / ALPHA_STEPS)


def blade(s):
    """State of the blade pair at slot coordinate s, or None if invisible.

    Returns (m, width, depth, alpha): m is the outward offset from the inner
    rest position in sub-cells (shared by both sides so the stack stays
    mirror-exact), width is the body width in sub-cells, depth indexes the
    colour ramp, alpha is the quantised opacity.
    """
    m = math.floor(SPACING * SUB * s + 0.5)
    d = INNER_D + m / SUB
    if d < 0.5:
        return None
    # bodies always stop short of the 2-cell centre gap, so a new pair opens as a slit
    width = max(0, min(3 * SUB, math.floor((d - 1.5) * SUB)))
    depth = 2 - max(0, min(2, math.floor(s + 0.5)))
    if s < 0:
        alpha = quantise((s + 0.8) / 0.8)  # fade in while opening
    elif s > 2:
        alpha = quantise(1 - (s - 2) / 0.8)  # fade out leaving, gone by s = 2.8
    else:
        alpha = 1.0
    if alpha <= 0:
        return None
    return m, width, depth, alpha


def text_cells():
    cells, x = set(), TEXT_X
    for ch in "RIG":
        rows = GLYPHS[ch]
        for dy, row in enumerate(rows):
            for dx, c in enumerate(row):
                if c == "X":
                    cells.add((x + dx, TEXT_Y + dy))
        x += len(rows[0]) + 4
    return cells


def to_sub(cells):
    """Cell coordinates -> (sub-cell x, cell row)."""
    return {(x * SUB + i, y) for x, y in cells for i in range(SUB)}


def render_frame(pal, txt, u, cursor_on):
    """One frame at slot offset u in [0, 1). Drawn at (sub-cell x, cell row)."""
    width_px = W_CELLS * SUB
    icon = {}
    for j in range(-1, 3):  # the pair being born, the three resting pairs
        st = blade(j + u)
        if st is None:
            continue
        m, width, depth, alpha = st
        for side in (-1, 1):
            # edge cell spans sub-cells x0 .. x0+SUB-1; the body grows inward from it
            x0 = LEFT_INNER_X * SUB - m if side < 0 else RIGHT_INNER_X * SUB + m
            if side > 0 and x0 > TEXT_GUARD * SUB:
                continue
            body0, step = (x0 + SUB, 1) if side < 0 else (x0 - 1, -1)
            for r in range(ICON_H):
                y = ICON_Y + r
                for e in range(SUB):
                    icon[(x0 + e, y)] = (pal["edge"], alpha)
                for b in range(min(ROW_WIDTHS[r] * SUB, width)):
                    icon[(body0 + b * step, y)] = (pal["ramp"][depth], alpha)
    icon = {(x, y): v for (x, y), v in icon.items() if 0 <= x < width_px}

    cursor = to_sub({(CURSOR_X + dx, TEXT_Y + r) for dx in (0, 1) for r in range(8)}) if cursor_on else set()

    img = Image.new("RGBA", (width_px, H_CELLS), (0, 0, 0, 0))
    px = img.load()
    assert px is not None
    solid = set(icon) | txt | cursor
    # 1-cell drop shadow; a fading blade casts a fading shadow
    for x, y in solid:
        sx, sy = x + SUB, y + 1
        if (sx, sy) in solid or not (0 <= sx < width_px and 0 <= sy < H_CELLS):
            continue
        a = icon[(x, y)][1] if (x, y) in icon else 1.0
        if a > px[sx, sy][3] / 255:
            px[sx, sy] = (*pal["shade"], round(a * 255))
    for (x, y), (rgb, a) in icon.items():
        px[x, y] = (*rgb, round(a * 255))
    for x, y in txt:
        px[x, y] = (*pal["ink"], 255)
    for x, y in cursor:
        px[x, y] = (*pal["accent"], 255)

    resample = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    return img.resize((W_CELLS * SCALE, H_CELLS * SCALE), resample)


def render(theme_name):
    pal, txt = THEMES[theme_name], to_sub(text_cells())
    steps = SPACING * SUB  # one frame per motion step
    return [render_frame(pal, txt, f / steps, f < steps // 2) for f in range(steps)]


def main():
    for theme_name in THEMES:
        frames = render(theme_name)
        duration = round(SLOT_SECONDS * 1000 / len(frames))
        out_anim = os.path.join(HERE, f"blade-zoom-{theme_name}.webp")
        frames[0].save(
            out_anim,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            lossless=True,
            background=(0, 0, 0, 0),
        )
        out_still = os.path.join(HERE, f"blade-zoom-{theme_name}-still.webp")
        frames[0].save(out_still, lossless=True, background=(0, 0, 0, 0))

        w, h = frames[0].size
        print(
            f"Generated {os.path.relpath(out_anim)}: {w}x{h} (shown at {w // 2}x{h // 2}), "
            f"{len(frames)} frames @ {duration} ms, {os.path.getsize(out_anim) / 1024:.1f} KiB"
        )


if __name__ == "__main__":
    main()
