#!/usr/bin/env python3
"""Generate the animated GitHub profile banner.

Run from the repository root:
    python scripts/banner/generate.py

Requires numpy, scipy and Pillow (see requirements.txt). The source portrait
at assets/source/ignacio.png must already have its background removed (a
one-off step, done locally — this script does not do background removal).

Started from github.com/emmi-lili/emmi-lili (scripts/banner/generate.py):
the portrait dithering, the optimal-transport morph between silhouettes, and
the terminal-panel SVG layout are all its technique. The icons, palette,
content and geophysics theming are this repo's own.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/source/ignacio.png"
LOGOS = Path(__file__).resolve().parent / "logos"
ASSETS = ROOT / "assets"
DATA = Path(__file__).resolve().parent / "data"

W, H = 1180, 610
INTRO_SECONDS = 3.2
TRAVELLER_COUNT = 2200
SEED = 314159

# Portrait crop box in assets/source/ignacio.png's own pixel space (head +
# shoulders, tight). Recompute this if the source photo changes: find the
# subject's alpha bounding box and re-center a 565x640 window (300:340
# aspect) on the head.
SOURCE_CROP = (243, 243, 243 + 565, 243 + 640)

ROWS = [
    ("Subject", "Ignacio Partarrieu"),
    ("Role", "Geofísico · Investigador"),
    ("Origin", "Concepción, Chile"),
    ("Education", "MSc Geofísica · U. de Concepción"),
    ("Status", "Disponible para nuevas oportunidades"),
    ("Core.Lang", "Python · MATLAB · C/C++"),
    ("Core.Domains", "Modelación · Energías Renovables"),
    ("Core.Tools", "WRF · ArcGIS/QGIS · R"),
    ("Affiliation", "MetGeo"),
    ("Grid.Mail", "ipartarrieua@gmail.com"),
    ("Grid.LinkedIn", "/in/ignacio-partarrieu-andrade"),
    ("Grid.GitHub", "IPartarrieu"),
]

THEMES = {
    "dark": {
        "bg": "#0A141F",
        "panel": "#0D1B28",
        "panel2": "#0F222F",
        "line": "#23394B",
        "muted": "#7E93A0",
        "text": "#DCEAF0",
        "portrait": "#4FB3D9",
        "chrome": "#5FD4C0",
        "accent": "#2F9E8F",
        "shadow": "#02060A",
    },
    "light": {
        "bg": "#F4F8FA",
        "panel": "#FFFFFF",
        "panel2": "#EAF3F5",
        "line": "#CBD9DE",
        "muted": "#57707A",
        "text": "#16232A",
        "portrait": "#1F7A99",
        "chrome": "#0E6B5C",
        "accent": "#1A7F37",
        "shadow": "#B9C7CD",
    },
}


def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    """Serpentine 1-bit Floyd-Steinberg diffusion; True means a lit pixel."""
    work = gray.astype(np.float32) / 255.0
    out = np.zeros_like(work, dtype=bool)
    height, width = work.shape
    for y in range(height):
        left_to_right = y % 2 == 0
        xs = range(width) if left_to_right else range(width - 1, -1, -1)
        direction = 1 if left_to_right else -1
        for x in xs:
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = bool(new)
            err = old - new
            nx = x + direction
            if 0 <= nx < width:
                work[y, nx] += err * 7 / 16
            if y + 1 < height:
                if 0 <= x - direction < width:
                    work[y + 1, x - direction] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= nx < width:
                    work[y + 1, nx] += err * 1 / 16
    return out


def portrait_points(theme: str, rng: np.random.Generator) -> np.ndarray:
    """Return sampled x/y banner coordinates from a 300x340 dither grid."""
    source = Image.open(SOURCE).convert("RGBA")
    crop = source.crop(SOURCE_CROP).resize((300, 340), Image.Resampling.LANCZOS)
    rgb = crop.convert("RGB")
    alpha = np.asarray(crop.getchannel("A"), dtype=np.float32) / 255.0

    if theme == "dark":
        lum = np.asarray(ImageOps.grayscale(rgb), dtype=np.float32)
        prepared = Image.fromarray(np.uint8(np.clip(lum * alpha, 0, 255)), "L")
        select_lit = True
    else:
        white = Image.new("RGBA", crop.size, "white")
        white.alpha_composite(crop)
        prepared = ImageOps.grayscale(white.convert("RGB"))
        select_lit = False

    if theme == "dark":
        mask = Image.fromarray(np.uint8((alpha > 0.08) * 255), "L")
        prepared = ImageOps.equalize(prepared, mask=mask)
    else:
        prepared = ImageOps.autocontrast(prepared, cutoff=1)
    prepared = ImageEnhance.Contrast(prepared).enhance(1.35)
    prepared = prepared.filter(ImageFilter.UnsharpMask(radius=2, percent=175, threshold=1))
    bits = floyd_steinberg(np.asarray(prepared))
    active = bits if select_lit else ~bits
    if theme == "dark":
        active &= alpha > 0.08

    ys, xs = np.where(active)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    points = np.column_stack((74 + xs, 154 + ys)).astype(np.float32)
    if len(points) > 18000:
        points = points[rng.choice(len(points), 18000, replace=False)]
    return points


def ribbon(
    points: list[tuple[float, float]],
    widths: list[float],
    left_extra: list[float] | None = None,
) -> list[tuple[float, float]]:
    """Thicken a centerline into a filled, variable-width polygon.

    A smoother alternative to stroking with overlapping circles (which reads
    as a bumpy rope): offset each point perpendicular to the local tangent by
    half its width, then close the outline via the mirrored return path.
    `left_extra`, if given, grows the left side only (leaving the right/outer
    edge untouched) — for thickening one face of a curve without disturbing
    its outer silhouette.
    """
    left, right = [], []
    n = len(points)
    for i in range(n):
        if i == 0:
            dv = np.array(points[1]) - np.array(points[0])
        elif i == n - 1:
            dv = np.array(points[-1]) - np.array(points[-2])
        else:
            dv = np.array(points[i + 1]) - np.array(points[i - 1])
        norm = np.linalg.norm(dv)
        perp = np.array([-dv[1], dv[0]]) / norm if norm > 1e-6 else np.array([0.0, 0.0])
        w = widths[i] / 2
        extra = left_extra[i] if left_extra else 0.0
        left.append(tuple(np.array(points[i]) + perp * (w + extra)))
        right.append(tuple(np.array(points[i]) - perp * w))
    return left + right[::-1]


def make_geo_icons() -> dict[str, Image.Image]:
    """Draw the geophysics-domain silhouettes: atmosphere, ocean, solid earth,
    and the planet itself.

    Plain black-on-transparent shapes built from primitives (no external
    assets), same approach the reference banner used for its own logos.
    """
    LOGOS.mkdir(parents=True, exist_ok=True)
    size = 400
    icons: dict[str, Image.Image] = {}

    # Sky (atmósfera): sun behind a rain cloud, composition inspired by a
    # classic "cloud + rain" pictogram — bumpy scalloped top, flatter base,
    # three teardrops hanging below.
    sky = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(sky)
    sun_c = np.array([280.0, 95.0])
    sun_r = 60
    for i in range(10):
        a = i * math.tau / 10
        inner = sun_c + (sun_r + 8) * np.array([math.cos(a), math.sin(a)])
        outer = sun_c + (sun_r + 74) * np.array([math.cos(a), math.sin(a)])
        perp = np.array([-math.sin(a), math.cos(a)]) * 17
        d.polygon(
            [tuple(inner - perp), tuple(inner + perp), tuple(outer)], fill="black"
        )
    d.ellipse(
        (sun_c[0] - sun_r, sun_c[1] - sun_r, sun_c[0] + sun_r, sun_c[1] + sun_r),
        fill="black",
    )
    # cloud body: three scalloped bumps over a flat-bottomed base, like a
    # thick-outline weather icon
    for cx, cy, r in [(120, 215, 78), (200, 175, 92), (270, 210, 72)]:
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill="black")
    d.rounded_rectangle((55, 210, 335, 300), radius=45, fill="black")
    # three raindrops hanging below the cloud
    for cx in (130, 200, 270):
        top = np.array([cx, 322.0])
        d.polygon(
            [tuple(top), (cx - 20, 355), (cx - 20, 370), (cx, 385), (cx + 20, 370), (cx + 20, 355)],
            fill="black",
        )
        d.ellipse((cx - 20, 340, cx + 20, 385), fill="black")
    icons["sky"] = sky

    # Wave (océano): one single, generic curving wave — a smooth spiral
    # sweeping up from the water line and curling over at the top, then
    # squashed vertically so it reads wide and low like a real breaking
    # wave (proportions matched to a reference photo) instead of a tall coil.
    wave = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(wave)
    n = 90
    cx, cy = 210.0, 195.0
    start_a, sweep = math.radians(130), math.radians(-310)
    r0, r1 = 270.0, 16.0
    raw, widths = [], []
    for i in range(n):
        t = i / (n - 1)
        a = start_a + sweep * t
        r = r0 * (1 - t) + r1 * t
        raw.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        widths.append(160 * (1 - t) + 10 * t)
    base_y, squash = 400.0, 0.68
    pts = [(x, base_y - (base_y - y) * squash) for x, y in raw]
    # widen the ascending right-hand wall (before the wave reaches its top),
    # growing only toward its inner/left face — 1.5x the plain width at the
    # peak of the bump — so the wave's own outer silhouette doesn't move.
    left_extra = [
        widths[i] * 0.5 * math.exp(-(((i / (n - 1) - 0.20) / 0.11) ** 2))
        for i in range(n)
    ]
    d.polygon(ribbon(pts, widths, left_extra), fill="black")
    d.rectangle((0, round(base_y - (base_y - 312) * squash), 400, 400), fill="black")
    icons["wave"] = wave

    # Volcano (tierra sólida): a flat-topped cone with a pointed lava cap
    # (a flame/teardrop silhouette, not a round dome) bulging out of the
    # summit, and a jagged zigzag seam cut between the lava and the rock
    # (drawn transparent over the fill, same cutout trick as before) so the
    # two parts read as clearly distinct.
    volcano = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(volcano)
    d.polygon(
        [(20, 392), (150, 175), (250, 175), (380, 392)],
        fill="black",
    )
    top_r, top_l = np.array([214.0, 68.0]), np.array([186.0, 68.0])

    def bez(p0, p1, p2, p3, n=24):
        return [
            tuple((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
                  + 3 * (1 - t) * t**2 * p2 + t**3 * p3)
            for t in np.linspace(0, 1, n)
        ]

    right = bez(top_r, np.array([225.0, 105.0]), np.array([275.0, 160.0]), np.array([262.0, 205.0]))
    left = bez(np.array([138.0, 205.0]), np.array([125.0, 160.0]), np.array([175.0, 105.0]), top_l)
    d.polygon(right + left, fill="black")
    cx0, cx1, teeth = 118, 282, 7
    seam_top, seam_bot = [], []
    for i in range(teeth + 1):
        x = cx0 + (cx1 - cx0) * i / teeth
        y = 195 if i % 2 == 0 else 215
        seam_top.append((x, y - 9))
        seam_bot.append((x, y + 9))
    d.polygon(seam_top + seam_bot[::-1], fill=(0, 0, 0, 0))
    icons["volcano"] = volcano

    for name, image in icons.items():
        image.save(LOGOS / f"{name}.png", optimize=True)
    return icons


def sample_logo_points(
    image: Image.Image, rng: np.random.Generator, count: int
) -> np.ndarray:
    """Sample a silhouette into the portrait frame's visual coordinate space."""
    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.where(alpha > 127)
    chosen = rng.choice(len(xs), count, replace=len(xs) < count)
    # Logo is fit (aspect preserved, "contain"-style) into a centered 270x270
    # box inside VISUAL.MAP, whatever the source raster's own resolution/shape.
    scale = min(270.0 / image.width, 270.0 / image.height)
    off_x = 89 + (270.0 - image.width * scale) / 2
    off_y = 188 + (270.0 - image.height * scale) / 2
    return np.column_stack(
        (off_x + xs[chosen] * scale, off_y + ys[chosen] * scale)
    ).astype(np.float32)


def transport(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Order target points by minimum-cost assignment from source points."""
    rows, cols = linear_sum_assignment(cdist(source, target, metric="sqeuclidean"))
    ordered = np.empty_like(target)
    ordered[rows] = target[cols]
    return ordered


def num(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def point_path(points: np.ndarray) -> str:
    """Aggregate adjacent horizontal one-pixel dots into compact SVG path runs."""
    if not len(points):
        return ""
    integer = np.rint(points).astype(int)
    unique = sorted({(int(x), int(y)) for x, y in integer}, key=lambda p: (p[1], p[0]))
    chunks: list[str] = []
    i = 0
    while i < len(unique):
        x0, y = unique[i]
        x1 = x0
        i += 1
        while i < len(unique) and unique[i][1] == y and unique[i][0] <= x1 + 1:
            x1 = unique[i][0]
            i += 1
        chunks.append(f"M{x0} {y}h{x1 - x0 + 1}")
    return "".join(chunks)


def dotted_leader(x1: float, x2: float, y: float) -> str:
    if x2 <= x1:
        return ""
    return "".join(f"M{x} {num(y)}h1" for x in np.arange(x1, x2, 5.0))


def text_width(text: str, font_size: float) -> float:
    return len(text) * font_size * 0.605


def animate_values(points: list[np.ndarray], index: int) -> str:
    return ";".join(f"{num(p[index, 0])} {num(p[index, 1])}" for p in points)


def render_svg(
    theme_name: str,
    portrait: np.ndarray,
    icon_points: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> str:
    t = THEMES[theme_name]
    n = min(TRAVELLER_COUNT, len(portrait))
    source = portrait[rng.choice(len(portrait), n, replace=False)]
    sky = transport(source, icon_points["sky"][:n])
    wave = transport(sky, icon_points["wave"][:n])
    volcano = transport(wave, icon_points["volcano"][:n])

    # Phase schedule, built up rather than hand-typed: portrait hold, then
    # each icon gets a 1.3s arrival + 2.0s hold, then a final 1.3s transition
    # back to portrait. Building every parallel array (frames, opacity, the
    # band-drift below) off this one list keeps them from ever drifting out
    # of sync with each other.
    times = [0.0, 3.0]
    frames = [source, source]
    for target in (sky, wave, volcano):
        times += [times[-1] + 1.3, times[-1] + 1.3 + 2.0]
        frames += [target, target]
    times.append(times[-1] + 1.3)
    frames.append(source)

    loop_seconds = times[-1]
    key_times = ";".join(num(v / loop_seconds) for v in times)
    opacity_values = ";".join(["0", "0"] + ["1"] * (len(times) - 3) + ["0"])

    parts: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
        'aria-labelledby="title desc">',
        "<title id=\"title\">Ignacio Partarrieu's live system profile</title>",
        '<desc id="desc">Animated terminal profile with a dithered portrait that '
        "morphs through the three domains of geophysics: atmosphere (sun and "
        "cloud), ocean (wave) and solid earth (volcano).</desc>",
        "<defs>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">'
        f'<feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" '
        'flood-opacity=".28"/></filter>',
        '<filter id="glow" x="-100%" y="-100%" width="300%" height="300%">'
        f'<feGaussianBlur stdDeviation="3" result="b"/><feFlood flood-color="{t["chrome"]}" '
        'flood-opacity=".35"/><feComposite in2="b" operator="in"/>'
        '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<clipPath id="visualClip"><rect x="49" y="124" width="390" height="414" rx="3"/></clipPath>',
        "</defs>",
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" '
        f'stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/>'
        '<circle cx="59" cy="38" r="6" fill="#FEBC2E"/>'
        '<circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'letter-spacing=".4">perfil.sh --live</text>',
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" '
        f'stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        f'<text x="49" y="111" fill="{t["chrome"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'font-weight="700" letter-spacing="1.2">VISUAL.MAP</text>',
        f'<text x="438" y="111" text-anchor="end" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">300×340 / 1-BIT</text>',
        f'<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12'
        f'M439 539h-12M439 539v-12" fill="none" stroke="{t["chrome"]}" opacity=".55"/>',
        '<g clip-path="url(#visualClip)" shape-rendering="crispEdges">',
        '<g opacity="1">',
    ]

    # Dense portrait drift: 94 independently noisy bands moving toward the sky's
    # centroid, visible only through the portrait's own hold (indices 0-1) and
    # snapping back only on the very last (the return to portrait) — same
    # opacity/translate shape as the travellers, built off the same `times`.
    sky_centroid = sky.mean(axis=0)
    band_ids = rng.integers(0, 94, size=len(portrait))
    noise = rng.normal(0, 4, size=(94, 2))
    band_opacity = ";".join(
        [".94", ".94"] + ["0"] * (len(times) - 3) + [".94"]
    )
    for band in range(94):
        pts = portrait[band_ids == band]
        if not len(pts):
            continue
        centroid = pts.mean(axis=0)
        delta = (sky_centroid - centroid) * 0.18 + noise[band]
        d = point_path(pts)
        drift = f"{num(delta[0])} {num(delta[1])}"
        band_translate = ";".join(
            ["0 0", "0 0", drift, drift] + ["0 0"] * (len(times) - 4)
        )
        parts.append(
            f'<path d="{d}" fill="none" stroke="{t["portrait"]}" stroke-width="1" '
            'opacity=".94">'
            f'<animateTransform attributeName="transform" type="translate" begin="{INTRO_SECONDS}s" '
            f'dur="{loop_seconds}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" values="{band_translate}"/>'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{loop_seconds}s" '
            f'repeatCount="indefinite" keyTimes="{key_times}" '
            f'values="{band_opacity}"/></path>'
        )

    for i in range(n):
        positions = animate_values(frames, i)
        parts.append(
            f'<path d="M-.65-.65h1.3v1.3h-1.3z" fill="{t["portrait"]}">'
            f'<animateTransform attributeName="transform" type="translate" begin="{INTRO_SECONDS}s" '
            f'dur="{loop_seconds}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" values="{positions}"/>'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{loop_seconds}s" '
            f'repeatCount="indefinite" calcMode="linear" keyTimes="{key_times}" '
            f'values="{opacity_values}"/></path>'
        )
    parts.append("</g>")

    intro_ids = rng.integers(0, 60, size=len(portrait))
    order = rng.permutation(60)
    starts = np.empty(60)
    starts[order] = np.linspace(0.05, 1.2, 60)
    for group in range(60):
        pts = portrait[intro_ids == group]
        if not len(pts):
            continue
        parts.append(
            f'<path d="{point_path(pts)}" fill="none" stroke="{t["portrait"]}" '
            'stroke-width="1" opacity="0">'
            f'<animate attributeName="opacity" begin="{num(starts[group])}s" dur=".8s" '
            'values="0;1" fill="freeze"/>'
            '<animate attributeName="opacity" begin="3.08s" dur=".12s" values="1;0" fill="freeze"/>'
            "</path>"
        )

    parts.extend(
        [
            "</g>",
            f'<text x="58" y="551" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="10">'
            f'PTS {len(portrait):05d} · FS/SERPENTINE</text>',
            f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{t["panel2"]}" '
            f'stroke="{t["line"]}"/>',
            f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
            f'<text x="490" y="111" fill="{t["chrome"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
            'font-weight="700" letter-spacing="1.2">SYSTEM.INFO</text>',
            '<g filter="url(#glow)"><circle cx="915" cy="106" r="4" fill="#FF4D5A">'
            '<animate attributeName="opacity" values="1;.3;1" dur="1.6s" repeatCount="indefinite"/>'
            '</circle></g>',
            '<text x="927" y="111" fill="#FF4D5A" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" '
            'font-weight="700">LIVE</text>',
            f'<rect x="982" y="94" width="146" height="24" rx="12" fill="{t["chrome"]}" opacity=".16" '
            f'stroke="{t["chrome"]}"/>',
            f'<text x="1055" y="111" text-anchor="middle" fill="{t["chrome"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="14" '
            'font-weight="700">@IPartarrieu</text>',
        ]
    )

    value_right = 1127.0
    row_y = 153.0
    for label, value in ROWS:
        value_len = text_width(value, 14)
        label_len = text_width(label, 14)
        leader_start = 491 + label_len + 12
        leader_end = value_right - value_len - 12
        parts.extend(
            [
                f'<text x="491" y="{num(row_y)}" fill="{t["muted"]}" '
                'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="14">'
                f"{html.escape(label)}</text>",
                f'<path d="{dotted_leader(leader_start, leader_end, row_y - 4)}" '
                f'fill="none" stroke="{t["line"]}" stroke-width="1" shape-rendering="crispEdges"/>',
                f'<text x="{num(value_right)}" y="{num(row_y)}" text-anchor="end" '
                f'fill="{t["text"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" '
                f'font-size="14" textLength="{num(value_len)}" lengthAdjust="spacingAndGlyphs">'
                f"{html.escape(value)}</text>",
            ]
        )
        row_y += 23

    parts.extend(
        [
            f'<path d="M490 530H1130" stroke="{t["line"]}"/>',
            f'<text x="491" y="548" fill="{t["accent"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">'
            "● ALL SYSTEMS NOMINAL</text>",
            f'<text x="1128" y="548" text-anchor="end" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">'
            "UTC-4 · CHILE</text>",
            "</svg>",
        ]
    )
    return "".join(parts)


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Missing source portrait: {SOURCE}")
    ASSETS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    icons = make_geo_icons()

    portraits: dict[str, np.ndarray] = {}
    for index, theme in enumerate(THEMES):
        rng = np.random.default_rng(SEED + index)
        points = portrait_points(theme, rng)
        portraits[theme] = points
        np.save(DATA / f"portrait-{theme}.npy", points)

    for index, theme in enumerate(THEMES):
        rng = np.random.default_rng(SEED + 100 + index)
        sampled = {
            name: sample_logo_points(image, rng, TRAVELLER_COUNT)
            for name, image in icons.items()
        }
        for name, points in sampled.items():
            np.save(DATA / f"{name}-{theme}.npy", points)
        svg = render_svg(theme, portraits[theme], sampled, rng)
        output = ASSETS / f"banner-{theme}.svg"
        output.write_text(svg, encoding="utf-8")
        byte_size = output.stat().st_size
        print(
            f"{output.relative_to(ROOT)}: {byte_size:,} bytes "
            f"({byte_size / 1024:.1f} KiB), {len(portraits[theme]):,} portrait dots, "
            f"{TRAVELLER_COUNT} travellers"
        )


if __name__ == "__main__":
    main()
