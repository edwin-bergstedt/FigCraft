#!/usr/bin/env python3

import configparser
import os
import sys

from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(SCRIPT_DIR, "photos")


def parse_color(
    s: str,
    default: tuple[int, int, int, int] = (0, 0, 0, 255),
) -> tuple[int, int, int, int]:
    """
    Parse color strings into RGBA.

    Accepts:
      - Named colors: 'white', 'red', ...
      - Hex: '#fff', '#ffffff', '#ffffffff'
      - rgb/rgba: 'rgb(255,255,255)', 'rgba(255,255,255,0.5)'
      - CSV/tuple: '255,255,255', '(255,255,255)', '(255,255,255,128)'

    Returns RGBA tuple or 'default' on failure.
    """

    def clamp8(x: float) -> int:
        return max(0, min(255, int(round(x))))

    if not s:
        return default

    s = s.strip()

    # Strip surrounding quotes if present
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()

    # Try Pillow's parser first (handles named, hex, rgb/rgba)
    try:
        r, g, b, a = ImageColor.getcolor(s, "RGBA")
        return (r, g, b, a)
    except Exception:
        pass

    # Try comma-separated numeric formats: 255,255,255[,a] or with brackets/parens
    try:
        t = s
        if t and t[0] in "([{":
            t = t[1:]
        if t and t[-1] in ")]}":
            t = t[:-1]
        parts = [p.strip() for p in t.split(",") if p.strip() != ""]
        if 3 <= len(parts) <= 4:
            vals: list[int] = []
            for i, p in enumerate(parts):
                if i == 3 and (("." in p) or ("e" in p.lower())):
                    # alpha as float 0..1
                    alpha01 = float(p)
                    vals.append(clamp8(alpha01 * 255.0))
                else:
                    vals.append(clamp8(float(p)))
            if len(vals) == 3:
                vals.append(255)
            r, g, b, a = vals[:4]
            return (r, g, b, a)
    except Exception:
        pass

    # Fall back
    return default


def load_font(path: str | None, size: int) -> ImageFont.FreeTypeFont:
    """Try to load the requested font; fall back to common fonts or default."""
    candidates = []
    if path and os.path.isfile(path):
        candidates.append(path)

    # Common fallbacks
    candidates.extend(
        [
            # DejaVu is commonly packaged with PIL on many systems
            os.path.join(os.path.dirname(ImageFont.__file__), "DejaVuSans.ttf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Arial.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        ],
    )

    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue

    # Last resort: default font (bitmap, not scalable)
    return ImageFont.load_default()


def parse_grid(config: configparser.ConfigParser) -> tuple[int, int]:
    g = config["GLOBAL"]
    rows = g.getint("rows", fallback=None)
    cols = g.getint("cols", fallback=None)
    grid_shape = g.get("grid_shape", fallback="").strip().lower()
    if (rows is None or cols is None) and grid_shape:
        if "x" in grid_shape:
            parts = grid_shape.split("x")
            if len(parts) == 2:
                try:
                    rows = int(parts[0])
                    cols = int(parts[1])
                except ValueError:
                    pass
    if rows is None or cols is None:
        raise ValueError(
            "Grid shape not specified. Use GLOBAL.grid_shape='RxC' or set rows & cols.",
        )
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive integers.")
    return rows, cols


def read_config(path: str) -> configparser.ConfigParser:
    # Allow full-line comments with ';' or '#'
    # Allow inline comments ONLY with ';' so hex colors like '#ffffff' are preserved
    config = configparser.ConfigParser(
        comment_prefixes=("#", ";"),
        inline_comment_prefixes=(";",),  # note the trailing comma to make it a tuple
        interpolation=None,
    )
    with open(path, encoding="utf-8") as f:
        config.read_file(f)

    if "IMAGES" not in config:
        config.add_section("IMAGES")
    return config


def get_image_order_and_positions(config, rows, cols):
    """
    Returns:
      ordered: list of filenames in placement order (row-major), possibly empty strings for explicit slots
      explicit_map: dict[(row,col)] = filename for images with explicit row/col (1-based)
      per_image: dict[filename] = dict of overrides

    """
    per_image: dict[str, dict] = {}
    explicit_map: dict[tuple[int, int], str] = {}

    # Gather per-image overrides
    for section in config.sections():
        if section.startswith("image:"):
            fname = section.split("image:", 1)[1].strip()
            s = config[section]
            per_image[fname] = {
                "text_string": s.get("text_string", fallback=None),
                "text_color": s.get("text_color", fallback=None),
                "text_size": s.getint("text_size", fallback=None),
                "text_font": s.get("text_font", fallback=None),
                "text_alignment": s.get("text_alignment", fallback=None),
                "text_position": s.get("text_position", fallback=None),
                "row": s.getint("row", fallback=None),
                "col": s.getint("col", fallback=None),
            }
            if per_image[fname]["row"] and per_image[fname]["col"]:
                r = per_image[fname]["row"]
                c = per_image[fname]["col"]
                if not (1 <= r <= rows and 1 <= c <= cols):
                    raise ValueError(
                        f"Explicit position out of bounds for {fname}: row={r}, col={c}",
                    )
                key = (r, c)
                if key in explicit_map:
                    raise ValueError(
                        f"Duplicate explicit position {key}: already occupied by {explicit_map[key]}",
                    )
                explicit_map[key] = fname

    # Build base ordered list from [IMAGES].order
    raw_order = config["IMAGES"].get("order", fallback="")
    # Split by lines, filter empties/comments
    ordered = []
    for line in raw_order.splitlines():
        line = line.strip()
        if not line or line.startswith(";") or line.startswith("#"):
            continue
        ordered.append(line)

    # Filter to those actually present in ./photos
    present_files = set(os.listdir(PHOTOS_DIR)) if os.path.isdir(PHOTOS_DIR) else set()
    ordered = [f for f in ordered if f in present_files]

    # Now fill a grid: place explicit positions first, then fill remaining slots row-major with the ordered list
    grid = [[None for _ in range(cols)] for _ in range(rows)]

    # Bookkeep which ordered items are already explicitly placed
    placed_set = set(explicit_map.values())

    # Place explicit
    for (r, c), fname in explicit_map.items():
        grid[r - 1][c - 1] = fname

    # Fill remaining by traversing the ordered list
    it = (f for f in ordered if f not in placed_set)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] is None:
                try:
                    grid[r][c] = next(it)
                except StopIteration:
                    grid[r][c] = None  # leave empty if not enough images

    # Flatten back row-major
    flat = []
    for r in range(rows):
        for c in range(cols):
            flat.append(grid[r][c])

    return flat, explicit_map, per_image


def get_global_defaults(config: configparser.ConfigParser):
    g = config["GLOBAL"]

    def _get_int(section, key, fallback):
        raw = section.get(key, fallback=None)
        if raw is None:
            return fallback
        try:
            return int(float(str(raw).strip()))
        except Exception:
            return fallback

    def _get_float01(section, key, fallback):
        """
        Get a float, accept '0.9' or '90%' and clamp to 0..1.
        """
        raw = section.get(key, fallback=None)
        if raw is None:
            return fallback
        s = str(raw).strip()
        try:
            if s.endswith("%"):
                val = float(s[:-1].strip()) / 100.0
            else:
                val = float(s)
            return max(0.0, min(1.0, val))
        except Exception:
            return fallback

    margin = _get_int(g, "margin", 16)
    cell_w = _get_int(g, "cell_width", 800)
    cell_h = _get_int(g, "cell_height", 600)
    output_file = g.get("output_file", fallback="collage_output.png")

    # --- Colors: handle empty values safely
    raw_bg = g.get("background_color", fallback="").strip()
    if not raw_bg:
        raw_bg = "#ffffff"
    bg_color = parse_color(raw_bg, default=(255, 255, 255, 255))

    text_string = g.get("text_string", fallback=None)

    raw_text_color = g.get("text_color", fallback="").strip()
    if not raw_text_color:
        raw_text_color = "#111111"
    text_color = parse_color(raw_text_color, default=(17, 17, 17, 255))

    text_size = _get_int(g, "text_size", 36)
    text_font = g.get("text_font", fallback=None)

    # --- Alignment normalization & validation
    raw_align = g.get("text_alignment", fallback="center")
    raw_align = (raw_align or "center").strip().lower()
    if raw_align == "middle":
        raw_align = "center"
    text_alignment = raw_align if raw_align in ("left", "center", "right") else "center"

    # --- Position normalization & validation: '<v>-<h>' where v in top/center/bottom, h in left/center/right
    raw_pos = g.get("text_position", fallback="bottom-center")
    raw_pos = (raw_pos or "bottom-center").strip().lower().replace("middle", "center")
    valid_vs = {"top", "center", "bottom"}
    valid_hs = {"left", "center", "right"}
    parts = raw_pos.split("-")
    if len(parts) == 2 and parts[0] in valid_vs and parts[1] in valid_hs:
        text_position = raw_pos
    else:
        text_position = "bottom-center"

    text_padding = _get_int(g, "text_padding", 12)

    # Accept '0.9' or '90%' and clamp to [0.05, 1.0] for sensible wrapping width
    max_width_pct = _get_float01(g, "text_box_max_width_pct", 0.9)
    max_width_pct = min(1.0, max(0.05, max_width_pct))

    dim_overlay_alpha = _get_int(g, "dim_overlay_alpha", 0)
    dim_overlay_alpha = max(0, min(255, dim_overlay_alpha))

    return {
        "margin": margin,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "output_file": output_file,
        "bg_color": bg_color,
        "text_string": text_string,
        "text_color": text_color,
        "text_size": text_size,
        "text_font": text_font,
        "text_alignment": text_alignment,
        "text_position": text_position,
        "text_padding": text_padding,
        "max_width_pct": max_width_pct,
        "dim_overlay_alpha": dim_overlay_alpha,
    }


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> str:
    """Simple greedy word wrap to fit max_width."""
    if not text:
        return ""
    words = text.split()
    if not words:
        return text
    lines = []
    current = words[0]
    for w in words[1:]:
        candidate = current + " " + w
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return "\n".join(lines)


def compute_anchor_rect(
    x0,
    y0,
    x1,
    y1,
    position: str,
    pad: int,
) -> tuple[int, int, int, int]:
    """
    Given a cell rectangle (x0,y0)-(x1,y1), return a padded anchor rect inside
    where text should be placed, based on position keyword like 'top-left' etc.
    """
    position = position.replace("middle", "center")
    # Pad the rectangle inward
    x0p, y0p = x0 + pad, y0 + pad
    x1p, y1p = x1 - pad, y1 - pad

    # We'll return the same rect but the alignment controls where inside it we anchor the text block.
    # Using the full padded rect means long captions can still wrap nicely.
    return (x0p, y0p, x1p, y1p)


def draw_caption(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    position: str,
    align: str,
    max_width_pct: float,
):
    """
    Draw text inside rect with positional anchor (top-/center-/bottom- × left/center/right).
    We use multiline text and compute exact placement.
    """
    if not text:
        return

    (rx0, ry0, rx1, ry1) = rect
    avail_w = max(1, int((rx1 - rx0) * max(0.05, min(max_width_pct, 1.0))))
    # Wrap
    wrapped = wrap_text(draw, text, font, avail_w)

    # Measure wrapped block
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align=align)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Parse position
    pos = position.lower().replace("middle", "center")
    vpos = "center"
    hpos = "center"
    if "top" in pos:
        vpos = "top"
    elif "bottom" in pos:
        vpos = "bottom"
    if "left" in pos:
        hpos = "left"
    elif "right" in pos:
        hpos = "right"

    # Compute (x, y) for the text block origin
    if hpos == "left":
        x = rx0
    elif hpos == "right":
        x = rx1 - w
    else:
        x = rx0 + (rx1 - rx0 - w) // 2

    if vpos == "top":
        y = ry0
    elif vpos == "bottom":
        y = ry1 - h
    else:
        y = ry0 + (ry1 - ry0 - h) // 2

    draw.multiline_text((x, y), wrapped, font=font, fill=fill, align=align)


def place_image_into_cell(
    img: Image.Image,
    cell_rect: tuple[int, int, int, int],
) -> tuple[Image.Image, tuple[int, int]]:
    """
    Resize img to fit into the cell_rect while preserving aspect ratio.
    Returns the (resized_img, top_left_position) for pasting.
    """
    x0, y0, x1, y1 = cell_rect
    cell_w = x1 - x0
    cell_h = y1 - y0

    # Contain within cell
    fitted = ImageOps.contain(img, (cell_w, cell_h), method=Image.Resampling.LANCZOS)
    fw, fh = fitted.size
    # Center
    px = x0 + (cell_w - fw) // 2
    py = y0 + (cell_h - fh) // 2
    return fitted, (px, py)


def build_collage(config_path: str):
    if not os.path.isdir(PHOTOS_DIR):
        raise FileNotFoundError(f"'photos' folder not found at {PHOTOS_DIR}")

    config = read_config(config_path)
    rows, cols = parse_grid(config)
    defaults = get_global_defaults(config)
    margin = defaults["margin"]
    cell_w = defaults["cell_w"]
    cell_h = defaults["cell_h"]
    output_file = defaults["output_file"]
    bg_color = defaults["bg_color"]
    global_text = defaults["text_string"]
    global_align = defaults["text_alignment"]
    global_pos = defaults["text_position"]
    global_color = defaults["text_color"]
    global_size = defaults["text_size"]
    global_font_path = defaults["text_font"]
    text_padding = defaults["text_padding"]
    max_width_pct = defaults["max_width_pct"]
    dim_alpha = defaults["dim_overlay_alpha"]

    # Compute canvas size with outer margins = margin
    canvas_w = cols * cell_w + (cols + 1) * margin
    canvas_h = rows * cell_h + (rows + 1) * margin

    # Create canvas
    canvas = Image.new("RGBA", (canvas_w, canvas_h), color=bg_color)
    draw = ImageDraw.Draw(canvas)

    # Prepare image order & per-image overrides
    order, explicit_map, per_image = get_image_order_and_positions(config, rows, cols)

    # For convenience, pre-load global font
    global_font = load_font(global_font_path, global_size)

    # Build grid
    idx = 0
    for r in range(rows):
        for c in range(cols):
            x0 = margin + c * (cell_w + margin)
            y0 = margin + r * (cell_h + margin)
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            cell_rect = (x0, y0, x1, y1)

            fname = order[idx] if idx < len(order) else None
            idx += 1

            if not fname:
                # Empty slot, draw a subtle border (optional)
                # draw.rectangle(cell_rect, outline=(200,200,200,255), width=1)
                continue

            fpath = os.path.join(PHOTOS_DIR, fname)
            if not os.path.isfile(fpath):
                # Missing file—skip
                continue

            # Open and place image
            with Image.open(fpath) as im:
                im = im.convert("RGBA")
                fitted, (px, py) = place_image_into_cell(im, cell_rect)
                canvas.alpha_composite(fitted, dest=(px, py))

                # Optional dim overlay to increase text legibility
                if dim_alpha > 0:
                    overlay = Image.new("RGBA", fitted.size, color=(0, 0, 0, dim_alpha))
                    canvas.alpha_composite(overlay, dest=(px, py))

            # Resolve per-image caption settings (fallback to global)
            p = per_image.get(fname, {})

            # --- Caption text resolution (per-image > global, ignore blanks) ---
            raw_text = p.get("text_string")

            if raw_text is not None:
                raw_text = str(raw_text).strip().strip('"').strip("'")

            # Explicit opt-out sentinel
            if isinstance(raw_text, str) and raw_text.lower() == "__none__":
                text_string = None
            else:
                text_string = (
                    raw_text if (raw_text and raw_text.strip() != "") else global_text
                )

            # If no caption at this point, skip drawing for this image
            if text_string is None or str(text_string).strip() == "":
                continue
            # --- Font ---
            text_size = (
                p.get("text_size") if p.get("text_size") is not None else global_size
            )
            text_font_path = (
                p.get("text_font")
                if p.get("text_font") is not None
                else global_font_path
            )
            font = (
                global_font
                if (text_font_path == global_font_path and text_size == global_size)
                else load_font(text_font_path, text_size)
            )

            # --- Color ---
            raw_override_color = p.get("text_color")
            if raw_override_color is not None:
                raw_override_color = str(raw_override_color).strip()
            text_color = (
                parse_color(raw_override_color, default=global_color)
                if raw_override_color
                else global_color
            )

            # --- Alignment (fallback to global, normalize/validate) ---
            raw_align = p.get("text_alignment")
            raw_align = str(raw_align).strip().lower() if raw_align is not None else ""
            align = raw_align if raw_align else global_align
            if align == "middle":
                align = "center"
            if align not in ("left", "center", "right"):
                align = "center"

            # --- Position (fallback to global, normalize/validate) ---
            raw_pos = p.get("text_position")
            raw_pos = str(raw_pos).strip().lower() if raw_pos is not None else ""
            position = raw_pos if raw_pos else global_pos
            position = (
                position.replace("middle", "center") if position else "bottom-center"
            )

            # Validate position '<v>-<h>'
            valid_vs = {"top", "center", "bottom"}
            valid_hs = {"left", "center", "right"}
            parts = position.split("-") if position else []
            if not (len(parts) == 2 and parts[0] in valid_vs and parts[1] in valid_hs):
                position = "bottom-center"

            # Compute caption rect inside the image area (not the full cell if image letterboxed)
            # We’ll place text relative to the actual image pixels
            img_rect = (px, py, px + fitted.size[0], py + fitted.size[1])
            caption_rect = compute_anchor_rect(
                *img_rect,
                position=position,
                pad=text_padding,
            )
            draw_caption(
                draw,
                caption_rect,
                str(text_string),
                font,
                text_color,
                position,
                align,
                max_width_pct,
            )

    # Convert to RGB if saving to JPEG
    ext = os.path.splitext(output_file)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        out_img = canvas.convert("RGB")
    else:
        out_img = canvas

    out_path = os.path.join(SCRIPT_DIR, output_file)
    out_img.save(out_path)
    print(f"Saved collage to: {out_path}")


def main():
    cfg = "config.txt"
    if len(sys.argv) > 1:
        cfg = sys.argv[1]
    config_path = os.path.join(SCRIPT_DIR, cfg)
    if not os.path.isfile(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    build_collage(config_path)


if __name__ == "__main__":
    main()
