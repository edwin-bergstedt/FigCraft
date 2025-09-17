🖼️ FigCraft

FigCraft is a Python tool that simplifies creating image collages for reports, presentations, and documentation.
It supports flexible layouts, text customization, and high-resolution export for professional results.

✅ Features

Flexible grid layouts (e.g., 2x2, 3x2, 4x1).
Global and per-image text customization:

Font, size, color
Position (top/middle/bottom × left/center/right)
Alignment (left, center, right)

Adjustable margins and spacing.
Configurable image order.
High-resolution export with DPI and scaling.
Optional text stroke (border) and shadow for readability.
Multiple configuration files for different layouts.


📂 Folder Structure

project/
├─ main.py
├─ config.txt
└─ photos/
   ├─ image1.jpg
   ├─ image2.jpg
   └─ ...


⚙️ Configuration Basics
All settings are defined in a config.txt file.
[GLOBAL]
grid_shape: Layout of the collage (e.g., 2x2, 3x2).
margin: Spacing between cells (in pixels).
text settings: Color, size, font, alignment, position.
output_file: Name of the exported collage image.

[IMAGES]
order: List of image filenames in the desired order.

Example:
[GLOBAL]
grid_shape = 2x2
margin = 24
text_color = white
text_size = 36
text_alignment = center
text_position = bottom-center
output_file = collage_output.png

[IMAGES]
order = 
	img1.jpg
	img2.jpg
	img3.jpg
	img4.jpg

🖼️ Image Ordering
Images are placed row by row, left to right:

2×2 grid:
1  2
3  4

4×1 grid:
1
2
3
4

⚙️ Install
To set up the project using uv:
1. Install uv (if not already installed)
	curl -LsSf https://astral.sh/uv/install.sh | sh
Or follow instructions from uv's GitHub page.

2. Sync dependencies and create virtual environment
	uv sync
This will:

Create a virtual environment
Install all required packages from pyproject.toml and requirements.txt (if present)

▶️ Usage
Run the script with the desired configuration file:
	uv run main.py
Or with a custom config file:
	uv run main.py config_example.txt

If no file is specified, it defaults to config.txt.

🔍 Advanced Features
✅ Per-Image Overrides
Override global settings for specific images:
[image:img2.jpg]
text_string = Custom caption for image 2
text_color = #ff0000
text_size = 48
text_alignment = left
text_position = top-left

✅ High-Resolution Export
For print-quality output:
render_scale = 2.0      ; doubles pixel dimensions
output_dpi = 300        ; embed 300 DPI for printing
jpeg_quality = 95
jpeg_subsampling = 0    ; best for text

✅ Text Styling
Enhance readability:
text_stroke_width = 2
text_stroke_color = black
text_shadow = 1
text_shadow_offset = 3,3
text_shadow_color = rgba(0,0,0,0.6)
text_shadow_radius = 1

✅ Dim Overlay
Darken images slightly to make text pop:
INIdim_overlay_alpha = 80   ; 0 = off, 255 = fully black

✅ Multiple Configurations
Create multiple .txt configs for different layouts or styles:

	python main.py config_presentation.txt
	python main.py config_report.txt

✅ Debug Mode
Enable debug guides to visualize caption placement:
	debug_caption_boxes = 1

🖨️ Print-Quality Tips

A4 @ 300 DPI → ~3508×2480 px → render_scale ≈ 1.4 for a 2×3 grid with 800 px cells.
A3 @ 300 DPI → ~4961×3508 px → render_scale ≈ 2.0.
Use PNG for sharp text or JPEG with subsampling=0 for smaller files.