# 🖼️ FigCraft<br>
FigCraft is a Python tool that simplifies creating image collages for reports, presentations, and documentation.<br>
It supports flexible layouts, text customization, and high-resolution export for professional results.<br>

## ✅ Features<br>
Flexible grid layouts (e.g., 2x2, 3x2, 4x1).

### Global and per-image text customization:<br>
Font, size, color<br>
Position (top/middle/bottom × left/center/right)<br>
Alignment (left, center, right)<br>
Adjustable margins and spacing.<br>
Configurable image order.<br>
High-resolution export with DPI and scaling.<br>
Optional text stroke (border) and shadow for readability.<br>
Multiple configuration files for different layouts.<br>

## 📂 Folder Structure<br>
project/<br>
&nbsp;&nbsp;&nbsp;&nbsp;├─ main.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;├─ config.txt<br>
&nbsp;&nbsp;&nbsp;&nbsp;└─ photos/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ image1.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ image2.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ ...<br>


## ⚙️ Configuration Basics<br>
All settings are defined in a config.txt file.<br>

### [GLOBAL]<br>
grid_shape: Layout of the collage (e.g., 2x2, 3x2).<br>
margin: Spacing between cells (in pixels).<br>
text settings: Color, size, font, alignment, position.<br>
output_file: Name of the exported collage image.<br>

### [IMAGES]<br>
order: List of image filenames in the desired order.<br>

## Example:<br>
### [GLOBAL]<br>
grid_shape = 2x2<br>
margin = 24<br>
text_color = white<br>
text_size = 36<br>
text_alignment = center<br>
text_position = bottom-center<br>
output_file = collage_output.png<br>

### [IMAGES]<br>
order = <br>
&nbsp;&nbsp;&nbsp;&nbsp;img1.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;img2.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;img3.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;img4.jpg<br>

## 🖼️ Image Ordering<br>
### Images are placed row by row, left to right:<br>
**2×2 grid:**<br>
&nbsp;&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;2<br>
&nbsp;&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;4<br>

**4×1 grid:**<br>
&nbsp;&nbsp;&nbsp;&nbsp;1<br>
&nbsp;&nbsp;&nbsp;&nbsp;2<br>
&nbsp;&nbsp;&nbsp;&nbsp;3<br>
&nbsp;&nbsp;&nbsp;&nbsp;4<br>

### Or if the per image override of position is used, the images can be placed freely:<br>
**3x3 grid:**<br>
&nbsp;&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;X&nbsp;&nbsp;X<br>
&nbsp;&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;5&nbsp;&nbsp;X<br>
&nbsp;&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;2&nbsp;&nbsp;6<br>

## ⚙️ Install<br>
### To set up the project using uv:

1. Git clone the FigCraft repository
```sh
	git clone https://github.com/edwin-bergstedt/FigCraft.git
```
2. Install uv (if not already installed)
```sh
	curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or follow instructions from uv's GitHub page: https://github.com/astral-sh/uv

3. cd to the FigCraft folder
4. Sync dependencies and create virtual environment
```python
	uv sync
```

### This will:<br>
Create a virtual environment<br>
Install all required packages from pyproject.toml and requirements.txt (if present)<br>

## ▶️ Usage<br>
### Run the script with the desired configuration file:<br>

	uv run main.py

Or with a custom config file:<br>

	uv run main.py config_example.txt

If no file is specified, it defaults to config.txt.<br>

## 🔍 Advanced Features<br>
## ✅ Per-Image Overrides<br>
### Override global settings for specific images:<br>

### [image:img2.jpg]<br>
text_string = Custom caption for image 2<br>
text_color = #ff0000<br>
text_size = 48<br>
text_alignment = left<br>
text_position = top-left<br>

## ✅ High-Resolution Export<br>
### For print-quality output:<br>
render_scale = 2.0      ; doubles pixel dimensions<br>
output_dpi = 300        ; embed 300 DPI for printing<br>
jpeg_quality = 95<br>
jpeg_subsampling = 0    ; best for text<br>

## ✅ Text Styling<br>
### Enhance readability:<br>
text_stroke_width = 2<br>
text_stroke_color = black<br>
text_shadow = 1<br>
text_shadow_offset = 3,3<br>
text_shadow_color = rgba(0,0,0,0.6)<br>
text_shadow_radius = 1<br>

## ✅ Dim Overlay<br>
### Darken images slightly to make text pop:<br>
INIdim_overlay_alpha = 80   ; 0 = off, 255 = fully black<br>

## ✅ Multiple Configurations<br>
### Create multiple .txt configs for different layouts or styles:<br>

	puv run main.py config_presentation.txt

or:<br>

	uv run main.py config_report.txt

## ✅ Debug Mode<br>
### Enable debug guides to visualize caption placement:<br>
debug_caption_boxes = 1

## 🖨️ Print-Quality Tips<br>
A4 @ 300 DPI → ~3508×2480 px → render_scale ≈ 1.4 for a 2×3 grid with 800 px cells.<br>
A3 @ 300 DPI → ~4961×3508 px → render_scale ≈ 2.0.<br>
Use PNG for sharp text or JPEG with subsampling=0 for smaller files.<br>