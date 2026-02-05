from PIL import Image, ImageDraw
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_gradcam(image_path, image_id):
    """
    Placeholder Grad-CAM generator.
    Simulates a heatmap over the lens area.
    """

    image = Image.open(image_path).convert("RGBA")
    width, height = image.size

    overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Simulated focus area (lens region)
    ellipse_box = [
        width * 0.25,
        height * 0.25,
        width * 0.75,
        height * 0.75
    ]

    draw.ellipse(ellipse_box, fill=(255, 0, 0, 90))

    heatmap = Image.alpha_composite(image, overlay)
    output_path = f"{OUTPUT_DIR}/gradcam_{image_id}"

    heatmap.convert("RGB").save(output_path)

    return output_path
