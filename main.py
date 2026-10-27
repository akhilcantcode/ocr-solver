# from transformers import VisionEncoderDecoderModel, TrOCRProcessor
# import torch
# from PIL import Image

# model_name = "anuashok/ocr-captcha-v3"

# # Load model and processor
# processor = TrOCRProcessor.from_pretrained(model_name)
# model = VisionEncoderDecoderModel.from_pretrained(model_name)

# # Load image
# image_path = "captcha_images/assam_20261023_195551_605ee0a5.png"
# image = Image.open(image_path).convert("RGBA")

# # Add white background to handle transparency
# background = Image.new("RGBA", image.size, (255, 255, 255, 255))
# combined = Image.alpha_composite(background, image).convert("RGB")

# # Preprocess image
# pixel_values = processor(
#     images=combined,
#     return_tensors="pt"
# ).pixel_values

# # Generate text
# generated_ids = model.generate(pixel_values)

# # Decode prediction
# generated_text = processor.batch_decode(
#     generated_ids,
#     skip_special_tokens=True
# )[0]

# print("Recognized text:", generated_text)



from pathlib import Path
import json
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# Paths
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "captcha_images"
OUTPUT_FILE = BASE_DIR / "results.json"

MODEL_NAME = "anuashok/ocr-captcha-v3"

# Load model and processor once
print("Loading OCR model...")

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

model.eval()

# Use GPU/MPS on Mac if available; otherwise CPU
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
model = model.to(device)

print(f"Using device: {device}")

# Supported image formats
extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
)

results = {}

print(f"Found {len(image_paths)} images.")

for index, image_path in enumerate(image_paths, start=1):
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGBA")

            # Add a white background for transparent images
            background = Image.new(
                "RGBA", image.size, (255, 255, 255, 255)
            )
            image = Image.alpha_composite(
                background, image
            ).convert("RGB")

        # Preprocess image
        pixel_values = processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(device)

        # Run inference without calculating gradients
        with torch.inference_mode():
            generated_ids = model.generate(pixel_values)

        prediction = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        results[image_path.name] = prediction

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name} -> {prediction}"
        )

    except Exception as error:
        results[image_path.name] = None
        print(f"Error processing {image_path.name}: {error}")

    # Save after every image so progress is not lost
    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

print(f"\nCompleted! Results saved to: {OUTPUT_FILE}")
