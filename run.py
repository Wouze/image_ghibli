from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import time  # Import the time module

# Load pipeline (using a fine-tuned Ghibli style model from HuggingFace)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "nitrosocke/ghibli-diffusion",  # Or use a locally downloaded model
    torch_dtype=torch.float32  # Use float32 instead of float16 for CPU compatibility
).to("cpu")

# Load the input image (or use Image.open("your-image.jpg"))
init_image = Image.open("input.png").convert("RGB")
init_image = init_image.resize((768, 512))  # Resize to match the model

# Prompt to use
prompt = "Convert this image to Studio Ghibli style illustration a peaceful anime village in Studio Ghibli style, soft colors, magical lighting, dreamy landscape"

# Start timing
start_time = time.time()

# Transform the image
# strength: Controls the degree of change from the original image (0.3 = less change, 0.8 = more change)
# guidance_scale: Controls adherence to the prompt (7.5 = high, 1.0 = low)
image = pipe(prompt=prompt, image=init_image, strength=0.3, guidance_scale=7.5).images[0]


# End timing
end_time = time.time()

# Save the result
image.save("ghibli_output.png")

print("✅ Image transformation complete: ghibli_output.png")
print(f"⏱️ Time taken for transformation: {end_time - start_time:.2f} seconds")
#120 secs
