"""Image encoding and VLM comparison utilities for alchemy runners."""

import base64
import os
import sys

# Import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.common import build_client


import io
from PIL import Image

def encode_image(image_path: str, compress: bool = True, quality: int = 95) -> str:
    """Encode image to base64 string for OpenAI API with optional compression.

    Args:
        image_path: Path to the image file.
        compress: Whether to compress the image to JPEG 95. Defaults to True.
        quality: JPEG compression quality (1-100). Defaults to 95.

    Returns:
        Base64 encoded string.
    """
    if compress:
        image = Image.open(image_path)
        img_byte_array = io.BytesIO()
        
        # JPEG doesn't support transparency, convert to RGB
        if image.mode in ['RGBA', 'LA', 'P']:
            if image.mode == 'P':
                image = image.convert('RGBA')
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode == 'LA':
                image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            
        image.save(img_byte_array, format='JPEG', quality=quality, optimize=True)
        img_byte_array.seek(0)
        return base64.b64encode(img_byte_array.read()).decode('utf-8')
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


def vlm_compare_images(image1_path: str, image2_path: str, target_path: str, model: str = "gpt-4o") -> int:
    """Use VLM to compare two images and determine which is closer to target.

    Args:
        image1_path: Path to first image.
        image2_path: Path to second image.
        target_path: Path to target image.
        model: Vision model to use.

    Returns:
        1 if image1 is closer to target, 2 if image2 is closer to target.
    """
    try:
        # Encode images
        image1_b64 = encode_image(image1_path)
        image2_b64 = encode_image(image2_path)
        target_b64 = encode_image(target_path)

        # Initialize OpenAI client
        client = build_client(model)

        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an expert at comparing 3D rendered images. I will show you two rendered images and a target image. Please determine which of the two rendered images is closer to the target image in terms of visual similarity, lighting, materials, geometry, and overall appearance. Respond with only '1' if the first image is closer to the target, or '2' if the second image is closer to the target."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{target_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Target image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image1_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Image 1:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image2_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Image 2:"
                    }
                ]
            }
        ]

        # Make API call
        response = client.chat.completions.create(model=model, messages=messages)

        # Parse response
        result = response.choices[0].message.content.strip()
        if result == "1":
            return 1
        elif result == "2":
            return 2
        else:
            # Default to image1 if response is unclear
            print(f"Unexpected VLM response: {result}, defaulting to image1")
            return 1

    except Exception as e:
        print(f"VLM comparison failed: {e}, defaulting to image1")
        return 1
