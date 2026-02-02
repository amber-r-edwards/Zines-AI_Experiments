import os
import pytesseract
from anthropic import Anthropic
from PIL import Image
from pathlib import Path
import base64


# Define directories
PROCESSED_IMGS_DIR = "processed_imgs/"  # Directory for color images
RESULTS_VISION_DIR = "results/Claude"  # Directory for Claude vision results

def resize_and_convert_to_jpeg(image_path, max_width=1024, max_height=1024):
    """
    Resize the image to reduce its size while maintaining aspect ratio.
    Convert the image to JPEG format to further reduce file size.
    """
    with Image.open(image_path) as img:
        img.thumbnail((max_width, max_height))  # Resize while maintaining aspect ratio
        jpeg_path = image_path.replace(".png", ".jpg")  # Save as JPEG
        img = img.convert("RGB")  # Ensure compatibility with JPEG
        img.save(jpeg_path, "JPEG", quality=90)  # Adjust quality as needed
        return jpeg_path


def encode_image(image_path):
    """
    Encode an image file to a base64 string.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def resolve_image_path(images_dir, filename):
    """
    Resolve the image path by checking for different file extensions.
    """
    # First check if the filename exists as-is
    candidate = os.path.join(images_dir, filename)
    if os.path.exists(candidate):
        return candidate

    stem = Path(filename).stem
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = os.path.join(images_dir, f"{stem}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def transcribe_with_vision_api(image_path, api_key):
    """
    Transcribe text from an image using the Anthropic Vision API.

    Args:
        image_path (str): Path to the image file.
        api_key (str): Anthropic API key.

    Returns:
        tuple: (transcribed_text, usage_info)
    """
    try:
        # Initialize Anthropic client
        client = Anthropic(api_key=api_key)
        
        # Resize and convert the image to JPEG
        resized_image_path = resize_and_convert_to_jpeg(image_path)

        # Encode the resized image to base64
        base64_image = encode_image(resized_image_path)
        if not base64_image:
            return None, None

        # Update media type after resize (always JPEG after conversion)
        media_type = "image/jpeg"

        # Send request to Anthropic Vision API
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            system="You are a precise transcription assistant specializing in historical documents from the women's liberation movement. Your task is to transcribe text exactly as it appears, without interpretation, correction, or addition.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """TRANSCRIPTION TASK - Read this carefully:

YOUR ONLY TASK: Copy the visible text exactly as it appears.

STRICT RULES (Breaking these is an error):
1. DO NOT add any text that is not visible in the image
2. DO NOT explain what you see
3. DO NOT summarize or paraphrase
4. DO NOT fix spelling or grammar errors
5. DO NOT complete partial words or sentences
6. DO NOT add punctuation that isn't there

WHAT TO INCLUDE:
- Every word, letter, and number you can see
- Original line breaks and spacing
- Original spelling (even if wrong)
- Headers, titles, dates, page numbers
- Handwritten notes (mark as [handwritten: text])

FORMAT:
Start your response immediately with the transcribed text.
Do not write "Here is the transcription:" or similar phrases.
Do not add explanations before or after the transcription.

Transcribe now:"""
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        }
                    ]
                }
            ]
        )

        # Extract usage information
        usage = response.usage
        usage_info = {
            'prompt_tokens': usage.input_tokens,
            'completion_tokens': usage.output_tokens,
            'total_tokens': usage.input_tokens + usage.output_tokens
        }

        # Extract the transcribed text
        transcribed_text = response.content[0].text.strip()
        
        # Print token usage and estimated cost
        print(f"📊 Token Usage: Prompt = {usage_info['prompt_tokens']}, Completion = {usage_info['completion_tokens']}, Total = {usage_info['total_tokens']}")
        
        # Calculate cost based on Claude pricing
        # Pricing as of the knowledge cutoff (check anthropic.com/pricing for current rates)
        # Update the model name below if you change the model above
        model = "claude-sonnet-4-5-20250929"
        pricing = {
            "claude-opus-4-5-20251101": {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
            "claude-sonnet-4-5-20250929": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
            "claude-haiku-4-5-20251001": {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000}
        }
        
        model_pricing = pricing.get(model, {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000})
        estimated_cost = (usage_info['prompt_tokens'] * model_pricing["input"]) + (usage_info['completion_tokens'] * model_pricing["output"])
        print(f"💰 Estimated Cost: ${estimated_cost:.6f}")
        
        return transcribed_text, usage_info

    except Exception as e:
        print(f"❌ Error calling Anthropic Vision API: {e}")
        return None, None
    
def main():
    """
    Main function to process images Anthropic Vision.
    """
    # Ensure output directory exists
    os.makedirs(RESULTS_VISION_DIR, exist_ok=True)

    # Retrieve the API key from the environment variable
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the CLAUDE_API_KEY environment variable.")

    # Initialize OpenAI client once for reuse
    client = Anthropic(api_key=api_key)

    # Tell user which processing method is being used
    print("\n=== OCR Processing ===")
    print("Anthropic Vision API - Claude")

    # Get images from processed_imgs directory only
    processed_imgs_dir = "processed_imgs"
    if not os.path.exists(processed_imgs_dir):
        print(f"Error: {processed_imgs_dir} directory not found.")
        return
    
    # Get only images from processed_imgs directory
    available_images = [f for f in os.listdir(processed_imgs_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
    
    if not available_images:
        print(f"No images found in {processed_imgs_dir} directory.")
        return
    
    # Sort for consistent ordering
    available_images.sort()
    
    # Ask user which images to process
    print(f"\n=== Select Images to Process from {processed_imgs_dir} ===")
    print("0. Process ALL images")
    for idx, img in enumerate(available_images, 1):
        print(f"{idx}. {img}")
    
    while True:
        image_choice = input("\nEnter image number(s) (comma-separated, or 0 for all): ").strip()
        try:
            # Parse the input
            if image_choice == '0':
                selected_images = available_images
                break
            else:
                # Split by comma and convert to integers
                indices = [int(x.strip()) for x in image_choice.split(',')]
                # Validate indices
                if all(1 <= idx <= len(available_images) for idx in indices):
                    selected_images = [available_images[idx - 1] for idx in indices]
                    break
                else:
                    print(f"Invalid selection. Please enter numbers between 1 and {len(available_images)}, or 0 for all.")
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")
    
    print(f"\n✅ Selected {len(selected_images)} image(s) for processing from {processed_imgs_dir}")

    # Process images with Anthropic Vision
    print("\n=== Starting Anthropic Vision processing ===")
    
    for image_file in selected_images:
        image_path = resolve_image_path(PROCESSED_IMGS_DIR, image_file)
        if not image_path:
            print(f"❌ Could not find image with any common extension: {image_file}")
            continue
        
        print(f"\nProcessing: {image_file}")
        transcribed_text, usage_info = transcribe_with_vision_api(image_path, api_key)
        if transcribed_text:
            try:
                # Save the transcribed text to a .txt file
                output_file = os.path.join(RESULTS_VISION_DIR, f"{Path(image_file).stem}_vision.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(transcribed_text)
                print(f"✅ Saved Vision OCR text to: {output_file}")
            except Exception as e:
                print(f"❌ Error processing {image_file} with Vision API: {e}")
    
    print("\n=== Processing Complete ===")


if __name__ == "__main__":
    main()
