import os
import pytesseract
from openai import OpenAI
from PIL import Image
from pathlib import Path
import base64


# Define directories
PROCESSED_IMGS_DIR = "processed_imgs/"  # Directory for color images
RESULTS_VISION_DIR = "results/OpenAI"  # Directory for Tesseract + OpenAI correction results

def create_correction_prompt(ocr_text, document_type="historical document"):
    """
    Create a prompt for OpenAI to correct OCR text.
    
    Args:
        ocr_text (str): The raw OCR text
        document_type (str): Type of document being processed
        
    Returns:
        str: Formatted prompt for OpenAI
    """
    prompt = f"""Please correct the following OCR text from a {document_type}. 
The text may contain OCR errors, missing punctuation, or formatting issues.

IMPORTANT: You must process the ENTIRE document from beginning to end. Do not stop early or truncate the text.

Please:
1. Fix obvious OCR errors (like '0' instead of 'O', '1' instead of 'l', etc.)
2. Add appropriate punctuation and capitalization
3. Fix spacing and line breaks where needed
4. Preserve the original meaning and historical context
5. If a word is unclear, make your best guess based on context
6. Do not add any additional text to the original text
7. Do not delete any text from the original text unless you are sure it is an error and you are correcting a typo
8. Process EVERY line of the original text - do not skip any content

Original OCR text:
{ocr_text}

Corrected text:"""

    return prompt


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
    Transcribe text from an image using the OpenAI Vision API.

    Args:
        image_path (str): Path to the image file.
        api_key (str): OpenAI API key.

    Returns:
        tuple: (transcribed_text, usage_info)
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Resize and convert the image to JPEG
        resized_image_path = resize_and_convert_to_jpeg(image_path)

        # Encode the resized image to base64
        base64_image = encode_image(resized_image_path)
        if not base64_image:
            return None, None

        # Determine image format
        image_format = Path(resized_image_path).suffix.lower()
        if image_format not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            print(f"Warning: {image_format} may not be supported by Vision API")

        # Send request to OpenAI Vision API using the responses endpoint
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise transcription assistant specializing in historical documents from the women's liberation movement. Your task is to transcribe text exactly as it appears, without interpretation, correction, or addition."
                },
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
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format[1:]};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_completion_tokens=6000,
        )

        # Extract usage information
        usage = response.usage
        usage_info = {
            'prompt_tokens': usage.prompt_tokens,
            'completion_tokens': usage.completion_tokens,
            'total_tokens': usage.total_tokens
        }

        # Extract the transcribed text
        transcribed_text = response.choices[0].message.content.strip()
        
        # Print token usage and estimated cost
        print(f"🔹 Token Usage: Prompt = {usage_info['prompt_tokens']}, Completion = {usage_info['completion_tokens']}, Total = {usage_info['total_tokens']}")
        
        # Calculate cost (adjust rates based on current OpenAI pricing)
        # For gpt-4o: $5 per 1M input tokens, $15 per 1M output tokens (example rates)
        cost_per_input_token = 5.00 / 1_000_000
        cost_per_output_token = 15.00 / 1_000_000
        estimated_cost = (usage_info['prompt_tokens'] * cost_per_input_token) + (usage_info['completion_tokens'] * cost_per_output_token)
        print(f"💰 Estimated Cost: ${estimated_cost:.6f}")
        
        return transcribed_text, usage_info

    except Exception as e:
        print(f"❌ Error calling OpenAI Vision API: {e}")
        return None, None
    
def main():
    """
    Main function to process images OpenAI Vision.
    """
    # Ensure output directory exists
    os.makedirs(RESULTS_VISION_DIR, exist_ok=True)

    # Retrieve the API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

    # Initialize OpenAI client once for reuse
    client = OpenAI(api_key=api_key)

    # Tell user which processing method is being used
    print("\n=== OCR Processing ===")
    print("OpenAI Vision API (for color images)")

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

    # Process images with OpenAI Vision
    print("\n=== Starting OpenAI Vision processing ===")
    
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
