import google.generativeai as genai
from PIL import Image
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
except Exception as e:
    print(f"Error configuring Google AI SDK: {e}")


def extract_nutrition_from_image(image_path: str) -> dict | None:
    """
    Extracts nutrition information from a nutrition label image using Gemini Vision.

    Args:
        image_path: Path to the nutrition label image file.

    Returns:
        A dictionary containing the extracted nutrition data in the specified format,
        or None if an error occurs.
        Example format:
        {'protein': ('0', 'g'), 'sodium': ('0', 'mg'), ...}
    """
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    # Set up prompt
    json_structure_example = """
    {
      "nutrient_name_1": ["value", "unit"],
      "nutrient_name_2": ["value", "unit"]
      // ... more nutrients ...
    }
    Example: {
      "calories": ["100", ""],
      "total_fat": ["8", "g"],
      "sodium": ["160", "mg"]
    }
    """

    prompt = f"""
    Analyze the provided nutrition label image. Extract all key nutritional values present (e.g., Calories, Total Fat, Saturated Fat, Trans Fat, Cholesterol, Sodium, Total Carbohydrate, Dietary Fiber, Total Sugars, Protein, Vitamin D, Calcium, Iron, Potassium, etc.).
    Return the result STRICTLY as a JSON object. Do not include any text before or after the JSON object.
    The keys of the JSON object should be the nutrient names, formatted consistently (e.g., lowercase with underscores like "total_fat", "dietary_fiber").
    The values should be arrays containing two strings: the numerical value and its unit (e.g., ["10", "g"], ["150", "mg"]). If a nutrient has no unit (like Calories), use an empty string for the unit (e.g., ["100", ""]).
    If a value is not clearly present for a standard nutrient typically found on labels, you may omit it or represent it appropriately (e.g., ["0", "g"] if explicitly stated as zero).

    Use the following general format:
    {json_structure_example}

    Ensure the output is only the JSON object.
    """

    # Configure model
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-preview-04-17",
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json"
        ),
    )

    try:
        # Generate nutrition info from image
        print(f"Sending image {image_path} to Gemini...")
        response = model.generate_content([prompt, img])
        print("Received response from Gemini.")

        # Clean the response text
        cleaned_response_text = re.sub(
            r"^```json\s*|\s*```$", "", response.text.strip(), flags=re.MULTILINE
        )

        cleaned_response_text = re.sub(
            r'\(\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\)',
            r'["\1", "\2"]',
            cleaned_response_text,
        )

        # Parse the JSON response
        nutrition_data = json.loads(cleaned_response_text)
        print("Successfully parsed nutrition data.")
        nutrition_data = {k: tuple(v) for k, v in nutrition_data.items()}

        return nutrition_data

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON response: {e}")
        print(f"Raw response text: {response.text}")
        return None
    except Exception as e:
        print(f"An error occurred during Gemini API call or processing: {e}")
        if "response" in locals() and hasattr(response, "text"):
            print(f"Raw response text: {response.text}")
        elif "response" in locals() and hasattr(response, "prompt_feedback"):
            print(f"Prompt Feedback: {response.prompt_feedback}")
        return None


if __name__ == "__main__":
    test_image_path = "./llm/test1.png"

    if os.path.exists(test_image_path):
        extracted_data = extract_nutrition_from_image(test_image_path)
        if extracted_data:
            print("\nExtracted Nutrition Data:")
            print(extracted_data)
        else:
            print("\nFailed to extract nutrition data.")
    else:
        print(f"\nTest image not found at: {test_image_path}")
        print("Please update the 'test_image_path' variable in the script.")
